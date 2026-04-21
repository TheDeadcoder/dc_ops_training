#!/usr/bin/env python
# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""
GRPO training for DC-Ops on AMD Instinct MI300X (ROCm 7.2).

This script is STANDALONE — it can run on a fresh GPU machine after SFT:
    1. Resolves the SFT source: prefers `sft_model_local` if the dir exists,
       else falls back to `sft_model_hub`.
    2. Recovers the rewritten system prompt: prefers a `system_prompt.txt`
       saved alongside the LoRA, else re-derives it from the raw HF dataset
       using the same rewriter SFT used. Either path produces a byte-identical
       string — the model doesn't see a distribution shift at RL time.
    3. Builds the GRPO prompt dataset from the live DC-Ops env (deterministic
       given the same seed config).
    4. Loads the model via Unsloth + vLLM fast-inference.
    5. Trains with the 4 reward functions, logging to wandb.

Usage:
    python scripts/run_grpo.py --config configs/grpo.yaml
    # or in background:
    ./launch/grpo.sh

Notes:
    - GRPO with vLLM is ~10–20× faster than plain `.generate()`. Keep
      `vllm.enabled=true` unless you have a specific reason not to.
    - If you OOM at init, drop `vllm.gpu_memory_utilization` from 0.75 → 0.65.

Import order matters:
    1. src.rocm_env          (HIP env vars before any GPU touch)
    2. unsloth.PatchFastRL   (must run BEFORE trl imports — this is what
                              makes max_prompt_length a valid GRPOTrainer kwarg
                              in the Unsloth-patched code path)
    3. unsloth.FastLanguageModel
    4. torch / trl / peft / etc.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1) ROCm env vars — MUST be set before any library touches the GPU.
# ---------------------------------------------------------------------------
import sys
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src import rocm_env  # noqa: F401  (applies env vars on import)

# ---------------------------------------------------------------------------
# 2) Unsloth — MUST be imported before trl/transformers/peft. PatchFastRL
#    MUST be called before importing GRPOTrainer (notebook cell 40 pattern).
# ---------------------------------------------------------------------------
from unsloth import FastLanguageModel, PatchFastRL  # noqa: E402
PatchFastRL("GRPO", FastLanguageModel)

# ---------------------------------------------------------------------------
# 2b) vLLM ROCm platform fix — MUST run after unsloth import, before LLM().
# ---------------------------------------------------------------------------
# Root cause: vllm.platforms initialises its `current_platform` singleton the
# instant vllm is first imported (which happens inside the unsloth import
# above). On this ROCm build the HIP-compat resolver doesn't fire at that
# moment, so the singleton lands on UnspecifiedPlatform (device_type="").
#
# Setting VLLM_TARGET_DEVICE=cuda in the environment doesn't help because
# vLLM 0.19.1+rocm721 doesn't consult that env-var for platform selection.
#
# The fix: directly inject the correct platform object into every module that
# has already captured it via `from vllm.platforms import current_platform`
# (a Python `from X import y` creates a separate name binding — replacing the
# module attribute alone never reaches those already-bound references).
def _fix_vllm_rocm_platform() -> None:
    """
    Replace vLLM's current_platform singleton with _MI300XPlatform — a subclass
    of RocmPlatform that:

      1. Overrides every known stub that raises NotImplementedError or returns
         None where vLLM expects a real value (check_if_supports_dtype,
         mem_get_info, get_punica_wrapper).
      2. Has a __getattribute__ safety-net: if ANY public method raises
         NotImplementedError at runtime, it is automatically retried on a
         CudaPlatform instance.  MI300X speaks CUDA-compat (HIP) natively,
         so CudaPlatform methods are always correct fallbacks.  This prevents
         future whack-a-mole iterations if more stubs surface later in the
         vLLM init path.
      3. Patches the new singleton into every module that captured
         current_platform via `from vllm.platforms import current_platform`
         (a Python name-binding, not a live reference to the module attribute).
    """
    import importlib
    import torch as _torch
    from vllm.lora.ops.triton_ops.lora_shrink_op import _lora_shrink as _triton_shrink
    try:
        _torch.library.impl("vllm::lora_shrink", "CUDA")(_triton_shrink)
    except RuntimeError:
        pass
    import vllm.platforms as _vp

    # ------------------------------------------------------------------
    # Build the platform bases
    # ------------------------------------------------------------------
    try:
        from vllm.platforms.rocm import RocmPlatform as _RocmBase
    except ImportError:
        from vllm.platforms.cuda import CudaPlatform as _RocmBase  # type: ignore[assignment]

    try:
        from vllm.platforms.cuda import CudaPlatform as _CudaBase
        _cuda_fallback = _CudaBase()
    except Exception:
        _cuda_fallback = None

    _SUPPORTED_DTYPES = {_torch.float16, _torch.bfloat16, _torch.float32}

    # ------------------------------------------------------------------
    # The patched platform class
    # ------------------------------------------------------------------
    class _MI300XPlatform(_RocmBase):
        """
        RocmPlatform with CUDA-compat shims for every unimplemented stub.
        Safe to use as a drop-in current_platform on MI300X / gfx942.
        """

        # ---- explicitly known broken stubs ---------------------------

        def check_if_supports_dtype(self, dtype: "_torch.dtype") -> None:
            """gpu_worker.init_device() — base raises NotImplementedError."""
            if dtype not in _SUPPORTED_DTYPES:
                raise ValueError(
                    f"MI300X (ROCm) does not support dtype {dtype}. "
                    f"Supported: {_SUPPORTED_DTYPES}"
                )

        def mem_get_info(self, device: "_torch.device") -> "tuple[int, int]":
            """mem_utils.MemorySnapshot — base returns None (not callable)."""
            return _torch.cuda.mem_get_info(device)

        def get_punica_wrapper(self) -> str:
            """punica_selector — base raises NotImplementedError.

            PunicaWrapperGPU uses custom `vllm::lora_shrink` C++/CUDA ops
            that are NOT compiled into the ROCm wheel.  On ROCm, vLLM ships
            Triton kernels for LoRA ops instead.  Prefer PunicaWrapperTriton;
            fall back to GPU wrapper only if the triton module is absent.
            """
            _triton_qualname = (
                "vllm.lora.punica_wrapper.punica_triton.PunicaWrapperTriton"
            )
            _gpu_qualname = (
                "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"
            )
            try:
                import importlib as _il
                _il.import_module("vllm.lora.punica_wrapper.punica_triton")
                return _triton_qualname
            except ImportError:
                return _gpu_qualname

        # ---- safety-net for any future unimplemented stubs -----------

        def __getattribute__(self, name: str):  # type: ignore[override]
            attr = super().__getattribute__(name)
            # Only intercept public, callable, non-dunder methods.
            if name.startswith("_") or not callable(attr):
                return attr

            def _with_cuda_fallback(*args, **kwargs):
                try:
                    return attr(*args, **kwargs)
                except NotImplementedError:
                    if _cuda_fallback is not None:
                        cuda_method = getattr(_cuda_fallback, name, None)
                        if cuda_method is not None:
                            return cuda_method(*args, **kwargs)
                    raise  # propagate if CUDA fallback also can't help

            return _with_cuda_fallback

    # ------------------------------------------------------------------
    # Install the singleton
    # ------------------------------------------------------------------
    _new = _MI300XPlatform()
    _vp.current_platform = _new

    # Patch every module that captured current_platform by name.
    # `from vllm.platforms import current_platform` creates an independent
    # name binding — updating the module attribute alone never reaches it.
    _targets = [
        "vllm.engine.arg_utils",
        "vllm.v1.engine.llm_engine",
        "vllm.v1.engine.async_llm",
        "vllm.v1.executor.gpu_executor",
        "vllm.v1.executor.abstract_executor",
        "vllm.v1.worker.gpu_worker",                  # check_if_supports_dtype
        "vllm.v1.worker.worker_base",
        "vllm.utils.mem_utils",                       # mem_get_info (MemorySnapshot)
        "vllm.lora.punica_wrapper.punica_selector",   # get_punica_wrapper
        "vllm.config.device",
    ]
    for _mod_name in _targets:
        try:
            _mod = importlib.import_module(_mod_name)
            if hasattr(_mod, "current_platform"):
                setattr(_mod, "current_platform", _new)
        except ImportError:
            pass  # module doesn't exist in this vLLM build — skip silently

    _dt = getattr(_new, "device_type", "?")
    print(f"[grpo] vLLM platform: _MI300XPlatform installed "
          f"(device_type='{_dt}', CUDA-fallback safety-net active)")

    # ------------------------------------------------------------------
    # Force PunicaWrapperTriton in punica_selector.
    #
    # Why: punica_selector.py selects PunicaWrapperGPU if it detects the
    # C++ ops registered (they are — for CPU/Meta only, NOT HIP/CUDA).
    # PunicaWrapperGPU then calls torch.ops.vllm.lora_shrink on the CUDA
    # backend, which raises NotImplementedError at profile_run time.
    #
    # PunicaWrapperTriton uses Triton JIT kernels that compile natively
    # on ROCm gfx942 and do not depend on the missing C++ op at all.
    #
    # We patch punica_selector.PunicaWrapper here (before LLM() is ever
    # called) so vllm instantiates PunicaWrapperTriton from the start.
    # ------------------------------------------------------------------
    try:
        import vllm.lora.punica_wrapper.punica_triton as _ptriton
        import vllm.lora.punica_wrapper.punica_selector as _psel
        _psel.PunicaWrapper = _ptriton.PunicaWrapperTriton
        # Also patch any sibling modules that imported PunicaWrapper directly.
        for _lmod in [
            "vllm.lora.layers.column_parallel_linear",
            "vllm.lora.layers.base_linear",
            "vllm.lora.worker_manager",
            "vllm.lora.models",
        ]:
            try:
                _lm = importlib.import_module(_lmod)
                if hasattr(_lm, "PunicaWrapper"):
                    _lm.PunicaWrapper = _ptriton.PunicaWrapperTriton
            except ImportError:
                pass
        print("[grpo] punica_selector.PunicaWrapper → PunicaWrapperTriton (ROCm override)")
    except ImportError as _e:
        print(f"[grpo] WARNING: could not redirect to PunicaWrapperTriton ({_e}); "
              f"falling back to PunicaWrapperGPU + PyTorch BMM patch")

_fix_vllm_rocm_platform()
del _fix_vllm_rocm_platform

# ---------------------------------------------------------------------------
# 2c) Patch punica LoRA ops for ROCm — safety-net only.
# ---------------------------------------------------------------------------
# Primary fix: _fix_vllm_rocm_platform() above already redirected
# punica_selector.PunicaWrapper → PunicaWrapperTriton (Triton JIT kernels
# that work on ROCm gfx942 without needing the C++ lora_shrink op).
#
# This block is a belt-and-suspenders fallback: if PunicaWrapperTriton was
# unavailable (ImportError) AND punica_selector fell back to PunicaWrapperGPU,
# we replace lora_shrink/lora_expand in punica_gpu's namespace with pure-
# PyTorch BMM implementations so that the NotImplementedError from the
# missing HIP backend is caught and handled gracefully.
#
# Confirmed: vllm::lora_shrink registers only CPU + Meta dispatch keys on
# this build — no HIP/CUDA kernel is present (see NotImplementedError in
# run log). PunicaWrapperGPU calling it would always fail on GPU.
def _patch_punica_lora_for_rocm() -> None:
    import torch as _torch

    try:
        import vllm.lora.punica_wrapper.punica_gpu as _pg
    except ImportError:
        return  # not available in this build

    _orig_shrink = getattr(_pg, "lora_shrink", None)
    _orig_expand = getattr(_pg, "lora_expand", None)

    def _pt_lora_shrink(*args, **kwargs):
        """
        Pure-PyTorch vllm::lora_shrink fallback for ROCm.

        Handles two calling conventions without relying on hardcoded
        positional indices (the layout changed across vllm builds):

          old (vllm ≤ 0.18, keyword-only):
              (inputs, lora_a_stacked, *, indices, out, scale)
          new (vllm 0.19.1+, positional, exact layout varies):
              args contain: 2-D input tensor, 3-D lora weight tensor(s),
              1-D integer index tensor, 2-D output tensor, float scale.
              We recover these by type + shape inspection.
        """
        if "indices" in kwargs:
            # ---- old keyword-only convention (vllm ≤ 0.18) ----
            inputs         = args[0]
            lora_a_stacked = args[1]
            indices = kwargs["indices"]
            out     = kwargs["out"]
            scale   = float(kwargs.get("scale", kwargs.get("scaling", 1.0)))
        else:
            # ---- vllm 0.19.1+ positional: recover by type/shape ----
            # Step 1: scaling scalar — first Python float OR 0/1-element tensor
            scale = 1.0
            for _a in args:
                if isinstance(_a, float):
                    scale = _a
                    break
                if isinstance(_a, _torch.Tensor) and _a.numel() == 1:
                    scale = _a.item()
                    break

            # Step 2: classify tensors by rank
            _td  = [_a for _a in args if isinstance(_a, _torch.Tensor)]
            _nd1 = [_a for _a in _td if _a.ndim == 1]            # candidates: indices
            _nd2 = [_a for _a in _td if _a.ndim == 2]            # candidates: inputs / out
            _nd3p= [_a for _a in _td if _a.ndim >= 3]            # candidates: lora_a weights

            if not _nd3p or not _nd1 or not _nd2:
                _shapes = [
                    (f"T{list(_a.shape)}" if isinstance(_a, _torch.Tensor)
                     else type(_a).__name__)
                    for _a in args
                ]
                print(f"[grpo][ERROR] _pt_lora_shrink: cannot identify args. "
                      f"Shapes: {_shapes}. Skipping lora contribution.")
                return

            # lora weights: all 3-D+ tensors
            lora_a_stacked = _nd3p

            # indices: first 1-D integer tensor
            indices = next(
                (_a for _a in _nd1 if not _a.is_floating_point()),
                _nd1[0],   # fallback to first 1-D tensor if all float
            )

            # inputs: 2-D tensor whose last dim matches lora hidden dim
            _h = lora_a_stacked[0].shape[-1]
            inputs = next((_a for _a in _nd2 if _a.shape[-1] == _h), _nd2[0])

            # out: 2-D tensor whose last dim equals the total rank across loras
            _r_total = sum(_la.shape[1] for _la in lora_a_stacked)
            out = next(
                (_a for _a in _nd2 if _a is not inputs and _a.shape[-1] == _r_total),
                next((_a for _a in _nd2 if _a is not inputs), _nd2[0]),
            )

        # ---- shared implementation ----
        _flat_idx = indices.view(-1)
        _valid    = _torch.where(_flat_idx >= 0)[0]
        if _valid.numel() == 0:
            return
        _idx = _flat_idx[_valid].long()
        _inp = inputs.view(-1, inputs.shape[-1])[_valid]     # [V, H]
        _out_flat = out.view(-1, out.shape[-1])               # [T, R_total]
        _r_off = 0
        for _la in lora_a_stacked:
            _R  = _la.shape[1]
            _wa = _la[_idx]                                   # [V, R, H]
            _out_flat[_valid, _r_off:_r_off + _R].add_(
                _torch.bmm(_inp.unsqueeze(1), _wa).squeeze(1).mul_(scale)
            )
            _r_off += _R

    def _pt_lora_expand(*args, **kwargs):
        """
        Pure-PyTorch vllm::lora_expand fallback for ROCm.

        Same dual-convention handling as _pt_lora_shrink above.
          old: (inputs, lora_b_stacked, *, indices, out, scale,
                offset_start=0, add_inputs=True, embed_indices=None)
          new: positional, layout inferred by type/shape.
        """
        if "indices" in kwargs:
            # ---- old keyword-only convention (vllm ≤ 0.18) ----
            inputs         = args[0]
            lora_b_stacked = args[1]
            indices      = kwargs["indices"]
            out          = kwargs["out"]
            scale        = float(kwargs.get("scale", kwargs.get("scaling", 1.0)))
            offset_start = int(kwargs.get("offset_start", 0))
        else:
            # ---- vllm 0.19.1+ positional: recover by type/shape ----
            scale = 1.0
            for _a in args:
                if isinstance(_a, float):
                    scale = _a
                    break
                if isinstance(_a, _torch.Tensor) and _a.numel() == 1:
                    scale = _a.item()
                    break

            _td  = [_a for _a in args if isinstance(_a, _torch.Tensor)]
            _nd1 = [_a for _a in _td if _a.ndim == 1]
            _nd2 = [_a for _a in _td if _a.ndim == 2]
            _nd3p= [_a for _a in _td if _a.ndim >= 3]

            if not _nd3p or not _nd1 or not _nd2:
                _shapes = [
                    (f"T{list(_a.shape)}" if isinstance(_a, _torch.Tensor)
                     else type(_a).__name__)
                    for _a in args
                ]
                print(f"[grpo][ERROR] _pt_lora_expand: cannot identify args. "
                      f"Shapes: {_shapes}. Skipping lora contribution.")
                return

            lora_b_stacked = _nd3p

            indices = next(
                (_a for _a in _nd1 if not _a.is_floating_point()),
                _nd1[0],
            )

            # expand: inputs have shape [T, R] (rank), out has shape [T, O] (output)
            _r = lora_b_stacked[0].shape[-1]   # lora_b shape: [L, O, R]
            inputs = next((_a for _a in _nd2 if _a.shape[-1] == _r), _nd2[0])

            _o_total = sum(_lb.shape[1] for _lb in lora_b_stacked)
            out = next(
                (_a for _a in _nd2 if _a is not inputs and _a.shape[-1] == _o_total),
                next((_a for _a in _nd2 if _a is not inputs), _nd2[0]),
            )

            # offset_start: first int arg that isn't a likely batch/seq/token count
            # (best-effort; defaults to 0 which is correct for most layers)
            offset_start = 0

        # ---- shared implementation ----
        _flat_idx = indices.view(-1)
        _valid    = _torch.where(_flat_idx >= 0)[0]
        if _valid.numel() == 0:
            return
        _idx = _flat_idx[_valid].long()
        _inp = inputs.view(-1, inputs.shape[-1])[_valid]       # [V, R]
        _out_flat = out.view(-1, out.shape[-1])                 # [T, O_total]
        _o_off = offset_start
        for _lb in lora_b_stacked:
            _O  = _lb.shape[1]
            _wb = _lb[_idx]                                     # [V, O, R]
            _out_flat[_valid, _o_off:_o_off + _O].add_(
                _torch.bmm(_wb, _inp.unsqueeze(-1)).squeeze(-1).mul_(scale)
            )
            _o_off += _O

    def _safe_shrink(*args, **kwargs):
        # ---------------------------------------------------------------
        # BUG FIX (vllm 0.19.1+rocm721): the old fixed signature
        #   (inputs, lora_a_stacked, *, indices, out, scale)
        # only accepted 2 positional args.  vllm 0.19.1 calls lora_shrink
        # with 11 positional args, so Python raised:
        #   TypeError: takes 2 positional arguments but 11 were given
        # — BEFORE the try-block body ran, so the except never fired.
        # Using *args/**kwargs makes the wrapper transparent to any
        # calling convention and lets the original C++ op (which IS
        # compiled in this ROCm build) handle the call directly.
        # ---------------------------------------------------------------
        try:
            _orig_shrink(*args, **kwargs)
        except (NotImplementedError, RuntimeError):
            _pt_lora_shrink(*args, **kwargs)

    def _safe_expand(*args, **kwargs):
        try:
            _orig_expand(*args, **kwargs)
        except (NotImplementedError, RuntimeError):
            _pt_lora_expand(*args, **kwargs)

    patched = []
    if _orig_shrink is not None:
        _pg.lora_shrink = _safe_shrink
        patched.append("lora_shrink")
    if _orig_expand is not None:
        _pg.lora_expand = _safe_expand
        patched.append("lora_expand")

    if patched:
        print(f"[grpo] punica_gpu ROCm patch: {patched} → pure-PyTorch BMM fallbacks")


_patch_punica_lora_for_rocm()
del _patch_punica_lora_for_rocm


import argparse                            # noqa: E402
import os                                  # noqa: E402
from collections import Counter            # noqa: E402
from pprint import pformat                 # noqa: E402

import torch                               # noqa: E402
import yaml                                # noqa: E402


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DC-Ops GRPO trainer (ROCm 7.2 / MI300X)")
    p.add_argument("--config", type=str, default="configs/grpo.yaml")
    p.add_argument("--dry-run", action="store_true",
                   help="Build everything, skip trainer.train()")
    return p.parse_args()


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_sft_source(mcfg: dict) -> str:
    """Prefer local dir if it exists, else the Hub id. Raise if neither."""
    local = mcfg.get("sft_model_local")
    hub   = mcfg.get("sft_model_hub")
    if local and os.path.isdir(local) and os.path.exists(
        os.path.join(local, "adapter_config.json")
    ):
        print(f"[grpo] using LOCAL SFT adapter at {local}")
        return local
    if hub:
        print(f"[grpo] using HUB SFT adapter: {hub}")
        return hub
    raise RuntimeError("No SFT source configured: set sft_model_local or sft_model_hub.")


def _get_system_prompt(sft_source: str, data_cfg: dict) -> str:
    """Recover the rewritten system prompt.

    Preference order:
      1. system_prompt.txt saved alongside the SFT LoRA (run_sft.py writes this).
      2. Rebuild from the HF dataset (or local jsonl) using src.prompts.
    """
    if os.path.isdir(sft_source):
        p = os.path.join(sft_source, "system_prompt.txt")
        if os.path.exists(p):
            with open(p) as f:
                content = f.read()
            print(f"[grpo] system prompt loaded from {p} ({len(content):,} chars)")
            return content

    print("[grpo] system_prompt.txt not found alongside LoRA — re-deriving from raw dataset")
    from datasets import load_dataset
    from src.prompts import rewrite_system_prompt

    if data_cfg.get("local_jsonl"):
        ds = load_dataset("json", data_files={"train": data_cfg["local_jsonl"]}, split="train")
    else:
        ds = load_dataset(
            "json",
            data_files={"train": f"hf://datasets/{data_cfg['hf_source']}/train.jsonl"},
            split="train",
        )
    return rewrite_system_prompt(ds[0]["conversations"][0]["value"])


# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    print("[grpo] config:\n" + pformat(cfg))

    # -------- wandb -----------------------------------------------------
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled"):
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError(
                "wandb.enabled=true but WANDB_API_KEY is not set. "
                "Export it or set wandb.enabled=false."
            )
        os.environ["WANDB_PROJECT"] = wandb_cfg.get("project", "dc-ops-amd")
        if wandb_cfg.get("run_name"):
            os.environ["WANDB_NAME"] = wandb_cfg["run_name"]
        if wandb_cfg.get("tags"):
            os.environ["WANDB_TAGS"] = ",".join(wandb_cfg["tags"])
        report_to = ["wandb"]
    else:
        report_to = "none"

    # -------- GPU sanity ------------------------------------------------
    assert torch.cuda.is_available(), "torch.cuda (ROCm) is not available"
    print(f"[grpo] device: {torch.cuda.get_device_name(0)}  |  "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # -------- HF login (if loading SFT LoRA from the Hub) --------------
    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        print("[grpo] logged in to HuggingFace Hub")

    # -------- Peft TP-sharding patch (notebook cell 40) ----------------
    # Required when loading a LoRA into Unsloth's vLLM-fast-inference path —
    # peft inspects torch.distributed.is_initialized() inside set_peft_model_state_dict
    # and fights with vLLM's tensor-parallel init. We trick peft into thinking
    # we're not in a distributed setting just for the duration of the load.
    import peft.utils.save_and_load as _sal
    _orig_set_peft = _sal.set_peft_model_state_dict

    def _patched_set_peft(model, state_dict, adapter_name="default", **kwargs):
        _orig_is_init = torch.distributed.is_initialized
        torch.distributed.is_initialized = lambda: False
        try:
            return _orig_set_peft(model, state_dict, adapter_name=adapter_name, **kwargs)
        finally:
            torch.distributed.is_initialized = _orig_is_init

    _sal.set_peft_model_state_dict = _patched_set_peft
    print("[grpo] applied peft TP-sharding patch")

    # -------- Reload SFT model with vLLM fast-inference -----------------
    mcfg = cfg["model"]
    vcfg = cfg["vllm"]
    sft_source = _resolve_sft_source(mcfg)

    print(f"[grpo] loading base+LoRA via Unsloth (fast_inference={vcfg['enabled']}, "
          f"gpu_memory_utilization={vcfg['gpu_memory_utilization']})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=sft_source,
        max_seq_length=mcfg["max_seq_length"],
        load_in_4bit=mcfg["load_in_4bit"],
        fast_inference=vcfg["enabled"],
        enforce_eager=vcfg.get("enforce_eager", True),
        max_lora_rank=mcfg["max_lora_rank"],
        gpu_memory_utilization=vcfg["gpu_memory_utilization"],
    )
    FastLanguageModel.for_training(model)
    print(f"[grpo] VRAM after model+vLLM init: "
          f"{torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

    # -------- System prompt + GRPO prompt dataset -----------------------
    dcfg = cfg["data"]
    system_prompt = _get_system_prompt(sft_source, dcfg)

    from src.grpo_data import build_grpo_prompts

    grpo_ds = build_grpo_prompts(
        tokenizer,
        system_prompt,
        num_initial=dcfg["num_initial_prompts"],
        num_midgame=dcfg["num_midgame_prompts"],
        seed=dcfg["seed"],
    )
    print(f"[grpo] built {len(grpo_ds):,} prompts")
    dist = Counter(grpo_ds["scenario_id"])
    for sid in sorted(dist.keys()):
        print(f"[grpo]   scenario {sid}: {dist[sid]} prompts")

    # Sanity: no <think> leaked into the chat template
    assert "<think>" not in grpo_ds[0]["prompt"], "system prompt still contains <think>!"
    n_with_warmup = sum(1 for r in grpo_ds if r["warmup_actions"])
    print(f"[grpo] mid-game prompts (carry warmup_actions): {n_with_warmup}")

    # -------- Reward functions ------------------------------------------
    from src.rewards import ALL_REWARD_FNS
    print(f"[grpo] using {len(ALL_REWARD_FNS)} reward functions")

    # -------- Build TRL GRPOConfig --------------------------------------
    from trl import GRPOConfig, GRPOTrainer

    gcfg = cfg["grpo"]
    _per_device = gcfg["per_device_train_batch_size"]
    _grad_accum = gcfg["gradient_accumulation_steps"]
    _num_gen    = gcfg["num_generations"]
    _max_compl  = gcfg["max_completion_length"]
    _max_prompt = gcfg["max_prompt_length"]

    # TRL constraint check — fail fast with a useful message
    eff_batch = _per_device * _grad_accum
    if eff_batch % _num_gen != 0:
        raise ValueError(
            f"TRL GRPO requires per_device_batch × grad_accum "
            f"({_per_device}×{_grad_accum}={eff_batch}) to be divisible by "
            f"num_generations ({_num_gen}). Fix configs/grpo.yaml."
        )

    # Unsloth vLLM sampling params (overrides generation_kwargs when set)
    vllm_sampling_params = None
    if vcfg["enabled"]:
        try:
            from unsloth import vLLMSamplingParams
            vllm_sampling_params = vLLMSamplingParams(
                temperature=vcfg["temperature"],
                top_p=vcfg["top_p"],
                max_tokens=_max_compl,
            )
            print("[grpo] vLLMSamplingParams configured")
        except ImportError:
            print("[grpo] vLLMSamplingParams not available — falling back to generation_kwargs")

    # Clear the model's stale generation-config max_length so TRL's padding
    # logic doesn't try to enforce the wrong cap.
    model.generation_config.max_length = None

    grpo_config = GRPOConfig(
        output_dir=gcfg["output_dir"],
        num_generations=_num_gen,
        per_device_train_batch_size=_per_device,
        gradient_accumulation_steps=_grad_accum,
        num_train_epochs=gcfg["num_train_epochs"],
        learning_rate=gcfg["learning_rate"],
        lr_scheduler_type=gcfg["lr_scheduler_type"],
        warmup_ratio=gcfg["warmup_ratio"],
        beta=gcfg["beta"],
        bf16=gcfg["bf16"],
        fp16=gcfg["fp16"],
        max_grad_norm=gcfg["max_grad_norm"],
        dataloader_num_workers=gcfg["dataloader_num_workers"],
        logging_steps=gcfg["logging_steps"],
        save_strategy=gcfg["save_strategy"],
        save_steps=gcfg["save_steps"],
        save_total_limit=gcfg["save_total_limit"],
        report_to=report_to,
        seed=gcfg["seed"],
        run_name=wandb_cfg.get("run_name"),
        generation_kwargs={
            "max_new_tokens": _max_compl,
            "temperature":    vcfg["temperature"],
            "do_sample":      True,
            "top_p":          vcfg["top_p"],
        },
    )
    if vllm_sampling_params is not None:
        grpo_config.vllm_sampling_params = vllm_sampling_params

    # Notebook cell 42 passes max_prompt_length / max_completion_length to
    # the GRPOTrainer constructor (these are kwargs accepted by the
    # PatchFastRL-patched trainer, alongside what's in GRPOConfig).
    grpo_trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=ALL_REWARD_FNS,
        args=grpo_config,
        train_dataset=grpo_ds,
        max_prompt_length=_max_prompt,
        max_completion_length=_max_compl,
    )

    # Hotfix a known Unsloth+TRL attribute gap (notebook cell 42 does this too)
    if not hasattr(grpo_trainer, "current_gradient_accumulation_steps"):
        grpo_trainer.current_gradient_accumulation_steps = _grad_accum
        print("[grpo] hotfixed missing 'current_gradient_accumulation_steps' on UnslothGRPOTrainer")

    # ------- Print the canonical config block (notebook cell 42 style) -
    print(f"[grpo] rollout config:")
    print(f"[grpo]   num_generations:       {_num_gen}")
    print(f"[grpo]   max_prompt_length:     {_max_prompt}")
    print(f"[grpo]   max_completion_length: {_max_compl}")
    print(f"[grpo]   temperature:           {vcfg['temperature']}")
    print(f"[grpo]   gpu_memory_util:       {vcfg['gpu_memory_utilization']}")
    print(f"[grpo]   per_device_batch:      {_per_device}")
    print(f"[grpo]   gradient_accum:        {_grad_accum}")
    print(f"[grpo]   effective_batch:       {eff_batch}  ({eff_batch // _num_gen} prompt(s) × {_num_gen} completions)")
    print(f"[grpo]   learning_rate:         {gcfg['learning_rate']}")
    print(f"[grpo]   beta (KL):             {gcfg['beta']}")
    print(f"[grpo]   max_grad_norm:         {gcfg['max_grad_norm']}")
    print(f"[grpo]   epochs:                {gcfg['num_train_epochs']}")
    print(f"[grpo]   reward fns:            format / env / command_quality / no_repeat")

    if args.dry_run:
        print("[grpo] --dry-run: skipping trainer.train()")
        return

    # -------- Train ------------------------------------------------------
    print("[grpo] starting training …")
    stats = grpo_trainer.train()
    print(f"[grpo] DONE. runtime = {stats.metrics['train_runtime']:.0f} s "
          f"({stats.metrics['train_runtime']/60:.1f} min)")

    # -------- Save final -------------------------------------------------
    save_dir = cfg["save"]["local_dir"]
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "system_prompt.txt"), "w") as f:
        f.write(system_prompt)
    print(f"[grpo] final LoRA + tokenizer + system_prompt saved → {save_dir}")


if __name__ == "__main__":
    main()