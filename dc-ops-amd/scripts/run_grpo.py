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
# 3) Standard library + everything else
# ---------------------------------------------------------------------------
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
