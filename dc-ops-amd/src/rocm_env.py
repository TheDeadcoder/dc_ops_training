# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""
ROCm 7.2 / MI300X runtime environment setup.

Import this module FIRST in every training script (before torch / vllm /
transformers) so the env vars land in the process before any CUDA/ROCm
symbol is touched.

What this is NOT:
    - No libnvJitLink preloading (that is NVIDIA-only).
    - No CUDA_DEVICE_MAX_CONNECTIONS / NCCL_P2P_DISABLE hacks from the
      original notebook; RCCL on a single MI300X doesn't need them.
    - No HSA_OVERRIDE_GFX_VERSION; MI300X is natively gfx942.
"""

from __future__ import annotations

import os


def _setdefault(key: str, value: str) -> None:
    """Set env var only if not already set by the user."""
    os.environ.setdefault(key, value)


def apply_rocm_env() -> None:
    """Apply all MI300X training env vars. Idempotent."""
    # --- GPU visibility (ROCm uses HIP_VISIBLE_DEVICES; both work) ---
    # We don't force to 0 because the user may want to pick a specific slot;
    # we only default if completely unset.
    _setdefault("HIP_VISIBLE_DEVICES", "0")

    # --- MI300X performance knobs (documented by AMD) ---
    # Fast kernel-arg path: eliminates a host-side copy, ~5-10% win on LLM training.
    _setdefault("HIP_FORCE_DEV_KERNARG", "1")
    # Prefer hipBLASLt over rocBLAS when both implement a kernel — better for
    # bf16 GEMMs on gfx942.
    _setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")
    # MIOpen tuning: enable auto-tune on new tensor shapes, cache in the
    # user database so subsequent runs hit warm kernels.
    _setdefault("MIOPEN_FIND_MODE", "3")
    _setdefault("MIOPEN_FIND_ENFORCE", "3")

    # --- PyTorch allocator (expandable segments reduces OOMs from
    # fragmentation during long GRPO runs; honoured on ROCm too) ---
    _setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

    # --- vLLM on ROCm ---
    # ROCm doesn't support fork(); vLLM must use spawn.
    _setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # Prefer the CK (Composable Kernel) flash-attn kernels over Triton on
    # MI300X — measurably faster for the head dims we use.
    _setdefault("VLLM_USE_TRITON_FLASH_ATTN", "0")
    # Unsloth-vLLM path: FlashInfer is NV-only, must stay disabled.
    _setdefault("UNSLOTH_VLLM_NO_FLASHINFER", "1")

    # --- General training hygiene ---
    _setdefault("TOKENIZERS_PARALLELISM", "false")
    _setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    _setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    _setdefault("TRANSFORMERS_VERBOSITY", "error")
    _setdefault("OMP_NUM_THREADS", "4")  # stop CPU oversubscription

    # --- RCCL (ROCm's NCCL-compat lib) — only matters for multi-GPU,
    # but these flags are safe on single-GPU too ---
    _setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    _setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")


# Apply on import — this is the whole point of the module existing.
apply_rocm_env()
