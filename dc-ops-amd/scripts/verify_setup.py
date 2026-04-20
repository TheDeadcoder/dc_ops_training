#!/usr/bin/env python
# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""
Post-install sanity check for the ROCm 7.2 / MI300X training stack.

Run this once after `./setup_env.sh` to catch broken installs *before* you
commit to a 3-hour training job.

Usage:
    source .venv/bin/activate
    python scripts/verify_setup.py

Exits 0 if everything that's required works, 1 if any required check fails.
Flash-Attn is treated as optional — training still works without it, it's just
slower.
"""

from __future__ import annotations

import sys
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src import rocm_env  # noqa: F401


# ---------- helpers ----------
GREEN = "\033[0;32m"; RED = "\033[0;31m"; YELLOW = "\033[1;33m"; NC = "\033[0m"

def ok(msg):   print(f"{GREEN}[ ok ]{NC} {msg}")
def fail(msg): print(f"{RED}[fail]{NC} {msg}")
def warn(msg): print(f"{YELLOW}[warn]{NC} {msg}")


def main() -> int:
    required_ok = True
    warnings = 0

    # 1) torch + ROCm ------------------------------------------------------
    try:
        import torch
        ok(f"torch {torch.__version__}  (HIP {torch.version.hip})")
        if not torch.cuda.is_available():
            fail("torch.cuda.is_available() is False — ROCm not detected")
            required_ok = False
        else:
            name  = torch.cuda.get_device_name(0)
            vram  = torch.cuda.get_device_properties(0).total_memory / 1e9
            bf16  = torch.cuda.is_bf16_supported()
            ok(f"GPU: {name}  |  VRAM {vram:.1f} GB  |  bf16={bf16}")
            if "MI300X" not in name:
                warn(f"expected MI300X, got {name!r}")
                warnings += 1
            if vram < 180:
                warn(f"VRAM {vram:.0f} GB < 180 GB — config tuned for 192 GB may OOM")
                warnings += 1
    except Exception as e:
        fail(f"torch import failed: {e}")
        return 1

    # 2) transformers + tokenizers + peft + trl + accelerate ---------------
    for pkg, required in [
        ("transformers", True),
        ("tokenizers",   True),
        ("peft",         True),
        ("trl",          True),
        ("accelerate",   True),
        ("datasets",     True),
    ]:
        try:
            mod = __import__(pkg)
            ok(f"{pkg} {getattr(mod, '__version__', '?')}")
        except Exception as e:
            (fail if required else warn)(f"{pkg} import failed: {e}")
            if required: required_ok = False
            else: warnings += 1

    # 3) bitsandbytes (required for 4-bit QLoRA on AMD) --------------------
    try:
        import bitsandbytes as bnb
        ok(f"bitsandbytes {bnb.__version__}  (ROCm fork)")
        # try a tiny 4-bit op
        x = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        from bitsandbytes.nn import Linear4bit
        lin = Linear4bit(16, 32, bias=False, quant_type="nf4").cuda().to(torch.bfloat16)
        y = lin(x)  # forward-only smoke test
        assert y.shape == (8, 32)
        ok("bitsandbytes 4-bit smoke test passed")
    except Exception as e:
        fail(f"bitsandbytes broken: {e}")
        fail("  → 4-bit QLoRA won't work. Re-run ./setup_env.sh or build it manually.")
        required_ok = False

    # 4) Unsloth ------------------------------------------------------------
    try:
        import unsloth
        ok(f"unsloth {unsloth.__version__}")
    except Exception as e:
        fail(f"unsloth import failed: {e}")
        required_ok = False

    # 5) vLLM (required for fast GRPO; SFT works without it) ---------------
    try:
        import vllm
        ok(f"vllm {vllm.__version__}")
    except Exception as e:
        warn(f"vllm import failed — GRPO fast-generation unavailable: {e}")
        warnings += 1

    # 6) Flash-Attention (optional) ----------------------------------------
    try:
        import flash_attn
        ok(f"flash-attn {flash_attn.__version__}  (CK ROCm)")
    except Exception as e:
        warn(f"flash-attn missing ({e}) — transformers will use SDPA (still fast on MI300X)")
        warnings += 1

    # 7) DC-Ops environment -------------------------------------------------
    try:
        from dc_ops_env.server.dc_ops_env_environment import DcOpsEnvironment
        from dc_ops_env.models import DcOpsAction
        env = DcOpsEnvironment()
        obs = env.reset(scenario="A2", seed=42)
        assert "CRAC" in obs.dashboard, "dashboard text looks wrong"
        obs = env.step(DcOpsAction(command="check_status"))
        assert obs.reward is not None
        ok("dc_ops_env import + A2 reset + 1 step works")
    except Exception as e:
        fail(f"dc_ops_env broken: {e}")
        fail("  → clone https://github.com/TheDeadcoder/dc_ops_environment to ../dc_ops_environment "
             "and `uv pip install -e ../dc_ops_environment/dc_ops_env`")
        required_ok = False

    # 8) wandb --------------------------------------------------------------
    try:
        import wandb
        ok(f"wandb {wandb.__version__}")
    except Exception as e:
        warn(f"wandb import failed: {e}")
        warnings += 1

    # 9) OpenEnv core -------------------------------------------------------
    try:
        import openenv
        ok(f"openenv-core {getattr(openenv, '__version__', '?')}")
    except Exception as e:
        warn(f"openenv-core import failed (not fatal if dc_ops_env works standalone): {e}")
        warnings += 1

    # ---------- summary ---------------------------------------------------
    print()
    if required_ok and warnings == 0:
        print(f"{GREEN}All checks passed. You're ready to train.{NC}")
        return 0
    if required_ok:
        print(f"{YELLOW}{warnings} warnings — non-fatal. Training should still run.{NC}")
        return 0
    print(f"{RED}Required checks failed — fix these before training.{NC}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
