#!/usr/bin/env python
# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""
Post-install sanity check for the ROCm 7.2 / MI300X training stack.

Run this once after `./setup_env.sh` (or after manually setting up the env)
to catch broken installs *before* you commit to a 3-hour training job.

Usage:
    source .venv/bin/activate
    python scripts/verify_setup.py

Exits 0 if all required checks pass, 1 if any required check fails.

The "known-good versions" printed at the end correspond to the
configuration the user verified working on their MI300X box (see
requirements.txt). Mismatches are flagged as warnings but don't fail
the check — versions can drift safely within minor bounds.
"""

from __future__ import annotations

import sys
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

# 1) Apply ROCm env vars first.
from src import rocm_env  # noqa: F401

# 2) Import Unsloth FIRST, before TRL/transformers/peft. We also touch
#    unsloth.__version__ here to force unsloth's lazy patching to fire NOW
#    (it's lazy — just `import unsloth` doesn't trigger the patching code,
#    so the "import unsloth first" warning would still fire later when
#    unsloth_zoo gets imported and finds transformers already loaded).
try:
    import unsloth  # noqa: F401
    _ = unsloth.__version__   # force the lazy zoo patch to run NOW
    import unsloth_zoo        # noqa: F401  (also forces patching)
except Exception:
    # Will be reported below by the proper check
    pass


# ---------- helpers ----------
GREEN  = "\033[0;32m"
RED    = "\033[0;31m"
YELLOW = "\033[1;33m"
NC     = "\033[0m"


def ok(msg):   print(f"{GREEN}[ ok ]{NC} {msg}")
def fail(msg): print(f"{RED}[fail]{NC} {msg}")
def warn(msg): print(f"{YELLOW}[warn]{NC} {msg}")


# Known-good versions on the user's MI300X box (Apr 2026).
# Mismatches don't fail — just get warned.
KNOWN_GOOD = {
    "torch":          "2.10.0+rocm7.1",
    "transformers":   "4.54.1",
    "tokenizers":     "0.21.0",
    "peft":           "0.19.1",
    "trl":            "0.21.0",
    "accelerate":     "1.13.0",
    "unsloth":        "2026.4.4",
    "unsloth_zoo":    "2026.4.8",
    "vllm":           "0.19.1+rocm721",
    "flash_attn":     "2.8.4",
    "bitsandbytes":   "0.43.3.dev",
    "triton":         "3.6.0",
    "datasets":       "4.3.0",
    "huggingface_hub":"0.36.2",
}


def _vercheck(pkg, mod):
    actual = getattr(mod, "__version__", "?")
    expected = KNOWN_GOOD.get(pkg)
    if expected and actual != expected:
        warn(f"  → {pkg} {actual} (known-good: {expected}) — non-fatal version drift")


def main() -> int:
    required_ok = True
    warnings_count = 0

    # 1) torch + ROCm ----------------------------------------------------
    try:
        import torch
        ok(f"torch {torch.__version__}  (HIP {torch.version.hip})")
        _vercheck("torch", torch)
        if not torch.cuda.is_available():
            fail("torch.cuda.is_available() is False — ROCm not detected")
            required_ok = False
        else:
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            bf16 = torch.cuda.is_bf16_supported()
            ok(f"GPU: {name}  |  VRAM {vram:.1f} GB  |  bf16={bf16}")
            if "MI300X" not in name and "gfx942" not in name:
                warn(f"  → expected MI300X / gfx942, got {name!r}")
                warnings_count += 1
            if vram < 180:
                warn(f"  → VRAM {vram:.0f} GB < 180 GB — config tuned for 192 GB may OOM")
                warnings_count += 1
    except Exception as e:
        fail(f"torch import failed: {e}")
        return 1

    # 2) Unsloth (must come before trl/transformers/peft) ---------------
    try:
        import unsloth
        ok(f"unsloth {unsloth.__version__}")
        _vercheck("unsloth", unsloth)
    except Exception as e:
        fail(f"unsloth import failed: {e}")
        required_ok = False

    try:
        import unsloth_zoo
        ok(f"unsloth_zoo {unsloth_zoo.__version__}")
        _vercheck("unsloth_zoo", unsloth_zoo)
    except Exception as e:
        warn(f"unsloth_zoo import failed: {e}")
        warnings_count += 1

    # 3) TRL / transformers / peft / accelerate / datasets / tokenizers -
    for pkg, required in [
        ("transformers",    True),
        ("tokenizers",      True),
        ("peft",            True),
        ("trl",             True),
        ("accelerate",      True),
        ("datasets",        True),
        ("huggingface_hub", True),
    ]:
        try:
            mod = __import__(pkg)
            ok(f"{pkg} {getattr(mod, '__version__', '?')}")
            _vercheck(pkg, mod)
        except Exception as e:
            (fail if required else warn)(f"{pkg} import failed: {e}")
            if required: required_ok = False
            else: warnings_count += 1

    # 4) bitsandbytes (required for 4-bit QLoRA on AMD) -----------------
    try:
        import bitsandbytes as bnb
        ok(f"bitsandbytes {bnb.__version__}  (ROCm fork)")
        _vercheck("bitsandbytes", bnb)
        # Tiny 4-bit smoke test
        x = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        from bitsandbytes.nn import Linear4bit
        lin = Linear4bit(16, 32, bias=False, quant_type="nf4").cuda().to(torch.bfloat16)
        y = lin(x)
        assert y.shape == (8, 32)
        ok("bitsandbytes 4-bit smoke test passed")
    except Exception as e:
        fail(f"bitsandbytes broken: {e}")
        fail("  → 4-bit QLoRA won't work. Reinstall via setup_env.sh.")
        required_ok = False

    # 5) vLLM ------------------------------------------------------------
    try:
        import vllm
        ok(f"vllm {vllm.__version__}")
        _vercheck("vllm", vllm)
    except Exception as e:
        warn(f"vllm import failed — GRPO fast-generation unavailable: {e}")
        warnings_count += 1

    # 6) Flash-Attention -------------------------------------------------
    try:
        import flash_attn
        ok(f"flash-attn {flash_attn.__version__}  (CK ROCm)")
        _vercheck("flash_attn", flash_attn)
    except Exception as e:
        warn(f"flash-attn missing ({e}) — transformers will use SDPA (still fast on MI300X)")
        warnings_count += 1

    # 7) Triton ----------------------------------------------------------
    try:
        import triton
        ok(f"triton {triton.__version__}")
        _vercheck("triton", triton)
    except Exception as e:
        warn(f"triton missing ({e}) — vLLM may degrade")
        warnings_count += 1

    # 8) DC-Ops environment ----------------------------------------------
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
        fail("  → clone https://github.com/TheDeadcoder/dc_ops_environment to "
             "../dc_ops_environment and `uv pip install -e ../dc_ops_environment/dc_ops_env`")
        required_ok = False

    # 9) wandb -----------------------------------------------------------
    try:
        import wandb
        ok(f"wandb {wandb.__version__}")
    except Exception as e:
        warn(f"wandb import failed: {e}")
        warnings_count += 1

    # 10) OpenEnv core ---------------------------------------------------
    try:
        import openenv
        ok(f"openenv-core {getattr(openenv, '__version__', '?')}")
    except Exception as e:
        warn(f"openenv-core import failed (not fatal if dc_ops_env works standalone): {e}")
        warnings_count += 1

    # ---------- summary ------------------------------------------------
    print()
    if required_ok and warnings_count == 0:
        print(f"{GREEN}All checks passed. You're ready to train.{NC}")
        return 0
    if required_ok:
        print(f"{YELLOW}{warnings_count} warning(s) — non-fatal. Training should still run.{NC}")
        return 0
    print(f"{RED}Required checks failed — fix these before training.{NC}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
