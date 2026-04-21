# Environment Verification

**Host compiler:** g++ 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1)

---

## Hardware

| Field | Value |
|---|---|
| GPU | AMD Instinct MI300X VF |
| VRAM | 205.8 GB |
| bf16 | Supported |
| ROCm / HIP | 7.1.25424 |

---

## Package Versions

| Package | Version | Status |
|---|---|---|
| torch | 2.10.0+rocm7.1 | PASS |
| unsloth | 2026.4.4 | PASS |
| unsloth_zoo | 2026.4.8 | PASS |
| transformers | 4.54.1 | PASS |
| tokenizers | 0.21.0 | PASS |
| peft | 0.19.1 | PASS |
| trl | 0.21.0 | PASS |
| accelerate | 1.13.0 | PASS |
| datasets | 4.3.0 | PASS |
| huggingface_hub | 0.36.2 | PASS |
| bitsandbytes | 0.43.3.dev (ROCm fork) | PASS |
| vllm | 0.19.1 | WARN |
| flash-attn | 2.8.4 (CK ROCm) | PASS |
| triton | 3.6.0 | PASS |
| wandb | 0.26.0 | PASS |
| openenv-core | 0.2.3 | PASS |

> **Warning:** vllm 0.19.1 detected; known-good version is `0.19.1+rocm721`. Non-fatal version drift.

---

## Functional Checks

| Check | Result |
|---|---|
| bitsandbytes 4-bit smoke test | PASS |
| dc_ops_env import + A2 reset + 1 step | PASS |

---

**All checks passed. Ready to train.**