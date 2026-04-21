g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Copyright (C) 2023 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/transformers/loss/loss_for_object_detection.py:28: UserWarning: A NumPy version >=1.23.5 and <2.3.0 is required for this version of SciPy (detected version 2.4.4)
  from scipy.optimize import linear_sum_assignment
🦥 Unsloth Zoo will now patch everything to make training faster!
[ ok ] torch 2.10.0+rocm7.1  (HIP 7.1.25424)
[ ok ] GPU: AMD Instinct MI300X VF  |  VRAM 205.8 GB  |  bf16=True
[ ok ] unsloth 2026.4.4
[ ok ] unsloth_zoo 2026.4.8
[ ok ] transformers 4.54.1
[ ok ] tokenizers 0.21.0
[ ok ] peft 0.19.1
[ ok ] trl 0.21.0
[ ok ] accelerate 1.13.0
[ ok ] datasets 4.3.0
[ ok ] huggingface_hub 0.36.2
[ ok ] bitsandbytes 0.43.3.dev  (ROCm fork)
[ ok ] bitsandbytes 4-bit smoke test passed
[ ok ] vllm 0.19.1
[warn]   → vllm 0.19.1 (known-good: 0.19.1+rocm721) — non-fatal version drift
[ ok ] flash-attn 2.8.4  (CK ROCm)
[ ok ] triton 3.6.0
[ ok ] dc_ops_env import + A2 reset + 1 step works
[ ok ] wandb 0.26.0
[ ok ] openenv-core 0.2.3

All checks passed. You're ready to train.