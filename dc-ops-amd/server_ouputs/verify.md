(.venv) root@7:~/dc_ops_training/dc-ops-amd# python scripts/verify_setup.py 
[ ok ] torch 2.10.0+rocm7.1  (HIP 7.1.25424)
[ ok ] GPU: AMD Instinct MI300X VF  |  VRAM 205.8 GB  |  bf16=True
[ ok ] transformers 4.54.1
[ ok ] tokenizers 0.21.0
Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.10.0+rocm7.1).
/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/transformers/loss/loss_for_object_detection.py:28: UserWarning: A NumPy version >=1.23.5 and <2.3.0 is required for this version of SciPy (detected version 2.4.4)
  from scipy.optimize import linear_sum_assignment
[ ok ] peft 0.19.1
[ ok ] trl 0.21.0
[ ok ] accelerate 1.13.0
[ ok ] datasets 4.3.0
g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Copyright (C) 2023 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

[ ok ] bitsandbytes 0.43.3.dev  (ROCm fork)
/root/dc_ops_training/dc-ops-amd/scripts/verify_setup.py:86: UserWarning: expandable_segments not supported on this platform (Triggered internally at /pytorch/c10/hip/HIPAllocatorConfig.h:40.)
  x = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
[ ok ] bitsandbytes 4-bit smoke test passed
/root/dc_ops_training/dc-ops-amd/scripts/verify_setup.py:99: UserWarning: WARNING: Unsloth should be imported before [trl, transformers, peft] to ensure all optimizations are applied. Your code may run slower or encounter memory issues without these optimizations.

Please restructure your imports with 'import unsloth' at the top of your file.
  import unsloth
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
[ ok ] unsloth 2026.4.4
[ ok ] vllm 0.19.1
[ ok ] flash-attn 2.8.4  (CK ROCm)
[ ok ] dc_ops_env import + A2 reset + 1 step works
[ ok ] wandb 0.26.0
[ ok ] openenv-core 0.2.3

All checks passed. You're ready to train.