# DC-Ops AMD — SFT + GRPO training pipeline

I use this repository to train `unsloth/Qwen2.5-7B-Instruct` for the DC-Ops environment on a single AMD Instinct MI300X. The workflow is fully script-driven: I install the ROCm stack with `setup_env.sh`, run supervised fine-tuning from `configs/sft.yaml`, optionally push the LoRA adapter to Hugging Face, and then continue with GRPO from `configs/grpo.yaml`.

I keep the stack ROCm-only. There is no CUDA path and I do not rely on any `nvidia-*` packages.

---

## What I packaged

- I keep ROCm runtime setup in `src/rocm_env.py` so every training process starts with the same environment variables.
- I keep data prep, prompt rewriting, and reward logic in `src/data_utils.py`, `src/prompts.py`, `src/grpo_data.py`, and `src/rewards.py`.
- I run SFT and GRPO from `scripts/run_sft.py` and `scripts/run_grpo.py`, both configured entirely from YAML.
- I launch long-running jobs through `launch/sft.sh` and `launch/grpo.sh` so `.env`, `nohup`, logs, and PID files are handled consistently.

## Hardware and OS I validated

| Item | Value |
|---|---|
| GPU | 1x AMD Instinct MI300X VF |
| Available VRAM at verification | 205.8 GB |
| GPU arch | gfx942 |
| Host ROCm target | ROCm 7.2 |
| Verified torch/HIP stack | torch 2.10.0+rocm7.1 |
| Python | 3.12 |
| OS | Ubuntu 22.04 / 24.04 |
| CPU / RAM | 20 vCPU / 240 GB |
| Disk | 5 TB NVMe scratch recommended for HF cache |

I tune the configs for a single GPU. If I move to a different card or a smaller memory budget, I retune the YAMLs first instead of trusting the defaults blindly.

## Verified software stack

I treat `python scripts/verify_setup.py` as the source of truth after installation. On the validated machine it reported the following stack:

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

The `vllm` warning is non-fatal version drift. The validated run still passed the rest of the checks.

## Dataset notes

I train on `Melikshah/dc-ops-sft-data`. The current EDA reported:

| Metric | Value |
|---|---|
| Raw episodes | 1,083 |
| Dropped `VAR_*` rows | 149 |
| Kept episodes | 934 |
| GPT turns in kept episodes | 7,107 |
| Tokenizer-measured p99 window length | ~2,700 tokens |
| Tokenizer-measured max window length | ~2,900 tokens |

I keep `model.max_seq_length: 3072` in both YAMLs because the EDA showed that going below `3072` would start truncating real windows.

### Scenario distribution

| Scenario | Count | Share |
|---|---|---|
| A1 | 145 | 15.5% |
| A2 | 225 | 24.1% |
| A4 | 181 | 19.4% |
| B1 | 120 | 12.8% |
| B3 | 160 | 17.1% |
| B4 | 103 | 11.0% |

### Most common commands

| Command | Count | Share |
|---|---|---|
| `set_rack_load` | 3,658 | 51.5% |
| `diagnose` | 1,128 | 15.9% |
| `check_status` | 853 | 12.0% |
| `adjust_setpoint` | 650 | 9.1% |

## Current training defaults

I landed on the following values after hitting OOM in both SFT and GRPO with more aggressive settings. If any older note or screenshot disagrees with this section, I treat `configs/sft.yaml` and `configs/grpo.yaml` as authoritative.

### SFT defaults

| Setting | Value |
|---|---|
| Base model | `unsloth/Qwen2.5-7B-Instruct` |
| `model.max_seq_length` | `3072` |
| 4-bit loading | `true` |
| LoRA rank / alpha | `16 / 32` |
| Gradient checkpointing | `"unsloth"` |
| `per_device_train_batch_size` | `10` |
| `gradient_accumulation_steps` | `2` |
| `num_train_epochs` | `2` |
| `learning_rate` | `7.0e-5` |
| Precision | `bf16=true`, `fp16=false` |
| Packing | `false` |
| Training output dir | `./outputs/sft` |
| Final LoRA dir | `./outputs/dc_ops_sft_lora` |

### GRPO defaults

| Setting | Value |
|---|---|
| SFT LoRA source | `./outputs/dc_ops_sft_lora` or `Melikshah/dc-ops-sft-lora-new` |
| `model.max_seq_length` | `3072` |
| `max_lora_rank` | `16` |
| Initial prompts | `30` |
| Midgame prompts | `55` |
| `num_generations` | `8` |
| `max_prompt_length` | `2048` |
| `max_completion_length` | `512` |
| `per_device_train_batch_size` | `8` |
| `gradient_accumulation_steps` | `2` |
| Gradient checkpointing | `"unsloth"` |
| `num_train_epochs` | `7` |
| `learning_rate` | `5.0e-6` |
| `beta` | `0.05` |
| `max_grad_norm` | `0.1` |
| `vllm.gpu_memory_utilization` | `0.65` |
| Training output dir | `./outputs/grpo` |
| Final GRPO dir | `./outputs/dc_ops_grpo_final` |

## Setup

I use `setup_env.sh` as the real installer. I keep `pyproject.toml` for Python-only metadata and tooling, but I do not rely on `uv sync` alone because the ROCm-specific stack comes from ROCm wheels and source builds.

### 1. Install system packages and clone the repositories

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git wget curl vim tmux htop nvtop cmake ninja-build
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

git clone <this-repo> dc-ops-amd
git clone https://github.com/TheDeadcoder/dc_ops_environment.git

cd dc-ops-amd
chmod +x setup_env.sh launch/sft.sh launch/grpo.sh
```

I keep `dc_ops_environment` beside this repository because `setup_env.sh` installs `../dc_ops_environment/dc_ops_env` in editable mode.

### 2. Run the full ROCm install inside `tmux`

```bash
tmux new -s rocm-setup
cd /root/dc_ops_training/dc-ops-amd
./setup_env.sh
```

I start the install inside `tmux` because the ROCm Flash-Attention build can take 40-50 minutes on a clean machine. If my SSH session drops mid-build, `tmux` keeps the process alive and I can reconnect later with:

```bash
tmux attach -t rocm-setup
```

### 3. Add the `bitsandbytes` compatibility symlink

```bash
cd /root/dc_ops_training/dc-ops-amd/.build/bitsandbytes/bitsandbytes
ln -sf libbitsandbytes_rocm72.so libbitsandbytes_rocm71.so
ls -l libbitsandbytes_rocm*.so
```

I keep this step explicit because the validated stack uses `torch==2.10.0+rocm7.1` on a ROCm 7.2 host. In practice, the ROCm fork can build `libbitsandbytes_rocm72.so` while the Python loader still looks for `libbitsandbytes_rocm71.so`. The symlink makes both names resolve to the same compiled library.

I expect the directory listing to look like this:

```text
libbitsandbytes_rocm72.so
libbitsandbytes_rocm71.so -> libbitsandbytes_rocm72.so
```

### 4. Create `.env`

```bash
cd /root/dc_ops_training/dc-ops-amd
cp .env.example .env
$EDITOR .env
```

I set these values before training:

- `HUGGINGFACE_TOKEN` for pushing the LoRA adapter or pulling a private adapter during GRPO
- `WANDB_API_KEY` because both YAMLs enable Weights & Biases by default

### 5. Verify the environment

```bash
cd /root/dc_ops_training/dc-ops-amd
source .venv/bin/activate
python scripts/verify_setup.py
```

I do not start training until this check passes. A `vllm` warning by itself is acceptable if the rest of the stack passes.

### 6. Inspect the data (optional)

```bash
cd /root/dc_ops_training/dc-ops-amd
source .venv/bin/activate
python scripts/eda.py
```

I use this when I want a fresh summary of scenario counts, command frequencies, and sequence lengths before changing any training window.

## Training workflow

### 1. Start SFT

```bash
cd /root/dc_ops_training/dc-ops-amd
./launch/sft.sh
tail -f logs/sft-*.log
```

I use the launcher instead of calling the trainer directly because it loads `.env`, activates `.venv`, writes a timestamped log file, and records the PID in `logs/sft.pid`. The SFT run writes checkpoints under `./outputs/sft` and the LoRA adapter under `./outputs/dc_ops_sft_lora`.

### 2. Push the SFT LoRA adapter to Hugging Face

```bash
cd /root/dc_ops_training/dc-ops-amd
source .venv/bin/activate
python scripts/push_to_hf.py \
  --local-dir outputs/dc_ops_sft_lora \
  --repo-id your-username/dc-ops-sft-lora \
  --private
```

I only need this step if I want GRPO to pull the adapter from the Hub or if I want to reuse the SFT adapter on another machine. If I push to a new repo, I update `model.sft_model_hub` in `configs/grpo.yaml` to match it.

### 3. Start GRPO

```bash
cd /root/dc_ops_training/dc-ops-amd
./launch/grpo.sh
tail -f logs/grpo-*.log
```

I keep `outputs/dc_ops_sft_lora` as the default local handoff, so GRPO prefers the local adapter automatically and only falls back to the Hub when the local directory does not exist.

## Repository layout

```text
dc-ops-amd/
├── README.md
├── .env.example
├── .gitignore
├── pyproject.toml
├── setup_env.sh
├── configs/
│   ├── grpo.yaml
│   └── sft.yaml
├── launch/
│   ├── grpo.sh
│   └── sft.sh
├── logs/
├── scripts/
│   ├── eda.py
│   ├── evaluate.py
│   ├── push_to_hf.py
│   ├── run_grpo.py
│   ├── run_sft.py
│   └── verify_setup.py
├── server_ouputs/
│   ├── eda.md
│   ├── eval_result.md
│   ├── sft.md
│   └── verify.md
├── src/
│   ├── __init__.py
│   ├── constants.py
│   ├── data_utils.py
│   ├── grpo_data.py
│   ├── prompts.py
│   ├── rewards.py
│   └── rocm_env.py
└── outputs/              # created during training
```

I also keep the DC-Ops environment as a sibling checkout:

```text
../dc_ops_environment/
```

## Troubleshooting

**`setup_env.sh` takes a very long time.**  
I expect the first install to be slow because `bitsandbytes` and especially ROCm `flash-attention` build from source. On my clean server, the Flash-Attention step can take 40-50 minutes, which is why I always start the install inside `tmux`.

**`bitsandbytes` cannot find `libbitsandbytes_rocm71.so`.**  
I create the compatibility symlink documented above. The compiled library can be named `libbitsandbytes_rocm72.so` on the host while the Python package still probes for the `rocm71` name because the validated torch stack is `2.10.0+rocm7.1`.

**`setup_env.sh` fails while building `bitsandbytes` with HIP or `hipcc` errors.**  
I install the ROCm development packages and rerun the installer:

```bash
sudo apt install -y rocm-dev hipblas-dev rocm-hip-sdk
cd /root/dc_ops_training/dc-ops-amd
./setup_env.sh
```

**`flash-attention` fails to build.**  
I treat Flash-Attention as an optimization, not a hard requirement. If that build keeps failing, I remove `.build/flash-attention`, skip that block in `setup_env.sh`, and let Transformers fall back to PyTorch SDPA on ROCm. Training still works; it is just less optimized.

**SFT runs out of memory.**  
The current SFT config is already the reduced version I settled on after OOMs: `max_seq_length=3072`, `per_device_train_batch_size=10`, `gradient_accumulation_steps=2`, `packing=false`, and gradient checkpointing through Unsloth. If I still OOM, I lower `per_device_train_batch_size` from `10` to `8` and then to `6`. I do not reduce `max_seq_length` below `3072` unless I am willing to accept truncation that the EDA already warned about.

**GRPO runs out of memory at startup.**  
The current GRPO config already uses `vllm.gpu_memory_utilization=0.65`, which is lower than the older, more aggressive settings. If initialization still OOMs, I reduce it again to `0.60` and then `0.55`.

**GRPO runs out of memory during training rather than at startup.**  
I first reduce `grpo.max_completion_length` from `512` to `384`, then to `256`. If needed, I reduce `grpo.per_device_train_batch_size` from `8` to `6` or `4`.

**`ImportError: cannot import name 'vLLMSamplingParams' from 'unsloth'`.**  
I treat this as non-fatal. `scripts/run_grpo.py` falls back to TRL generation kwargs.

**`verify_setup.py` reports `vllm 0.19.1` instead of `0.19.1+rocm721`.**  
I treat that as version drift, not an automatic failure. The verified environment report already marked it as a warning while keeping the rest of the stack green.

**`wandb` errors at startup.**  
I confirm that `WANDB_API_KEY` is set in `.env`. The launchers read `.env` automatically, but they do not invent missing credentials.

**`dc_ops_env` import or reset fails.**  
I confirm that the sibling repository exists and is installed in editable mode:

```bash
cd /root/dc_ops_training/dc_ops_environment/dc_ops_env
uv pip install -e .
```

## Credits

- Environment: [TheDeadcoder/dc_ops_environment](https://github.com/TheDeadcoder/dc_ops_environment)
- Base model: [unsloth/Qwen2.5-7B-Instruct](https://huggingface.co/unsloth/Qwen2.5-7B-Instruct)
- Teacher data: [Melikshah/dc-ops-sft-data](https://huggingface.co/datasets/Melikshah/dc-ops-sft-data)
- Frameworks: [Unsloth](https://github.com/unslothai/unsloth), [TRL](https://github.com/huggingface/trl), [OpenEnv](https://github.com/meta-pytorch/OpenEnv)
