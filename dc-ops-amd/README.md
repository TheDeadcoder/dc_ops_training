# DC-Ops AMD — SFT + GRPO training pipeline

Port of the `dc-ops-training-v3` notebook to a script-based pipeline tuned for
**AMD Instinct MI300X (ROCm 7.2)**. Trains `Qwen2.5-7B-Instruct` with 4-bit
QLoRA via Unsloth + TRL, using the physics-based [DC-Ops OpenEnv environment](https://github.com/TheDeadcoder/dc_ops_environment).

No CUDA. No nvidia-*. Works on bare-metal ROCm 7.2 with `uv`.

---

## Contents

| Path | What it is |
|---|---|
| `setup_env.sh` | One-shot installer for the full ROCm 7.2 stack via `uv` |
| `pyproject.toml` | Python deps (`uv sync`-compatible; complements `setup_env.sh`) |
| `configs/sft.yaml` | SFT hyperparameters (MI300X-tuned, with memory math in comments) |
| `configs/grpo.yaml` | GRPO hyperparameters (MI300X-tuned) |
| `scripts/run_sft.py` | SFT trainer |
| `scripts/push_to_hf.py` | Upload SFT LoRA to the HF Hub |
| `scripts/run_grpo.py` | GRPO trainer — **standalone**, pulls SFT LoRA from HF |
| `scripts/verify_setup.py` | Post-install sanity check |
| `scripts/eda.py` | Dataset EDA (scenarios, commands, token lengths) |
| `launch/sft.sh` | `nohup` background launcher for SFT |
| `launch/grpo.sh` | `nohup` background launcher for GRPO |
| `src/` | Shared modules (data pipeline, rewards, prompts, env-vars) |

---

## Hardware / software assumed

| | |
|---|---|
| GPU | 1× AMD Instinct MI300X (gfx942, 192 GB HBM3) |
| ROCm | 7.2.0 (driver amdgpu 6.16.13) |
| Python | 3.12 |
| OS | Ubuntu 22.04 / 24.04 |
| CPU / RAM | 20 vCPU / 240 GB |
| Scratch disk | 5 TB NVMe — recommended for HF cache (see `.env.example`) |

The scripts assume a **single GPU**. The config is tuned for that. It will
likely work on MI300A / MI325X / MI355X as well — just update `PYTORCH_ROCM_ARCH`
in `setup_env.sh`.

---

## Quick-start (happy path)

```bash
# 1) get the repos side-by-side
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git wget curl vim tmux htop nvtop
apt update && apt install -y cmake ninja-build
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
git clone <this-repo>             dc-ops-amd
git clone https://github.com/TheDeadcoder/dc_ops_environment.git

cd dc-ops-amd
# 2) install everything (~15 min on first run: bitsandbytes + flash-attn
#    compile from source). Re-runnable.
chmod +x setup_env.sh

tmux new -s rocm-setup
cd /root/dc_ops_training/dc-ops-amd   
source .venv/bin/activate
./setup_env.sh
Detach with Ctrl+B then D — the build continues in the background.
Reattach later with tmux attach -t rocm-setup.

# 3) secrets
cp .env.example .env
$EDITOR .env       # set HUGGINGFACE_TOKEN and WANDB_API_KEY

# 4) sanity-check
source .venv/bin/activate
cp -r ../dc_ops_environment/dc_ops_env/ .
python scripts/verify_setup.py

# 5) (optional) look at the data
python scripts/eda.py

# 6) SFT — runs in background, survives SSH disconnect
chmod +x launch/sft.sh launch/grpo.sh
./launch/sft.sh
tail -f logs/sft-*.log            # follow locally
# or watch the wandb project 'dc-ops-amd' for live curves

# 7) push SFT LoRA to HF (foreground, takes ~1 min — it's tiny)
python scripts/push_to_hf.py \
    --local-dir outputs/dc_ops_sft_lora \
    --repo-id   your-username/dc-ops-sft-lora \
    --private

# 8) update configs/grpo.yaml → model.sft_model_hub to your-username/dc-ops-sft-lora
#    (or keep `sft_model_local: ./outputs/dc_ops_sft_lora` if training on the
#    same box — the script prefers local over Hub automatically.)

# 9) GRPO — also backgrounded
./launch/grpo.sh
tail -f logs/grpo-*.log
```

---

## Key changes from the notebook

| Aspect | Notebook | This pipeline |
|---|---|---|
| Platform | Modal (A100 80GB) + gVisor | Bare-metal ROCm 7.2 / MI300X |
| CUDA env-vars / libnvJitLink preload | present | **removed** |
| `PYTORCH_CUDA_ALLOC_CONF` | present | replaced with `PYTORCH_HIP_ALLOC_CONF` |
| `max_seq_length` | 4096 | **2048** (EDA shows p99 ≈ 1680 tokens) |
| SFT batch | 16 (per_device 16, accum 1) | **32** (per_device 32, accum 1) |
| `use_gradient_checkpointing` | `"unsloth"` | **`False`** (192 GB VRAM = we don't need it; ~25–40% faster) |
| GRPO `num_generations` | 8 | 8 |
| GRPO per_device × accum | 4 × 2 = 8 | **8 × 2 = 16** (2 prompts/step, cleaner advantages) |
| `gpu_memory_utilization` (vLLM) | 0.75 | **0.80** |
| wandb | `report_to="none"` | **enabled** via config |
| `<think>` rewrite | in notebook cells | in `src/prompts.py` (shared) |
| Standalone GRPO | requires re-running SFT cells | **yes** — pulls SFT LoRA from HF |

**I kept, unchanged:**
the 4 reward functions, the scenario filter (A1/A2/A4/B1/B3/B4), the warmup-
replay GRPO dataset builder, the `<think>`-stripping logic, the system-prompt
rewrite, LoRA rank / target modules, LR schedules, GRPO β / max_grad_norm.

---

## Hyperparameter memory math (MI300X, 192 GB HBM3)

### SFT (`configs/sft.yaml`)

At `max_seq_length=2048`, `per_device_train_batch_size=32`, `packing=true`:

| Item | Memory |
|---|---|
| Base model (Qwen2.5-7B, 4-bit NF4) | ≈ 3.5 GB |
| LoRA adapter (r=64, 7 modules, bf16) | ≈ 0.8 GB |
| LoRA gradients | ≈ 0.8 GB |
| AdamW states (fp32 m/v on LoRA only) | ≈ 3.2 GB |
| Activations (flash-attn, bf16, batch 32) | ≈ 24 GB |
| Workspace / fragmentation | ≈ 5 GB |
| **Total** | **≈ 37 GB** of 192 GB — ~155 GB headroom |

→ We can push `per_device` to 64 if you want more throughput. 32 is comfortable.

### GRPO (`configs/grpo.yaml`)

At `gpu_memory_utilization=0.80`, `per_device=8`, `num_generations=8`:

| Item | Memory |
|---|---|
| vLLM inference pool (KV cache + weights) | ≈ 153 GB |
| Base 4-bit + LoRA + grads + AdamW | ≈ 8 GB |
| Activations (prompt 1792 + completion 384, batch 8) | ≈ 22 GB |
| Workspace | ≈ 5 GB |
| **Total** | **≈ 188 GB** — tight but fits |

→ If you OOM at init, drop `gpu_memory_utilization` from 0.80 → 0.70.

---

## How GRPO on a fresh machine works

The GRPO script is standalone so you can run SFT on machine A, push, then run
GRPO on a different machine B. On launch it:

1. **Resolves the SFT source.** Prefers `sft_model_local` if the directory
   exists, else falls back to `sft_model_hub`. Logs which it picked.
2. **Recovers the system prompt.** Prefers a `system_prompt.txt` saved next to
   the LoRA (the SFT script writes one); falls back to re-rewriting it from
   the raw HF dataset. Either way the string is byte-identical to what SFT
   saw — the model doesn't see a distribution shift at RL time.
3. **Builds the GRPO prompt dataset** from the live DC-Ops env (env resets
   are seeded, so the prompts are deterministic given the same config).
4. **Loads the model via Unsloth with vLLM fast-inference enabled** — that's
   the 10–20× speedup vs plain `.generate()` for the rollout phase.
5. **Trains** with all 4 reward functions, logging to wandb.

---

## Wandb

Both configs have `wandb.enabled: true` by default. Curves you'll see:

- `train/loss`, `eval/loss` (SFT)
- `train/reward`, `train/reward_std` (GRPO — the main one to watch)
- per-reward-function metrics under `rewards/*` (format, env, command_quality,
  no_repeat) — these should all move. In the notebook's v2 run, `format` and
  `env` both saturated; the v3 reward design in this pipeline fixes that.
- `train/completion_length` (watch this — if it pegs at `max_completion_length=384`,
  raise it)
- `train/learning_rate`, `train/grad_norm`

To disable wandb, set `wandb.enabled: false` in the yaml.

---

## Troubleshooting

**`setup_env.sh` fails on the bitsandbytes build** with a hipcc error.
You need `rocm-dev` and `hipblas-dev` on the system:
```bash
sudo apt install rocm-dev hipblas-dev rocm-hip-sdk
```
Then re-run `./setup_env.sh` — it's idempotent.

**Flash-attn build fails or takes forever.**
It's optional. Delete `.build/flash-attention/`, re-run setup without FA, and
add `# skip` or comment out the flash-attn step. Transformers will fall back
to PyTorch SDPA on ROCm (aotriton-backed), which is fast enough.

**`ImportError: cannot import name 'vLLMSamplingParams' from 'unsloth'`.**
Harmless — `scripts/run_grpo.py` falls back to TRL's `generation_kwargs`.

**GRPO OOMs at init.**
Drop `vllm.gpu_memory_utilization` from 0.80 → 0.70 → 0.65 in
`configs/grpo.yaml` until it fits. Each 0.05 = 9.6 GB freed for training-side.

**GRPO OOMs mid-training** (not at init).
`max_completion_length` is almost certainly too large. Our default is 384
which is already ~2.5× the teacher's p99. If a reward function starves the
model and it generates junk up to the cap, completions balloon. Drop to 256.

**wandb complains about permissions / auth.**
`wandb login` once interactively; after that the key is cached. Or
`export WANDB_API_KEY=…` / put it in `.env`.

**GRPO loss NaNs after step ~5.**
Drop `grpo.learning_rate` from 5e-6 → 2e-6, and/or reduce `grpo.max_grad_norm`
from 0.1 → 0.05. GRPO on a small-reward-variance problem is finicky.

**Training hangs for several minutes at the start.**
Normal — MIOpen is auto-tuning new tensor shapes. With `MIOPEN_FIND_MODE=3`
(set by `src/rocm_env.py`), the tuned configs are cached to disk and the
next run starts fast.

**`amd-smi` shows 0% GFX util but training is running.**
amd-smi samples at ~1 Hz and can miss peaks. Use `watch -n 0.2 amd-smi monitor`
or look at wandb `train/train_tokens_per_second` instead.

---

## Project layout

```
dc-ops-amd/
├── README.md                    ← you are here
├── setup_env.sh                 ← ROCm 7.2 stack installer
├── pyproject.toml               ← Python deps (uv-compatible)
├── .env.example                 ← template for secrets
├── .gitignore
│
├── configs/
│   ├── sft.yaml
│   └── grpo.yaml
│
├── src/
│   ├── __init__.py
│   ├── rocm_env.py              ← ROCm-only env vars, imported first everywhere
│   ├── constants.py             ← scenarios, command verbs, warmup actions
│   ├── prompts.py               ← <think>-strip + system-prompt rewrite
│   ├── data_utils.py            ← SFT dataset load + windowing
│   ├── rewards.py               ← 4 GRPO reward functions
│   └── grpo_data.py             ← GRPO prompt builder
│
├── scripts/
│   ├── run_sft.py               ← SFT trainer
│   ├── push_to_hf.py            ← upload SFT LoRA to HF Hub
│   ├── run_grpo.py              ← GRPO trainer (standalone)
│   ├── verify_setup.py          ← post-install sanity check
│   └── eda.py                   ← dataset EDA dumper
│
├── launch/
│   ├── sft.sh                   ← nohup SFT launcher
│   └── grpo.sh                  ← nohup GRPO launcher
│
├── logs/                        ← auto-created (stdout + PID files)
└── outputs/                     ← auto-created (SFT + GRPO checkpoints)
```

**Not in this repo** — clone these alongside:

```
../dc_ops_environment/           ← the custom OpenEnv environment
```

---

## Credits

- Environment: [TheDeadcoder/dc_ops_environment](https://github.com/TheDeadcoder/dc_ops_environment)
- Base model: [unsloth/Qwen2.5-7B-Instruct](https://huggingface.co/unsloth/Qwen2.5-7B-Instruct)
- Teacher data: [Melikshah/dc-ops-sft-data](https://huggingface.co/datasets/Melikshah/dc-ops-sft-data)
- Frameworks: [Unsloth](https://github.com/unslothai/unsloth), [TRL](https://github.com/huggingface/trl), [OpenEnv](https://github.com/meta-pytorch/OpenEnv)
