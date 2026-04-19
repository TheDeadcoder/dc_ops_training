#!/usr/bin/env bash
# =============================================================================
# install.sh — ROCm 7.2 / MI300X bootstrap for Unsloth SFT + vLLM GRPO
# =============================================================================
# Tested stack (verified Jan 2026):
#   • ROCm 7.2.x runtime on host
#   • PyTorch 2.9.x (ROCm 7.1 stable wheels) or 2.10 nightly (ROCm 7.2)
#   • Unsloth AMD branch   (supports MI300X / gfx942)
#   • TRL 0.22.2, transformers 4.57.1  (Qwen3 + assistant mask verified)
#   • vLLM ROCm wheels     (needed later for GRPO fast_inference)
#   • bitsandbytes 0.45+   (multi-backend, ROCm supported)
#
# Usage:
#   chmod +x install.sh
#   ./install.sh            
# =============================================================================
set -euo pipefail

log() { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31mERR:\033[0m %s\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# 0. Sanity checks
# -----------------------------------------------------------------------------
command -v python3 >/dev/null 2>&1 || die "python3 not found."

# Detect ROCm version. MI300X is gfx942, ROCm 7.x.
ROCM_VER="unknown"
if command -v amd-smi >/dev/null 2>&1; then
    ROCM_VER=$(amd-smi version 2>/dev/null | awk -F'ROCm version: ' 'NF>1 {split($2,a," "); print a[1]; exit}' || true)
fi
if [[ "$ROCM_VER" == "unknown" ]] && [[ -r /opt/rocm/.info/version ]]; then
    ROCM_VER=$(cat /opt/rocm/.info/version 2>/dev/null || true)
fi
log "Detected ROCm: $ROCM_VER (expected 7.2.x)"

# -----------------------------------------------------------------------------
# 1. Virtualenv
# -----------------------------------------------------------------------------
if [[ ! -d .venv ]]; then
    log "Creating uv venv (.venv, python 3.12)"
    uv venv --python 3.12 .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# hf_transfer for fast HF downloads
export HF_HUB_ENABLE_HF_TRANSFER=1
# Prevent a non-RoCm torch from sneaking in
export PIP_EXTRA_INDEX_URL=""

# -----------------------------------------------------------------------------
# 2. PyTorch + triton for ROCm
# -----------------------------------------------------------------------------
# We use ROCm 7.1 stable wheels (torch 2.9.x) — these are the ABI that
# vLLM ROCm wheels target, and Unsloth's AMD branch has been validated
# against them. They also work on ROCm 7.2 runtime (minor rev, same ABI).
#
# If you want bleeding edge, swap to nightly rocm7.2:
#   --index-url https://download.pytorch.org/whl/nightly/rocm7.2
log "Installing PyTorch (ROCm 7.1 stable, torch 2.9.1)"
uv pip install --upgrade --force-reinstall \
    torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/rocm7.1

python -c "import torch; print(f'torch {torch.__version__}, cuda_available={torch.cuda.is_available()}, hip={torch.version.hip}')" \
    || die "torch import failed"

# -----------------------------------------------------------------------------
# 3. Core training deps (pinned)
# -----------------------------------------------------------------------------
log "Installing training deps (transformers, trl, peft, accelerate, datasets, wandb, ...)"
uv pip install \
    "transformers==4.57.1" \
    "trl==0.22.2" \
    "peft>=0.13.0" \
    "accelerate>=1.0.0" \
    "datasets>=3.0.0,<5.0.0" \
    "huggingface_hub>=0.26.0" \
    "safetensors>=0.4.5" \
    "sentencepiece>=0.2.0" \
    "protobuf>=4.25.0" \
    "tiktoken>=0.7.0" \
    "hf_transfer>=0.1.8" \
    "wandb>=0.19.0" \
    "pyyaml>=6.0.1" \
    "omegaconf>=2.3.0" \
    "tqdm>=4.66.0" \
    "rich>=13.7.0" \
    "pydantic>=2.0.0" \
    "orjson>=3.10.0" \
    "xformers==0.0.33.post1" \
    "cut_cross_entropy" \
    || die "Core deps install failed"

# bitsandbytes multi-backend (includes ROCm). Needed by adamw_8bit optimizer.
# If this wheel install fails on your ROCm version, fall back to:
#   git clone -b rocm_enabled_multi_backend https://github.com/ROCm/bitsandbytes
#   cd bitsandbytes && pip install -r requirements-dev.txt && \
#       cmake -DCOMPUTE_BACKEND=hip -S . && make -j && pip install .
log "Installing bitsandbytes (multi-backend)"
uv pip install "bitsandbytes>=0.45.0" || log "bitsandbytes install warning — continuing"

# -----------------------------------------------------------------------------
# 4. Unsloth (AMD branch)
# -----------------------------------------------------------------------------
# Official path per https://unsloth.ai/docs/get-started/install/amd
# We install with --no-deps on the first step to prevent unsloth-zoo
# from dragging a CUDA torch wheel in.
log "Installing Unsloth (AMD / MI300X)"
uv pip install --no-deps "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo.git"
uv pip install --no-deps "unsloth[amd] @ git+https://github.com/unslothai/unsloth"

python -c "import unsloth; print(f'unsloth {unsloth.__version__}')" \
    || die "unsloth import failed"

# -----------------------------------------------------------------------------
# 5. vLLM ROCm (only needed for GRPO later; safe to install now)
# -----------------------------------------------------------------------------
# Pre-built ROCm wheels live at https://wheels.vllm.ai/rocm/. These bundle
# a matched torch — we install with --no-deps to preserve our torch.
# If you are ONLY doing SFT on this box, you can comment this out to save
# ~2 GB of install.
log "Installing vLLM (ROCm wheels, no-deps)"
uv pip install --no-deps "vllm" --extra-index-url https://wheels.vllm.ai/rocm/ \
    || log "vLLM install skipped (OK for SFT-only box)"

# -----------------------------------------------------------------------------
# 6. Install this project (for imports)
# -----------------------------------------------------------------------------
log "Installing dc-ops-sft (editable)"
uv pip install -e . --no-deps

# -----------------------------------------------------------------------------
# 7. (Optional) dc_ops_env — for env_eval / eval_compare.py
# -----------------------------------------------------------------------------
# Needed only if you want to run `scripts/eval_compare.py` locally.
# Pass the path to your dc_ops_environment checkout as arg $1, or set
# DC_OPS_ENV_PATH env var, else we skip.
DC_OPS_ENV_PATH="${DC_OPS_ENV_PATH:-${1:-}}"
if [[ -n "$DC_OPS_ENV_PATH" ]] && [[ -d "$DC_OPS_ENV_PATH" ]]; then
    log "Installing dc_ops_env from $DC_OPS_ENV_PATH"
    uv pip install -e "$DC_OPS_ENV_PATH" --no-deps
    uv pip install "openenv-core[core]>=0.2.1" "fastapi>=0.115.0" "uvicorn>=0.24.0"
else
    log "DC_OPS_ENV_PATH not set — skipping env install. Set it if you want eval_compare."
fi

# -----------------------------------------------------------------------------
# 8. Final verification
# -----------------------------------------------------------------------------
log "Verification:"
python - <<'PY'
import torch, transformers, trl, peft, datasets
import unsloth
print(f"  torch          : {torch.__version__}  (hip={torch.version.hip})")
print(f"  cuda_available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device_name    : {torch.cuda.get_device_name(0)}")
    print(f"  device_count   : {torch.cuda.device_count()}")
    free, total = torch.cuda.mem_get_info(0)
    print(f"  vram           : {free/2**30:.1f} / {total/2**30:.1f} GB free")
print(f"  transformers   : {transformers.__version__}")
print(f"  trl            : {trl.__version__}")
print(f"  peft           : {peft.__version__}")
print(f"  datasets       : {datasets.__version__}")
print(f"  unsloth        : {unsloth.__version__}")
try:
    import vllm
    print(f"  vllm           : {vllm.__version__}")
except Exception as e:
    print(f"  vllm           : NOT INSTALLED ({e})")
try:
    import bitsandbytes as bnb
    print(f"  bitsandbytes   : {bnb.__version__}")
except Exception as e:
    print(f"  bitsandbytes   : {e}")
PY

log "Install complete. Activate with: source .venv/bin/activate"