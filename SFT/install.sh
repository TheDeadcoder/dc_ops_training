#!/usr/bin/env bash
# =============================================================================
# install.sh — ROCm 7.2 / MI300X bootstrap (FIXED)
# =============================================================================
# Stack (verified against PyTorch + Unsloth current pyproject.toml, Apr 2026):
#   • torch 2.10.0         from https://download.pytorch.org/whl/rocm7.1
#                          (PyTorch's rocm7.1 index is the newest stable index;
#                           wheels there are ABI-compatible with ROCm 7.2 runtime)
#   • transformers 4.57.1  (inside Unsloth's allowed range)
#   • trl 0.22.2           (assistant-mask Qwen3 auto-patch works here)
#   • peft >= 0.18.0       (Unsloth main currently requires this)
#   • datasets >= 3.4.1, <4.4.0
#   • unsloth + unsloth_zoo AMD path (installed --no-deps to keep our torch)
#   • vLLM ROCm (optional; only needed later for GRPO)
#   • NO xformers — on ROCm, Unsloth falls back to SDPA (AOTriton on MI300X).
#   • NO bitsandbytes — we default to adamw_torch_fused (ROCm bnb has NaN risk).
# =============================================================================
set -euo pipefail

log()  { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m!!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[1;31mERR:\033[0m %s\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# 0. Sanity checks
# -----------------------------------------------------------------------------
command -v uv >/dev/null 2>&1 || die "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
command -v python3 >/dev/null 2>&1 || die "python3 not found."

# -----------------------------------------------------------------------------
# 0.1 ROCm detection (INFORMATIONAL ONLY — install uses a fixed wheel index)
# -----------------------------------------------------------------------------
detect_rocm() {
    local v=""
    # amd-smi table format ("ROCm version: 7.2.0")
    if command -v amd-smi >/dev/null 2>&1; then
        v=$(amd-smi 2>/dev/null | awk -F'ROCm version: ' 'NF>1 {split($2,a,"[ |]"); print a[1]; exit}' || true)
        [[ -n "$v" ]] && { echo "$v (amd-smi)"; return; }
        v=$(amd-smi version 2>/dev/null | awk -F'ROCm version: ' 'NF>1 {split($2,a,"[ |]"); print a[1]; exit}' || true)
        [[ -n "$v" ]] && { echo "$v (amd-smi version)"; return; }
    fi
    # /opt/rocm/.info/version (may lag the kernel driver in a container)
    if [[ -r /opt/rocm/.info/version ]]; then
        v=$(cat /opt/rocm/.info/version 2>/dev/null | head -1 | tr -d ' ')
        [[ -n "$v" ]] && { echo "$v (/opt/rocm/.info/version)"; return; }
    fi
    if command -v hipconfig >/dev/null 2>&1; then
        v=$(hipconfig --version 2>/dev/null || true)
        [[ -n "$v" ]] && { echo "$v (hipconfig)"; return; }
    fi
    echo "unknown"
}
ROCM_INFO=$(detect_rocm)
log "Detected ROCm: $ROCM_INFO"
log "(We install torch 2.10.0 from rocm7.1 index — ABI-compatible with ROCm 7.0-7.2)"

# -----------------------------------------------------------------------------
# 1. Virtualenv
# -----------------------------------------------------------------------------
if [[ ! -d .venv ]]; then
    log "Creating uv venv (.venv, python 3.12)"
    uv venv --python 3.12 .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

export HF_HUB_ENABLE_HF_TRANSFER=1
export PIP_EXTRA_INDEX_URL=""

# -----------------------------------------------------------------------------
# 2. PyTorch for ROCm
# -----------------------------------------------------------------------------
# torch 2.10.0 is what's actually on pytorch.org's rocm7.1 index (2.9.1 isn't).
# Works fine on ROCm 7.2 runtime — minor ROCm revs are ABI-stable.
log "Installing torch==2.10.0 (ROCm 7.1 wheels, works on ROCm 7.2 runtime)"
uv pip install --upgrade --force-reinstall \
    "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0" \
    --index-url https://download.pytorch.org/whl/rocm7.1 \
    || die "torch install failed. See NOTES at bottom for AMD wheel fallback."

python -c "
import torch
print(f'torch = {torch.__version__}')
print(f'cuda.is_available = {torch.cuda.is_available()}')
print(f'hip = {torch.version.hip}')
" || die "torch imported but something is off."

# -----------------------------------------------------------------------------
# 3. Core HF training deps (pinned inside Unsloth's supported ranges)
# -----------------------------------------------------------------------------
log "Installing transformers, trl, peft, accelerate, datasets, wandb, ..."
uv pip install \
    "transformers==4.57.1" \
    "trl==0.22.2" \
    "peft>=0.18.0" \
    "accelerate>=1.0.0" \
    "datasets>=3.4.1,<4.4.0" \
    "huggingface_hub>=0.34.0" \
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
    "typer" \
    "nest-asyncio" \
    "diffusers" \
    "cut_cross_entropy" \
    || die "HF deps install failed."

# -----------------------------------------------------------------------------
# 4. Unsloth (install --no-deps to preserve our torch)
# -----------------------------------------------------------------------------
# Note: we do NOT use `unsloth[amd]` because that extra pulls in its own torch
# from repo.radeon.com. We already have torch from pytorch.org and it works.
# At import time, Unsloth detects our ROCm platform and uses SDPA (no xformers
# needed on MI300X — AOTriton provides the fast-attention path).
log "Installing Unsloth from git (--no-deps)"
uv pip install --no-deps "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git"
uv pip install --no-deps "unsloth @ git+https://github.com/unslothai/unsloth"

python -c "import unsloth_zoo; print(f'unsloth_zoo = {unsloth_zoo.__version__}')"
python -c "import unsloth;     print(f'unsloth      = {unsloth.__version__}')"

# -----------------------------------------------------------------------------
# 5. vLLM ROCm (OPTIONAL — only for later GRPO, not for SFT)
# -----------------------------------------------------------------------------
log "Installing vLLM (ROCm wheels, --no-deps). Non-fatal if it fails."
if uv pip install --no-deps "vllm" \
       --extra-index-url https://wheels.vllm.ai/rocm/ 2>&1 | tail -5; then
    log "vLLM installed."
else
    warn "vLLM install failed — OK for an SFT-only box. Install on the GRPO machine."
fi

# -----------------------------------------------------------------------------
# 6. This project (editable)
# -----------------------------------------------------------------------------
log "Installing dc-ops-sft (editable, --no-deps)"
uv pip install -e . --no-deps

# -----------------------------------------------------------------------------
# 7. OPTIONAL: dc_ops_env for eval_compare.py
# -----------------------------------------------------------------------------
DC_OPS_ENV_PATH="${DC_OPS_ENV_PATH:-${1:-}}"
if [[ -n "$DC_OPS_ENV_PATH" ]] && [[ -d "$DC_OPS_ENV_PATH" ]]; then
    log "Installing dc_ops_env from $DC_OPS_ENV_PATH"
    uv pip install -e "$DC_OPS_ENV_PATH" --no-deps
    uv pip install "openenv-core[core]>=0.2.1" "fastapi>=0.115.0" "uvicorn>=0.24.0"
else
    warn "DC_OPS_ENV_PATH not set — skipping env install."
    warn "(Only needed for scripts/eval_compare.py. Pass the path as the first arg.)"
fi

# -----------------------------------------------------------------------------
# 8. Verification
# -----------------------------------------------------------------------------
log "Final verification:"
python - <<'PY'
import torch, transformers, trl, peft, datasets, unsloth, unsloth_zoo
print(f"  torch          : {torch.__version__}  (hip={torch.version.hip})")
print(f"  cuda_available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device_name    : {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info(0)
    print(f"  vram           : {free/2**30:.1f} / {total/2**30:.1f} GB free")
print(f"  transformers   : {transformers.__version__}")
print(f"  trl            : {trl.__version__}")
print(f"  peft           : {peft.__version__}")
print(f"  datasets       : {datasets.__version__}")
print(f"  unsloth        : {unsloth.__version__}")
print(f"  unsloth_zoo    : {unsloth_zoo.__version__}")
try:
    import vllm
    print(f"  vllm           : {vllm.__version__}")
except Exception as e:
    print(f"  vllm           : not installed ({type(e).__name__})")
PY

log "Install complete. Activate with: source .venv/bin/activate"
log "Next step: python -m dc_ops_sft.data unsloth/Qwen3-8B  (smoke check)"

# =============================================================================
# NOTES — fallback if step 2 (torch 2.10 / rocm7.1) fails on your box:
# =============================================================================
# Option A — AMD's manylinux wheels (exact-match ROCm 7.2.1):
#   BASE=https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1
#   uv pip install --force-reinstall \
#       $BASE/torch-2.9.1+rocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl \
#       $BASE/torchvision-0.24.0+rocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl \
#       $BASE/torchaudio-2.9.0+rocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl \
#       $BASE/triton-3.5.1+rocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl
#   uv pip install 'numpy<2.0'   # required with these wheels
#
# Option B — pytorch nightly rocm7.2:
#   uv pip install --pre torch torchvision torchaudio \
#       --index-url https://download.pytorch.org/whl/nightly/rocm7.2
# =============================================================================
