#!/usr/bin/env bash
# ==============================================================================
# DC-Ops Training — ROCm 7.2 (MI300X) environment bootstrap
# ==============================================================================
# One-shot setup for the full training stack using `uv`. Designed for:
#   - AMD Instinct MI300X (gfx942), ROCm 7.2.0
#   - Ubuntu 22.04 / 24.04, Python 3.12
#   - No CUDA, no nvidia-* packages anywhere
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh
#
# What it installs (in this order, pinned for version-compat):
#   1. uv itself (if missing)
#   2. A Python 3.12 venv at ./.venv
#   3. PyTorch 2.10.0 + ROCm 7.1 wheels
#      (ROCm 7.1 torch wheels are forward-compat with ROCm 7.2 runtime —
#       this is the currently-stable pairing; rocm7.2 nightly exists but
#       breaks more often. We use this deliberately.)
#   4. Triton 3.6.0 (matches torch 2.10 on ROCm)
#   5. bitsandbytes — ROCm fork from github.com/ROCm/bitsandbytes
#      (built-from-source; required for 4-bit QLoRA on AMD)
#   6. flash-attn — ROCm fork (CK backend, fastest on MI300X)
#   7. vLLM — ROCm wheel (needed for GRPO fast-generation)
#   8. xformers — ROCm-compatible build
#   9. transformers / tokenizers / peft / accelerate / trl — pinned to known-
#      good versions from the notebook
#  10. unsloth + unsloth_zoo — pulls the official AMD install path
#  11. OpenEnv core (for dc_ops_env)
#  12. wandb, datasets, etc.
#
# Exits non-zero on any failure; re-runnable (idempotent where possible).
# ==============================================================================

set -euo pipefail

# ---------- colours ----------
BLUE='\033[0;34m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${BLUE}[setup]${NC} $*"; }
ok()   { echo -e "${GREEN}[ ok ]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
err()  { echo -e "${RED}[fail]${NC} $*" >&2; exit 1; }

# ---------- 0. sanity ----------
log "checking ROCm / GPU visibility"
command -v amd-smi >/dev/null 2>&1 || err "amd-smi not found — is ROCm 7.2 installed?"
amd-smi list 2>/dev/null | grep -q "MI300X" || warn "MI300X not detected in amd-smi list (continuing anyway)"

ROCM_VERSION=$(amd-smi version 2>/dev/null | awk -F'ROCm version: ' 'NF>1{print $2; exit}' | awk '{print $1}')
log "detected ROCm version: ${ROCM_VERSION:-unknown}"

# gfx942 is MI300X; verify
if command -v rocminfo >/dev/null 2>&1; then
  GFX=$(rocminfo 2>/dev/null | grep -m1 "Name: *gfx" | awk '{print $2}')
  log "detected GPU gfx arch: ${GFX:-unknown}"
  [[ "$GFX" == "gfx942" ]] || warn "Expected gfx942 for MI300X, got $GFX"
fi

# ---------- 1. uv ----------
if ! command -v uv >/dev/null 2>&1; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
ok "uv $(uv --version)"

# ---------- 2. venv (Python 3.12) ----------
if [[ ! -d .venv ]]; then
  log "creating Python 3.12 venv at ./.venv"
  uv venv --python 3.12 .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
ok "venv active: $(which python) ($(python --version))"

# always use --python from the venv for uv pip
UV_PIP="uv pip install --python .venv/bin/python"

# ---------- 3. PyTorch + torchvision + torchaudio for ROCm ----------
# torch 2.10.0 + rocm7.1 wheels run fine on ROCm 7.2 userspace (forward compat).
# This is the set used by vLLM's ROCm 7.1 constraints file (triton 3.6, etc.).
log "installing PyTorch 2.10.0 + rocm7.1 wheels"
$UV_PIP install --upgrade pip wheel setuptools
$UV_PIP install \
  torch==2.10.0 \
  torchvision==0.25.0 \
  torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/rocm7.1

# verify GPU visible
python - <<'PY' || err "torch.cuda (ROCm) not available"
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
print(f"  torch {torch.__version__}  |  HIP {torch.version.hip}  |  device {torch.cuda.get_device_name(0)}")
PY
ok "torch sees the GPU"

# ---------- 4. Triton (matched to torch 2.10 / rocm7.1) ----------
log "installing triton 3.6.0"
$UV_PIP install --upgrade triton==3.6.0

# ---------- 5. Core training libs (pinned from the notebook) ----------
log "installing transformers / tokenizers / peft / accelerate / trl"
$UV_PIP install \
  "transformers==4.54.1" \
  "tokenizers==0.21.0" \
  "peft>=0.13,<0.17" \
  "huggingface-hub>=0.34.0,<1.0" \
  "accelerate>=1.2.0" \
  "datasets>=3.0" \
  "sentencepiece" \
  "protobuf" \
  "safetensors"

# trl — pinned to a version known to work with transformers 4.54 + GRPO
$UV_PIP install "trl==0.21.0"

# ---------- 6. xformers (ROCm-compatible build) ----------
# xformers 0.0.34 pairs with torch 2.10 on ROCm 7.1 (per vLLM constraints).
# We install it --no-build-isolation so it picks up the ROCm torch we already have.
log "installing xformers 0.0.34 (ROCm build)"
PYTORCH_ROCM_ARCH="gfx942" \
  $UV_PIP install --no-build-isolation "xformers==0.0.34" || warn "xformers install failed — SDPA will fallback to PyTorch native. Training still works."

# ---------- 7. bitsandbytes — ROCm fork ----------
# Needed for 4-bit QLoRA (load_in_4bit=True). The upstream bnb does not support
# ROCm; we must build from the ROCm/bitsandbytes rocm_enabled_multi_backend branch.
log "installing bitsandbytes (ROCm fork, built from source — ~2-4 min)"
BNB_DIR="$PWD/.build/bitsandbytes"
if [[ ! -d "$BNB_DIR" ]]; then
  mkdir -p "$(dirname "$BNB_DIR")"
  git clone --recurse-submodules --branch rocm_enabled_multi_backend \
    https://github.com/ROCm/bitsandbytes "$BNB_DIR"
fi
(
  cd "$BNB_DIR"
  $UV_PIP install -r requirements-dev.txt
  # ROCm 7.2 uses amdclang; force gfx942
  cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx942" -S . -B build
  cmake --build build -j"$(nproc)"
  $UV_PIP install -e .
)
python - <<'PY' || err "bitsandbytes not usable"
import bitsandbytes as bnb
print(f"  bitsandbytes {bnb.__version__}")
PY
ok "bitsandbytes ROCm build installed"

# ---------- 8. Flash-Attention — ROCm CK backend ----------
# The CK (Composable Kernel) flash-attn is the fastest option on MI300X.
# We build with GPU_ARCHS=gfx942 only, which cuts compile from ~30min to ~8min.
log "installing flash-attention (ROCm CK backend, gfx942-only — ~5-10 min)"
FA_DIR="$PWD/.build/flash-attention"
if [[ ! -d "$FA_DIR" ]]; then
  git clone --recurse-submodules https://github.com/ROCm/flash-attention "$FA_DIR"
fi
(
  cd "$FA_DIR"
  # pin to a tag known to build against torch 2.10 on ROCm 7.1/7.2
  git checkout main_perf 2>/dev/null || git checkout main
  GPU_ARCHS="gfx942" \
  MAX_JOBS="$(nproc)" \
  FLASH_ATTENTION_FORCE_BUILD="TRUE" \
    $UV_PIP install . --no-build-isolation
)
python - <<'PY' || warn "flash-attn import failed — transformers will fallback to SDPA (still fast on MI300X)"
import flash_attn
print(f"  flash-attn {flash_attn.__version__}")
PY
ok "flash-attention installed"

# ---------- 9. vLLM for ROCm ----------
# vLLM on ROCm 7.2 is available as a prebuilt wheel at wheels.vllm.ai/rocm/.
# We install --no-deps to stop it from yanking in a CPU-only torch from pypi.
log "installing vllm (ROCm wheel, --no-deps)"
$UV_PIP install --no-deps --upgrade \
  vllm \
  --extra-index-url https://wheels.vllm.ai/rocm/ || \
$UV_PIP install --no-deps --upgrade \
  vllm \
  --extra-index-url https://download.pytorch.org/whl/rocm7.1
# pull the few missing runtime bits (ray, openai, etc) — without torch override
$UV_PIP install \
  ray \
  "openai>=1.0" \
  outlines \
  prometheus_client \
  py-cpuinfo \
  msgspec \
  uvloop \
  "fastapi>=0.110" \
  "aiohttp>=3.9"

python - <<'PY' || warn "vllm import failed — GRPO fast-generation will be unavailable"
import vllm
print(f"  vllm {vllm.__version__}")
PY

# ---------- 10. Unsloth (AMD/ROCm path) ----------
# Official install per unsloth.ai/docs/get-started/install/amd.
# unsloth_zoo must come BEFORE unsloth.
log "installing unsloth_zoo + unsloth (AMD path)"
$UV_PIP install unsloth_zoo
$UV_PIP install --no-deps "unsloth==2026.4.4"

# ---------- 11. OpenEnv + DC-Ops environment ----------
log "installing OpenEnv core"
$UV_PIP install --upgrade "openenv-core[core]>=0.2.1"

# dc_ops_env is installed from the sibling ../dc_ops_environment clone.
# If the user hasn't cloned it yet, we do it here.
DC_OPS_DIR="$PWD/../dc_ops_environment"
if [[ ! -d "$DC_OPS_DIR" ]]; then
  log "cloning dc_ops_environment alongside this repo"
  git clone https://github.com/TheDeadcoder/dc_ops_environment.git "$DC_OPS_DIR" || \
    warn "couldn't clone dc_ops_environment — please copy it to $DC_OPS_DIR manually"
fi
if [[ -d "$DC_OPS_DIR/dc_ops_env" ]]; then
  log "installing dc_ops_env in editable mode"
  $UV_PIP install -e "$DC_OPS_DIR/dc_ops_env"
fi

# ---------- 12. Training-side utilities ----------
log "installing wandb, pyyaml, etc."
$UV_PIP install \
  wandb \
  pyyaml \
  python-dotenv \
  rich \
  tqdm

# ---------- 13. Pin lock — re-assert the core versions in case some transitive
#              dep (unsloth_zoo in particular) tried to bump them.
log "re-pinning transformers / tokenizers / hf-hub to avoid transitive upgrades"
$UV_PIP install --force-reinstall --no-deps \
  "transformers==4.54.1" \
  "tokenizers==0.21.0" \
  "huggingface-hub>=0.34.0,<1.0"

# ---------- 14. Final verification ----------
log "running final import check"
python - <<'PY'
import torch, transformers, peft, tokenizers, trl, accelerate
print(f"torch        {torch.__version__}  (HIP {torch.version.hip})")
print(f"transformers {transformers.__version__}")
print(f"tokenizers   {tokenizers.__version__}")
print(f"peft         {peft.__version__}")
print(f"trl          {trl.__version__}")
print(f"accelerate   {accelerate.__version__}")
try:
    import bitsandbytes as bnb; print(f"bitsandbytes {bnb.__version__}")
except Exception as e: print(f"bitsandbytes MISSING: {e}")
try:
    import unsloth; print(f"unsloth      {unsloth.__version__}")
except Exception as e: print(f"unsloth MISSING: {e}")
try:
    import vllm; print(f"vllm         {vllm.__version__}")
except Exception as e: print(f"vllm MISSING: {e}")
try:
    import flash_attn; print(f"flash-attn   {flash_attn.__version__}")
except Exception as e: print(f"flash-attn MISSING (not fatal, SDPA will be used): {e}")

assert torch.cuda.is_available()
print(f"GPU          {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
print(f"bf16 support {torch.cuda.is_bf16_supported()}")
PY

ok "--------------------------------------------------"
ok "Setup complete. Activate with:  source .venv/bin/activate"
ok "Then see README.md for how to run SFT → push → GRPO."
ok "--------------------------------------------------"
