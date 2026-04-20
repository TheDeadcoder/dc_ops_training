#!/usr/bin/env bash
# ==============================================================================
# Background launcher for GRPO training
# ------------------------------------------------------------------------------
# Identical pattern to launch/sft.sh but for GRPO. See that file for the
# full explanation of `nohup` / logging / PID handling.
#
# Usage:
#   chmod +x launch/grpo.sh
#   ./launch/grpo.sh                          # uses configs/grpo.yaml
#   ./launch/grpo.sh configs/my_grpo.yaml     # or a custom config
# ==============================================================================

set -euo pipefail

CONFIG="${1:-configs/grpo.yaml}"

cd "$(dirname "$0")/.."
ROOT="$PWD"
echo "[grpo.sh] repo root: $ROOT"
echo "[grpo.sh] config:    $CONFIG"

[[ -f "$CONFIG" ]] || { echo "[grpo.sh] config not found: $CONFIG"; exit 1; }

# ---------- load .env ----------
if [[ -f .env ]]; then
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
  echo "[grpo.sh] loaded .env"
fi

# ---------- activate venv ----------
if [[ ! -d .venv ]]; then
  echo "[grpo.sh] .venv not found — run ./setup_env.sh first" >&2
  exit 1
fi
# shellcheck source=/dev/null
source .venv/bin/activate
echo "[grpo.sh] venv python: $(which python)"

# ---------- sanity ----------
if grep -Eq '^\s*enabled:\s*true' "$CONFIG" | head -1 >/dev/null; then
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[grpo.sh] WARNING: wandb.enabled=true but WANDB_API_KEY is empty"
    echo "           training may error at init. Set it in .env."
  fi
fi

# GRPO will try to pull the SFT LoRA from HF if no local copy exists → need HF token.
if [[ ! -d outputs/dc_ops_sft_lora ]] && [[ -z "${HUGGINGFACE_TOKEN:-}${HF_TOKEN:-}" ]]; then
  echo "[grpo.sh] WARNING: no local SFT LoRA at outputs/dc_ops_sft_lora"
  echo "           AND no HUGGINGFACE_TOKEN set. Hub download will fail for"
  echo "           private repos. Set HUGGINGFACE_TOKEN in .env or push the LoRA"
  echo "           to a public repo first."
fi

# ---------- launch ----------
mkdir -p logs
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="logs/grpo-${STAMP}.log"
PID_FILE="logs/grpo.pid"

nohup stdbuf -oL -eL python -u scripts/run_grpo.py --config "$CONFIG" \
    > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "[grpo.sh] launched PID=$PID"
echo "[grpo.sh] log:     $LOG"
echo "[grpo.sh] to follow:  tail -f $LOG"
echo "[grpo.sh] to kill:    kill \$(cat $PID_FILE)"
echo "[grpo.sh] wandb:      check the project configured in $CONFIG"
