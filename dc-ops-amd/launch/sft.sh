#!/usr/bin/env bash
# ==============================================================================
# Background launcher for SFT training
# ------------------------------------------------------------------------------
# What it does:
#   1. Activates the .venv created by setup_env.sh
#   2. Loads .env (HUGGINGFACE_TOKEN, WANDB_API_KEY) if present
#   3. Runs scripts/run_sft.py under `nohup`, detached from the shell
#   4. Writes stdout+stderr to logs/sft-<timestamp>.log
#   5. Writes the PID to logs/sft.pid so you can `kill $(cat logs/sft.pid)`
#
# After launching, you can:
#   - Close the SSH session; the job keeps running.
#   - `tail -f logs/sft-*.log` to follow locally.
#   - Check wandb (configs/sft.yaml → wandb.project, wandb.run_name) for live
#     loss/eval curves.
#
# Usage:
#   chmod +x launch/sft.sh
#   ./launch/sft.sh                          # uses configs/sft.yaml
#   ./launch/sft.sh configs/my_sft.yaml      # or a custom config
# ==============================================================================

set -euo pipefail

CONFIG="${1:-configs/sft.yaml}"

# ---------- cd to repo root (so relative paths in the config work) ----------
cd "$(dirname "$0")/.."
ROOT="$PWD"
echo "[sft.sh] repo root: $ROOT"
echo "[sft.sh] config:    $CONFIG"

[[ -f "$CONFIG" ]] || { echo "[sft.sh] config not found: $CONFIG"; exit 1; }

# ---------- load .env if present ----------
if [[ -f .env ]]; then
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
  echo "[sft.sh] loaded .env"
fi

# ---------- activate venv ----------
if [[ ! -d .venv ]]; then
  echo "[sft.sh] .venv not found — run ./setup_env.sh first" >&2
  exit 1
fi
# shellcheck source=/dev/null
source .venv/bin/activate
echo "[sft.sh] venv python: $(which python)"

# ---------- sanity: wandb key is present if wandb is enabled ----------
if grep -Eq '^\s*enabled:\s*true' "$CONFIG"; then
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[sft.sh] ERROR: wandb.enabled=true in $CONFIG but WANDB_API_KEY is empty"
    echo "          set it in .env or disable wandb in the config"
    exit 1
  fi
fi

# ---------- launch ----------
mkdir -p logs
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="logs/sft-${STAMP}.log"
PID_FILE="logs/sft.pid"

# stdbuf -oL makes output line-buffered so tail -f is responsive
nohup stdbuf -oL -eL python -u scripts/run_sft.py --config "$CONFIG" \
    > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "[sft.sh] launched PID=$PID"
echo "[sft.sh] log:     $LOG"
echo "[sft.sh] to follow:  tail -f $LOG"
echo "[sft.sh] to kill:    kill \$(cat $PID_FILE)"
echo "[sft.sh] wandb:      check the project configured in $CONFIG"
