#!/usr/bin/env bash
# Launch generate.py in the background, fully detached from the terminal.
# - Logs go to data_out/generate.log (stdout + stderr merged)
# - PID is written to data_out/generate.pid
# - Survives SSH disconnect (uses setsid + nohup)
#
# Usage:
#     bash run_in_background.sh                  # uses NUM_WORKERS=28 by default
#     NUM_WORKERS=24 bash run_in_background.sh   # override
#     DC_OPS_ENV_PATH=/path bash run_in_background.sh
#
# After launch:
#     bash status.sh           # one-shot snapshot
#     tail -f data_out/generate.log
#     bash stop.sh             # graceful stop

set -euo pipefail

OUT_DIR="${OUTPUT_DIR:-data_out}"
mkdir -p "$OUT_DIR"

LOG_FILE="$OUT_DIR/generate.log"
PID_FILE="$OUT_DIR/generate.pid"
START_FILE="$OUT_DIR/generate.started_at"

# Refuse to start if a previous run is still alive
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "ERROR: a generation run is already active (PID $OLD_PID)."
        echo "       Use 'bash stop.sh' first, or delete $PID_FILE if it is stale."
        exit 1
    else
        echo "Stale PID file removed (process $OLD_PID is no longer running)."
        rm -f "$PID_FILE"
    fi
fi

NUM_WORKERS="${NUM_WORKERS:-28}"
export NUM_WORKERS

echo "Launching generate.py with NUM_WORKERS=$NUM_WORKERS ..."
echo "  logs : $LOG_FILE"
echo "  pid  : $PID_FILE"

# setsid + nohup + & = process detaches cleanly from terminal session.
# python -u disables output buffering so tail -f sees progress immediately.
setsid nohup python -u generate.py >"$LOG_FILE" 2>&1 &
RUN_PID=$!

echo "$RUN_PID" >"$PID_FILE"
date -u +"%Y-%m-%dT%H:%M:%SZ" >"$START_FILE"

# Brief sanity check — give it 2 seconds to confirm it didn't crash on startup
sleep 2
if ! kill -0 "$RUN_PID" 2>/dev/null; then
    echo
    echo "ERROR: process exited immediately. Last 30 log lines:"
    echo "----------------------------------------"
    tail -n 30 "$LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi

echo "Started OK. PID $RUN_PID is running."
echo
echo "Next steps:"
echo "  bash status.sh             # progress snapshot"
echo "  tail -f $LOG_FILE          # live log stream"
echo "  bash stop.sh               # graceful shutdown"
