#!/usr/bin/env bash
# Gracefully stop the background generate.py run.
# - Sends SIGTERM (asyncio cleanup runs, in-flight episodes finish writing)
# - Waits up to 30s
# - Falls back to SIGKILL if needed
# - Removes PID file

set -euo pipefail

OUT_DIR="${OUTPUT_DIR:-data_out}"
PID_FILE="$OUT_DIR/generate.pid"

if [[ ! -f "$PID_FILE" ]]; then
    echo "No PID file at $PID_FILE — nothing to stop."
    exit 0
fi

PID=$(cat "$PID_FILE")
if ! kill -0 "$PID" 2>/dev/null; then
    echo "Process $PID is not running. Cleaning up stale PID file."
    rm -f "$PID_FILE"
    exit 0
fi

echo "Sending SIGTERM to PID $PID ..."
kill -TERM "$PID" || true

# Also signal the entire process group so worker subprocesses get the signal too
PGID=$(ps -o pgid= -p "$PID" | tr -d ' ' 2>/dev/null || echo "")
if [[ -n "$PGID" ]]; then
    kill -TERM -"$PGID" 2>/dev/null || true
fi

# Wait up to 30s for graceful exit
for i in $(seq 1 30); do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "Process exited cleanly after ${i}s."
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
done

echo "Did not exit after 30s. Sending SIGKILL ..."
kill -KILL "$PID" 2>/dev/null || true
if [[ -n "$PGID" ]]; then
    kill -KILL -"$PGID" 2>/dev/null || true
fi
sleep 1
rm -f "$PID_FILE"
echo "Killed."
