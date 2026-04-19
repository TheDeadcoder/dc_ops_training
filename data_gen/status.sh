#!/usr/bin/env bash
# Print a snapshot of generation progress: PID alive?, episodes written so
# far across all shards, planned total, est rate / ETA, last log line.
#
# Safe to run any time. Read-only.

set -euo pipefail

OUT_DIR="${OUTPUT_DIR:-data_out}"
PID_FILE="$OUT_DIR/generate.pid"
LOG_FILE="$OUT_DIR/generate.log"
START_FILE="$OUT_DIR/generate.started_at"
SHARD_DIR="$OUT_DIR/raw_episodes"

echo "=== DC-Ops generation status ==="

# --- Process state
if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Process : RUNNING  (PID $PID)"
    else
        echo "Process : NOT RUNNING (stale PID $PID — finished or crashed)"
    fi
else
    echo "Process : no PID file (no run started, or stop.sh was used)"
fi

# --- Wall time
if [[ -f "$START_FILE" ]]; then
    START_ISO=$(cat "$START_FILE")
    START_EPOCH=$(date -d "$START_ISO" +%s 2>/dev/null || echo 0)
    NOW_EPOCH=$(date -u +%s)
    if [[ "$START_EPOCH" -gt 0 ]]; then
        ELAPSED_S=$((NOW_EPOCH - START_EPOCH))
        ELAPSED_MIN=$((ELAPSED_S / 60))
        echo "Started : $START_ISO  (${ELAPSED_MIN} min ago)"
    fi
fi

# --- Shard counts
if [[ -d "$SHARD_DIR" ]]; then
    SHARD_COUNT=$(find "$SHARD_DIR" -name 'shard_*.jsonl' -type f | wc -l | tr -d ' ')
    if [[ "$SHARD_COUNT" -gt 0 ]]; then
        # Count total episodes (lines) across all shards
        EPISODES=$(cat "$SHARD_DIR"/shard_*.jsonl 2>/dev/null | wc -l | tr -d ' ')
        # Count error episodes (those with "error" key)
        ERRORS=$(cat "$SHARD_DIR"/shard_*.jsonl 2>/dev/null | grep -c '"error":' || true)
        OK=$((EPISODES - ERRORS))
        echo "Shards  : $SHARD_COUNT files in $SHARD_DIR/"
        echo "Episodes: $EPISODES total ($OK ok, $ERRORS errored)"
    else
        echo "Episodes: 0 (no shards yet)"
        EPISODES=0
    fi
else
    echo "Episodes: 0 (raw_episodes dir not created yet)"
    EPISODES=0
fi

# --- Planned total (parse from log)
if [[ -f "$LOG_FILE" ]]; then
    PLANNED=$(grep -oE 'total jobs: [0-9]+' "$LOG_FILE" | head -1 | grep -oE '[0-9]+' || echo "?")
    echo "Planned : $PLANNED"

    # Rate + ETA from elapsed time and episode count
    if [[ "${ELAPSED_S:-0}" -gt 0 && "$EPISODES" -gt 0 && "$PLANNED" != "?" ]]; then
        # Episodes per minute (integer arithmetic)
        EPS_PER_MIN=$(awk -v e="$EPISODES" -v t="$ELAPSED_S" 'BEGIN{printf "%.2f", e*60/t}')
        REMAINING=$((PLANNED - EPISODES))
        if [[ "$REMAINING" -gt 0 ]]; then
            ETA_MIN=$(awk -v r="$REMAINING" -v rate="$EPS_PER_MIN" 'BEGIN{if(rate>0) printf "%.0f", r/rate; else print "?"}')
            ETA_HOUR=$(awk -v m="$ETA_MIN" 'BEGIN{printf "%.1f", m/60}')
            echo "Rate    : $EPS_PER_MIN eps/min"
            echo "ETA     : ~$ETA_MIN min (~${ETA_HOUR}h) for remaining $REMAINING episodes"
        else
            echo "ETA     : DONE"
        fi
    fi
fi

# --- Latest progress line from python
if [[ -f "$LOG_FILE" ]]; then
    LAST_PROGRESS=$(grep -E '\[progress\]' "$LOG_FILE" | tail -1 || true)
    if [[ -n "$LAST_PROGRESS" ]]; then
        echo
        echo "Latest log:"
        echo "  $LAST_PROGRESS"
    fi
    # Also surface the last 3 non-progress lines (errors, warnings)
    LAST_OTHER=$(grep -vE '\[progress\]' "$LOG_FILE" | tail -3 || true)
    if [[ -n "$LAST_OTHER" ]]; then
        echo
        echo "Last 3 non-progress lines:"
        echo "$LAST_OTHER" | sed 's/^/  /'
    fi
fi

# --- Per-scenario completion breakdown
if [[ "${EPISODES:-0}" -gt 0 ]]; then
    echo
    echo "Per-scenario completed (from shard metadata):"
    cat "$SHARD_DIR"/shard_*.jsonl 2>/dev/null \
        | python3 -c "
import json, sys
from collections import Counter
c = Counter()
err = Counter()
for line in sys.stdin:
    try:
        rec = json.loads(line)
        key = rec.get('metadata', {}).get('scenario_key') or rec.get('scenario_key', '?')
        if 'error' in rec:
            err[key] += 1
        else:
            c[key] += 1
    except Exception:
        continue
for k in sorted(set(list(c.keys()) + list(err.keys()))):
    print(f'  {k:20s} ok={c[k]:4d}  err={err[k]:3d}')
" 2>/dev/null || echo "  (could not summarize)"
fi
