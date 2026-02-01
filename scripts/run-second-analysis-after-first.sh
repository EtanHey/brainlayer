#!/usr/bin/env bash
# Waits for the first analysis to finish, then runs a second one to a different location.
# Usage: nohup ./scripts/run-second-analysis-after-first.sh > /tmp/second-analysis.log 2>&1 &

OUTPUT_DIR="/tmp/style-analysis-2"
LOG_FILE="/tmp/second-analysis-run.log"

# Find zikaron analyze-evolution process (exclude this script's parent)
find_zikaron_pid() {
  pgrep -f "zikaron analyze-evolution" | head -1
}

echo "[$(date)] Starting. Will sleep 1 hour, then poll until first run finishes."
sleep 3600

ZIKARON_PID=$(find_zikaron_pid)
while [ -n "$ZIKARON_PID" ]; do
  echo "[$(date)] First run still in progress (PID $ZIKARON_PID). Sleeping 30 minutes..."
  sleep 1800
  ZIKARON_PID=$(find_zikaron_pid)
done

echo "[$(date)] First run finished. Starting second analysis to $OUTPUT_DIR"
cd /Users/etanheyman/Gits/zikaron && source .venv/bin/activate
echo "y" | zikaron analyze-evolution \
  --claude-export /tmp/claude-export/conversations.json \
  --output "$OUTPUT_DIR" \
  --granularity half \
  --model qwen3-coder-64k 2>&1 | tee "$LOG_FILE"

echo "[$(date)] Second analysis complete. Output in $OUTPUT_DIR"
