#!/usr/bin/env bash
# Waits for the first analysis to finish, then runs a second one to a different location.
# Usage: nohup ./scripts/run-second-analysis-after-first.sh > /tmp/second-analysis.log 2>&1 &

OUTPUT_DIR="/tmp/style-analysis-2"
LOG_FILE="/tmp/second-analysis-run.log"

# Find brainlayer analyze-evolution process (exclude this script's parent)
find_brainlayer_pid() {
  pgrep -f "brainlayer analyze-evolution" | head -1
}

echo "[$(date)] Starting. Will sleep 1 hour, then poll until first run finishes."
sleep 3600

BRAINLAYER_PID=$(find_brainlayer_pid)
while [ -n "$BRAINLAYER_PID" ]; do
  echo "[$(date)] First run still in progress (PID $BRAINLAYER_PID). Sleeping 30 minutes..."
  sleep 1800
  BRAINLAYER_PID=$(find_brainlayer_pid)
done

echo "[$(date)] First run finished. Starting second analysis to $OUTPUT_DIR"
cd /path/to/brainlayer && source .venv/bin/activate
echo "y" | brainlayer analyze-evolution \
  --claude-export /tmp/claude-export/conversations.json \
  --output "$OUTPUT_DIR" \
  --granularity half \
  --model qwen3-coder-64k 2>&1 | tee "$LOG_FILE"

echo "[$(date)] Second analysis complete. Output in $OUTPUT_DIR"
