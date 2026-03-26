#!/usr/bin/env bash
# backfill_orchestrate.sh — Full pipeline orchestrator for Gemini batch enrichment
# Handles: resume submitted → fix broken-completed → resubmit failed+unsubmitted → resume again
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/../.venv/bin/python3"
BACKFILL="$SCRIPT_DIR/cloud_backfill.py"
LOG="/tmp/backfill_orchestrate.log"
API_KEY="${GOOGLE_API_KEY:-}"

if [[ -z "$API_KEY" ]]; then
    echo "ERROR: GOOGLE_API_KEY not set"
    exit 1
fi

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== ORCHESTRATOR START ==="
cd "$SCRIPT_DIR/.."

# PHASE 1: Resume all submitted batches (may already be running)
log "Phase 1: Resume submitted batches..."
GOOGLE_API_KEY="$API_KEY" "$PYTHON" -u "$BACKFILL" --resume 2>&1 | tee -a "$LOG"

# PHASE 2: Fix "completed" batches that had 0 imports (broken download bug)
# The broken run completed all its batches at 2026-03-14T12:21 UTC.
# Current (correct) run completes batches after 2026-03-14T12:22 UTC.
log ""
log "Phase 2: Checking for incorrectly-completed batches (0 imports due to broken download)..."
"$PYTHON" -u - <<'PYEOF' 2>&1 | tee -a "$LOG"
import apsw
from pathlib import Path

cp_db = Path.home() / '.local/share/brainlayer/enrichment_checkpoints.db'
conn = apsw.Connection(str(cp_db))
conn.setbusytimeout(10000)

# Find completed batches from the broken 3rd run (completed at 12:21 UTC)
# These had 0 imports due to Files.download(name=) bug that has since been fixed
rows = list(conn.cursor().execute("""
    SELECT batch_id, chunk_count, completed_at
    FROM enrichment_checkpoints
    WHERE status = 'completed'
    AND completed_at < '2026-03-14T12:22:00'
"""))

print(f"Found {len(rows)} broken 'completed' batches (before fix at 12:22 UTC)")
if rows:
    # Reset them to 'submitted' so --resume will re-process them
    conn.cursor().execute("""
        UPDATE enrichment_checkpoints
        SET status = 'submitted', completed_at = NULL
        WHERE status = 'completed'
        AND completed_at < '2026-03-14T12:22:00'
    """)
    print(f"Reset {len(rows)} batches to 'submitted'")
else:
    print("No broken batches found - all completed batches imported correctly")

conn.close()
PYEOF

# PHASE 3: Resume the reset batches
log ""
log "Phase 3: Resume reset batches..."
GOOGLE_API_KEY="$API_KEY" "$PYTHON" -u "$BACKFILL" --resume 2>&1 | tee -a "$LOG"

# PHASE 4: Resubmit failed batches + submit unsubmitted JSONL files
log ""
log "Phase 4: Resubmit failed/unsubmitted batches..."
GOOGLE_API_KEY="$API_KEY" "$PYTHON" -u "$BACKFILL" --submit-only 2>&1 | tee -a "$LOG"

log ""
log "Phase 4 done. Waiting 60s for jobs to register before polling..."
sleep 60

# PHASE 5: Resume all newly submitted batches
log ""
log "Phase 5: Resume newly submitted batches (poll + import)..."
GOOGLE_API_KEY="$API_KEY" "$PYTHON" -u "$BACKFILL" --resume 2>&1 | tee -a "$LOG"

# Final stats
log ""
log "=== FINAL STATUS ==="
GOOGLE_API_KEY="$API_KEY" "$PYTHON" -u "$BACKFILL" --status 2>&1 | tee -a "$LOG"

log "=== ORCHESTRATOR COMPLETE ==="
