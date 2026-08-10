# Wave 3a two-host LIVE runbook

Status: rehearsal-proven; LIVE execution scheduled for Tuesday 2026-08-11 at 20:00 IDT or later with Etan present.

This runbook is the source-class segment of the Wave 3 maintenance sitting. Execute it on Main first, reconcile and probe Main, then repeat it on M1. Do not begin a host without Etan's explicit go. Keep enrichment off wherever the pause sentinel applies.

## Pinned artifacts and rehearsal measurements

- Migration code commit: `be877608ae5ff14de079ccd40949ef14f88aa578`.
- Command: `scripts/migrate_source_class.py` from a checkout containing that commit.
- Rehearsal source: online SQLite backup from canonical opened `mode=ro` into gitignored `.tmp-wave3-3a/` (contingency route, not the LIVE primary route).
- Timed contingency replay: 26.16 seconds wall-clock while live drain writes continued; 744,396 rows, 4,251,874 pages, 17,415,675,904 bytes. The pristine rehearsal copy passed `PRAGMA quick_check` before migration.
- Rehearsal rows: 744,335 before and after.
- Rehearsal migration: 198.82 seconds wall-clock on reviewed migration commit `be877608ae5ff14de079ccd40949ef14f88aa578` (`duration_seconds=198.79003841709346`).
- Rehearsal distribution: NULL 69,581; brain-worker 84; cli-agent 536,851; desktop 2,696; fleet-coordination 97,678; subagent 37,445.
- Rehearsal migration WAL: 0 bytes before, 29,181,992-byte observed peak, 0 bytes at close after checkpoints.
- Rehearsal rollback: restore by APFS re-copy in 0.00 seconds; restored row count 744,335 and `source_class` absent.

## Host receipt table

Fill every blank on each host. Do not copy Main values into M1.

| Field | Main | M1 |
|---|---|---|
| Operator / Etan go time |  |  |
| Hostname |  |  |
| Canonical DB path |  |  |
| Checked-out commit |  |  |
| BrainBar version (must be 1.5.4+) |  |  |
| Free bytes |  |  |
| Active labels receipt file |  |  |
| Pause sentinel contents |  |  |
| Rows before |  |  |
| WAL bytes before stop |  |  |
| Checkpoint result / WAL bytes after |  |  |
| Backup-pipeline JSON receipt |  |  |
| Verified local raw backup ref |  |  |
| Verified Drive file id / MD5 |  |  |
| Backup wall-clock |  |  |
| Migration wall-clock |  |  |
| Rows after |  |  |
| NULL / cli / desktop / subagent / brain-worker / fleet |  |  |
| Both ledger SHAs |  |  |
| Migration WAL bytes after |  |  |
| SQL reconcile |  |  |
| Real service probes |  |  |
| Restarted labels |  |  |
| Etan host acceptance |  |  |

## 1. Host preflight and writer inventory

From the pinned checkout, set only host-local absolute paths:

```bash
export WAVE3A_REPO="$PWD"
export WAVE3A_PYTHON="$WAVE3A_REPO/.venv/bin/python"
export WAVE3A_CLI="$WAVE3A_REPO/.venv/bin/brainlayer"
export PYTHONPATH="$WAVE3A_REPO/src"
export WAVE3A_DB="$($WAVE3A_PYTHON -c 'from brainlayer.paths import get_db_path; print(get_db_path())')"
export WAVE3A_SHA="be877608ae5ff14de079ccd40949ef14f88aa578"
export WAVE3A_ACTOR="etan+brainlayerCodex"
export WAVE3A_HOST="$(hostname -s | tr -cd 'A-Za-z0-9_-')"
export WAVE3A_RUN_DIR="$HOME/.local/share/brainlayer/wave3-live-2026-08-11/$WAVE3A_HOST"
mkdir -p "$WAVE3A_RUN_DIR"
test -x "$WAVE3A_PYTHON"
test -x "$WAVE3A_CLI"
test "$(git rev-parse "$WAVE3A_SHA")" = "$WAVE3A_SHA"
git merge-base --is-ancestor "$WAVE3A_SHA" HEAD
test -f "$WAVE3A_DB"
/usr/libexec/PlistBuddy -c 'Print:CFBundleShortVersionString' /Applications/BrainBar.app/Contents/Info.plist
df -k "$(dirname "$WAVE3A_DB")"
stat -f '%z' "$WAVE3A_DB"
stat -f '%z' "$WAVE3A_DB-wal" 2>/dev/null || true
cat "$HOME/.local/share/brainlayer/pause.sentinel" 2>/dev/null || true
pgrep -af '[b]rain_digest|[b]rainlayer.*digest' && exit 1 || true
pgrep -af '[b]rainlayer.backup_daily|[b]ackup-daily.sh' && exit 1 || true
```

Require more than 40 GiB free, no digest, no concurrent backup, and BrainBar 1.5.4 or later. Record the exact active set before changing launchd:

```bash
launchctl list | awk '$3 ~ /^com\.brainlayer\./ {print $3}' | sort > "$WAVE3A_RUN_DIR/active-labels.before"
: > "$WAVE3A_RUN_DIR/stopped-labels"
cat "$WAVE3A_RUN_DIR/active-labels.before"
grep -Fxq 'com.brainlayer.brainbar-daemon' "$WAVE3A_RUN_DIR/active-labels.before"
```

Stop auto-restart/watchdog labels first, then all DB writers and scheduled maintenance. Do not start enrichment if it was absent or sentinel-paused.

```bash
for label in \
  com.brainlayer.tier0-watchdog \
  com.brainlayer.throughput-watchdog \
  com.brainlayer.health-check \
  com.brainlayer.backup-daily \
  com.brainlayer.maintenance-nightly \
  com.brainlayer.maintenance-weekly \
  com.brainlayer.wal-checkpoint \
  com.brainlayer.repair-fts \
  com.brainlayer.decay \
  com.brainlayer.index \
  com.brainlayer.watch \
  com.brainlayer.drain \
  com.brainlayer.hotlane-brainbar \
  com.brainlayer.enrichment \
  com.brainlayer.brainbar; do
  if grep -Fxq "$label" "$WAVE3A_RUN_DIR/active-labels.before"; then
    launchctl bootout "gui/$(id -u)/$label"
    printf '%s\n' "$label" >> "$WAVE3A_RUN_DIR/stopped-labels"
  fi
done
```

Confirm no DB-writing process remains. Leave `com.brainlayer.brainbar-daemon` loaded only for the backup-pipeline request.

## 2. Checkpoint

```bash
/usr/bin/time -p "$WAVE3A_PYTHON" scripts/wal_checkpoint.py --mode TRUNCATE --retry-busy --json \
  | tee "$WAVE3A_RUN_DIR/checkpoint.json"
stat -f '%z' "$WAVE3A_DB-wal" 2>/dev/null || true
```

Stop if the checkpoint reports busy or returns nonzero.

## 3. Primary LIVE backup route: backup pipeline

This is the required Tuesday route. It depends on deployed BrainBar 1.5.4 carrying the core-profile gate fix for `brain_backup_vacuum_into`.

```bash
export BRAINLAYER_BACKUP_FULL_VERIFY=1
export BRAINLAYER_BACKUP_SQLITE_CHECK_TIMEOUT_SECONDS=3600
/usr/bin/time -p "$WAVE3A_PYTHON" - "$WAVE3A_HOST" 2>&1 <<'PY' | tee "$WAVE3A_RUN_DIR/backup-pipeline.out"
import json, sys
from brainlayer.backup_daily import run_backup
result = run_backup(
    date_stamp=f"2026-08-11-wave3a-{sys.argv[1]}",
    remove_local_after_upload=False,
)
print(json.dumps(result, sort_keys=True))
if not result.get("verified") or not result.get("uploaded"):
    raise SystemExit(1)
PY
tail -n 1 "$HOME/.local/share/brainlayer/logs/backup-daily.log" \
  | tee "$WAVE3A_RUN_DIR/backup-pipeline.json"
```

Before migration, require all of the following in the JSON receipt: `backup_log_provenance=real`, `verified=true`, `uploaded=true`, a Drive file id, matching verification checks, and a non-null `local_uncompressed_snapshot`. Assign that exact raw path and verify its row count read-only:

```bash
export WAVE3A_BACKUP_RAW="<local_uncompressed_snapshot from receipt>"
test -f "$WAVE3A_BACKUP_RAW"
"$WAVE3A_PYTHON" - "$WAVE3A_BACKUP_RAW" <<'PY'
import sqlite3, sys
p = sys.argv[1]
c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
print(c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
print(c.execute("PRAGMA quick_check").fetchone()[0])
c.close()
PY
```

### Contingency only: rehearsal-proven online backup

Do not choose this branch merely to save time. Use it only if the 1.5.4 backup pipeline fails, after Etan explicitly authorizes the contingency. It was rehearsal-proven with canonical opened `mode=ro`; page retries while drain was writing only stretched wall-clock time.

```bash
export WAVE3A_BACKUP_RAW="$WAVE3A_RUN_DIR/online-backup-contingency.db"
test ! -e "$WAVE3A_BACKUP_RAW"
/usr/bin/time -p "$WAVE3A_PYTHON" - "$WAVE3A_DB" "$WAVE3A_BACKUP_RAW" <<'PY'
import sqlite3, sys
source, destination = sys.argv[1:3]
src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
dst = sqlite3.connect(destination)
src.backup(dst, pages=4096, sleep=0.050)
dst.close()
src.close()
PY
"$WAVE3A_PYTHON" - "$WAVE3A_BACKUP_RAW" <<'PY'
import sqlite3, sys
c = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
print(c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
print(c.execute("PRAGMA quick_check").fetchone()[0])
c.close()
PY
```

After either backup route succeeds, stop the remaining daemon before migration:

```bash
if grep -Fxq 'com.brainlayer.brainbar-daemon' "$WAVE3A_RUN_DIR/active-labels.before"; then
  launchctl bootout "gui/$(id -u)/com.brainlayer.brainbar-daemon"
  printf '%s\n' 'com.brainlayer.brainbar-daemon' >> "$WAVE3A_RUN_DIR/stopped-labels"
fi
"$WAVE3A_PYTHON" scripts/wal_checkpoint.py --mode TRUNCATE --retry-busy --json \
  | tee "$WAVE3A_RUN_DIR/checkpoint-after-backup.json"
```

## 4. Migrate in one supervised sitting

The environment gate is intentionally scoped to this command. Without it the script refuses the configured canonical path.

```bash
/usr/bin/time -p env BRAINLAYER_OFFLINE_MIGRATOR_GATED_SWAP=1 \
  "$WAVE3A_PYTHON" scripts/migrate_source_class.py \
  --db "$WAVE3A_DB" \
  --git-sha "$WAVE3A_SHA" \
  --actor "$WAVE3A_ACTOR" \
  --batch-size 5000 \
  | tee "$WAVE3A_RUN_DIR/source-class-migration.json"
```

If additional approved Wave 3 migrations are scheduled for the same maintenance window, execute their pinned commands now, before reconcile and restart. Never invent or substitute an unpinned 3b/3c command.

## 5. Reconcile and verify

Run this read-only receipt query:

```bash
"$WAVE3A_PYTHON" - "$WAVE3A_DB" "$WAVE3A_SHA" <<'PY'
import json, sqlite3, sys
db, expected_sha = sys.argv[1:3]
c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
distribution = dict(("NULL" if k is None else k, v) for k, v in c.execute(
    "SELECT source_class, COUNT(*) FROM chunks GROUP BY source_class"
))
schema = json.loads(c.execute(
    "SELECT details FROM schema_migrations WHERE name='2026_08_10_source_class_v1'"
).fetchone()[0])
event = c.execute(
    "SELECT commit_hash, actor, status FROM migration_events WHERE id='schema:2026_08_10_source_class_v1'"
).fetchone()
receipt = {
    "quick_check": c.execute("PRAGMA quick_check").fetchone()[0],
    "rows": c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
    "distribution": distribution,
    "invalid_classes": c.execute("SELECT COUNT(*) FROM chunks WHERE source_class IS NOT NULL AND source_class NOT IN ('cli-agent','desktop','subagent','brain-worker','fleet-coordination')").fetchone()[0],
    "brain_worker_fts_rows": c.execute("SELECT COUNT(*) FROM chunk_fts_rowids r JOIN chunks c ON c.id=r.chunk_id WHERE c.source_class='brain-worker'").fetchone()[0],
    "schema_sha": schema.get("git_sha"),
    "event": event,
}
c.close()
print(json.dumps(receipt, indent=2, sort_keys=True))
assert receipt["quick_check"] == "ok"
assert receipt["invalid_classes"] == 0
assert receipt["brain_worker_fts_rows"] == 0
assert receipt["schema_sha"] == expected_sha
assert event[0] == expected_sha and event[2] == "success"
PY
stat -f '%z' "$WAVE3A_DB-wal" 2>/dev/null || true
```

Required visibility probes before restart: one exact stored id from each class plus one NULL row. Exact expansion must work for all six; default search must show cli-agent, subagent, fleet-coordination, and NULL, while hiding desktop and brain-worker. The internal desktop opt-in must reveal desktop but never brain-worker:

```bash
"$WAVE3A_PYTHON" scripts/verify_source_class_visibility.py --db "$WAVE3A_DB" \
  | tee "$WAVE3A_RUN_DIR/source-class-visibility.json"
```

## 6. Restart exactly the prior active set and probe the real service

Restore labels from the saved inventory, except that enrichment remains skipped when absent before or named by the pause sentinel:

```bash
while IFS= read -r label; do
  if [ "$label" = "com.brainlayer.enrichment" ] && [ -f "$HOME/.local/share/brainlayer/pause.sentinel" ]; then
    continue
  fi
  plist="$HOME/Library/LaunchAgents/$label.plist"
  test -f "$plist"
  launchctl bootstrap "gui/$(id -u)" "$plist"
done < <(tail -r "$WAVE3A_RUN_DIR/stopped-labels")
```

Then prove the executing binary serves a real request:

```bash
launchctl print "gui/$(id -u)/com.brainlayer.brainbar-daemon" > "$WAVE3A_RUN_DIR/brainbar-daemon.print"
QUERY=agent-html TIMEOUT_SECONDS=30 scripts/smoke/firstturn-brainlayer-smoke.sh \
  | tee "$WAVE3A_RUN_DIR/real-mcp-smoke.out"
"$WAVE3A_CLI" search "source class" -n 3 --text \
  | tee "$WAVE3A_RUN_DIR/real-cli-search.out"
```

Stop and rollback if any SQL reconcile, visibility probe, or real-service probe fails.

## 7. Rollback = re-copy the verified backup

Keep all writers stopped. If failure is detected after services were restarted, repeat the stop phase for the saved active set and confirm the database has no writer before continuing. Preserve the failed database rather than deleting it. Re-copy the verified raw backup on the same volume, quick-check the restore candidate, then atomically put it at the canonical path:

```bash
export WAVE3A_FAILED_DB="$WAVE3A_RUN_DIR/failed-after-wave3a.db"
export WAVE3A_RESTORE_DB="$(dirname "$WAVE3A_DB")/.brainlayer-wave3a-restore.db"
test ! -e "$WAVE3A_FAILED_DB"
test ! -e "$WAVE3A_FAILED_DB-wal"
test ! -e "$WAVE3A_FAILED_DB-shm"
test ! -e "$WAVE3A_RESTORE_DB"
cp -p "$WAVE3A_BACKUP_RAW" "$WAVE3A_RESTORE_DB"
"$WAVE3A_PYTHON" - "$WAVE3A_RESTORE_DB" <<'PY'
import sqlite3, sys
c = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
print(c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
assert c.execute("PRAGMA quick_check").fetchone()[0] == "ok"
c.close()
PY
if [ -e "$WAVE3A_DB-wal" ]; then mv "$WAVE3A_DB-wal" "$WAVE3A_FAILED_DB-wal"; fi
if [ -e "$WAVE3A_DB-shm" ]; then mv "$WAVE3A_DB-shm" "$WAVE3A_FAILED_DB-shm"; fi
mv "$WAVE3A_DB" "$WAVE3A_FAILED_DB"
mv "$WAVE3A_RESTORE_DB" "$WAVE3A_DB"
test ! -e "$WAVE3A_DB-wal"
test ! -e "$WAVE3A_DB-shm"
```

Checkpoint, restart only the prior active labels, and rerun the real-service probes. Record the rollback wall-clock and restored row count in the host table.

## Rehearsal receipt

The PR handoff fills the final-code rerun values here and in the collab append:

| Check | Measured result |
|---|---|
| Canonical source writes | none |
| Copy route | SQLite online backup; source `mode=ro` |
| Contingency online-backup wall-clock | 26.16 seconds under live drain writes; source opened `mode=ro` |
| Rows before / after | 744,335 / 744,335 |
| Migration wall-clock | 198.82 seconds (`duration_seconds=198.79003841709346`) |
| WAL bytes before / observed peak / after | 0 / 29,181,992 / 0 |
| Distribution | NULL 69,581; brain-worker 84; cli-agent 536,851; desktop 2,696; fleet-coordination 97,678; subagent 37,445 |
| Ledgers | exact final migration code SHA in both rows |
| Quick check | `ok` in 503.49 seconds; both ledgers pin `be877608ae5ff14de079ccd40949ef14f88aa578`; zero invalid classes / brain-worker FTS rows |
| Class visibility / expansion | six source buckets green against real copied rows |
| Rollback | APFS re-copy, 0.00 seconds; 744,335 rows; `source_class` absent |
| Repository gates | focused 198 passed; broad 3,969 passed / 10 skipped / 62 deselected / 2 xfailed; one unrelated MPS arbitration test separately diagnosed and deselected after a 997.19s no-progress compute call |
