# Wave 3a two-host LIVE runbook

Status: rehearsal-proven; LIVE execution scheduled for Tuesday 2026-08-11 at 20:00 IDT or later with Etan present.

This runbook is the source-class segment of the Wave 3 maintenance sitting. Execute it on Main first, reconcile and probe Main, then repeat it on M1. Do not begin a host without Etan's explicit go. Keep enrichment off wherever the pause sentinel applies.

## Pinned artifacts and rehearsal measurements

- Migration/source-class code commit: `3964412f8291a083150a424e38df08ece817783d`.
- Required deployed release before either host migrates: BrainLayer + BrainBar **exactly v1.5.6**. v1.5.5 carries the backup-race fix; v1.5.6 adds the Swift default-search source-class gate.
- Command: `scripts/migrate_source_class.py` from a checkout containing that commit.
- Rehearsal source: online SQLite backup from canonical opened `mode=ro` into gitignored `.tmp-wave3-3a/` (contingency route, not the LIVE primary route).
- Timed contingency replay: 26.16 seconds wall-clock while live drain writes continued; 744,396 rows, 4,251,874 pages, 17,415,675,904 bytes. The pristine rehearsal copy passed `PRAGMA quick_check` before migration.
- Rehearsal rows: 744,335 before and after.
- Final-code rehearsal migration: 252.02 seconds wall-clock on commit `3964412f8291a083150a424e38df08ece817783d` (`duration_seconds=251.79996774997562`); exact-SHA idempotent rerun: 0.65 seconds.
- Rehearsal distribution: NULL 69,581; brain-worker 84; cli-agent 536,851; desktop 2,696; fleet-coordination 105,383; subagent 29,740.
- Earlier monitored rehearsal migration WAL: 0 bytes before, 29,181,992-byte observed peak, 0 bytes at close after checkpoints; the final-code rerun was not separately peak-sampled.
- Rehearsal rollback: restore by APFS re-copy in 0.00 seconds; restored row count 744,335 and `source_class` absent.

## Host receipt table

Fill every blank on each host. Do not copy Main values into M1.

| Field | Main | M1 |
|---|---|---|
| Operator / Etan go time |  |  |
| Hostname |  |  |
| Canonical DB path |  |  |
| Checked-out commit |  |  |
| Installed BrainLayer / BrainBar versions (must be exactly 1.5.6) |  |  |
| Installed CLI path + SHA-256 |  |  |
| Executing BrainBarDaemon PID / path + SHA-256 |  |  |
| Executing daemon `source_class` capability probe |  |  |
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
| No-write observer before / after |  |  |
| Migration wall-clock |  |  |
| Rows after |  |  |
| NULL / cli / desktop / subagent / brain-worker / fleet |  |  |
| Both ledger SHAs |  |  |
| Migration WAL bytes after |  |  |
| SQL reconcile |  |  |
| Real service probes |  |  |
| Restarted labels |  |  |
| Etan host acceptance |  |  |

## 0. Deploy v1.5.6 first and prove the executing binaries

The order is mandatory on **each host**: install v1.5.6, restart and probe the installed formula and BrainBarDaemon, then stop writers, back up, and migrate. The new Python watcher INSERT and Swift search clause both guard for an absent `source_class` column, so deploying v1.5.6 before the schema migration is safe. The reverse order is forbidden: migration must not make the contract live while an older BrainBar still exposes desktop rows.

Run the supported release installers with Etan present. Do not substitute a worktree build:

```bash
set -euo pipefail
export WAVE3A_REQUIRED_VERSION="1.5.6"
brew update
if brew list --formula brainlayer >/dev/null 2>&1; then
  brew upgrade etanhey/layers/brainlayer || brew reinstall etanhey/layers/brainlayer
else
  brew install etanhey/layers/brainlayer
fi
scripts/brainlayer-update-brainbar.sh
export WAVE3A_INSTALLED_CLI="$(command -v brainlayer)"
test -x "$WAVE3A_INSTALLED_CLI"
"$WAVE3A_INSTALLED_CLI" --version | tee "$HOME/.local/share/brainlayer/wave3a-installed-cli-version.txt"
grep -F "$WAVE3A_REQUIRED_VERSION" "$HOME/.local/share/brainlayer/wave3a-installed-cli-version.txt"
/usr/libexec/PlistBuddy -c 'Print:CFBundleShortVersionString' /Applications/BrainBar.app/Contents/Info.plist \
  | tee "$HOME/.local/share/brainlayer/wave3a-brainbar-version.txt"
grep -Fx "$WAVE3A_REQUIRED_VERSION" "$HOME/.local/share/brainlayer/wave3a-brainbar-version.txt"
```

Restart the daemon from the installed bundle, then bind the receipt to the process that is actually serving MCP:

```bash
launchctl kickstart -k "gui/$(id -u)/com.brainlayer.brainbar-daemon"
sleep 2
export WAVE3A_DAEMON_PID="$(launchctl print "gui/$(id -u)/com.brainlayer.brainbar-daemon" | awk '/^[[:space:]]*pid = / {print $3; exit}')"
test -n "$WAVE3A_DAEMON_PID"
ps -p "$WAVE3A_DAEMON_PID" -o command= | tee "$HOME/.local/share/brainlayer/wave3a-daemon-command.txt"
grep -F '/Applications/BrainBar.app/Contents/MacOS/BrainBarDaemon' "$HOME/.local/share/brainlayer/wave3a-daemon-command.txt"
shasum -a 256 "$WAVE3A_INSTALLED_CLI" /Applications/BrainBar.app/Contents/MacOS/BrainBarDaemon \
  | tee "$HOME/.local/share/brainlayer/wave3a-installed-sha256.txt"
strings /Applications/BrainBar.app/Contents/MacOS/BrainBarDaemon | grep -F 'source_class' \
  | tee "$HOME/.local/share/brainlayer/wave3a-daemon-source-class-capability.txt"
QUERY=agent-html DEADLINE_SECS=30 scripts/smoke/firstturn-brainlayer-smoke.sh \
  | tee "$HOME/.local/share/brainlayer/wave3a-pre-migration-real-mcp-smoke.out"
```

Do not enter host preflight unless all commands pass. Re-run this section independently on M1; never copy Main's receipt.

## 1. Host preflight and writer inventory

From the pinned checkout, set only host-local absolute paths:

```bash
export WAVE3A_REPO="$PWD"
export WAVE3A_PYTHON="$WAVE3A_REPO/.venv/bin/python"
export WAVE3A_CLI="$(command -v brainlayer)"
export PYTHONPATH="$WAVE3A_REPO/src"
export WAVE3A_DB="$($WAVE3A_PYTHON -c 'from brainlayer.paths import get_db_path; print(get_db_path())')"
export WAVE3A_SHA="3964412f8291a083150a424e38df08ece817783d"
export WAVE3A_ACTOR="etan+brainlayerCodex"
export WAVE3A_HOST="$(hostname -s | tr -cd 'A-Za-z0-9_-')"
export WAVE3A_RUN_DIR="$HOME/.local/share/brainlayer/wave3-live-2026-08-11/$WAVE3A_HOST"
mkdir -p "$WAVE3A_RUN_DIR"
test -x "$WAVE3A_PYTHON"
test -x "$WAVE3A_CLI"
test "$(git rev-parse "$WAVE3A_SHA")" = "$WAVE3A_SHA"
git merge-base --is-ancestor "$WAVE3A_SHA" HEAD
test -f "$WAVE3A_DB"
/usr/libexec/PlistBuddy -c 'Print:CFBundleShortVersionString' /Applications/BrainBar.app/Contents/Info.plist | grep -Fx '1.5.6'
df -k "$(dirname "$WAVE3A_DB")"
stat -f '%z' "$WAVE3A_DB"
stat -f '%z' "$WAVE3A_DB-wal" 2>/dev/null || true
cat "$HOME/.local/share/brainlayer/pause.sentinel" 2>/dev/null || true
pgrep -af '[b]rain_digest|[b]rainlayer.*digest' && exit 1 || true
pgrep -af '[b]rainlayer.backup_daily|[b]ackup-daily.sh' && exit 1 || true
```

Require more than 40 GiB free, no digest, no concurrent backup, and both installed/executing v1.5.6 proofs from §0. Record the exact active set before changing launchd. The broader pattern is intentional: `com.mcplayer.brainlayer-proxy` is a write-capable bridge even though its label does not start with `com.brainlayer`.

```bash
launchctl list | awk '$3 ~ /^(com\.brainlayer\.|com\.mcplayer\.)/ {print $3}' | sort > "$WAVE3A_RUN_DIR/active-labels.before"
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
  com.brainlayer.jsonl-backup \
  com.brainlayer.maintenance-nightly \
  com.brainlayer.maintenance-weekly \
  com.brainlayer.wal-checkpoint \
  com.brainlayer.repair-fts \
  com.brainlayer.decay \
  com.brainlayer.p0-counter \
  com.brainlayer.index \
  com.brainlayer.watch \
  com.brainlayer.drain \
  com.brainlayer.hotlane-brainbar \
  com.brainlayer.enrichment \
  com.brainlayer.brainbar \
  com.brainlayer.gemini-loopback \
  com.mcplayer.brainlayer-proxy; do
  if grep -Fxq "$label" "$WAVE3A_RUN_DIR/active-labels.before"; then
    launchctl bootout "gui/$(id -u)/$label"
    printf '%s\n' "$label" >> "$WAVE3A_RUN_DIR/stopped-labels"
  fi
done
```

`com.mcplayer.bus` may remain loaded: it is the event bus, not the BrainLayer MCP proxy and not a DB writer. Confirm no other DB-writing process or TCP bridge remains. Leave `com.brainlayer.brainbar-daemon` loaded only for the backup-pipeline request.

```bash
launchctl list | awk '$3 ~ /^(com\.brainlayer\.|com\.mcplayer\.)/ {print $3}' | sort \
  > "$WAVE3A_RUN_DIR/active-labels.after-stop"
for label in \
  com.brainlayer.gemini-loopback \
  com.brainlayer.p0-counter \
  com.brainlayer.jsonl-backup \
  com.brainlayer.watch \
  com.brainlayer.drain \
  com.brainlayer.enrichment \
  com.mcplayer.brainlayer-proxy; do
  ! grep -Fxq "$label" "$WAVE3A_RUN_DIR/active-labels.after-stop"
done
! lsof -nP -iTCP:48123 -sTCP:LISTEN
pgrep -af '[b]rainlayer.*(watch|drain|enrich|p0-counter|jsonl-backup)' && exit 1 || true
```

## 2. Checkpoint

```bash
set -o pipefail
/usr/bin/time -p "$WAVE3A_PYTHON" scripts/wal_checkpoint.py --mode TRUNCATE --retry-busy --json \
  | tee "$WAVE3A_RUN_DIR/checkpoint.json"
stat -f '%z' "$WAVE3A_DB-wal" 2>/dev/null || true
```

Stop if the checkpoint reports busy or returns nonzero.

Start one persistent read-only SQLite observer before either backup route. Its connection records `PRAGMA data_version` before the snapshot and again only after BrainBarDaemon is stopped. Any intervening commit makes the rollback snapshot non-authoritative, so the observer fails the gate and migration must not start.

```bash
set -euo pipefail
export WAVE3A_FENCE_READY="$WAVE3A_RUN_DIR/no-write-observer.ready.json"
export WAVE3A_FENCE_RELEASE="$WAVE3A_RUN_DIR/no-write-observer.release"
export WAVE3A_FENCE_RESULT="$WAVE3A_RUN_DIR/no-write-observer.result.json"
rm -f "$WAVE3A_FENCE_READY" "$WAVE3A_FENCE_RELEASE" "$WAVE3A_FENCE_RESULT"
"$WAVE3A_PYTHON" - "$WAVE3A_DB" "$WAVE3A_FENCE_READY" "$WAVE3A_FENCE_RELEASE" "$WAVE3A_FENCE_RESULT" <<'PY' &
import json, os, sqlite3, sys, time
db, ready, release, result = sys.argv[1:]
conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=60)
conn.execute("PRAGMA query_only = ON")
before = int(conn.execute("PRAGMA data_version").fetchone()[0])
payload = {"pid": os.getpid(), "data_version_before": before}
tmp = ready + ".tmp"
open(tmp, "w", encoding="utf-8").write(json.dumps(payload, sort_keys=True) + "\n")
os.replace(tmp, ready)
while not os.path.exists(release):
    time.sleep(0.05)
after = int(conn.execute("PRAGMA data_version").fetchone()[0])
conn.close()
payload["data_version_after"] = after
payload["unchanged"] = before == after
open(result, "w", encoding="utf-8").write(json.dumps(payload, sort_keys=True) + "\n")
raise SystemExit(0 if before == after else 1)
PY
export WAVE3A_FENCE_PID=$!
cleanup_wave3a_fence() {
  : > "$WAVE3A_FENCE_RELEASE"
  wait "$WAVE3A_FENCE_PID" 2>/dev/null || true
}
trap cleanup_wave3a_fence EXIT
while [ ! -f "$WAVE3A_FENCE_READY" ]; do
  kill -0 "$WAVE3A_FENCE_PID"
  sleep 0.1
done
cat "$WAVE3A_FENCE_READY"
```

## 3. Primary LIVE backup route: backup pipeline

This is the required Tuesday route. Deployed v1.5.6 includes v1.5.5's backup-race fix and the v1.5.6 Swift source-class gate.

```bash
set -o pipefail
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

Do not choose this branch merely to save time. Use it only if the deployed v1.5.6 backup pipeline fails, after Etan explicitly authorizes the contingency. It was rehearsal-proven with canonical opened `mode=ro`; page retries while drain was writing only stretched wall-clock time.

```bash
set -o pipefail
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
set -euo pipefail
if grep -Fxq 'com.brainlayer.brainbar-daemon' "$WAVE3A_RUN_DIR/active-labels.before"; then
  launchctl bootout "gui/$(id -u)/com.brainlayer.brainbar-daemon"
  printf '%s\n' 'com.brainlayer.brainbar-daemon' >> "$WAVE3A_RUN_DIR/stopped-labels"
fi
: > "$WAVE3A_FENCE_RELEASE"
wait "$WAVE3A_FENCE_PID"
unset WAVE3A_FENCE_PID
trap - EXIT
cat "$WAVE3A_FENCE_RESULT"
"$WAVE3A_PYTHON" scripts/wal_checkpoint.py --mode TRUNCATE --retry-busy --json \
  | tee "$WAVE3A_RUN_DIR/checkpoint-after-backup.json"
```

## 4. Migrate in one supervised sitting

The environment gate is intentionally scoped to this command. Without it the script refuses the configured canonical path.

```bash
set -o pipefail
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
import sqlite_vec
db, expected_sha = sys.argv[1:3]
c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
c.enable_load_extension(True)
c.load_extension(sqlite_vec.loadable_path())
c.enable_load_extension(False)
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
    "brain_worker_float_vector_rows": c.execute("SELECT COUNT(*) FROM chunk_vectors v JOIN chunks c ON c.id=v.chunk_id WHERE c.source_class='brain-worker'").fetchone()[0],
    "brain_worker_binary_vector_rows": c.execute("SELECT COUNT(*) FROM chunk_vectors_binary v JOIN chunks c ON c.id=v.chunk_id WHERE c.source_class='brain-worker'").fetchone()[0],
    "schema_sha": schema.get("git_sha"),
    "event": event,
}
c.close()
print(json.dumps(receipt, indent=2, sort_keys=True))
assert receipt["quick_check"] == "ok"
assert receipt["invalid_classes"] == 0
assert receipt["brain_worker_fts_rows"] == 0
assert receipt["brain_worker_float_vector_rows"] == 0
assert receipt["brain_worker_binary_vector_rows"] == 0
assert receipt["schema_sha"] == expected_sha
assert event[0] == expected_sha and event[2] == "success"
PY
stat -f '%z' "$WAVE3A_DB-wal" 2>/dev/null || true
```

Required visibility probes before restart: one exact stored id from each class plus one NULL row. Exact expansion must work for all six; default search must show cli-agent, subagent, fleet-coordination, and NULL, while hiding desktop and brain-worker. The internal desktop opt-in must reveal desktop but never brain-worker:

```bash
set -o pipefail
"$WAVE3A_PYTHON" scripts/verify_source_class_visibility.py --db "$WAVE3A_DB" \
  | tee "$WAVE3A_RUN_DIR/source-class-visibility.json"
```

## 6. Restart exactly the prior active set and probe the real service

Restore labels from the saved inventory, except that enrichment remains skipped when absent before or named by the pause sentinel:

```bash
while IFS= read -r label; do
  if [ "$label" = "com.brainlayer.enrichment" ]; then
    if "$WAVE3A_PYTHON" - "$label" <<'PY'
import sys
from datetime import UTC, datetime
from brainlayer.pause import DEFAULT_PAUSE_SENTINEL_PATH, pause_applies_to_label, pause_sentinel_state
label = sys.argv[1]
payload, active, _stale = pause_sentinel_state(DEFAULT_PAUSE_SENTINEL_PATH, datetime.now(UTC))
raise SystemExit(0 if active and pause_applies_to_label(payload, label) else 1)
PY
    then
      continue
    fi
  fi
  plist="$HOME/Library/LaunchAgents/$label.plist"
  test -f "$plist"
  launchctl bootstrap "gui/$(id -u)" "$plist"
done < <(tail -r "$WAVE3A_RUN_DIR/stopped-labels")
```

Then prove the executing binary serves a real request:

```bash
set -euo pipefail
launchctl print "gui/$(id -u)/com.brainlayer.brainbar-daemon" > "$WAVE3A_RUN_DIR/brainbar-daemon.print"
export WAVE3A_DAEMON_PID="$(awk '/^[[:space:]]*pid = / {print $3; exit}' "$WAVE3A_RUN_DIR/brainbar-daemon.print")"
test -n "$WAVE3A_DAEMON_PID"
ps -p "$WAVE3A_DAEMON_PID" -o command= | tee "$WAVE3A_RUN_DIR/brainbar-daemon.command"
grep -F '/Applications/BrainBar.app/Contents/MacOS/BrainBarDaemon' "$WAVE3A_RUN_DIR/brainbar-daemon.command"
shasum -a 256 /Applications/BrainBar.app/Contents/MacOS/BrainBarDaemon \
  | tee "$WAVE3A_RUN_DIR/brainbar-daemon.sha256"
"$WAVE3A_PYTHON" - "$WAVE3A_RUN_DIR/source-class-visibility.json" "$WAVE3A_RUN_DIR/live-probes.env" <<'PY'
import json, shlex, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
values = {
    "WAVE3A_VISIBLE_ID": receipt["cli-agent"]["chunk_id"],
    "WAVE3A_VISIBLE_TOKEN": receipt["cli-agent"]["token"],
    "WAVE3A_DESKTOP_ID": receipt["desktop"]["chunk_id"],
    "WAVE3A_DESKTOP_TOKEN": receipt["desktop"]["token"],
}
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    for key, value in values.items():
        handle.write(f"export {key}={shlex.quote(str(value))}\n")
PY
source "$WAVE3A_RUN_DIR/live-probes.env"
QUERY="$WAVE3A_VISIBLE_TOKEN" NUM_RESULTS=100 RAW_OUTPUT_PATH="$WAVE3A_RUN_DIR/real-mcp-visible.raw.jsonl" \
  DEADLINE_SECS=30 scripts/smoke/firstturn-brainlayer-smoke.sh \
  | tee "$WAVE3A_RUN_DIR/real-mcp-visible.out"
grep -F "$WAVE3A_VISIBLE_ID" "$WAVE3A_RUN_DIR/real-mcp-visible.raw.jsonl"
QUERY="$WAVE3A_DESKTOP_TOKEN" NUM_RESULTS=100 RAW_OUTPUT_PATH="$WAVE3A_RUN_DIR/real-mcp-desktop.raw.jsonl" \
  DEADLINE_SECS=30 scripts/smoke/firstturn-brainlayer-smoke.sh \
  | tee "$WAVE3A_RUN_DIR/real-mcp-desktop.out"
! grep -F "$WAVE3A_DESKTOP_ID" "$WAVE3A_RUN_DIR/real-mcp-desktop.raw.jsonl"
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
| Tuesday LIVE primary route | deployed v1.5.6 backup pipeline; verified local raw snapshot plus uploaded Drive receipt required |
| Rehearsal-proven contingency | SQLite online backup; source opened `mode=ro` |
| Contingency online-backup wall-clock | 26.16 seconds under live drain writes; source opened `mode=ro` |
| Rows before / after | 744,335 / 744,335 |
| Migration wall-clock | 252.02 seconds (`duration_seconds=251.79996774997562`); exact-SHA idempotent rerun 0.65 seconds |
| WAL bytes before / observed peak / after | 0 / 29,181,992 / 0 in the earlier monitored rehearsal; final-code peak not separately sampled |
| Distribution | NULL 69,581; brain-worker 84; cli-agent 536,851; desktop 2,696; fleet-coordination 105,383; subagent 29,740 |
| Ledgers | schema and event rows both pin `3964412f8291a083150a424e38df08ece817783d`; event status `success` |
| Quick check | `ok` in 712.79 seconds; zero invalid classes; zero brain-worker FTS, float-vector, and binary-vector rows |
| Class visibility / expansion | six source buckets green against real copied rows; aggregate desktop audit sampled 72 tokens with zero leaked IDs |
| Rollback | APFS re-copy, 0.00 seconds; 744,335 rows; `source_class` absent |
| Repository gates | focused source-class 251 passed; final search/cache 42 passed; additional search scopes 92 passed / 1 xfailed with the protected production-DB latency probe refused by the test-path guard; Swift Database/KG 119 passed; full Swift reached 850 passed / 10 skipped with 7 unrelated timing failures while another lane held 24 deliberate CPU spinners |
