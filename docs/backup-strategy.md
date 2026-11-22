# BrainLayer Backup Strategy

Status: implemented for daily database snapshots.

## Decision

Use SQLite's online backup API, gzip the resulting snapshot, and upload it directly to Google Drive using the existing `~/.config/google-drive-mcp` OAuth credentials.

Target folder:

`Brain Drive/06_ARCHIVE/backups/brainlayer-db/YYYY-MM-DD.db.gz`

Encryption posture:

Backups are encrypted in transit by HTTPS and at rest by Google's infrastructure. Google holds the provider-side encryption keys. The database can contain user messages, code snippets, file paths, and agent memory, so client-side encryption should be added before upload if the threat model requires protection from the Drive account/provider layer. Recommended upgrade path: encrypt the gzip with `age` or GPG using a key stored in 1Password, then upload `YYYY-MM-DD.db.gz.age` and document the recovery key location.

Schedule:

Daily at 03:17 local time via `com.brainlayer.backup-daily`.

Retention (owner ruling, 2026-11-22):

**Drive backups stay unchanged.** Nothing on Drive is trashed, moved, renamed or
permanently deleted by retention. Additive uploads remain a separate operation.
`BRAINLAYER_BACKUP_DRIVE_RETENTION` cannot enable remote cleanup.

**Local copies:** keep the newest **3 verified compressed daily snapshots on the M1**,
or **1 on the M4**. A local archive can go only when Drive currently holds the **same
snapshot**, with the exact filename, size and stored-byte MD5 proven by its latest
receipt and live listing. The current folder and machine property must both match.
A missing listing, revoked receipt, changed archive or unknown file authorizes no deletion.
Cleanup hashes the local archive again immediately before unlinking it. Unknown files
do not displace the verified cohort, so they can leave more files than the target count.
Host detection comes from `backup_heavy_gate`.

The file named by `.weekly-backup-pin.json` stays protected until its owner replaces or
removes the pin. A malformed pin refuses cleanup. One-off `*-pre-*.db` migration copies
are never automatically deleted. A newer verified snapshot reports older copies by path,
byte size and `superseded_by` in `one_off_copies[]`; this timestamp comparison does not
prove the migration's own verification. A person must rule on deleting them.

| Environment variable | Meaning |
| --- | --- |
| `BRAINLAYER_BACKUP_LOCAL_KEEP` | M1 local count; default above |
| `BRAINLAYER_BACKUP_LOCAL_KEEP_M4` | M4 local count; default above |

Counts must be positive integers; invalid values refuse cleanup. Library callers can
opt out of local cleanup with `remove_local_after_upload=False`. Filename-only legacy
pruning entrypoints do nothing: filenames cannot establish a live verified Drive copy.
A newer day's upload never authorizes deleting an unverified older day.

Drive policy comparisons are **read-only reports**, not cleanup plans with authority.
`backup_retention.drive_retention_report` reports counts, bytes and date ranges for
keeping seven daily objects, and for keeping those plus the latest daily from each of
the previous three completed calendar months. Historical dated raw `.db` files can
appear in that report, but local cleanup covers only exact compressed daily names
(`YYYY-MM-DD.db.gz` or `YYYY-MM-DD.db.zst`). The report has no service or mutation API.
The owner must separately choose any future Drive policy.

Local cleanup reads a fresh listing when the receipt-proven local cohort exceeds its
cap; otherwise `local_retention_inventory=within_cap` explains why no listing was
needed. Failed measurement records an error and a `backup-retention` alert while the
verified backup keeps its successful outcome. A changed same-size archive is retained.

## Why This Approach

The database runs in WAL mode and has active writers from BrainBar, watch, and maintenance jobs. Copying `brainlayer.db` directly can miss WAL contents or capture an inconsistent file pair. SQLite's online backup API reads through SQLite itself, so the backup is a consistent snapshot without stopping the live services.

Direct Google Drive API upload is used because the post-repair machine no longer has Google Drive Desktop mounted at the old CloudStorage path. Historical DriveFS logs show the previous path was:

`~/Library/CloudStorage/you@example.com/My Drive/Brain Drive`

That mount is not present after repair, and `/Applications/Google Drive.app` is also absent. The API path avoids depending on that local mount.

## Implementation

Repo files:

- `src/brainlayer/backup_daily.py`: creates and verifies the SQLite backup, compresses it, uploads it to Drive, and applies the retention rule above.
- `src/brainlayer/backup_retention.py`: pure daily-name, receipt-MD5 and per-machine selection used by cleanup and read-only dry-runs.
- `scripts/launchd/backup-daily.sh`: launchd wrapper installed to `~/.local/lib/brainlayer/backup-daily.sh`.
- `scripts/launchd/com.brainlayer.backup-daily.plist`: LaunchAgent template.
- `scripts/launchd/install.sh backup`: installs the wrapper and LaunchAgent.

Local logs:

- `~/.local/share/brainlayer/logs/backup-daily.log`
- `~/.local/share/brainlayer/logs/backup-daily.err`

Each JSONL result in `backup-daily.log` includes `backup_log_provenance`. Launchd/manual backup
runs use `real`; pytest sets `BRAINLAYER_BACKUP_LOG_PATH` to a temporary file and
`BRAINLAYER_BACKUP_LOG_PROVENANCE=pytest`. Heartbeat monitors should treat only `real` entries in
the production log as evidence of a successful scheduled backup.

Manual run:

```bash
PYTHONPATH=~/Gits/brainlayer/src python3 -m brainlayer.backup_daily
```

## Runbook: a stalled backup blocks new backups (low disk space)

**What you see.** BrainBar's Backups attention reads *"Latest backup stopped; last verified backup:
<date>. Not enough disk space; stalled backup kept: YYYY-MM-DD.db.zst. Move it to another disk, then
run the backup again."* Technical details shows error code `retained_stalled_backup_capacity`. The
failing receipt in `backup-daily.log` lists every retained file in `retained_stalled_backups`, and any
whose identity could not be confirmed in `retained_stalled_unconfirmed`.

**Why it does not clear by itself.** When a Drive upload stalls, its compressed snapshot (about
9.5 GB) stays in the backup staging folder, `~/.local/share/brainlayer/backups/` by default, so the
next run can resume the same upload. After 36 hours it can no longer be resumed. A newer day's
upload does not authorize removing this day's only copy. It stays until its own snapshot is
verified live on Drive or a person moves it. A new backup needs the free space the stalled file is
using. So while the disk stays full, every run stops here. BrainLayer does not delete the file itself,
because it has no verified copy of that day anywhere else.

**Safe action.**

1. Make sure no backup is running: `pgrep -af '[b]rainlayer.backup_daily'` prints nothing.
2. Move only the files the message names out of the staging folder, onto another disk (an external
   drive, for example). Leave every other file in that folder alone: `.backup.lock`, raw `.db`
   snapshots and attempt files are handled by their own rules.
3. Let the next scheduled backup run. It takes a fresh snapshot.
4. Once a newer backup shows as verified, the moved file is an extra point-in-time copy that never
   reached Drive. Keep it or delete it; BrainLayer no longer tracks it.

A file marked *identity unconfirmed* changed after its receipt was written (different size, path,
link count or name). BrainLayer cannot tell what it holds, so it never resumes or removes it. The
same action applies.

## Restore Drill

1. Pick the newest good snapshot from Google Drive. New snapshots are zstd
   (`YYYY-MM-DD.db.zst`); older ones, and any made with
   `BRAINLAYER_BACKUP_COMPRESSOR=gzip`, are gzip (`YYYY-MM-DD.db.gz`):

   `Brain Drive/06_ARCHIVE/backups/brainlayer-db/YYYY-MM-DD.db.zst`

2. Download it to a local scratch path, for example:

   `/tmp/brainlayer-restore/YYYY-MM-DD.db.zst`

3. Decompress and verify integrity:

   ```bash
   mkdir -p /tmp/brainlayer-restore
   zstd -d -c /tmp/brainlayer-restore/YYYY-MM-DD.db.zst > /tmp/brainlayer-restore/brainlayer.db
   # legacy gzip snapshot instead:
   # gunzip -c /tmp/brainlayer-restore/YYYY-MM-DD.db.gz > /tmp/brainlayer-restore/brainlayer.db
   sqlite3 /tmp/brainlayer-restore/brainlayer.db 'PRAGMA integrity_check; SELECT count(*) FROM chunks;'
   ```

4. Stop writers before replacing the live DB. Enrichment is retired; the legacy unload
   below only quiesces an older installed job and must not be paired with a restart:

   ```bash
   launchctl unload ~/Library/LaunchAgents/com.brainlayer.brainbar.plist 2>/dev/null || true
   launchctl unload ~/Library/LaunchAgents/com.brainlayer.enrichment.plist 2>/dev/null || true
   launchctl unload ~/Library/LaunchAgents/com.brainlayer.watch.plist 2>/dev/null || true
   launchctl unload ~/Library/LaunchAgents/com.brainlayer.decay.plist 2>/dev/null || true
   ```

5. Preserve the corrupted DB and install the restored copy:

   ```bash
   ts="$(date +%Y%m%d-%H%M%S)"
   mkdir -p ~/.local/share/brainlayer/corrupt-$ts
   ls -lh ~/.local/share/brainlayer/brainlayer.db ~/.local/share/brainlayer/brainlayer.db-wal ~/.local/share/brainlayer/brainlayer.db-shm 2>/dev/null || true
   mv ~/.local/share/brainlayer/brainlayer.db* ~/.local/share/brainlayer/corrupt-$ts/
   ls -lh ~/.local/share/brainlayer/corrupt-$ts/
   cp /tmp/brainlayer-restore/brainlayer.db ~/.local/share/brainlayer/brainlayer.db
   ```

   The `brainlayer.db*` move preserves the main database plus SQLite auxiliary files: `brainlayer.db`,
   `brainlayer.db-wal`, and `brainlayer.db-shm`. Verify the `ls` output before and after the move; the
   wildcard will also move any other similarly named files in that directory.

6. Verify the restored DB before re-enabling services:

   ```bash
   sqlite3 ~/.local/share/brainlayer/brainlayer.db 'PRAGMA integrity_check; SELECT count(*) FROM chunks;'
   ```

7. Re-enable the retained services. Keep retired enrichment jobs down
   (history: [retirement details](enrichment.md)):

   ```bash
   launchctl load ~/Library/LaunchAgents/com.brainlayer.brainbar.plist
   launchctl load ~/Library/LaunchAgents/com.brainlayer.watch.plist
   launchctl load ~/Library/LaunchAgents/com.brainlayer.decay.plist
   ```

8. Run a post-restore WAL checkpoint:

   ```bash
   brainlayer wal-checkpoint --mode TRUNCATE
   ```

## Monthly Drill

Once per month:

1. Download the newest snapshot from Drive.
2. Restore it into `/tmp/brainlayer-restore`.
3. Run `PRAGMA integrity_check` and `SELECT count(*) FROM chunks`.
4. Record the snapshot date, chunk count, and command output in the maintenance log.

Do not replace the live DB during a drill unless the live DB is actually corrupted.
