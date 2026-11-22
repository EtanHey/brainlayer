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

Retention:

Keep the newest **3 verified compressed daily snapshots locally**, or **1 on the M4** once
Drive has verified that copy or a newer one. A day's only verified local copy is never
removed until that day's upload has verified. Unknown, changed, unverified or pinned
files can therefore leave more than the target count. Cleanup hashes a local archive
again before removing it; a new unverified day cannot displace older verified copies.
The M4/M1 distinction uses `backup_heavy_gate`'s shared host detection.

Drive keeps the newest **7 verified daily snapshots per machine**, across gzip and zstd.
Weekly and daily runs share this count. Selection requires both the exact machine folder
and `appProperties.brainlayer_machine`, plus a latest successful receipt whose stored-byte
MD5, filename and size still match the current Drive object. Prior machine-ID folders are
available for recovery, but retention does not cross into them. Files without this proof
are kept and do not count toward the cap. An unverified newest upload never displaces the
newest verified backup. Daily names must be exactly `YYYY-MM-DD.db.gz` or `YYYY-MM-DD.db.zst`.
Named one-offs, other machines' objects and the file named by `.weekly-backup-pin.json`
are protected. A pin remains protected until its owner replaces or removes it.

Drive cleanup **trashes**, never permanently deletes; Drive normally retains trashed files
for 30 days. A cleanup failure does not fail an already verified backup: its receipt
records the error and any files already trashed, and `backup-retention` stays alerted until
a later successful cleanup. An unreadable weekly pin refuses cleanup.

Configuration (positive integer counts; invalid values refuse cleanup):

| Environment variable | Meaning |
| --- | --- |
| `BRAINLAYER_BACKUP_LOCAL_KEEP` | Local count on the M1; default above |
| `BRAINLAYER_BACKUP_LOCAL_KEEP_M4` | Local count on the M4; default above |
| `BRAINLAYER_BACKUP_DRIVE_KEEP` | Drive count per machine; default above |
| `BRAINLAYER_BACKUP_DRIVE_RETENTION` | Enabled by default; `0` disables Drive cleanup |

Library callers can opt out of local cleanup with `remove_local_after_upload=False`.
One-off `*-pre-*.db` migration copies are **never automatically deleted**. A verified
newer snapshot reports older copies by path, byte size and `superseded_by` in
`one_off_copies[]`. The timestamp comparison does not prove the migration's own
verification; a person must rule on deletion.

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
next run can resume the same upload. After 36 hours it can no longer be resumed. BrainLayer removes
it only once a newer, fresh backup has uploaded and verified, and it hashes the file first to prove it
is the snapshot the receipt describes. That newer backup needs the free space the stalled file is
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
