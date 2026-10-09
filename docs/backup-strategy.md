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

Keep the latest 7 daily snapshots. Weekly maintenance uses the same Drive API path with a 4-snapshot retention policy.

## Why This Approach

The database runs in WAL mode and has active writers from BrainBar, watch, and maintenance jobs. Copying `brainlayer.db` directly can miss WAL contents or capture an inconsistent file pair. SQLite's online backup API reads through SQLite itself, so the backup is a consistent snapshot without stopping the live services.

Direct Google Drive API upload is used because the post-repair machine no longer has Google Drive Desktop mounted at the old CloudStorage path. Historical DriveFS logs show the previous path was:

`~/Library/CloudStorage/you@example.com/My Drive/Brain Drive`

That mount is not present after repair, and `/Applications/Google Drive.app` is also absent. The API path avoids depending on that local mount.

## Implementation

Repo files:

- `src/brainlayer/backup_daily.py`: creates the SQLite backup, gzips it, uploads it to Drive, verifies the uploaded file, removes the local staging copy, and prunes retention.
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

## Restore Drill

### Recover a preserved failed upload

After the recovery source has been reviewed and installed, use the explicit operator
command for an existing `YYYY-MM-DD.db` and matching `.db.gz`. This operation holds
the same exclusive, nonblocking `.backup.lock` as the daily job. It never takes a new
snapshot, creates a restore copy, deletes candidates, or runs retention.

```bash
brainlayer backup recover-existing --snapshot-date YYYY-MM-DD \
  --source-receipt-line LINE --expected-raw-bytes MEASURED_RAW_SIZE --no-retention
```

The CLI invokes the installed environment wrapper and pinned keg Python. `LINE` is
the one-based original failed JSONL record in the resolved canonical backup log;
`MEASURED_RAW_SIZE` is the existing raw file's measured size. The receipt must match
the configured machine, Drive folder, dated staging path, compressed size and sentinel.
The raw and gzip must pass streamed byte equivalence, gzip CRC, hashes, sentinel
count and a read-only bounded `quick_check` (default 300 seconds). The ordinary backup
wall-clock timeout also applies; zero/skipped SQLite checks cannot satisfy recovery.

The original Drive folder is resolved read-only. Exactly one complete object matching
parent, original name, machine, size and MD5 is reused. With no complete match, upload
uses `YYYY-MM-DD-recovery-<unique-attempt-id>.db.gz` without overwriting any candidate.
Ambiguous or unknown ownership fails. Both preserved files remain after success or failure.
Remote metadata is read again and verified; the canonical attempt receipt is appended
and fsynced before the backup alert is cleared. A recovery receipt records the new
`attempted_at` separately from `source_attempted_at` and `snapshot_date`; freshness
readers use the source time. Recovery proves the preserved snapshot, not a new capture.

This command cannot resume the original in-memory resumable upload session. Source
tests do not establish installed behavior or actual recovery; those need a later
accepted operator run. Never use `brainlayer.backup_daily` arguments for recovery:
that module starts the ordinary daily snapshot flow.

1. Pick the newest good snapshot from Google Drive:

   `Brain Drive/06_ARCHIVE/backups/brainlayer-db/YYYY-MM-DD.db.gz`

2. Download it to a local scratch path, for example:

   `/tmp/brainlayer-restore/YYYY-MM-DD.db.gz`

3. Decompress and verify integrity:

   ```bash
   mkdir -p /tmp/brainlayer-restore
   gunzip -c /tmp/brainlayer-restore/YYYY-MM-DD.db.gz > /tmp/brainlayer-restore/brainlayer.db
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
