# BrainLayer Backup Strategy

Status: implemented for daily database snapshots.

## Decision

Use SQLite's online backup API, gzip the resulting snapshot, and upload it directly to Google Drive using the existing `~/.config/google-drive-mcp` OAuth credentials.

Target folder:

`Brain Drive/06_ARCHIVE/backups/brainlayer-db/YYYY-MM-DD.db.gz`

Schedule:

Daily at 03:17 local time via `com.brainlayer.backup-daily`.

Retention:

Keep the latest 30 daily snapshots plus the latest snapshot for each of the latest 12 months.

## Why This Approach

The database runs in WAL mode and has active writers from BrainBar, enrichment, watch, and maintenance jobs. Copying `brainlayer.db` directly can miss WAL contents or capture an inconsistent file pair. SQLite's online backup API reads through SQLite itself, so the backup is a consistent snapshot without stopping the live services.

Direct Google Drive API upload is used because the post-repair machine no longer has Google Drive Desktop mounted at the old CloudStorage path. Historical DriveFS logs show the previous path was:

`~/Library/CloudStorage/GoogleDrive-etanface@gmail.com/My Drive/Brain Drive`

That mount is not present after repair, and `/Applications/Google Drive.app` is also absent. The API path avoids depending on that local mount.

## Implementation

Repo files:

- `src/brainlayer/backup_daily.py`: creates the SQLite backup, gzips it, uploads it to Drive, and prunes retention.
- `scripts/launchd/backup-daily.sh`: launchd wrapper installed to `~/.local/lib/brainlayer/backup-daily.sh`.
- `scripts/launchd/com.brainlayer.backup-daily.plist`: LaunchAgent template.
- `scripts/launchd/install.sh backup`: installs the wrapper and LaunchAgent.

Local logs:

- `~/.local/share/brainlayer/logs/backup-daily.log`
- `~/.local/share/brainlayer/logs/backup-daily.err`

Manual run:

```bash
PYTHONPATH=~/Gits/brainlayer/src python3 -m brainlayer.backup_daily
```

## Restore Drill

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

4. Stop writers before replacing the live DB:

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
   mv ~/.local/share/brainlayer/brainlayer.db* ~/.local/share/brainlayer/corrupt-$ts/
   cp /tmp/brainlayer-restore/brainlayer.db ~/.local/share/brainlayer/brainlayer.db
   ```

6. Re-enable services:

   ```bash
   launchctl load ~/Library/LaunchAgents/com.brainlayer.brainbar.plist
   launchctl load ~/Library/LaunchAgents/com.brainlayer.enrichment.plist
   launchctl load ~/Library/LaunchAgents/com.brainlayer.watch.plist
   launchctl load ~/Library/LaunchAgents/com.brainlayer.decay.plist
   ```

7. Confirm the DB is usable:

   ```bash
   sqlite3 ~/.local/share/brainlayer/brainlayer.db 'PRAGMA integrity_check; SELECT count(*) FROM chunks;'
   brainlayer wal-checkpoint --mode TRUNCATE
   ```

## Monthly Drill

Once per month:

1. Download the newest snapshot from Drive.
2. Restore it into `/tmp/brainlayer-restore`.
3. Run `PRAGMA integrity_check` and `SELECT count(*) FROM chunks`.
4. Record the snapshot date, chunk count, and command output in the maintenance log.

Do not replace the live DB during a drill unless the live DB is actually corrupted.
