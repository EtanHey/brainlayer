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
Host selection uses the shared detector in strict mode: an explicit `m1`/`m4` override
or a recognised Apple M1/M4 CPU brand on macOS. Any other host keeps everything
and raises `backup-retention`. An unreadable weekly pin also keeps everything and alerts.
Stalled uploads without their own verified live Drive copy remain until a person moves
them (about 9.5 GB each on the M4). They can exceed the local count; the disk gate
then defers and alerts rather than consuming the M4's 40 GiB reserve.
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
The configured preview below still needs the owner's explicit instruction before
any future Drive cleanup can be enabled.

Local cleanup reads a fresh listing when the receipt-proven local cohort exceeds its
cap; otherwise `local_retention_inventory=within_cap` explains why no listing was
needed. Failed measurement records an error and a `backup-retention` alert while the
verified backup keeps its successful outcome. A changed same-size archive is retained.

## Database Drive backup rule (read-only)

The default rule is **7 daily + one per month for 3 months, per Mac**. The monthly
representative is the last dated daily backup in each of the previous three full
calendar months; the current partial month does not count. Daily and monthly
cohorts are combined without counting the same Drive object twice.

| Runtime setting | Default | Accepted values |
| --- | --- | --- |
| `BRAINLAYER_BACKUP_DRIVE_DAILY_KEEP` | `7` | Positive integer |
| `BRAINLAYER_BACKUP_DRIVE_MONTHLY_KEEP` | `3` | Integer from `0` to `120` |

These settings change the read-only selection report, independently for each Mac's
owned folder. They do not authorize Drive changes. The separate Drive-write gate
`backup_daily._drive_retention_enabled()` stays hard-coded **false**. Existing
`BRAINLAYER_BACKUP_DRIVE_RETENTION`, JSONL retention opt-ins, and a similarly named
write-enable environment variable cannot turn this database gate on. Enabling a
future cleanup needs Etan's explicit instruction and a separately reviewed change;
this implementation has no Drive cleanup actor. Local retention still uses its
own verified-copy requirements and per-host counts.

The backup records `drive_retention_selection` when its existing local-retention
inventory path reads Drive. A failed listing records the attempt time and error;
a missing report is not a zero-file inventory. `observability.json` reads those
records without contacting Drive and adds optional `backups.drive_retention` in
schema v2. Schema v1 is unchanged. The configuration remains visible even when
backup health is unmeasurable.

`drive_retention.rule` contains `daily_keep`, `monthly_keep`, `per_machine: true`
and `description`; `writes_enabled` is false and `config_error` is null or an
explanation. `machines[]` contains `machine_id`, `state`, `selected_at`, `error`,
`on_drive`, `named_one_offs_count`, `kept`, `kept_by_rule`, `unverified`, `pinned`, `would_remove` and
`monthly_representatives`. Each summary contains `count`, `bytes`, `oldest` and
`newest`; dates are ISO dates, and `selected_at` is the last selection attempt's
ISO timestamp. `on_drive` describes dated daily `.db`, `.db.gz` and `.db.zst`
objects only; named one-offs are counted separately and excluded from selection.

States are `measured`, `unmeasurable`, `not_run` and `stale`. Unknown or failed
selections have null summaries/counts. A rule change or calendar-month rollover makes the old report stale
until another inventory is read; old candidate counts are withheld. Invalid
configuration is reported rather than silently falling back. Reports are cached
observations at `selected_at`, not a claim that a fresh Drive listing was fetched
when settings opened.

Only receipt-verified dailies matching the live listing's exact ID, name, size,
machine, folder and MD5 can enter `would_remove`. The weekly pin and unverified
daily objects stay in `kept`; historical raw `.db` files without this proof also
stay kept. Only verified dailies fill the newest-N or monthly slots, using the same
receipt eligibility and ordering as the deletion selector. An unverified newest or
month-end object cannot displace the Mac's verified copy. `kept` is total retained,
including pins and unverified files; `kept_by_rule` is only the verified daily/monthly
cohort, and `unverified` separately describes the unverified dated objects. These two
summaries and `pinned` are disjoint buckets of `kept`, not additional files in the total.
`pinned` counts only verified weekly-pin files outside the rule cohort. If a verified
pin is also selected daily/monthly, `kept_by_rule` wins; an unverified pin stays only
in `unverified`. For both counts and bytes, `kept = kept_by_rule + unverified + pinned`
and `on_drive = kept + would_remove`. Old cached reports
without these summaries are unmeasurable until inventory refreshes. The unused legacy
`keep_7` comparison was removed; configured selection is the preview consumers use.
A missing monthly cohort is skipped, never backfilled from an older month.
An invalid Drive setting or report-only error records a timestamped selection
failure and `config_error` where applicable; it does not pause local cleanup.
`selected_at` is required on every measured or failed selection so BrainBar can
display its age. A measured report may remain weeks old within the same month.

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
