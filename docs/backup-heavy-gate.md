# Daily backup admission

The daily LaunchAgent (`scripts/launchd/com.brainlayer.backup-daily.plist`, 03:17)
and the maintenance calendar invoke `brainlayer.backup_daily`. Its parent waits
for fleet admission before spawning the supervised backup child. The lock covers
snapshot, verification, compression and upload, including upload-only recovery.
The child does not acquire a second lock.

Admission uses atomic `mkdir ~/.m4-heavy.lock` or `~/.m1-heavy.lock`. CPU-brand
detection selects M1 for Apple M1, otherwise M4. Each acquisition/release writes
`brainlayer backup_daily took HH:MM` / `released HH:MM` to the host's heavy log.
Swap must be **under 9 GiB**, one-minute load **under 12** on both hosts. The
gate also requires raw free space for the backup's peak allocation plus a host
reserve: **40 GiB on M4, zero extra on M1**. It calls the existing
`_peak_backup_required_bytes` with the current DB size, estimated compressed
size and full-verify schedule, using the conservative non-reclaiming peak.
That helper already includes the backup's 2 GiB raw floor. Raw
`free_now_bytes` is measured on the staging volume before and under the lock;
purgeable capacity, future pruning and streaming reclamation do not count.
The 44 GiB build/install floor does not apply to backups.
Failed or unreadable probes defer work. Pressure is probed before taking a lock,
then rechecked under it; only admitted work logs took/released. A held lock is
never removed, including a stale one.
Only orc may clear another lane's stale lock after its age/process checks.

The parent retries every ten minutes within a six-hour pressure window, logging
each reason. At the end of that window it runs despite swap/load pressure, but
still waits for and takes the host lock. It logs that it is running past the
window and the last deferral reason. This prevents pressure alone from causing
a missed nightly backup. **The disk reserve is never waived**: a disk shortage
continues to defer past the pressure window and alerts with "needs N GiB, has
M GiB". The lock remains held through upload.
The first deferral raises a durable `backup-daily` (or `backup-snapshot`) job
alert naming its reason and start time. It survives a stop, SIGKILL or power
loss; only a later successful backup clears it. Disk shortages atomically
refresh the same alert's reason without clearing it or sending repeat notices.
A calendar admission deadline
can still return exit 75 while preserving the alert: pressure overrides cannot
extend the calendar's hard budget or preempt another lane's lock. Admission reserves at least
60 seconds of work (or the initial computed budget, if smaller) beside the
snapshot receipt reserve. Time spent waiting is subtracted from the child's
timeout so it cannot overrun the calendar's outer budget. A weekly snapshot
can spend its budget waiting and cause later maintenance steps to defer past
the quiet window; this preserves fleet admission even on that night. Signals
unwind an owned lock after the supervisor
has reaped its child. SIGKILL/power loss may leave the lock for orc to assess.

Configuration (numeric values must be finite and positive, except the host
reserve which also accepts zero):

| Environment variable | Default |
| --- | --- |
| `BRAINLAYER_BACKUP_HEAVY_HOST` | CPU detection; override `m1` or `m4` |
| `BRAINLAYER_BACKUP_HEAVY_LOCK_PATH` | `~/.<host>-heavy.lock` |
| `BRAINLAYER_BACKUP_HEAVY_LOG_PATH` | `~/.<host>-heavy.log` |
| `BRAINLAYER_BACKUP_HEAVY_WINDOW_SECONDS` | 21600 |
| `BRAINLAYER_BACKUP_HEAVY_RETRY_SECONDS` | 600 |
| `BRAINLAYER_BACKUP_HEAVY_SWAP_GIB` | 9 |
| `BRAINLAYER_BACKUP_HEAVY_LOAD` | 12 |
| `BRAINLAYER_BACKUP_HEAVY_HOST_FREE_GIB` | 40 on M4, 0 on M1; extra reserve above peak |

Focused tests use fake resource probes and deterministic time with real lock
directories inside `.build.noindex/pytest`. They never run a real backup or open
the live database.
