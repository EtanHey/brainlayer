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
backup's existing snapshot-space checks and 2 GiB raw floor remain responsible
for disk admission; the 44 GiB build/install floor does not apply to backups.
Failed or unreadable probes defer work. Pressure is probed before taking a lock,
then rechecked under it; only admitted work logs took/released. A held lock is
never removed, including a stale one.
Only orc may clear another lane's stale lock after its age/process checks.

The parent retries every ten minutes for up to six hours, logging each reason.
The first deferral raises a durable `backup-daily` (or `backup-snapshot`) job
alert naming its reason and start time. It survives a stop, SIGKILL or power
loss; only a later successful backup clears it. Expiry logs the elapsed wait
and returns exit 75, retaining that first alert. Admission reserves at least
60 seconds of work (or the initial computed budget, if smaller) beside the
snapshot receipt reserve. Time spent waiting is subtracted from the child's
timeout so it cannot overrun the calendar's outer budget. A weekly snapshot
can spend its budget waiting and cause later maintenance steps to defer past
the quiet window; this preserves fleet admission even on that night. Signals
unwind an owned lock after the supervisor
has reaped its child. SIGKILL/power loss may leave the lock for orc to assess.

Configuration (all numeric values must be finite and positive):

| Environment variable | Default |
| --- | --- |
| `BRAINLAYER_BACKUP_HEAVY_HOST` | CPU detection; override `m1` or `m4` |
| `BRAINLAYER_BACKUP_HEAVY_LOCK_PATH` | `~/.<host>-heavy.lock` |
| `BRAINLAYER_BACKUP_HEAVY_LOG_PATH` | `~/.<host>-heavy.log` |
| `BRAINLAYER_BACKUP_HEAVY_WINDOW_SECONDS` | 21600 |
| `BRAINLAYER_BACKUP_HEAVY_RETRY_SECONDS` | 600 |
| `BRAINLAYER_BACKUP_HEAVY_SWAP_GIB` | 9 |
| `BRAINLAYER_BACKUP_HEAVY_LOAD` | 12 |

Focused tests use fake resource probes and deterministic time with real lock
directories inside `.build.noindex/pytest`. They never run a real backup or open
the live database.
