# Daily backup admission

The daily LaunchAgent (`scripts/launchd/com.brainlayer.backup-daily.plist`, 03:17)
and the maintenance calendar invoke `brainlayer.backup_daily`. Its parent waits
for fleet admission before spawning the supervised backup child. The lock covers
snapshot, verification, compression and upload, including upload-only recovery.
The child does not acquire a second lock.

Admission uses atomic `mkdir ~/.m4-heavy.lock` or `~/.m1-heavy.lock`. CPU-brand
detection selects M1 for Apple M1, otherwise M4. Each acquisition/release writes
`brainlayer backup_daily took HH:MM` / `released HH:MM` to the host's heavy log.
Swap must be **under 9 GiB**, one-minute load **under 12**; M1 also needs at least
44 GiB free. Failed or unreadable probes defer work. A closed gate releases its
own lock before waiting; a held lock is never removed, including a stale one.
Only orc may clear another lane's stale lock after its age/process checks.

The parent retries every ten minutes for up to six hours, logging each reason.
Expiry raises a durable `backup-daily` (or `backup-snapshot`) job alert visible
to BrainBar and returns exit 75. A later successful backup clears that alert.
The calendar's absolute snapshot deadline shortens admission to preserve the
existing receipt reserve. Calendar outer run limits still apply; this admission
window does not extend them. Signals unwind an owned lock after the supervisor
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
| `BRAINLAYER_BACKUP_HEAVY_FREE_GIB` | 44 (M1) |

Focused tests use fake resource probes and deterministic time with real lock
directories inside `.build.noindex/pytest`. They never run a real backup or open
the live database.
