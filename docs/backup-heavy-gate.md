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
size and full-verify schedule. On confirmed APFS source/staging volumes it uses
the [planned clone peak](backup-apfs-clone.md); otherwise it uses the conservative
non-reclaiming VACUUM peak. A failed clone rechecks the larger fallback budget.
That helper already includes the backup's 2 GiB raw floor. Raw
`free_now_bytes` is measured on the staging volume before and under the lock;
purgeable capacity, future pruning and streaming reclamation do not count.
Peak allocation is phase-aware. Snapshot-only reserves the clone or VACUUM
snapshot/compression peak, with no Drive download/restore. Upload-only creates neither file: the
existing archive is already counted in measured free space. Pending receipt
metadata selects its verification budget without hashing payloads or changing
the journal. Streaming hash verification allocates no local payload; legacy
verification reserves a DB-sized restore and, on full verification, a downloaded
archive. A batch reserves the largest candidate's additional space; an explicit
recovery target reserves only that target. The existing 2 GiB floor is retained.
The 44 GiB build/install floor does not apply to backups.
Failed or unreadable probes defer work. Pressure is probed before taking a lock,
then rechecked under it; only admitted work logs took/released. A held lock is
never removed, including a stale one.
Only orc may clear another lane's stale lock after its age/process checks.

On M1 only, the gate participates in the binding `~/.m1-heavy.next`
reservation protocol (orc, 26 November 2026). It publishes
`brainlayer-backup <job_key> <pid> <ISO time>` when the slot is absent or empty,
only after resource probes pass and a held lock blocks admission. A failed
resource probe on the next retry withdraws its owned reservation. A foreign reservation blocks
admission even with a free lock. The gate may reclaim it only after observing
the lock free for **more than ten minutes**; observing a held lock resets that
timer, as does a change in reservation content, device, inode or nanosecond
modification time. Reclamation rechecks that same identity immediately before
unlinking. An old reservation timestamp alone is insufficient evidence. This may
conservatively add ten minutes after a gate restart.
Publication uses a temporary file in the same directory and an exclusive hard
link, so a foreign reservation published concurrently cannot be overwritten.
Empty slots are removed with the same identity check before exclusive publication.
Symlinks and non-regular reservation files
defer admission and are never followed. Real admission, after the post-lock probes,
removes the backup's own reservation. A post-lock probe bounce retains an existing
reservation through release and retry sleep; the next resource sample withdraws it
if resources still defer admission. Deadline, signal and exception unwinding also remove it if its
content still matches. A replacement belonging to another lane is retained.
Cleanup checks content and file identity, then checks identity again with lstat
immediately before unlink. POSIX path-based unlink still has a final scheduling
window after that check; it does not provide atomic compare-and-delete.
SIGKILL/power loss can leave a reservation for the stale-slot rule to handle.
M4 neither reads nor writes `.next`.

The parent retries every ten minutes within a six-hour pressure window, logging
each reason. At the end of that window it runs despite swap/load pressure, but
still waits for and takes the host lock. It logs that it is running past the
window and the last deferral reason. This prevents pressure alone from causing
a missed nightly backup. **The disk reserve is never waived**: a disk shortage
continues to defer past the pressure window and alerts with "needs N GiB, has
M GiB". The lock remains held through upload.
The first deferral raises a durable `backup-daily` (or `backup-snapshot`) job
alert naming its reason and start time. It survives a stop, SIGKILL or power
loss while the backup waits. Every deferral atomically
refreshes the same alert to its current disk, pressure, lock or time-window
cause without clearing it or sending repeat notices. When the gate admits the
run, including past the pressure window, it clears the alert only if this
invocation raised a wait, so a running upload is never reported as waiting; the
heavy log keeps the last reason. The backup's end then clears it on success or
records "Database backup failed". An alert this invocation did not raise (for
example, the previous night's failure when this run never waited) stays until
that end.
A calendar admission deadline
can still return exit 75 while preserving the alert: pressure overrides cannot
extend the calendar's hard budget or preempt another lane's lock. Admission
reserves the phase's expected work duration beside the snapshot receipt
reserve: 40 minutes for snapshot, 30 for upload, 70 for all. These configurable
estimates cover the review's observed snapshot/verify/compression and upload
ranges; actual network completion remains variable. A smaller available budget
defers to the next window with an alert; it never shrinks the expected duration
to fit the remaining time. Time spent waiting is subtracted from the child's
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
| `BRAINLAYER_BACKUP_HEAVY_EXPECTED_SNAPSHOT_SECONDS` | 2400 |
| `BRAINLAYER_BACKUP_HEAVY_EXPECTED_UPLOAD_SECONDS` | 1800 |
| `BRAINLAYER_BACKUP_HEAVY_EXPECTED_ALL_SECONDS` | 4200 |

Focused tests use fake resource probes and deterministic time with real lock
directories inside `.build.noindex/pytest`. They never run a real backup or open
the live database.
