# APFS daily snapshots

Daily snapshots use an APFS clone when the source and staging directory share an APFS volume.
`BRAINLAYER_BACKUP_SNAPSHOT_METHOD=vacuum` selects the existing VACUUM path explicitly; the default
is `auto`. Unsupported filesystems, a busy writer, an outgrown WAL budget, or a failed clone log a reason
and use VACUUM only after checking its larger peak plus the host reserve again.

## Consistent DB and WAL

The snapshot process opens the source only through SQLite. `BEGIN IMMEDIATE` holds the WAL writer
lock across two separate-process `/bin/cp -c` calls, for the DB and its WAL. Commits and WAL restart
or truncation cannot pass that lock. A PASSIVE checkpoint can copy only the stable WAL frames;
replaying the frozen WAL on the clone overlays those pages consistently. DB inode and WAL
identity/size/mtime are checked before and after copying. A source change rejects the attempt.

Copying runs in another process because closing a raw source descriptor in SQLite's process could
release its POSIX locks. Source checkpoint-on-close is disabled with
`SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE`; an older Python/SQLite runtime without that control falls back.
The source transaction rolls back and closes before integrity checking or compression.

The private clone rebuilds its own SHM, replays its private WAL, and becomes a standalone DELETE-mode
DB. Only that owned clone is checkpointed. Full `integrity_check` must pass before create-exclusive
publication. Raw SHA-256 and the decompressed archive SHA-256 must match before the raw clone is
unlinked. Peer files are never replaced or removed.

SQLite's [file-copy discussion](https://sqlite.org/forum/forumpost/905eb5e564) describes the writer
barrier and separate-process constraint. The [checkpoint API](https://www.sqlite.org/c3ref/wal_checkpoint_v2.html)
documents PASSIVE versus writer-locking checkpoint modes.

## Space budget and verification

The clone budget is compressed bytes plus the observed WAL size, 64 MiB of WAL growth slack, and a
divergence allowance: planned write bytes/second times expected snapshot seconds. The existing
2 GiB raw floor and 1 MiB copy allowance are additional. The gate retains the M4's configurable
40 GiB host reserve (M1: zero extra). A WAL that outgrows its admission budget triggers a rechecked
VACUUM fallback rather than unbudgeted private recovery.

`BRAINLAYER_BACKUP_CLONE_WRITE_BYTES_PER_SECOND` defaults to 2 MiB/s, from the read-only M4 sample
on 2026-11-22 04:42:15–04:45:15 UTC: 1,733,624 B/s average, rounded up. The sampler used DB/WAL
stat metadata and `proc_pid_rusage` v4 for processes observed holding the files, comparing logical
write-byte ledger increases with physical write-byte increases every five seconds. It never opened
a database. WAL size remained 97,920,072 bytes while its and the DB's mtimes advanced. The largest
five-second write rate was 18,175,039 B/s. Process-wide counters include other files, and newly seen
holders and short-lived processes limit attribution; this is a planning sample, not a future bound.
Apple's [kernel ledger definition](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/kern/task.c)
specifies logical-write units as bytes. Physical counters alone miss cached writes.

Expected duration uses `BRAINLAYER_BACKUP_HEAVY_EXPECTED_SNAPSHOT_SECONDS` (default 2400): this gives
4.6875 GiB divergence allowance. For a 12 GiB archive and the sampled WAL, the allocation plan is
about 16.84 GiB, before the separate raw/host floors. Real write activity makes a zero-divergence
13 GB promise inappropriate. Operators can remeasure and configure the rate/time estimate.

Clones pin extents when the source changes. During integrity, hashing and compression an absolute
free-space guard preserves 40 GiB plus the existing 2 GiB raw floor on M4, or 2 GiB on M1. Whole-volume
growth is recorded for observation and never used as an abort threshold. A floor breach removes this
attempt's owned clone/partial archive and rechecks VACUUM capacity. If that fits, backup continues;
otherwise the job defers with a durable alert, returning exit 75. Clone attempts never hole-punch
shared extents.

Both ordinary and weekly clone verification stream the archive without restoring a DB file.
Weekly full verification also streams Drive's compressed and decompressed hashes. Newly created
weekly VACUUM artifacts bind their full raw integrity proof to SHA-256 and stream too. Legacy
receipts without that binding retain the restore verification path and its upload-phase budget.

## Regression proof

`tests/test_backup_clone.py` uses a separate real SQLite writer and checkpoint process. It attempts
commits between file copies and proves they are busy, then succeed after release. A deliberately
removed barrier permits a commit and WAL truncation between copies; the pair is rejected rather
than published. Other cases cover clone failure, corrupt clone rejection, stop cleanup, a last
source connection that must not checkpoint, non-APFS fallback, and the fallback capacity check.
`tests/test_backup_heavy_gate.py` proves clone admission keeps the M4 reserve.

The M1 copy-only acceptance check bootstraps an APFS copy through a read-only canonical connection,
rejecting changed DB/WAL identities or data versions. Snapshotting, private checkpointing, full
integrity and compression run only against that copy. A separate restore checks full integrity and
row counts for every table. A separate writer updates an owned synthetic table at the planned
2 MiB/s payload rate, including a five-second burst at the observed 18,175,039 B/s peak. Sequence and
payload checks on the restored synthetic rows supplement full integrity and table counts. The
restore is acceptance work, not part of the production peak.
