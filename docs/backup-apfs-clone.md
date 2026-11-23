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

Run-deadline and stop signals propagate through integrity checking, including SQLite's
exception-clearing callback boundary; they never select a VACUUM fallback. A failed clone
integrity check raises the `backup-clone-integrity` job alert even if VACUUM succeeds. Only a
later healthy clone integrity check clears that diagnostic.

Under the staging lock, admission sweeps aged, same-user private clone directories whose
contents are recognized regular files with one hardlink and whose recorded process is gone. Recent,
live, linked, foreign and unknown entries remain. Their full possible COW divergence is
reserved before admission, including retained dated raw snapshots, without crediting any
existing bytes as free space.

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

Snapshot sleep observations span a capacity-checked VACUUM fallback, including sleep during the
discarded clone attempt. The clone's owned identity is recorded before the post-publication observer,
so an observer failure still removes that attempt's clone. Regression tests cover both boundaries.

The M1 copy-only acceptance check bootstraps an APFS copy through a read-only canonical connection,
rejecting changed DB/WAL identities or data versions. Snapshotting, private checkpointing, full
integrity and compression run only against that copy. A separate restore checks full integrity and
row counts for every table. A separate writer updates an owned synthetic table at the planned
2 MiB/s payload rate, including a five-second burst at the observed 18,175,039 B/s peak. Sequence and
payload checks on the restored synthetic rows supplement full integrity and table counts. The
restore is acceptance work, not part of the production peak.

### M1 copy acceptance, 2026-11-22

One run against source hashes from `056846f6` (the M1 checkout's base HEAD was `797ad572`;
the three changed source-file hashes were asserted before and after the run).

| Observation | Result |
| --- | --- |
| Owned source copy, including synthetic writer table | 4,738,621,440 bytes |
| Compressed archive | 1,792,023,691 bytes |
| Sampled volume-free delta peak during backup | 2,076,516,352 bytes |
| Snapshot / restored full integrity | Both `ok` |
| Per-table row counts | All 93 equal |
| Restored writer sequence/payload invariant | Passed |
| Concurrent writer | 17,559 commits, 287,686,656 payload bytes in 98.857 s |
| Writer pacing | 2 MiB/s sustained; 18,175,039 B/s for the first five seconds |
| Local archive proof | SHA-256 matches the integrity-checked snapshot |

The sampled peak includes unrelated volume activity and excludes the separate restore check.
This proves the copy path with an active writer on M1; it does not measure M4 production peak or
Drive transport. The owned DB copies and archive were removed after recording the receipt.

### M4 copy acceptance, 2026-11-23

The renewed check used source `47cd54db12620381142049d78ae8537b9d93e052`, asserting the daily, clone, gate and
sleep module hashes before and after. Bootstrap used only `cp -c` of the canonical DB and WAL;
both files' identity, size, mtime and ctime stayed unchanged across those copies. The proof process
rejected SQLite connections outside its owned scratch directory. All recovery, snapshotting,
writing, integrity checks and queries used owned copies.

| Observation | Result |
| --- | --- |
| Owned source, including writer table | 28,334,350,336 bytes |
| Compressed archive | 12,693,817,094 bytes |
| Sampled backup-phase volume delta peak | 15,364,497,408 bytes |
| Snapshot / restored full integrity | Both `ok` |
| Per-table row counts | All 114 equal |
| Restored writer sequence/payload invariant | Passed |
| Concurrent writer | 82,309 commits, 1,348,550,656 payload bytes in 604.714 s |
| Writer schedule | 2 MiB/s sustained; first five seconds at 18,175,039 B/s |
| Writer busy results | 0 |
| Local archive proof | SHA-256 matches the integrity-checked snapshot |

The peak includes other volume activity and excludes bootstrap and the separate restore acceptance.
Restore enforced a 42 GiB free-space reserve. The run began with 105 GiB free; after all owned DBs,
WAL/SHM files, the archive and scratch directory were removed, the measured free space was 107 GiB.
Deletion used one named file per command. This is a copy-only proof, not a scheduled backup or a
Drive upload. Small numerical receipts remain; no copied database or archive remains.

Clone-specific race tests require SQLite's checkpoint-on-close control. Runtimes without that
control use the explicit VACUUM fallback, which has its own test rather than claiming clone coverage.
