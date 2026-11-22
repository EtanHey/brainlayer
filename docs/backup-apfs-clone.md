# APFS daily snapshots

Daily snapshots use an APFS clone when the source and staging directory share an APFS volume.
`BRAINLAYER_BACKUP_SNAPSHOT_METHOD=vacuum` selects the existing VACUUM path explicitly; the default
is `auto`. Unsupported filesystems, a busy writer, an oversized WAL, or a failed clone log a reason
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

The clone budget is compressed bytes plus 320 MiB: at most 64 MiB of WAL recovery and a 256 MiB CoW
margin. The existing 2 GiB raw floor and 1 MiB copy allowance are additional; the gate also retains
the M4's configurable 40 GiB host reserve (M1: zero extra). A source WAL larger than 64 MiB uses the
capacity-checked VACUUM fallback instead of claiming a small recovery peak.

Clones pin extents when the source changes. A volume-free-space guard checks during integrity,
hashing and compression, stopping if growth beyond the archive exceeds the margin or the reserve
is breached. This is a conservative sampled volume measurement, including unrelated writes; it is
not an attribution of every allocated block, nor a guarantee against arbitrary concurrent disk
writers. A stopped run follows the existing failure/alert path. Clone attempts never hole-punch
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
row counts for every table. The restore is acceptance work, not part of the production peak.
