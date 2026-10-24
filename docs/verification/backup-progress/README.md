# Backup invocation ownership and terminal projection

`BRAINLAYER_BACKUP_INVOCATION_ID` carries a fresh UUID hex value from the calendar
backup-daily invocation to its child. Spawn and finished journal rows carry the
optional `backup_invocation_id`. The backup body copies it into artifact receipts
only after staging-lock admission; lock losers cannot stamp an admitted receipt.
A retry gets a new invocation. Existing rows remain valid without the field.

The reader requires one complete spawn/finished pair with the receipt's exact
valid ID, finite ordered event timestamps, and no duplicate or interrupted pair.
Time containment alone never supplies ownership. Legacy/manual/unbound receipts
remain unattributed. Their raw in_progress checkpoint is preserved; the pure
status projection says activity is unknown, rather than advertising a live
upload. The stopped calendar job still has its independent failure status.

Calendar decoding is line-oriented over bytes. Complete valid prefixes survive
a partial, bad-JSON or non-UTF-8 tail. Each skipped line is recorded in inputs and
acts as a barrier to an unfinished pair. It cannot stitch evidence across a gap;
a complete proven pair before a damaged later append remains available. No raw
journal is repaired or rewritten by the reader. Duplicate matching IDs fail closed.

## R1 model fidelity correction

The old model/logs at commit6e5a0d5f stipulated receiptAttempt=attempt. Its
756-distinct-state green run validated that stipulated projection, **not actual
ownership**, and missed the calendar-spawn-to-lock gap. Original logs remain
preserved; they are not evidence against the independent R1 counterexample.

The replacement model has two sequential calendar invocations and an independent
manual contender. Spawn, staging-lock admission/loss, snapshot/checkpoint,
uploaded-before-verification, verification, death and calendar finish are separate
actions. Death releases the admitted body lock before calendar finish, so another
actor can acquire it in that gap. Receipt owner is independent ground truth. A receipt copies its admitted
actor's inherited nonce; the binding algorithm must discover the owner from
journal evidence. Missing legacy IDs, an interior journal gap, a damaged later
append, duplicate IDs, stale receipts and a later retry are represented. Pause /
resume permit stuttering. `receiptSpan` records which calendar intervals contained
body admission: an order abstraction of the OLD timestamp-containment algorithm,
not ownership. Old and corrected configs share transitions and finite bounds;
Corrected changes the reader's candidate/projection rule only.

`NoWrongOwner` compares the inferred binding against the independent actual lock
owner. `OrdinaryTerminalHandled` checks provable unfinished terminal runs;
`CompletedReceiptNotRevoked` covers uploaded/verified receipts. `HistoryForwarded`
only checks the pure projection forwards modeled verification history unchanged;
it does not prove storage or archive bytes. `UnprovenIsNotLive` covers the status
projection. Python/Swift tests separately execute actual reader/writer contracts.

Assumptions: trusted local writers, fresh non-colliding invocation IDs, and their
normal inheritance (no deliberate forgery). Missing or corrupt proof declines
attribution. The model makes no fairness/eventual delivery, real signal delivery,
Drive, filesystem durability, storage-byte or deployed-system verification claim.
A death without a durable terminal record remains unconfirmed.

Run from repo root with the existing toolchain:

```sh
java -XX:+UseParallelGC -Xmx512m -cp "$TLA2TOOLS" tlc2.TLC -workers 1 -seed 123 -fp 0 -metadir .build.noindex/tlc-owner-old -config docs/verification/backup-progress/old.cfg docs/verification/backup-progress/BackupProgress.tla
java -XX:+UseParallelGC -Xmx512m -cp "$TLA2TOOLS" tlc2.TLC -workers 1 -seed 123 -fp 0 -metadir .build.noindex/tlc-owner-active -config docs/verification/backup-progress/old-upload.cfg docs/verification/backup-progress/BackupProgress.tla
java -XX:+UseParallelGC -Xmx512m -cp "$TLA2TOOLS" tlc2.TLC -workers 1 -seed 123 -fp 0 -metadir .build.noindex/tlc-owner-corrected -config docs/verification/backup-progress/corrected.cfg docs/verification/backup-progress/BackupProgress.tla
```

Actual TLC2 2.19 (08 August2024, revision5a47802), Java17.0.19, one worker:

- Old timestamp rule: exit12,2177 generated/849 distinct, depth7. Calendar spawns;
  manual wins lock; calendar loses and finishes failed; manual receipt is falsely
  bound to that failed contender. Manual still owns the staging lock.
- Exact in_progress variant: exit12,6969 generated/2487 distinct, depth8. Adds
  the manual checkpoint; the unrelated calendar failure revokes its progress.
- Corrected binding: exit0,9943909 generated/1201546 distinct, depth29, empty
  queue; no invariant violation. These are bounded source-model results only.
