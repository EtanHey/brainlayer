# Terminal backup progress projection

The producer previously projected a durable `pending`/`in_progress` receipt even
after its calendar step ended. `observability_backup._terminal_upload_error`
now binds the receipt's timezone-aware attempt time to a backup-daily spawn /
finished interval. It preserves the terminal status and exit code as an
incomplete-upload error. A verified upload, another job's success, or a later
retry has its own evidence; none rewrites the preceding failure receipt.

`BackupProgress.tla` abstracts that projection, not the deployed backup system.
Bounds: two serial attempts; one latest receipt; snapshot/upload/death/terminal
phases; stopped, failed and successful calendar exits; a competing job; pause
and resume. A new attempt can leave an old receipt visible until its checkpoint.
`Death` can precede `RecordDeath`, representing a lost final producer receipt.
Pause/resume alone cannot complete an upload. No fairness or eventual-delivery
claim is made; these are safety checks with stuttering allowed.

Assumptions: trustworthy ordered journals, timezone-aware timestamps that bind
an attempt to its serial owning run, and no ambiguous overlapping attempt
intervals. Missing, malformed or unbound calendar history cannot establish a
terminal result. Neither the model nor this repair proves process liveness when
no matching terminal record exists. Filesystem durability, wall-clock rollback,
Drive behavior, cancellation delivery and real backup recovery are outside the
model. The Python regression and producer-to-Swift fixture test exercise the
implementation separately.

Run with Java and an existing `tla2tools` jar (the local filename is not its
reported TLC version):

```sh
java -XX:+UseParallelGC -Xmx512m -cp "$TLA2TOOLS" tlc2.TLC -workers 1 -seed 123 -fp 0 -metadir .build.noindex/tlc-backup-old -config docs/verification/backup-progress/old.cfg docs/verification/backup-progress/BackupProgress.tla
java -XX:+UseParallelGC -Xmx512m -cp "$TLA2TOOLS" tlc2.TLC -workers 1 -seed 123 -fp 0 -metadir .build.noindex/tlc-backup-upload -config docs/verification/backup-progress/old-upload.cfg docs/verification/backup-progress/BackupProgress.tla
java -XX:+UseParallelGC -Xmx512m -cp "$TLA2TOOLS" tlc2.TLC -workers 1 -seed 123 -fp 0 -metadir .build.noindex/tlc-backup-corrected -config docs/verification/backup-progress/corrected.cfg docs/verification/backup-progress/BackupProgress.tla
```

Executed with TLC2 2.19 (08 August 2024, revision 5a47802), Java 17.0.19:

- Old projection: exit 12, invariant violation, 29 generated / 20 distinct
  states, depth 4. Counterexample: Start, Snapshot, Finish(stopped), while the
  receipt remains pending.
- Old upload projection: exit 12, 83 generated / 44 distinct states, depth 5.
  Counterexample adds Checkpoint before Finish(stopped); the receipt remains
  in_progress, matching the observed symptom.
- Corrected projection: exit 0, 3,208 generated / 756 distinct states, depth 9,
  empty queue, no invariant violation. This is bounded model evidence only.
