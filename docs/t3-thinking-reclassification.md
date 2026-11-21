# Historical T3 thinking migration

`scripts/reclassify_t3_thinking.py` labels old T3 assistant thinking rows with
`source_class='agent-thinking'`, `metadata.source_class='agent-thinking'`,
`metadata.thinking=true`, and the original class in `metadata.thinking_source_class`.
Default search hides that class; the existing internal hidden-source opt-in reveals it.
No transcript content is deleted.

Selection requires `source='t3'`, `sender='assistant'`, a class other than
`agent-thinking` or `brain-worker`, and the exact native thinking item ID predicate used by #1824
(`brainlayer.ingest.t3._is_thinking_message`). Text mentioning thinking is irrelevant.
Sol reconciled the counts against the original **17 November 21:36 UTC** census:
**10,286 + 1,154 newer rows + 10 older hashed-ID rows − 4 false positives = 11,446**.
The old query matched a thinking substring anywhere in the whole chunk ID. It
missed hashed IDs and matched a thinking-policy thread's user/reply rows. The new
predicate checks the native message ID in metadata. The difference includes both
ingestion growth and the old selector's precision/coverage limitations.

## Lead procedure

The lead runs it on the M4 after a fresh backup and review (Sol + one Claude,
orc 21 November 14:06). Quiesce all writers first. Workers use only an APFS clone.
Use the reviewed worktree's Python environment with `PYTHONPATH=src` so the
predicate comes from the reviewed source. Supply an absolute DB path.

```console
python scripts/reclassify_t3_thinking.py --db /path/to/copy.db
python scripts/reclassify_t3_thinking.py --db /path/to/copy.db --apply
python scripts/reclassify_t3_thinking.py --db /path/to/copy.db --undo --run-id PRINTED-ID
```

Dry-run is the default and opens read-only. Mutations refuse the canonical DB,
symlinks and hardlinks to it unless the lead explicitly supplies `--live`.
Missing paths are never created. Apply and undo use only PASSIVE checkpoints
before, after, and every three batches. Checkpoints never wait for readers or
writers; pending frames and checkpoint errors are reported separately from the
committed result. A checkpoint error does not turn committed changes into a
failed migration. Writes retain 5,000-row transactions and bounded SQLITE_BUSY
retries; those transactions briefly hold the writer lock.

Every batch atomically journals original source_class and raw metadata bytes in
`t3_thinking_reclassification`, keyed by run ID and chunk ID. Keep the printed ID,
especially if a run is interrupted. Undo restores those exact values and marks
the journal records undone; it retains the audit journal. A second apply changes
zero rows. A completed undo is also idempotent. Without `--run-id`, undo accepts
only a single active run; multiple active runs require an explicit choice.

Undo skips rows whose class or metadata changed since apply and reports each ID
and changed field (`source_class`, `metadata`, or `row` for a missing row). Other
rows are restored. Skipped journal records remain active for later review/retry;
the summary prints restored and skipped counts. A successful process exit means
the reported changes committed, not that every row was restored: read the skip
report. Undo never overwrites newer metadata.
Quiescence remains required: each batch checks its inputs, but earlier completed
batches remain committed after an error. Undo each printed run ID to reverse a
partial migration or a resumed apply that generated another run ID.

## Validation

The initial four tests failed on the unchanged base because the script did not
exist. Eleven migration tests cover apply, read-only dry-run, idempotence, exact undo,
non-T3/user/non-thinking exclusions, canonical and hardlink refusal, later-edit
per-row skip reporting, SQLITE_BUSY rollback/retry, interrupted-batch recovery,
PASSIVE checkpoints with a held reader and a 25 ms writer, and separate reporting
of checkpoint errors after committed work. Together with the existing source
visibility tests: 36 passed after round 1.

The initial M4 Max rehearsal used one `cp -c` clone of the 26 GiB DB and its WAL/SHM.
`df -g /` showed 62 GiB available immediately before cloning. Initial
`PRAGMA quick_check` returned `ok`.

| Stage | Eligible old thinking | T3 cli-agent | T3 agent-thinking | Changed |
| --- | ---: | ---: | ---: | ---: |
| Dry-run | 11,446 | 258,438 | 20 | 0 |
| Apply | 0 | 246,992 | 11,466 | 11,446 |
| Apply again | 0 | 246,992 | 11,466 | 0 |
| Undo | 11,446 | 258,438 | 20 | 11,446 |

A real readonly `VectorStore.search` query returned one selected legacy row
before apply. After apply, the query returned no selected rows by default and
returned that sample with the existing hidden-source opt-in.

The initial rehearsal harness mixed stdlib SQLite and APSW connections in one
process and exited SIGBUS (138) during undo. A standalone CLI undo restored all
11,446 rows. The repeated rehearsal uses separate search subprocesses and
explicitly closes all stdlib connections. The CLI also explicitly closes its
connection after each invocation. This records the observed failure and recovery;
it does not claim a proven native-crash root cause.

The isolated full cycle passed: the SHA-256 stream of every chunk's ID, class,
and raw metadata matched exactly after undo, original search visibility returned,
and the final `PRAGMA quick_check` returned `ok`. The rehearsal process exited 0.
The clone was removed after verification. This is clone-backed migration proof;
the live migration remains the lead's reviewed, backed-up step.

## Round 1 rehearsal

After the routed Sol/Claude reviews, the F1–F3 changes passed 36 focused tests.
Four regression cases failed on the reviewed version before these fixes.
One new APFS DB/WAL/SHM clone was rehearsed under `nice -n 15` throughout;
disk had 89 GiB available before cloning. The initial `quick_check` returned `ok`.

| Stage | Eligible old thinking | T3 cli-agent | T3 agent-thinking | Changed |
| --- | ---: | ---: | ---: | ---: |
| Dry-run | 11,446 | 258,811 | 22 | 0 |
| Apply | 0 | 247,365 | 11,468 | 11,446 |
| Apply again | 0 | 247,365 | 11,468 | 0 |
| Undo | 11,446 | 258,811 | 22 | 11,446 restored, 0 skipped |

All before/batch/after checkpoints used PASSIVE and reported complete. Real
readonly search again hid selected rows by default and returned the baseline
sample with the internal opt-in. The all-chunk ID/class/raw-metadata hash matched
exactly after undo: `322c1c0b0e2ded17c5202caec896132b2b073c31ff73bef3007dd5367effd985`.

Post-undo `quick_check` returned `ok`; the rehearsal process exited 0. The
DB/WAL/SHM clone was removed after verification. No live migration or merge.
