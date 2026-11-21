# Historical T3 thinking migration

`scripts/reclassify_t3_thinking.py` labels old T3 assistant thinking rows with
`source_class='agent-thinking'`, `metadata.source_class='agent-thinking'`,
`metadata.thinking=true`, and the original class in `metadata.thinking_source_class`.
Default search hides that class; the existing internal hidden-source opt-in reveals it.
No transcript content is deleted.

Selection requires `source='t3'`, `sender='assistant'`, a class other than
`agent-thinking`, and the exact native thinking item ID predicate used by #1824
(`brainlayer.ingest.t3._is_thinking_message`). Text mentioning thinking is irrelevant.
The rehearsal census differs from the original estimate because it measures the
current snapshot rather than assuming an earlier row count.

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
Missing paths are never created. Apply and undo checkpoint FULL before and after,
use 5,000-row transactions, checkpoint every three batches, and retry SQLITE_BUSY
with bounded backoff. An unsuccessful checkpoint fails loudly.

Every batch atomically journals original source_class and raw metadata bytes in
`t3_thinking_reclassification`, keyed by run ID and chunk ID. Keep the printed ID,
especially if a run is interrupted. Undo restores those exact values and marks
the journal records undone; it retains the audit journal. A second apply changes
zero rows. A completed undo is also idempotent. Without `--run-id`, undo accepts
only a single active run; multiple active runs require an explicit choice.

Undo refuses rows edited since apply instead of overwriting newer metadata.
Quiescence remains required: each batch checks its inputs, but earlier completed
batches remain committed after an error. Undo each printed run ID to reverse a
partial migration or a resumed apply that generated another run ID.

## Validation

The initial four tests failed on the unchanged base because the script did not
exist. Six migration tests cover apply, read-only dry-run, idempotence, exact undo,
non-T3/user/non-thinking exclusions, canonical and hardlink refusal, later-edit
refusal, SQLITE_BUSY rollback/retry, and interrupted-batch recovery. Together with
the existing source visibility tests: 31 passed.

The M4 Max rehearsal used one `cp -c` clone of the 26 GiB DB and its WAL/SHM.
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
