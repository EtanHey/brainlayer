# Calendar index convergence

Calendar `index` passes retain durable input offsets in `index-checkpoint-<identity>.json`
beside the resolved database. `index-backlog-<identity>.json` holds the latest pass's
inventory, remaining files/chunks, stalled-pass count, and per-file failure counters.
Neither sidecar contains source text or exception messages.

`INDEX_CHECKPOINTED` includes `files_remaining`, `chunks_remaining`, and
`uncounted_files`. Inventory parsing and fingerprinting share the pass watchdog budget;
unchanged file counts are cached. If counting is interrupted, the chunk count is a
**known lower bound**, with the number of uncounted files printed explicitly.
`INDEX_BACKLOG` is also emitted on success, deferral, or failure after checkpoint setup.

A shrinking backlog keeps the normal rc0 checkpointed yield. After an initial baseline,
`BRAINLAYER_INDEX_STALL_PASSES` consecutive non-shrinking comparisons (default **3**)
produce `INDEX_BACKLOG_STALLED` and rc1, which the calendar records as failed/attention.
A reduction in remaining files, known chunks, or uncounted files resets that streak.
Zero remaining files also resets it. No custom alarm delivery is required for this
nonzero calendar result to remain visible.

Index rc75 deferrals use the calendar's existing `defer_after` streak (default 3
consecutive nights), reported through the `index` job-alert key. Rc0, including a
checkpointed yield, resets the deferral streak and clears its notice. A deferred
pass does not advance the CLI backlog-stall streak: the calendar owns that cause,
so persistent writer contention produces one deferral escalation, not a second
backlog alarm. Real rc1 non-convergence still follows the failed-job report path.

An append resumes at the saved input offset only on the same device/inode, with at
least the saved byte length, a newline-terminated saved prefix, and a matching SHA-256
of that prefix. Truncation, replacement, or prefix rewrite replays from zero. Older
checkpoint entries without this identity/fingerprint proof replay conservatively.
Fingerprints are captured before embedding so a deadline after COMMIT cannot block
publication of an already proven complete batch.

Recursive MCP-output chunks are rejected before embedding, counted, and consumed at
their original input positions, including entirely rejected batches. The canonical
store's rejection contract remains unchanged.

`BRAINLAYER_INDEX_QUARANTINE_PASSES` (default **3**) controls quarantine after identical
ValueError/RuntimeError failures at the same input offset. Cap interrupts, SQLite busy
errors, and I/O errors are not quarantined. Appending to the same failing prefix does
not erase its streak. A quarantined file is skipped so later files can proceed, but it
remains in the backlog, is named by `INDEX_QUARANTINED`, and forces rc1 until resolved.
Rewriting/replacing the failing prefix retries it; `--force` clears checkpoint/failure
history and replays all inputs. Quarantine never moves or deletes source data.

Unreadable or malformed checkpoint/history JSON emits a warning and restarts from
safe replay/measurement. Replay uses the existing idempotent upsert path. If sidecar
publication itself fails, the pass still fails visibly.
