# Calendar index convergence

Calendar `index` passes retain durable input offsets in `index-checkpoint-<identity>.json`
beside the resolved database. `index-backlog-<identity>.json` holds the latest pass's
inventory, remaining files/chunks, stalled-pass count, best comparison backlog, and per-file failure counters.
Neither sidecar contains source text or exception messages.

`INDEX_CHECKPOINTED` includes `files_remaining`, `chunks_remaining`, and
`uncounted_files`. Inventory parsing and fingerprinting share the pass watchdog budget;
unchanged file counts are cached. If counting is interrupted, the chunk count is a
**known lower bound**, with the number of uncounted files printed explicitly.
`INDEX_BACKLOG` is also emitted on success, deferral, or failure after checkpoint setup.

A shrinking, fully measured backlog keeps the normal rc0 checkpointed yield.
Progress means `uncounted_files=0` and a smaller `(chunks_remaining, files_remaining)`
pair, ordered by chunks first and files only to break ties, than the best held
fully measured pair. Growth never moves that baseline upward: files 5,4,6,5,7
with equal chunk counts stall after the improvement to 4. A partial inventory's
lower chunk bound or falling uncounted count never establishes progress. The first
fully measured pass after a partial baseline establishes a comparable pair without
resetting an existing stall count.

After an initial non-deferred baseline, `BRAINLAYER_INDEX_STALL_PASSES` non-improving
non-deferred passes (default **3**) produce `INDEX_BACKLOG_STALLED` and rc1, which
the calendar records as failed/attention. Genuine improvement resets the streak.
Zero remaining files resets both the streak and comparison baseline; later new
work starts a new measurement cycle. Flat churn that consumes inputs but leaves
the same backlog still counts as stalled. No custom alarm delivery is required
for this nonzero calendar result to remain visible.

Index rc75 deferrals use the calendar's existing `defer_after` streak (default 3
consecutive nights), reported through the `index` job-alert key. Rc0, including a
checkpointed yield, resets the deferral streak and clears its notice. A deferred
pass carries the CLI stall count and its held non-deferred comparison baseline
forward unchanged, and emits no backlog-stall alarm. The next non-deferred pass
compares against that held baseline. Intermittent rc75 cannot erase non-convergence;
persistent writer contention still produces only the calendar deferral escalation.
Real rc1 non-convergence follows the failed-job report path.

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

Unreadable or malformed checkpoint/history JSON, including invalid signature scalar
types or negative byte lengths, emits a warning and restarts from safe replay/measurement.
Replay uses the existing idempotent upsert path. If sidecar
publication itself fails, the pass still fails visibly.
