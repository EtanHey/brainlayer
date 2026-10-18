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
incrementing or resetting an existing stall count. Incomplete passes carry both the
stall count and best baseline unchanged and emit no backlog-stall alarm, like deferrals.
Quarantined files remain in `files_remaining`, but their unknown totals do not count
in `uncounted_files`; quarantine itself forces `INDEX_FILE_QUARANTINED` and rc1,
with priority over a backlog-stall label.

After an initial non-deferred baseline, `BRAINLAYER_INDEX_STALL_PASSES` non-improving
fully measured, non-deferred passes (default **3**) produce `INDEX_BACKLOG_STALLED` and rc1, which
the calendar records as failed/attention. Genuine improvement resets the streak.
A dip followed by a decline still above the held minimum can fire the alarm.
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
of that prefix. Append-resume also requires the saved `pipeline_version` to equal
`INDEX_PIPELINE_VERSION` in `pipeline/chunk.py` (initially **1**); a missing or
mismatched stamp replays the changed file from zero. Unchanged signatures retain
their saved state, including done files, across both package and pipeline versions.
Package-only releases therefore do not re-embed the corpus. Bump this explicit
constant whenever parsing, classification, chunking, or admission changes positional
output. The small parse→classify→chunk→admit fixture in
`test_r2_pipeline_golden_admitted_output` hashes ordered admitted rows, including
positions, metadata and content; changed output requires a version bump and a
reviewed hash for that new version. Truncation, replacement, or prefix rewrite also
replays from zero. Older checkpoint entries without identity/fingerprint proof
cannot append-resume. Backlog inventory/baselines use the same constant: a pipeline
upgrade recounts instead of trusting old chunk totals.
Fingerprints are captured before embedding so a deadline after COMMIT cannot block
publication of an already proven complete batch.

Recursive MCP-output chunks are rejected before embedding, counted, and consumed at
their original input positions, including entirely rejected batches. The canonical
store's rejection contract remains unchanged.

`BRAINLAYER_INDEX_QUARANTINE_PASSES` (default **3**) controls quarantine after identical
ValueError/RuntimeError failures at the same input offset. Cap interrupts, SQLite busy
errors, and resource failures are not quarantined. Resource detection walks preserved
exception causes/contexts, so OSError (including ENOSPC), MemoryError, or a loaded
PyTorch OutOfMemoryError remains retryable even when an encoder and `embed_chunks`
wrap it in RuntimeError. These errors still fail the pass visibly. Appending to the same failing prefix does
not erase its streak. A quarantined file is skipped so later files can proceed, but it
remains in the backlog, is named by `INDEX_QUARANTINED`, and forces rc1 until resolved.
Failure entries carry the same explicit `INDEX_PIPELINE_VERSION`: a pipeline upgrade clears the
old failure streak and retries the unchanged file, starting at count 1 if it still
fails. Rewriting/replacing the failing prefix also retries it; `--force` clears
checkpoint/failure history and replays all inputs. For a targeted retry without a
corpus replay, with no index pass running, remove only the affected absolute-path
key from `failures` in the matching `index-backlog-<identity>.json` and its path from
the derived `quarantined` list. Keep the checkpoint and remaining history intact;
the next pass recomputes quarantine and resumes from the durable checkpoint.
Quarantine never moves or deletes source data.

Unreadable or malformed checkpoint/history JSON, including invalid signature scalar
types or negative byte lengths, emits a warning and restarts from safe replay/measurement.
Replay uses the existing idempotent upsert path. If sidecar
publication itself fails, the pass still fails visibly.
