# D1 oversized-ingestion report

## Status

Implemented and committed on `fix/d1-oversized-ingestion` as
`95f6d22ff8aaa2d1ce7efae56c585e518f5821ed` (`fix: stream oversized watcher ingestion`). No PR was opened and
the branch was not pushed. Production data was not modified and the already-abandoned bytes were not backfilled.

## Independent re-verification

At base `cc1d21f66534747ee4a50367c290858972629a70`, the watcher declared a 100 MiB cap at
`src/brainlayer/watcher.py:49`, applied it to pending bytes at `src/brainlayer/watcher.py:1219` and
`src/brainlayer/watcher.py:1237`, then wrote the full file size into the registry before logging a warning at
`src/brainlayer/watcher.py:1249-1263`. That is the defect: unread bytes became acknowledged bytes, while the only
surface was a log warning.

I independently re-read the live offset registry and statted its live paths. I also opened the canonical production
database only with SQLite URI `mode=ro`; no production write was issued. The measurement reproduced the stated blast
radius:

- 17 live oversized JSONL files.
- All 17 had `offset == size`; none had `offset < size` or `offset > size`.
- Total checkpointed size was 5,833,367,241 bytes (5.432747 GiB).
- The largest was 2,436,056,705 bytes.

This is evidence of prior abandonment, not a backfill result. Those 5.43 GiB remain a separate tracked recovery job.

The launchd scheduling policy was not changed: `launchd/com.brainlayer.watch.plist:44-52` at
`95f6d22ff8aaa2d1ce7efae56c585e518f5821ed` still uses `Nice=10` and `LowPriorityIO=true`.

## Design

This is not another skip-hardening patch. The size-based skip/checkpoint path was removed. The legacy
`BRAINLAYER_WATCH_MAX_FILE_BYTES` value is now a per-file, per-poll read window, while reads use 64 KiB chunks and
stop after the configured line or byte budget (`src/brainlayer/watcher.py:49-79`,
`src/brainlayer/watcher.py:682-738` at `95f6d22ff8aaa2d1ce7efae56c585e518f5821ed`). File size therefore controls
latency/fairness, never eligibility. Offsets advance only across complete parsed records
(`src/brainlayer/watcher.py:744-802` at the same commit).

A separate `BRAINLAYER_WATCH_MAX_RECORD_BYTES` ceiling (128 MiB by default) bounds one incomplete record in memory.
Crossing it freezes the offset and raises a health-surfaced watcher alarm; it does not skip the bytes
(`src/brainlayer/watcher.py:51-97`, `src/brainlayer/watcher.py:693-730`, and
`src/brainlayer/watcher.py:764-789` at `95f6d22ff8aaa2d1ce7efae56c585e518f5821ed`).

Malformed complete records are copied byte-for-byte into a collision-checked quarantine using file and directory
`fsync` before the tailer may advance over exactly that record. A quarantine write failure leaves the buffer and
offset untouched (`src/brainlayer/watcher.py:1223-1299` at the same commit). Failures raise through `raise_alarm`,
are deduplicated, and appear in bounded health details with overflow and quarantine counters
(`src/brainlayer/watcher.py:1180-1218`, `src/brainlayer/watcher.py:1376-1417`). Poll/normalization failures restore the
tailer snapshot so a later poll retries the same bytes (`src/brainlayer/watcher.py:1573-1649`).

Durable watermarks now carry source inode and rewind-generation provenance. A retained batch from an old inode or
pre-rewind generation cannot poison the current file's registry offset
(`src/brainlayer/watcher.py:1047-1067`, `src/brainlayer/watcher.py:1621-1627` at
`95f6d22ff8aaa2d1ce7efae56c585e518f5821ed`).

## What input can this still silently drop?

For D1's path: **no ingest-eligible record is silently dropped because its file or pending tail is large**. A large
file is incrementally drained. An over-ceiling single record is loudly deferred with its offset frozen. A read,
parse, quarantine, or normalization failure either retries unchanged bytes or creates an alarm plus health state.
A malformed complete record is not indexed, but is durably quarantined byte-for-byte and loudly surfaced, so it is
not a silent loss.

The following boundaries remain and must not be conflated with that guarantee:

- Denylisted paths and provider control/tool records are intentional policy exclusions.
- Valid JSON values that are not objects are invalid watcher schema and continue to be consumed without indexing.
- A pre-existing append-only assumption remains: an in-place rewrite that keeps the same inode and grows beyond the
  current cursor can hide rewritten earlier bytes from the tailer. That is a real residual silent-loss class, but it
  is not caused by file size and was not expanded into this D1 change.
- Repeated downstream flush failures use the existing durable flush quarantine and critical logging; those events
  are not yet represented in the new per-file ingestion-failure health fields.

The fresh reviewer explicitly probed non-object JSON and the stale-watermark case, and accepted the D1 change with
these boundaries documented. The final post-hardening verdict is `ACCEPT` in `PAIR_REVIEW_POST_R5.md:1-8` at
`95f6d22ff8aaa2d1ce7efae56c585e518f5821ed`.

## Test and review evidence

The implementation followed red-green TDD:

- Initial D1-focused tests: 8 failed before production changes, then 8 passed.
- Reviewer-hardening tests: 6 failed before their implementation, then 6 passed.
- Normalization rollback regression: 1 failed before the rollback, then passed.
- Old-inode stale-watermark regression: 1 failed with registry offset 2050 instead of replacement size 111, then
  passed after provenance filtering. The committed regression is at `tests/test_jsonl_watcher.py:1459-1508` at
  `95f6d22ff8aaa2d1ce7efae56c585e518f5821ed`.

Committed coverage includes full indexing beyond the read window and never checkpointing unparsed bytes
(`tests/test_jsonl_watcher.py:1281-1385`), inode replacement and rewind
(`tests/test_jsonl_watcher.py:1387-1535`), loud health-surfaced failures and retry rollback
(`tests/test_jsonl_watcher.py:1612-1689`), and durable malformed-record quarantine
(`tests/test_jsonl_watcher.py:1691-1798`), all at `95f6d22ff8aaa2d1ce7efae56c585e518f5821ed`.

Final local verification after the last code change:

- `python3 -m pytest tests/test_jsonl_watcher.py -q`: **107 passed, 1 warning**.
- 14-file watcher/health/bridge group: **338 passed, 1 warning**.
- `python3 -m ruff check src/brainlayer/watcher.py tests/test_jsonl_watcher.py`: passed.
- `python3 -m ruff format --check src/brainlayer/watcher.py tests/test_jsonl_watcher.py`: 2 files already formatted.
- `git diff --check`: passed.
- Full suite with `ulimit -n 4096; python3 -m pytest -q -p no:randomly`: **3709 passed, 60 skipped, 5 xfailed,
  2 failed** in 505.30 seconds.

The two full-suite failures were only
`tests/test_think_recall_integration.py::TestSessionsReal::test_sessions_returns_data` and
`tests/test_think_recall_integration.py::TestSessionsReal::test_sessions_golems_project`. Both query live production
session data from the last 90 days and received an empty result. Re-running exactly those tests in isolation produced
the same two failures in 0.11 seconds. The committed diff touches only watcher code, watcher tests, and review/report
artifacts; I am not representing the full suite as green.

Three fresh Claude passes were performed. The first requested hardening (`CHANGES_REQUESTED` in `PAIR_REVIEW.md:1-3`
at `95f6d22ff8aaa2d1ce7efae56c585e518f5821ed`). The second accepted after independently reproducing the required
behavior and measured about 77x less CPU and 2,500x less resident buffer than the old whole-window preload
(`PAIR_REVIEW_FINAL.md:1-12` at the same commit). Its stale-watermark observation was fixed even though classified
nonblocking. The third review independently reproduced the old failure, confirmed the new inode/rewind gate, ran 107
watcher tests and the 338-test relevant group, found no throughput regression, and returned `ACCEPT`
(`PAIR_REVIEW_POST_R5.md:1-8`, `PAIR_REVIEW_POST_R5.md:41-42`, and
`PAIR_REVIEW_POST_R5.md:293-307` at the same commit).

## Next

The separate backfill worker still owns recovery of the already-abandoned 5.43 GiB. Follow-up defects worth tracking
independently are same-inode in-place rewrite detection, flush-quarantine health visibility, and generation-aware
discarded-progress checkpointing for permanently retained stale batches. None should be folded into or used to delay
the D1 correctness fix.
