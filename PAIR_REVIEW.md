# D1 Oversized Ingestion — Fresh Pair Review

**Verdict: `CHANGES_REQUESTED`**

The core defect is genuinely fixed and the tests are real, not decorative. The cap no longer decides
*whether* a file is read, and the offset registry no longer crosses unparsed bytes. That is the eighth
commit this path needed and it is not another skip-hardening patch.

Two things block acceptance, both of which the review contract names explicitly:

1. the read window bounds per-poll **disk reads** but not **resident memory** — a file whose unread region
   contains no newline is accumulated whole-file in RAM, silently (contract: *"without introducing …
   unbounded whole-file reads"*);
2. a single malformed line now stops ingestion of that file permanently, with no bounded recovery — loud,
   but unbounded and unrecoverable (contract: *"eventually fully indexed"*).

Neither is a return to silent loss. Both are new failure surfaces created by the fix.

---

## What was reviewed

- Worktree: `/Users/etanheyman/Gits/worktrees/bl-d1-oversized-ingestion`, branch `fix/d1-oversized-ingestion`
- Baseline: `cc1d21f66534747ee4a50367c290858972629a70`
- Implementation is **uncommitted** in this worktree, so new-code citations cannot carry a commit SHA.
  Reviewed blobs, for reproducibility:
  - `src/brainlayer/watcher.py` → `git hash-object` = `f043f536412469dfacb8c0c7459aa2e761527749`
  - `tests/test_jsonl_watcher.py` → `git hash-object` = `89088be229105177a0656f124125897f45d1c6f9`
  - `git diff cc1d21f6 -- src/brainlayer/watcher.py tests/test_jsonl_watcher.py | shasum -a 256`
    = `d78120f7d1461f75386cedd07b89cb147ad5802c9c63bcac9c19c55e85026bc8`
- Baseline citations below are `watcher.py:N @ cc1d21f6`; new-code citations are `watcher.py:N @ blob f043f536`.

---

## Independent re-verification

**Baseline coordinates — confirmed, all four.** `git show cc1d21f6:src/brainlayer/watcher.py`:

| line | content |
|---|---|
| `watcher.py:49` | `_DEFAULT_WATCH_MAX_FILE_BYTES = 100 * 1024 * 1024` |
| `watcher.py:1219` | `if pending_bytes <= self.max_file_bytes:` |
| `watcher.py:1237` | `if pending_bytes <= self.max_file_bytes:` |
| `watcher.py:1249` | `self.registry.set(filepath, file_stat.st_size, file_stat.st_ino)` ← the defect |
| `watcher.py:1256-1263` | `logger.warning("Oversized JSONL checkpointed and skipped: …")` |

**Blast radius — re-measured independently** against `~/.local/share/brainlayer/offsets.json` and `os.path.getsize`
(read-only; no DB touched):

- registry entries with a live file on disk: **5,313**
- files > 100 MB: **17**, totalling **5,833,367,241 B (5.43 GiB)**
- of those, `offset >= size` (abandoned): **17 of 17**, **5,833,367,241 B** — every one
- largest: `rollout-2026-07-17T09-53-25-019f6ed9….jsonl`, 2,436,056,705 B, `offset == size`
- second: `rollout-2026-07-30T02-39-08-019fb03f….jsonl`, 391,880,632 B, `offset == size`

This matches phase-1 `findings.md` exactly. The plan's numbers are now also mine.

**Consequence the fix does not change:** those 17 registry entries still sit at `offset == size`, so the new
code reads **zero** bytes from them. The go-forward path is fixed; the 5.43 GiB is not recovered. That is
correct per scope (Phase 2), and it must not be read as "the data is back."

---

## Commands run and exact results

All from the worktree root. `pyproject.toml:118` sets `pythonpath = ["src"]`, and I confirmed at runtime that
`brainlayer.watcher` resolves to `…/bl-d1-oversized-ingestion/src/brainlayer/watcher.py`, not the main checkout
(which is what a bare `import brainlayer` picks up here).

| command | result |
|---|---|
| the 5 required tests (brief §"Run at minimum") | **5 passed** |
| `pytest tests/test_jsonl_watcher.py -q` | **97 passed** |
| watcher/health group (10 files, listed below) | **224 passed** |
| `ruff check` + `ruff format --check` on both changed files | **All checks passed / 2 files already formatted** |
| full suite, `ulimit -n 4096`, `pytest -q -p no:randomly` | *(see "Full suite" below)* |

Watcher/health group = `test_jsonl_watcher.py test_alarm.py test_watch_backfill_cli.py test_watcher_bridge.py
test_watcher_provenance_ingest.py test_ingest_denylist.py test_ingest_guard.py test_throughput_watchdog.py
test_cli_index_watchdog.py test_drain_health.py`.

### Red-before-green — verified, not assumed

I copied the tree to a scratch dir, replaced only `src/brainlayer/watcher.py` with the `cc1d21f6` version, and
ran the new tests against it:

```
FAILED test_poll_fully_indexes_file_larger_than_read_window
FAILED test_poll_never_checkpoints_past_unparsed_window
FAILED test_file_processing_failure_raises_alarm_and_surfaces_in_health
FAILED test_poll_indexes_large_inode_replacement_from_start
FAILED test_poll_indexes_large_same_inode_rewind_from_start
FAILED TestJSONLTailer::test_corrupt_line_stops_before_unparsed_bytes
6 failed
```

**6 failed at baseline, 6 pass on the change.** These are real regression tests.

---

## Findings

### HIGH-1 — The read window bounds disk reads, not memory: a record with no newline is buffered whole-file, silently

`watcher.py:659-670 @ blob f043f536`

```python
f.seek(self.offset + len(self._buffer))
new_data = f.read() if max_bytes is None or max_bytes <= 0 else f.read(max_bytes)
...
self._buffer += new_data
```

`max_bytes` caps one `read()`, but the seek is `offset + len(self._buffer)` and the result is **appended**.
When the window contains no `\n`, `has_complete_buffered_line()` (`watcher.py:672-674`) is False, so
`poll_once` (`watcher.py:1310`) takes the read path again next poll and appends another window. The buffer
grows by `max_bytes` per poll with no ceiling until a newline appears or EOF.

Measured, window = 1024 B, one 20,507-byte record with no trailing newline:

```
buffer len per poll: [1024, 2048, 3072, 4096, 5120, 6144] … [20507, 20507, 20507]
file size: 20507   final buffer: 20507
tailer.last_error: None      watcher._file_ingestion_failures: {}
```

Same input at `cc1d21f6`: `buffer = 0` (it skipped the file — the bug being fixed). So this exposure is
**new**. At the production default (`_DEFAULT_WATCH_MAX_FILE_BYTES = 100 MB`, `watcher.py:49`) the same shape
pulls an entire 2.44 GB file into the watcher's RSS, plus a transient second copy during `+=`.

It is also the quietest of the new paths: `last_error` stays `None` and `_file_ingestion_failures` stays empty,
so no `file_ingestion_failure` alarm and no per-file health entry. The only signal is the generic
`offset_lag` reason (`watcher.py:211-215`), which fires after 300 s for *any* backlog > 1 MB and cannot
distinguish "catching up" from "buffering a file whole into memory."

Aggregate is per-file, not global — measured with window 4096 and 5 backlogged files: 3,864 B resident in each
of 5 tailers simultaneously. At the production window that is `N_backlogged × 100 MB`.

Fix shape: cap `len(self._buffer)`; when a single record exceeds the window, fail it loudly through
`_record_file_ingestion_failure` (as an oversized-record failure) rather than growing without bound.

### HIGH-2 — One corrupt line stops the file forever, with no bounded recovery

`watcher.py:692-696 @ blob f043f536`

```python
except (json.JSONDecodeError, UnicodeDecodeError) as error:
    self.last_error = error
    break
```

Refusing to advance is correct — it is exactly the invariant the brief demands. But there is no path back:
the malformed bytes stay at the head of the buffer, `has_complete_buffered_line()` stays True, the drain path
is taken every poll (`watcher.py:1324-1326`), `read_buffered_lines` breaks immediately, and the file never
progresses again.

Measured — file = `good \n corrupt \n good`, 5 polls, then 4 appends with a poll after each:

```
flushed: ['first']                      ← the record after the corrupt line: never
registry: (37, …)   file size: 99       ← offset correctly frozen; invariant holds
alarms: [('watcher_file_ingestion_failed', 99)]
health: alerting=True  reasons=['file_ingestion_failure']  count=1
after 4 further appends → flushed still ['first']
```

At `cc1d21f6` the tailer skipped the one bad line and kept going (`test_corrupt_line_skipped`, deleted by this
diff). The change trades "lose one line quietly" for "lose the entire remainder of the session, loudly,
forever." For a live session file that is unbounded and needs manual intervention to clear.

This is not silent, so it does not reopen D1. It does defeat *"a file larger than the configured window is
eventually fully indexed"* for any file that ever contains one torn record.

Fix shape: the quarantine machinery already exists (`BRAINLAYER_WATCHER_QUARANTINE_DIR`,
`watcher.py:807-816`). Write the raw malformed bytes there, advance exactly past that one record, count it in
the health payload, and keep the file moving. That satisfies both invariants — nothing lost without a durable
record, and no wedge.

### MEDIUM-3 — Alarm de-dup keyed on file size ⇒ one alarm per poll for a blocked, growing file

`watcher.py:1035-1045 @ blob f043f536` — the fingerprint includes `file_size_bytes`, which changes on every
append, so the `previous == fingerprint` short-circuit never engages for a live file.

Measured: 4 polls with one append each on a corrupt-blocked file → **4 alarms**. At `poll_interval_s = 1.0`
(`watcher.py:878`) that is one alarm per second per blocked file. Each `raise_alarm` does
`logger.critical` + a `stderr` write + spawns a fresh daemon thread to POST to Axiom
(`alarm.py:66-99`). A root-wide failure across N files multiplies it by N.

Drop the volatile size from the fingerprint, or add a cool-down.

### MEDIUM-4 — Health payload has no cap on `file_ingestion_failures`

`watcher.py:1133-1136` and `watcher.py:1163-1164 @ blob f043f536`. One entry per failing live file, each
carrying the full `error` string, serialized into `watcher-health.json` on **every** poll
(`watcher.py:1167-1171`). A systemic failure — unreadable root, unmounted volume, fd exhaustion — produces one
entry per discovered file (5,313 registry entries exist today) and rewrites a multi-MB JSON once per second.
Cap the list and keep an overflow count.

### LOW-5 — Draining a window is O(lines × buffer): each consumed line re-slices the whole buffer

`watcher.py:699 @ blob f043f536` — `self._buffer = self._buffer[nl_idx + 1 :]` copies the remaining buffer for
every record.

Measured at the production window (91.8 MB file, 34,000 records of ~2.9 KB, `max_lines_per_file = 100`,
`watcher.py:884`):

```
poll #1 (read 100 MB + parse 100 lines): 0.181 s, buffer 91.6 MB
subsequent drain polls (100 lines each): 0.139 0.140 0.137 0.137 0.134 … s
avg 0.135 s/poll → 340 polls and ~46 s CPU to drain one window
```

So a backlogged file drains at roughly **270 KB/s** and costs ~13 % of a core while it does. That is a
throughput ceiling rather than a break, but the poll loop is sequential over all files
(`watcher.py:1301`), so several backlogged files can push a cycle past the 1 s interval and add latency to
live sessions. An index cursor or `memoryview` instead of re-slicing removes the term entirely.

### LOW-6 — Residual (pre-existing heuristic, wider blind spot now): in-place rewrite at ≥ the cursor is not a rewind

`check_rewind` (`watcher.py:624-645`) only fires on `file_size < offset + len(buffer)`. Measured — 20 records
rewritten in place, same inode, new size 750 B > confirmed offset 185 B:

```
rewinds detected: []
indexed: old-0 … old-19   (drained from the stale pre-rewrite buffer)
NEW-0 present? False      → all 20 rewritten records never ingested
```

The heuristic is unchanged from baseline, but the blind spot is materially wider now: for large files the
offset used to be pinned at `size` (so any shrink was caught), and now it legitimately lags far behind.
Session logs are append-only in practice, so I rate this low — but it is the one remaining path where
ingest-eligible bytes vanish with no signal at all.

### INFO-7 — Inode replacement re-ingests from 0 without firing `on_rewind` ⇒ duplicate chunks

`watcher.py:1215-1219` / `watcher.py:1227-1229 @ blob f043f536` call `registry.mark_rewind` but not
`_handle_rewind`, so the archival callback never runs. Measured: 5 records, atomic `os.replace` with a
6-record file → 11 records indexed, `on_rewind` fired 0 times.

**Verified identical at `cc1d21f6`** (same probe, same 11/0 result), so this is pre-existing, not a
regression. Logging it because the fix makes the re-read path actually reachable for large files.

### INFO-8 — `BRAINLAYER_WATCH_MAX_FILE_BYTES=0` still means "no window"

`watcher.py:660` — `max_bytes <= 0` falls through to a bare `f.read()`. Deliberate and covered by
`test_zero_watch_read_window_disables_bounding`, but it is an unbounded-read switch that now reads rather
than skips.

---

## What is right, and worth saying plainly

- **The cap is no longer a correctness boundary.** `_skip_oversized_file` and the
  `registry.set(filepath, file_stat.st_size, …)` at `watcher.py:1249 @ cc1d21f6` are gone. The window is a
  per-poll read bound (`watcher.py:1330-1333`), nothing more.
- **The offset advances strictly over parsed bytes.** `read_buffered_lines` moves `self.offset` only after
  `json.loads` succeeds (`watcher.py:698-700`), and the reorder — compute `line_end_offset`, then slice, then
  assign — means a mid-loop break leaves both buffer and offset consistent.
- **Discarded bytes are only crossed after everything indexable is durably confirmed.**
  `_checkpoint_discarded_progress` (`watcher.py:1004-1015`) refuses to advance while the indexer still holds
  buffered entries for that file or while the registry is behind the highest indexable line end.
- **Deferrals are health-visible, not log-only.** `file_ingestion_failure_count` /
  `file_ingestion_failures` in the payload, plus `alerting: true` and an appended `alert_reasons` entry
  (`watcher.py:1137-1144`) — and a `raise_alarm` through `alarm.py`. This is the part the previous seven
  commits never did.
- **`mark_rewind` on inode change (`watcher.py:1217`, `watcher.py:1228`) closes a real trap:** without it, a
  replacement smaller than the stale offset would be permanently suppressed by the
  `offset >= current_offset` guard in `_advance_confirmed_offsets` (`watcher.py:930`).
- **`current_inode != 0` guards (`watcher.py:1215`, `watcher.py:1227`)** stop a transient `stat` failure from
  being read as a replacement and forcing a spurious re-ingest from zero.
- **No production DB writes.** The only DB access in the diff's blast radius is
  `_db_realtime_inserts_since_window_start`, which opens `file:{db}?mode=ro` (`watcher.py:1074`). No new
  writes, no new locks, no new threads on the poll path (the alarm thread is per-alarm, see MEDIUM-3).
- **Offset lag became truthful.** Under the old skip, `offset == size` meant `max_offset_lag_bytes == 0` for
  exactly the files that were being abandoned — the watchdog was blind by construction. Now the lag is real
  and `offset_lag` can fire.

---

## The acceptance question: what ingest-eligible input can this still silently drop?

**Through the oversized path: nothing.** That path is deleted, not hardened. Verified by test and by probe.

**Still silently dropped — no alarm, no health entry, offset advances over it:**

1. **Non-`claude` provider entries the normalizer refuses.** `_normalize_lines` (`watcher.py:985-988`) falls
   back to `dict(line)` for `claude`, so nothing is lost there — but for `codex` and other roots,
   `normalize_provider_entry` returns `None` for every `response_item` that is not a `message`
   (`watcher.py:139-141`) and for anything whose role is not user/assistant or whose text renders empty
   (`watcher.py:158-163`). Reasoning traces, `function_call`, and `function_call_output` records are dropped and
   the offset is advanced over them by `_checkpoint_discarded_progress`. **This is the intentional-filtering
   boundary and it is correct by design** — but it is invisible in health, so an operator cannot distinguish
   "filtered 90 % of a codex rollout on purpose" from "lost 90 % of a codex rollout." A discarded-record
   counter in the health payload would make the distinction auditable; today only the deleted-by-design
   `logger` breadcrumbs existed and even those are gone.
2. **JSON lines that parse to a non-dict** (`watcher.py:701` — `if isinstance(parsed, dict)`): the offset
   advances, the record is dropped, nothing is recorded. Pre-existing.
3. **Records that repeatedly fail to flush.** After `BRAINLAYER_WATCHER_FLUSH_RETAIN_LIMIT` (default 3)
   attempts they are written to `~/.brainlayer/quarantine` and dropped from the buffer
   (`watcher.py:818-825`); a later confirmed watermark then advances the registry past them
   (`watcher.py:930`). Loud in the log and durable on disk, but **absent from the health payload** — the one
   loss class that has a file but no surface. Pre-existing.
4. **In-place rewrite at ≥ the read cursor** — LOW-6 above. Measured: 20 records, zero signal.
5. **Denylisted paths** — `brain-worker` subagents and `wf_*` (`ingest_denylist.py`); `poll_once` removes them
   from tailers and registry (`watcher.py:1302-1306`). Intentional, and explicit in the code.

**Not silent, but permanently deferred (loss grows without bound):**

6. Everything after the first malformed record in a file — HIGH-2.
7. Everything in a file whose unread region has no newline, while the whole file accumulates in RAM — HIGH-1,
   surfaced only by the generic `offset_lag`.

**Out of scope but must not be misread:** the 17 files / 5,833,367,241 B already at `offset == size` are not
recovered by this change. The fix stops the bleeding; it does not restore the blood.

---

## Residual risk register

| Area | Risk | Evidence |
|---|---|---|
| Memory | `N_backlogged × 100 MB` resident; unbounded for a newline-free region (whole file into RSS) | HIGH-1, measured |
| Throughput | ~270 KB/s per backlogged file; ~0.135 s CPU per poll against a 92 MB buffer; sequential poll loop can exceed the 1 s interval and delay live sessions | LOW-5, measured |
| Offsets | Invariant holds: the registry never crosses unparsed bytes, and never crosses indexable bytes before durable confirmation | `watcher.py:698-700`, `watcher.py:1004-1015`; test `test_poll_never_checkpoints_past_unparsed_window` |
| Concurrency | No new locks or shared state; `_file_ingestion_failures` is poll-thread-only. One daemon thread per alarm — MEDIUM-3 makes that one per second per blocked file | `alarm.py:88-96`, measured |
| Health surface | Correct and genuinely new, but uncapped (MEDIUM-4) and alarm-storming (MEDIUM-3); silent classes 1–4 above have no counter at all | `watcher.py:1133-1164` |
| Prod DB | Untouched. Read-only `mode=ro` probe only; my own measurements read `offsets.json` and `os.path.getsize` only | `watcher.py:1074` |
| Recovery | No path back from a wedged file; requires manual intervention | HIGH-2, measured |

---

## What would flip this to ACCEPT

1. Bound `JSONLTailer._buffer` and fail an over-window single record loudly instead of accumulating it (HIGH-1).
2. Quarantine-and-advance past exactly one malformed record so a torn line cannot wedge a session forever
   (HIGH-2).
3. Drop `file_size_bytes` from the alarm fingerprint or add a cool-down (MEDIUM-3).
4. Cap `file_ingestion_failures` in the health payload with an overflow count (MEDIUM-4).

MEDIUM-3 and MEDIUM-4 are small; the two HIGHs are the real work. A discarded-record counter for silent
class 1 would be the difference between "filtering is intentional" and "filtering is auditable" — recommended,
not required.

DONE_D1_PAIR_REVIEW
