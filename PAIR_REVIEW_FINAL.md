# D1 Oversized Ingestion — Fresh Acceptance Re-review (final)

**Verdict: `ACCEPT`**

All four acceptance blockers from the first review are resolved, and I confirmed each one by independent
probe rather than by reading the tests. The cap is a per-poll read budget, not a correctness boundary; the
offset registry never crosses unparsed, unquarantined, or unconfirmed indexable bytes; and the two failure
surfaces the first review created (unbounded buffering, permanent wedge on one torn line) are now bounded
and recoverable.

The change is also a **throughput improvement, not a regression** — 77× less CPU and 2,500× less resident
memory than baseline on the same 25 MB drain (measured below).

Residual risks are real but none of them reopen D1. The silent-drop classes that remain are all
byte-identical to `cc1d21f6` and I verified that at the baseline blob.

---

## What was reviewed

- Worktree: `/Users/etanheyman/Gits/worktrees/bl-d1-oversized-ingestion`, branch `fix/d1-oversized-ingestion`
- Baseline: `cc1d21f66534747ee4a50367c290858972629a70`
- Implementation is still **uncommitted**, so new-code citations carry blob hashes, not a commit SHA:
  - `src/brainlayer/watcher.py` → `git hash-object` = `16802db50a1ed5dc2c2a27a0f9939f59fa02e57f`
  - `tests/test_jsonl_watcher.py` → `git hash-object` = `b32f3051e6e82f1d9385ea4134b627507c5dff2c`
  - `git diff cc1d21f6 -- src/brainlayer/watcher.py tests/test_jsonl_watcher.py | shasum -a 256`
    = `54eb8660a4b4e121bb0a3b867c782acb71916cacc277c463d087c0895687297f`
  - diffstat: `+724 / -176` across 2 files
- **These are not the blobs the first review saw** (`f043f536` / `89088be2`). The implementation moved.
- Citations below: `watcher.py:N @ 16802db5` for new code, `watcher.py:N @ cc1d21f6` for baseline.
- Module resolution verified under pytest at runtime: `brainlayer.watcher` →
  `…/bl-d1-oversized-ingestion/src/brainlayer/watcher.py` (a bare `python3 -c "import brainlayer"` in this
  shell resolves to the *main* checkout — `pyproject.toml` `pythonpath = ["src"]` is what makes pytest correct).

---

## Commands run and exact counts

| command | result |
|---|---|
| the 13 tests named in the brief §"Run at minimum" | **13 passed** |
| `pytest tests/test_jsonl_watcher.py -q` | **106 passed** |
| watcher/health/alarm/bridge/watchdog/denylist/doctor/status group (14 files) | **337 passed** |
| `ruff check` on both changed files | **All checks passed!** |
| `ruff format --check` on both changed files | **2 files already formatted** |

The 14-file group = `test_jsonl_watcher test_alarm test_watch_backfill_cli test_watcher_bridge
test_watcher_provenance_ingest test_ingest_denylist test_ingest_guard test_throughput_watchdog
test_cli_index_watchdog test_drain_health test_doctor test_status_truthfulness test_fts5_health
test_stability_health_check`.

**The implementer's claims check out.** `106 passed` on `test_jsonl_watcher.py` is exact. `325` across the
named modules is within my 337 — the difference is which files I included for "health"; every one is green.

### Red-before-green — verified, not assumed

Copied the tree to a scratch dir, replaced **only** `src/brainlayer/watcher.py` with the `cc1d21f6` version,
ran the full new test file:

```
21 failed, 85 passed
```

Failures include all 13 required tests plus `test_corrupt_line_stops_before_unparsed_bytes`,
`test_read_new_lines_limits_bytes_per_call`, `test_poll_bounds_large_file_without_starving_healthy_file`,
`test_poll_ingests_append_after_large_file_is_fully_consumed`,
`test_file_processing_failure_raises_alarm_and_surfaces_in_health`.

Being precise: **3 of the 21** (`test_zero/negative/invalid_watch_read_window_falls_back_to_default`) fail at
baseline partly because they call the renamed `_watch_read_window_bytes` helper, so they are API-surface
artifacts as well as behaviour tests. The other **18 are genuine regression tests** — they fail on baseline
for behavioural reasons.

---

## The four blockers — each independently confirmed resolved

### Blocker 1 — bounded incomplete record, alarm + health, offset frozen, zero-window closed ✅

**Bound:** `watcher.py:51-52` introduces `BRAINLAYER_WATCH_MAX_RECORD_BYTES` (default 128 MB).
`watcher.py:689-690` refuses to read at all once the partial record already exceeds the ceiling;
`watcher.py:705-709` clamps each 64 KB chunk (`_WATCH_READ_CHUNK_BYTES`, `watcher.py:53`) to the remaining
record capacity, so the buffer stops at exactly `ceiling + 1`. `watcher.py:783-784` raises
`OversizedJSONLRecordError` (`watcher.py:630-631`) for the newline-free case; `watcher.py:760-762` for a
complete-but-oversized record.

Probe — `BRAINLAYER_WATCH_MAX_RECORD_BYTES=4096`, one 20,060-byte record with **no newline**, 1 KB window,
8 polls (this is the exact shape review #1 measured as unbounded):

```
buffer len per poll: [1024, 2048, 3072, 4096, 4097, 4097, 4097, 4097]   ← ceiling+1, then flat
file size: 20060
last_error: OversizedJSONLRecordError JSONL record exceeds 4096 bytes
tailer.offset: 0   registry: (0, 0)                    ← offset never moved
health alerting: True  reasons: ['file_ingestion_failure']  count: 1  ['OversizedJSONLRecordError']
```

Review #1's measurement on the old blob was `[1024, 2048, … 20507, 20507]` with `last_error: None` and an
empty failure map. That is now bounded and loud.

**Zero read-window closed:** `watcher.py:71-79` returns the default for any `parsed_value <= 0`.

```
env='0'  -> window=104857600      env='-1' -> window=104857600      env='abc' -> window=104857600
```

`poll_once` always passes this value (`watcher.py:1573`), so there is no reachable whole-file-read switch.
The test renamed from `test_zero_watch_max_file_bytes_disables_checkpointing` to
`test_zero_watch_read_window_falls_back_to_default` records the semantic change.

### Blocker 2 — byte-for-byte quarantine before crossing, health-visible, later records continue ✅

`_quarantine_failed_record` (`watcher.py:1189-1265`) writes to a `NamedTemporaryFile` in the quarantine dir,
`flush()` + `os.fsync()`, `Path.replace()`, then `fsync`s the **directory fd** (`watcher.py:1211-1227`) —
durable before anything moves. Only then does `watcher.py:1260` call `discard_failed_record`
(`watcher.py:788-798`) to cross the bytes. A pre-existing path with different bytes raises rather than
overwriting (`watcher.py:1228-1229`).

Probe — `good \n malformed \n good`:

```
flushed contents: ['first', 'second']              ← later records DO continue
registry offset: (188, …)   file size: 188          ← fully consumed
quarantine file: watcher-parse-s-66-6fda54ccc357face.jsonl.bad
byte-for-byte match: True   bytes: b'{"type":"user","message":{"role":"user","content":"tor\n'
health alerting: True  alert_reasons: ['quarantined_record']
quarantined_record_count_total: 1
quarantined_records: [{start_offset:66, end_offset:121, record_bytes:55, sha256:6fda54cc…, quarantine_path:…}]
alarm count: 1
```

Review #1 measured `flushed: ['first']` forever with the file wedged. That is gone.

**Write-failure clause — buffer and registry unchanged.** `watcher.py:1230-1235` catches `OSError`, unlinks
the temp file, and returns `False` **without touching buffer or offset**. Probe with `mkdir` raising
`ENOSPC`:

```
registry: (62, …)  (file size 133)                 ← frozen at the start of the bad record
tailer.offset: 62   buffer starts with broken record: True
buffer: b'{"broken\n{"type": "user", …"a"}}\n'      ← the trailing good record is retained too
last_error: OSError [Errno 28] No space left on device
alarms: 1 ['OSError']
-- after the disk recovers --
registry: (133, …)   flushed: ['a', 'a']   quarantine: watcher-parse-s-62-609c19b57e17849b.jsonl.bad
```

It retries and recovers on its own. Nothing lost, nothing crossed while the write was failing.

**Quarantined offset waits for prior confirmation.** `_advance_quarantined_offsets` (`watcher.py:1035-1054`)
holds `(start, end)` pairs in `_pending_quarantined_offsets` and refuses to advance while
`current_offset < start_offset` (`watcher.py:1043-1045`). Combined with `_checkpoint_discarded_progress`
(`watcher.py:1119-1138`), which refuses to cross while `indexer.has_buffered_source(filepath)` is true or the
registry is behind the highest indexable line end, the registry never crosses a quarantined record before the
bytes preceding it are durably confirmed.

### Blocker 3 — one de-duplicated alarm for a growing blocked record ✅

`watcher.py:1165-1172`: the fingerprint is now
`(error_type, error, confirmed_offset, read_offset, disposition, quarantine_path)`.
**`file_size_bytes` was removed from the fingerprint** — it is still in the emitted context
(`watcher.py:1160`), so the payload stays truthful, but it no longer defeats the
`previous == fingerprint` short-circuit at `watcher.py:1175-1176`.

Probe — blocked record, 6 polls, a 400-byte append after every poll:

```
polls=6 with an append each -> alarms: 1
fingerprint fields: [('OversizedJSONLRecordError', 0, 0)]     ← stable
file_size_bytes seen: [651]                                    ← context still carries it
registry: (0, 0)   file size: 3051   buffer len: 513
```

Review #1 measured 4 alarms for 4 polls. Now 1 for 6.

### Blocker 4 — health failure detail capped at 100, truthful total, overflow count ✅

`watcher.py:54` `_MAX_HEALTH_FAILURE_DETAILS = 100`; `watcher.py:1342-1347` slices the detail list and
computes the overflow; `watcher.py:1378-1380` emits all three fields. Quarantine details are separately
capped at 100 (`watcher.py:55`, `watcher.py:1249`) with a truthful `quarantined_record_count_total`
(`watcher.py:1381`).

Probe — 130 files each blocked on an over-ceiling record, one poll:

```
file_ingestion_failure_count (truthful total): 130
len(file_ingestion_failures) (capped detail):  100
file_ingestion_failures_overflow_count:         30
health.json size: 32,811 bytes                  ← not multi-MB
```

---

## The rest of the verification contract

**Files larger than the read window are eventually fully indexed; the cap is not a correctness boundary.**
97,890-byte file against a **2,048-byte** window, driven through the real `poll_once`:

```
indexed 1500 / 1500 records in 48 polls; final registry offset 97890 / 97890
contents complete & ordered: True
offset-invariant violations: NONE
```

**Confirmed offsets never cross unparsed, unquarantined, or unconfirmed indexable bytes.** In the run above
I asserted on **every poll** that (a) the registry offset lands exactly on a newline boundary in the raw file
and (b) `registry_offset <= tailer.offset`. Zero violations across 48 polls. Structurally: `tailer.offset`
advances only by `consumed_bytes` after a successful `json.loads` (`watcher.py:770-778`), and the reorder —
compute `line_end_offset`, then a single slice at the end — means a mid-loop break leaves buffer and offset
consistent.

**Per-line-budget reading no longer loads 100 MB to parse 100 small lines.** 100.7 MB file, 34,000 records of
2,961 bytes, production window (100 MB), `max_lines=100`, instrumented `read()`:

```
poll #1: parsed=100  disk_bytes_read=327,680 (5 read() calls)  resident_buffer=31,580 B  0.0007 s
full drain: polls=616  total_parsed=34,000  peak_buffer=34,536 B  elapsed=0.11 s  final_offset=100,674,000/100,674,000
```

Review #1 measured 0.181 s for poll #1 with a 91.6 MB resident buffer, and ~46 s of CPU to drain one window.
This is **~400× cheaper** and the buffer never exceeds 35 KB.

**No throughput regression — a large improvement.** Identical 25.2 MB / 20,000-record drain, baseline
`watcher.py` vs the change, same harness:

```
BASELINE  parsed=20000 polls=201 cpu=3.467s  0.01725 s/poll  peak_buffer=25.09 MB
CHANGE    parsed=20000 polls=386 cpu=0.045s  0.00012 s/poll  peak_buffer= 0.01 MB
```

77× less CPU, 2,500× less resident memory. (Poll count roughly doubles — see residual R2.)

**Normalization exception rolls back the tailer.** `watcher.py:1605-1611`: on any exception with
`read_accepted` still false, `tailer.offset, tailer._buffer = tailer_snapshot` restores the pre-read state and
the error is recorded through `_record_file_ingestion_failure`. `read_accepted` flips true only after
`indexer.add()` succeeds (`watcher.py:1589-1590`), so accepted data is never rolled back and unaccepted data
is never checkpointed past. Covered by `test_normalization_failure_retries_without_crossing_failed_record`
(fails on baseline).

**Rewind / inode replacement — all three restart safely from byte zero.** Driven through `poll_once` on files
larger than the window:

```
I) same-inode rewind (300 recs -> truncate+rewrite 200):
   rewind callbacks: 1 | re-indexed from zero: ['new0','new1'] count=200 | all 200 present: True | offset 13290/13290
J) confirmed inode replacement (300 -> os.replace with 250):
   re-indexed count=250 from ['b0'] | all present: True | offset 16140/16140
K) replacement while the OLD tailer buffer is still unconfirmed (8 unparsed bytes held):
   old-file leakage after replacement: NONE | new file fully indexed: True count=50 | offset 3190/3190
```

`_ensure_tailer` (`watcher.py:1433-1441`, `watcher.py:1450-1453`) and the drain-path guard
(`watcher.py:1545-1561`) both check the inode before draining and drop `_pending_quarantined_offsets` on
replacement, which is what makes case K clean.

**Denylist / provider filtering is separated from ingest-eligible loss.** Denylisted paths are removed from
tailers *and* the registry before any read (`watcher.py:1525-1538`); `provider_for_file` returns `"unknown"`
for them (`watcher.py:1057-1058`). Provider filtering happens later and separately in `_normalize_lines`
(`watcher.py:1103-1117`). The two are not conflated anywhere in the offset path.

**No production DB writes, no new concurrency.** Grepping the diff for `sqlite3|threading|Thread|Lock|
mode=ro|conn\.|execute\(|INSERT|UPDATE|DELETE` returns **zero** added or removed lines. The only DB access in
the blast radius is the pre-existing read-only `file:{db}?mode=ro` probe (`watcher.py:1283`). No new locks,
no new threads on the poll path. My own measurements read `offsets.json` and `os.path.getsize` only; every
probe used a temp registry and temp quarantine dir, and the test file sets
`BRAINLAYER_WATCHER_QUARANTINE_DIR` in all 4 places it needs to (no writes landed in
`~/.brainlayer/quarantine`).

**Blast radius re-measured independently** (read-only, against the live `offsets.json`):

```
registry entries with a live file on disk: 5,316
files > 100 MB: 17   total: 5,833,367,241 B (5.43 GiB)
of those, offset >= size (abandoned): 17 of 17 — every one
largest: rollout-2026-07-17T09-53-25-019f6ed9….jsonl  2,436,056,705 B, offset == size
```

Matches phase-1 `findings.md` and the first review exactly. **Those 17 entries still sit at `offset == size`,
so the fixed code reads zero bytes from them.** Correct per scope (Phase 2), and it must not be read as "the
data is back."

---

## The acceptance question: what ingest-eligible input can this still silently drop?

**Through the oversized path: nothing.** The path is deleted, not hardened. `_skip_oversized_file` and
`registry.set(filepath, file_stat.st_size, file_stat.st_ino)` (`watcher.py:1249 @ cc1d21f6`) are gone.

**The two loss classes review #1 blocked on are closed.** A newline-free region is bounded and alarms; a torn
record is quarantined byte-for-byte and the file keeps moving. Neither can grow without bound and neither
needs manual intervention.

**Still silently dropped — offset advances, no alarm, no health entry, no artifact.** All four are
byte-identical to `cc1d21f6`; I checked the baseline blob for each:

1. **JSON lines that parse to a non-dict.** `watcher.py:772` gates on `isinstance(parsed, dict)`; the offset
   has already advanced at `watcher.py:771`. Probed — a `[...]` line and a bare `42` line:
   `flushed: ['keep','keep']`, `registry offset 176/176`, `alarms: []`, `failures: 0`, `reasons: []`.
   **Pre-existing** — verified the identical `isinstance(parsed, dict)` gate with the same
   advance-then-check ordering in `read_buffered_lines @ cc1d21f6`.
2. **Non-`claude` provider entries the normalizer refuses.** `normalize_provider_entry` returns `None` for
   every codex `response_item` that is not a `message` (`watcher.py:163-165`) and for anything whose role is
   not user/assistant or whose text renders empty (`watcher.py:183-187`); `_checkpoint_discarded_progress`
   then advances over them. Probed — 3 codex records (reasoning, function_call, message) → 1 indexed, offset
   326/326, `alarms: []`, `reasons: []`. **Correct by design**, but still invisible: an operator cannot
   distinguish "filtered 2 of 3 on purpose" from "lost 2 of 3." A discarded-record counter in the health
   payload would make it auditable. Recommended, not required — the brief asks only that this be *separated*
   from ingest-eligible loss, and it is.
3. **In-place rewrite at or above the read cursor.** `check_rewind` (`watcher.py:655-676`) only fires on
   `file_size < offset + len(buffer)`. I diffed the function against `cc1d21f6`: **byte-identical**. Probed —
   3 records indexed, then the file rewritten in place with 20 different records (same inode, larger):
   `rewinds detected: 0`, `NEW0..NEW2 -> NONE (LOST)`, 17 of 20 indexed, `alarms: []`, `reasons: []`.
   This is the one remaining path where ingest-eligible bytes vanish with no signal at all. Session logs are
   append-only in practice, and the heuristic is unchanged by this diff, so it does not block — but it should
   be its own tracked defect.
4. **Records that repeatedly fail to flush.** After `BRAINLAYER_WATCHER_FLUSH_RETAIN_LIMIT` (default 3)
   attempts they are written to the quarantine dir and dropped from the buffer (`watcher.py:915-922`); a later
   confirmed watermark advances the registry past them. Durable on disk and `logger.critical` in the log, but
   **absent from the health payload** — unlike parse-quarantine, flush-quarantine has no counter.
   Pre-existing; now the odd one out, since the new parse-quarantine surface shows exactly how to fix it.

**Intentional and explicit:** denylisted paths (`brain-worker` subagents, `wf_*`) are removed from tailers and
registry at `watcher.py:1525-1538`.

**Not silent, and no longer unbounded:** an over-ceiling record defers that file with a de-duplicated alarm
and a standing health entry until an operator raises `BRAINLAYER_WATCH_MAX_RECORD_BYTES` or the record is
repaired. That is a deferral, not a drop — the bytes stay on disk and the offset stays put.

**Out of scope, must not be misread:** the 17 files / 5,833,367,241 B already at `offset == size` are not
recovered by this change.

---

## Residual risk register

| # | Area | Risk | Evidence |
|---|---|---|---|
| R1 | Memory | Bounded, but the ceiling is high and paid ~2×. `bytearray(self._buffer)` + `bytes(combined)` (`watcher.py:698`, `watcher.py:727`) double-copies each poll. Measured at an 8 MB ceiling: resident 8.4 MB, tracemalloc peak 17.3 MB = **2.1× ceiling**. At the shipped 128 MB default that is **~277 MB per blocked file**, and it is per-file, so N blocked files multiply it. Consider defaulting `BRAINLAYER_WATCH_MAX_RECORD_BYTES` well below 128 MB. | measured |
| R2 | Throughput (backlog wall-clock) | CPU per poll is now negligible, but `max_lines_per_file = 100` (`watcher.py:981`) × `poll_interval_s = 1.0` (`watcher.py:975`) caps drain at ~100 records/s/file — and the chunked reader alternates a full-100 poll with a small remainder poll (386 polls for 20,000 records vs baseline's 201; 34,000 records in 616 polls), so the effective rate is **~50 records/s/file**. The 2.44 GB backlog file would take on the order of a day of wall-clock to drain even after Phase 2 resets its offset. "Eventually fully indexed" is true; "quickly" is not. | measured |
| R3 | Health surface | `alerting` is `or bool(self._quarantined_record_count_total)` (`watcher.py:1355-1357`) and that counter never resets, so **one quarantined record makes the watcher permanently alerting until restart**. Loud by design, but a signal that cannot return to green erodes. | `watcher.py:1351-1358` |
| R4 | Health surface | `_quarantined_record_count_total` and `_quarantined_records` are in-memory only (`watcher.py:1011-1012`) — quarantine history vanishes on restart while the `.bad` files persist on disk. The health surface and the durable artifacts disagree after any restart. | `watcher.py:1011-1012` |
| R5 | Offsets | A retained flush-failure batch from an old inode, flushed **after** an inode replacement, can write a watermark beyond the new file's size, because `_advance_confirmed_offsets` (`watcher.py:1031`) only compares against the registry, not the file. Reproduced: `registry offset=1310` vs `new file size=195`. **No data was lost in any variant I could build** — the live tailer keeps its own correct offset and `check_rewind` self-heals on restart when the file is smaller. Loss would require flush-failure + replacement + a blocked record + growth past the stale offset + a restart in that window; I could not construct it end-to-end. Logged as a poisoned-watermark hazard, not a defect. | measured |
| R7 | Health surface | Watchdog `offset_lag` fires after 300 s for any backlog > 1 MB (`watcher.py:235-239`). Now that lag is truthful rather than masked by `offset == size`, a legitimately draining large file will hold `alerting: true` for the whole drain. Correct, but expect it. | `watcher.py:235-239` |
| R8 | Silent loss | Classes 1–4 in the section above (non-dict lines, provider filtering, in-place rewrite, flush-quarantine) have no counter on the health surface. All pre-existing and unchanged; item 3 is the only one that loses genuine ingest-eligible bytes. | probed |
| R9 | Concurrency | No new locks, threads, or shared state; `_file_ingestion_failures` and `_pending_quarantined_offsets` are poll-thread-only. Alarm de-dup (Blocker 3) removes the one-daemon-thread-per-second-per-blocked-file storm review #1 measured. | diff grep: zero `threading`/`sqlite3` lines |
| R10 | Prod DB | Untouched. Only the pre-existing read-only `mode=ro` probe. | `watcher.py:1283` |

---

## What is right, and worth saying plainly

- The cap governs **how much is read per cycle**, never **whether the file is read**. A 97 KB file with a
  2 KB window indexes 1500/1500 records with zero offset-invariant violations.
- Every remaining defer path is loud: `raise_alarm` through `alarm.py` **plus** a health-payload entry
  **plus**, for malformed records, a durable byte-for-byte artifact on disk. A log file is not a surface, and
  this is not a log file.
- The de-dup fix is the difference between a signal and a denial-of-service on the alarm channel: 6 polls with
  6 appends now emit 1 alarm instead of 6.
- The quarantine write is genuinely durable — temp file, `fsync`, atomic `replace`, **directory `fsync`** —
  and the failure path mutates nothing. That is the ordering the contract demanded and it is implemented
  correctly.
- The chunked reader is not just a bound, it is a 77× throughput win over baseline with a 2,500× smaller
  resident buffer.
- Offset lag became truthful. Under the old skip, `offset == size` meant `max_offset_lag_bytes == 0` for
  exactly the files being abandoned — the watchdog was blind by construction.

The two HIGHs that blocked the first review are closed with evidence, both MEDIUMs are closed, and the LOW-5
re-slicing cost is closed as a side effect of the chunked reader. LOW-6 (in-place rewrite) and INFO-7 remain,
both verified byte-identical to baseline and neither introduced here.

**Recommended follow-ups, none blocking:** lower the default record ceiling (R1); reset or window the
quarantine alerting flag (R3); persist quarantine counters (R4); add a discarded-record counter so intentional
provider filtering is auditable; track the in-place-rewrite blind spot (silent class 3) as its own defect.

DONE_D1_PAIR_REVIEW_FINAL
