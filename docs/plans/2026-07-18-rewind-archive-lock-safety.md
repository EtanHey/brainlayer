# Rewind Archive Lock Safety Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Keep watcher ingestion running through SQLite lock contention while making rewind archival durable, retryable, and byte-offset bounded.

**Architecture:** The watcher must never open the BrainLayer database for rewind archival. It batches complete rewind intents and atomically writes them to the existing JSONL queue on a periodic watcher tick; the single-writer drain applies those intents under its existing rollback/preserve/retry contract. Watcher chunks persist their source line-end byte offset so archival only affects the reverted interval, while drain schema preparation happens before `BEGIN IMMEDIATE` to avoid holding the exclusive writer lock during schema/FTS initialization.

**Tech Stack:** Python 3.11-3.13, APSW/SQLite WAL, durable JSONL queue, pytest.

---

### Task 1: Make rewind batching queue-only and tick-driven

**Files:**
- Modify: `src/brainlayer/cli/__init__.py`
- Modify: `src/brainlayer/watcher.py`
- Modify: `src/brainlayer/queue_io.py`
- Test: `tests/test_rewind_batch_archival.py`
- Test: `tests/test_jsonl_watcher.py`

**Step 1: Write failing tests**

- Assert `_RewindArchiveBatcher` retains filepath, session id, old offset, and new offset.
- Assert `flush()` writes `rewind_archive` JSONL events without constructing `VectorStore` or touching a locked DB.
- Assert a failed queue write preserves pending intents for the next tick.
- Assert `JSONLWatcher.poll_once()` invokes `on_tick` even when no new rewind occurs.

**Step 2: Run tests to verify RED**

Run: `pytest -q tests/test_rewind_batch_archival.py tests/test_jsonl_watcher.py`

Expected: failures for missing queue-only batcher fields and missing tick callback.

**Step 3: Implement the minimum queue-only path**

- Add `enqueue_rewind_archive_batch(events, queue_dir=None)` using `enqueue_jsonl_batch`.
- Replace the batcher's `VectorStore` factory with an injected enqueue function.
- Clear pending intents only after the atomic queue write succeeds.
- Add `on_tick` to `JSONLWatcher` and call it after `indexer.tick()` with exception isolation.
- Keep shutdown flush best-effort so archival failure cannot replace the watcher's exit reason.

**Step 4: Run tests to verify GREEN**

Run: `pytest -q tests/test_rewind_batch_archival.py tests/test_jsonl_watcher.py`

Expected: pass.

### Task 2: Persist exact watcher source offsets

**Files:**
- Modify: `src/brainlayer/queue_io.py`
- Modify: `src/brainlayer/watcher_bridge.py`
- Modify: `src/brainlayer/vector_store.py`
- Modify: `src/brainlayer/drain.py`
- Test: `tests/test_watcher_bridge.py`
- Test: `tests/test_drain.py`

**Step 1: Write failing tests**

- Assert watcher queue events include `_line_end_offset` as `source_end_offset`.
- Assert new chunks persist the offset.
- Assert repeated same-file chunks keep the earliest offset and reactivate a row that reappears after rewind.
- Assert legacy rows with NULL offsets remain untouched.

**Step 2: Run focused tests to verify RED**

Run: `pytest -q tests/test_watcher_bridge.py tests/test_drain.py -k 'offset or watcher'`

Expected: failures for missing event/schema fields.

**Step 3: Implement offset persistence**

- Add nullable `source_end_offset INTEGER` to the chunks schema and drain's lightweight migration.
- Pass the entry line-end offset through watcher queue events.
- After watcher dedupe/merge, keep the minimum same-source offset and reactivate matching realtime-watcher rows.

**Step 4: Run focused tests to verify GREEN**

Run: `pytest -q tests/test_watcher_bridge.py tests/test_drain.py -k 'offset or watcher'`

Expected: pass.

### Task 3: Apply rewind archival in the drain with retry-not-wedge semantics

**Files:**
- Modify: `src/brainlayer/drain.py`
- Test: `tests/test_drain.py`
- Replace: `tests/test_rewind_batch_archival.py` DB-write assertions with drain assertions.

**Step 1: Write failing tests**

- Assert only rows for the exact source file with `new_offset < source_end_offset <= old_offset` are archived.
- Assert rows before the boundary, rows beyond the recorded old offset, unrelated sources, and legacy NULL offsets remain live.
- Hold an external `BEGIN IMMEDIATE`; assert drain returns without deleting the rewind queue file, then succeeds after lock release.

**Step 2: Run tests to verify RED**

Run: `pytest -q tests/test_drain.py tests/test_rewind_batch_archival.py`

Expected: missing `rewind_archive` dispatcher and bounds failures.

**Step 3: Implement drain dispatch**

- Add `rewind_archive` to event routing and realtime telemetry classification.
- Apply the bounded update inside the drain transaction.
- Set all available archival lifecycle fields consistently.
- Preserve the queue file on any BusyError through the existing retry path.

**Step 4: Run tests to verify GREEN**

Run: `pytest -q tests/test_drain.py tests/test_rewind_batch_archival.py`

Expected: pass.

### Task 4: Shorten exclusive-lock windows during drain initialization

**Files:**
- Modify: `src/brainlayer/drain.py`
- Test: `tests/test_drain.py`

**Step 1: Write a failing trace-order regression**

Record APSW statements and assert lightweight schema/dedupe preparation completes before `BEGIN IMMEDIATE`.

**Step 2: Run the regression to verify RED**

Run: `pytest -q tests/test_drain.py -k 'schema_before_begin or lock_window'`

Expected: current ordering places `ensure_dedupe_schema` after BEGIN.

**Step 3: Move schema work outside the transaction**

Run all idempotent schema checks and trigger/schema preparation before acquiring the exclusive writer transaction. Keep event mutation and commit atomic.

**Step 4: Run the regression to verify GREEN**

Run: `pytest -q tests/test_drain.py -k 'schema_before_begin or lock_window'`

Expected: pass.

### Task 5: Make watchdog recovery terminate the wedged watcher PID

**Files:**
- Modify: `scripts/launchd/throughput-watchdog.py`
- Test: `tests/test_throughput_watchdog.py`

**Step 1: Write failing recovery tests**

- Assert recovery resolves the currently loaded watcher PID before mutation.
- Assert it sends `SIGKILL` to that exact PID and waits for its disappearance before kickstarting the label.
- Assert a PID that changed between lookup and kill is not targeted.
- Assert timeout/failure remains bounded and produces an actionable recovery result.

**Step 2: Run tests to verify RED**

Run: `pytest -q tests/test_throughput_watchdog.py -k 'kill or recovery'`

Expected: current kickstart-only implementation does not issue the required PID-targeted SIGKILL.

**Step 3: Implement PID-safe SIGKILL escalation**

- Resolve and validate the watcher PID from launchd.
- Send `SIGKILL` to the exact stale PID before `launchctl kickstart -k`.
- Bound the disappearance wait and re-check identity to avoid killing a replacement process.
- Keep recovery cooldown/logging behavior intact.

**Step 4: Run tests to verify GREEN**

Run: `pytest -q tests/test_throughput_watchdog.py -k 'kill or recovery'`

Expected: pass.

### Task 6: Verify and publish the follow-up PR

**Files:**
- Verify all modified files.

**Step 1: Run focused suites**

Run: `pytest -q tests/test_rewind_batch_archival.py tests/test_jsonl_watcher.py tests/test_watcher_bridge.py tests/test_drain.py tests/test_writer_telemetry_misc_paths.py tests/test_throughput_watchdog.py`

Expected: pass.

**Step 2: Run static checks**

Run: `ruff check src/brainlayer/cli/__init__.py src/brainlayer/queue_io.py src/brainlayer/watcher.py src/brainlayer/watcher_bridge.py src/brainlayer/drain.py scripts/launchd/throughput-watchdog.py tests/test_rewind_batch_archival.py tests/test_jsonl_watcher.py tests/test_watcher_bridge.py tests/test_drain.py tests/test_throughput_watchdog.py`

Run: `git diff --check`

Expected: clean.

**Step 3: Run the full suite**

Run: `pytest -q`

Expected: pass, with any pre-existing failures explicitly compared against `origin/main`.

**Step 4: Commit, push, and open the PR**

- Include `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` in every commit.
- Push `rescue/apsw-rider-wip`.
- Open the follow-up PR and request Codex, Cursor/Bugbot, and CodeRabbit review.
- Do not merge; layerSpec owns the merge gate.
