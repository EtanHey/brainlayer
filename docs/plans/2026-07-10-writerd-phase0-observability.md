# Writerd Phase 0 Observability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add fail-open local telemetry that identifies every current writer transaction, its active SQL fingerprint, queue delay, WAL/FTS context, and outcome without changing write semantics.

**Architecture:** A new `writer_telemetry` module owns JSONL rotation, SQL fingerprinting, APSW `trace_v2` statement aggregation, and a pid/DB-specific active-transaction marker. Existing writer paths create spans immediately before their current `BEGIN IMMEDIATE`, `SAVEPOINT`, or implicit init/autocommit operation and finish them only after the existing commit/rollback path. The CLI reads telemetry files only; no Phase 1 coordination or write-path behavior is introduced.

**Tech Stack:** Python 3.11+, APSW `trace_v2`, Typer, pytest, Ruff.

---

### Task 1: Core telemetry sink, fingerprint, span, and heartbeat

**Files:**
- Create: `src/brainlayer/writer_telemetry.py`
- Create: `tests/test_writer_telemetry.py`
- Modify: `tests/conftest.py`

**Step 1: Write failing sink/fingerprint tests**

Cover stable whitespace-normalized fingerprints, literal/binding safety, JSONL append, capped rotation, and fail-open behavior when directory creation or append fails.

**Step 2: Run the tests and verify RED**

Run: `pytest -q tests/test_writer_telemetry.py`

Expected: collection/import failure because `brainlayer.writer_telemetry` does not exist.

**Step 3: Implement the minimal sink and configuration**

Add environment-backed defaults for enabled state, `~/.local/share/brainlayer/logs/writer-telemetry.jsonl`, rotation bytes/backups, heartbeat path, and FTS sample TTL. All public instrumentation entrypoints catch telemetry errors and return no-op state; telemetry exceptions never propagate into a writer.

Because telemetry is enabled by default in production, add an autouse test fixture that redirects the log and heartbeat directory to `tmp_path`; no test may write to the user's live telemetry paths.

**Step 4: Write failing span/heartbeat tests**

Using only `tmp_path` SQLite databases, assert:

- the active marker exists before `BEGIN IMMEDIATE` executes;
- `trace_v2` records the current normalized SQL fingerprint and aggregates profile duration/status;
- commit and rollback clear the marker and emit an outcome;
- WAL bytes/frames, total row changes, executor PID, monotonic/wall starts, producer/lane/operation, and sampled FTS segment counts are present;
- disabled instrumentation installs no trace and writes no files.

**Step 5: Run the tests and verify RED**

Run: `pytest -q tests/test_writer_telemetry.py`

Expected: behavioral assertion failures for missing span/heartbeat APIs.

**Step 6: Implement the minimal span**

Use APSW `trace_v2(SQLITE_TRACE_STMT | SQLITE_TRACE_PROFILE, ..., id=...)` so existing exec traces remain untouched. Keep per-fingerprint aggregates in memory and emit one bounded transaction-end record. Write the active JSON atomically before yielding to the caller; update it for an open transaction through one process-level polling thread; clear it only after the caller records commit/rollback/completed/error.

**Step 7: Run tests and verify GREEN**

Run: `pytest -q tests/test_writer_telemetry.py`

Expected: all tests pass.

### Task 2: Read-only writer-telemetry CLI

**Files:**
- Modify: `src/brainlayer/cli/__init__.py`
- Create: `tests/test_cli_writer_telemetry.py`

**Step 1: Write failing CLI tests**

Use `typer.testing.CliRunner` and tmp JSONL fixtures to require:

- `brainlayer writer-telemetry tail --lines N --path ...` prints the newest N raw events;
- `brainlayer writer-telemetry summary --lines N --path ...` reports counts by producer/lane/outcome plus max/p95 duration;
- missing files return an empty, successful read-only result.

**Step 2: Verify RED**

Run: `pytest -q tests/test_cli_writer_telemetry.py`

Expected: command-not-found failures.

**Step 3: Implement and verify GREEN**

Add one Typer command with a `tail|summary` action and delegate parsing/summarization to pure functions in `writer_telemetry.py`.

Run: `pytest -q tests/test_cli_writer_telemetry.py tests/test_writer_telemetry.py`

Expected: all tests pass.

### Task 3: VectorStore init/upsert and MCP direct-store spans

**Files:**
- Modify: `src/brainlayer/vector_store.py`
- Modify: `src/brainlayer/store.py`
- Modify: `tests/test_vector_store_upsert_transactions.py`
- Create: `tests/test_writer_telemetry_store_paths.py`

**Step 1: Write failing path tests**

Assert init emits a `vector_store/init` operation span whose current statement changes during schema/FTS work; each PR #570 sub-batch emits its own `index/upsert_chunks` transaction with planned/applied row counts; direct `store_memory` and re-embed transactions emit `mcp/store_memory` spans; retries produce rollback then commit without changing existing DB results.

**Step 2: Verify RED**

Run: `pytest -q tests/test_writer_telemetry_store_paths.py tests/test_vector_store_upsert_transactions.py`

Expected: no writer telemetry events from these paths.

**Step 3: Instrument without owning transactions**

Start spans immediately before the existing transaction or implicit init work, leave all BEGIN/COMMIT/ROLLBACK/retry statements in their current order, and finish spans only after those existing statements. Mark init as `span_kind=writer_operation` and `transaction_mode=implicit_per_statement` so telemetry never falsely claims its entire duration held the SQLite writer lock.

**Step 4: Verify GREEN**

Run: `pytest -q tests/test_writer_telemetry_store_paths.py tests/test_vector_store_upsert_transactions.py`

Expected: all tests pass with the existing transaction-count assertions unchanged.

### Task 4: Drain queue wait and hotlane spans

**Files:**
- Modify: `src/brainlayer/drain.py`
- Modify: `scripts/hotlane_brainbar_daemon.py`
- Modify: `tests/test_drain_health.py`
- Modify: `tests/test_hotlane_brainbar_daemon.py`
- Create: `tests/test_writer_telemetry_drain.py`

**Step 1: Write failing drain/hotlane tests**

Require one span per burn/live drain attempt and hotlane vector transaction. Drain events must include producer `drain`, a lane/source derived from queued event metadata, and queue wait measured from the spool file mtime at apply time. Busy retries must close the failed attempt before opening the next.

**Step 2: Verify RED**

Run: `pytest -q tests/test_writer_telemetry_drain.py tests/test_hotlane_brainbar_daemon.py`

Expected: missing transaction events/queue fields.

**Step 3: Instrument existing seams and verify GREEN**

Wrap, but do not move, the existing BEGIN/COMMIT/ROLLBACK blocks. Compute queue metadata before BEGIN and pass it to the span; do not query SQLite solely for queue timing.

Run: `pytest -q tests/test_writer_telemetry_drain.py tests/test_drain_health.py tests/test_hotlane_brainbar_daemon.py`

Expected: all tests pass.

### Task 5: Watcher rewind and direct enrichment-apply spans

**Files:**
- Modify: `src/brainlayer/cli/__init__.py`
- Modify: `src/brainlayer/enrichment_controller.py`
- Modify: `tests/test_cli_direct_sqlite.py`
- Modify: `tests/test_enrichment_controller.py`
- Create: `tests/test_writer_telemetry_misc_paths.py`

**Step 1: Write failing tests**

Require rewind flush to emit `watcher/rewind_archive` with session/row counts and direct `_apply_enrichment` to emit `enrichment/apply` around the existing SAVEPOINT transaction. Error tests must preserve the original exception and record a rollback/error outcome.

**Step 2: Verify RED**

Run: `pytest -q tests/test_writer_telemetry_misc_paths.py tests/test_cli_direct_sqlite.py tests/test_enrichment_controller.py`

Expected: telemetry assertions fail while existing behavior tests remain green.

**Step 3: Instrument and verify GREEN**

Start/finish observation around existing code only. Do not enable direct enrichment, change queue defaults, alter commit boundaries, or add retries.

Run: `pytest -q tests/test_writer_telemetry_misc_paths.py tests/test_cli_direct_sqlite.py tests/test_enrichment_controller.py`

Expected: all tests pass.

### Task 6: Deliberately slow FTS gate, full verification, and PR loop

**Files:**
- Create: `tests/test_writer_telemetry_gate.py`
- Modify: `docs/plans/2026-07-10-writerd-phase0-observability.md` only if implementation receipts require clarification

**Step 1: Write the failing gate test**

Create a tmp SQLite database with a synthetic FTS5 corpus. Run an intentionally slowed FTS write statement on a worker thread/connection. While it is open, assert the independent marker contains the transaction start and that statement's fingerprint; after completion, assert JSONL contains its start, duration, profile counters, and committed outcome.

**Step 2: Verify RED, implement only missing telemetry behavior, then verify GREEN**

Run: `pytest -q tests/test_writer_telemetry_gate.py`

Expected RED: the gate exposes any missing live marker/fingerprint/duration field. Expected GREEN: the gate passes without touching production or `tests/test_vector_store.py` / `tests/test_engine.py`.

**Step 3: Run required verification**

Run:

```bash
pytest -q \
  tests/test_writer_telemetry.py \
  tests/test_cli_writer_telemetry.py \
  tests/test_writer_telemetry_store_paths.py \
  tests/test_writer_telemetry_drain.py \
  tests/test_writer_telemetry_misc_paths.py \
  tests/test_writer_telemetry_gate.py \
  tests/test_vector_store_upsert_transactions.py \
  tests/test_drain_health.py \
  tests/test_hotlane_brainbar_daemon.py \
  tests/test_enrichment_controller.py \
  tests/test_cli_direct_sqlite.py
ruff check src/ tests/
ruff format src/ tests/
git diff --check
```

Do not run `tests/test_vector_store.py` or `tests/test_engine.py`.

**Step 4: Complete the authorized PR loop**

Run bounded local CodeRabbit, commit intentional files, push with `BRAINLAYER_PREPUSH_SCOPE=changed-only`, open a ready PR, post the PR immediately to the buddy channel, invoke CodeRabbit/Codex/Cursor/Bugbot plus the required Claude review, address findings through clean re-review, merge only after the reviewer matrix and CI are clean, and verify the remote merge contains the latest pushed tree.

**Step 5: Write the handoff report**

Create `docs.local/handoffs/2026-07-10-brainlayerLead-w3-phase0-REPORT.md` containing branch, PR URL, changed files, slow-FTS gate receipt, verification counts, review/merge receipt, and exact final line `DONE_W3_PHASE0_OBSERVABILITY`.
