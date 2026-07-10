# Index Run Watchdog Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bound `brainlayer index` wall-clock runtime and exit loudly after releasing the current write transaction.

**Architecture:** The CLI computes one monotonic deadline from `BRAINLAYER_INDEX_MAX_RUNTIME_S` (default 1800 seconds) and checks it at file/entry boundaries while passing it through embedding batches to `VectorStore.upsert_chunks`. The upsert loop checks outside each transaction and installs an APSW progress handler during statements; deadline interruption rolls back the active sub-batch before a typed exception carries earlier committed progress back to the CLI. The CLI emits the existing alarm telemetry/stderr signal and exits non-zero.

**Tech Stack:** Python, Typer, APSW, pytest, Rich, existing BrainLayer alarm telemetry.

---

### Task 1: Prove transaction-boundary deadline behavior

**Files:**
- Modify: `tests/test_vector_store_upsert_transactions.py`
- Modify: `src/brainlayer/vector_store.py`

1. Add a test with `BRAINLAYER_INDEX_TXN_BATCH=2` and a controlled monotonic clock.
2. Call `upsert_chunks(..., deadline_monotonic=...)` with four unique chunks.
3. Assert a typed deadline exception after the first committed sub-batch, including `processed_count == 2`.
4. Assert exactly two rows remain committed in the temporary database.
5. Run the test and confirm it fails because deadline support does not exist.
6. Add the optional deadline argument, processed-count tracking, and outside-transaction checks.
7. Re-run the focused transaction tests.

### Task 2: Prove CLI deadline signaling and fast-run behavior

**Files:**
- Create: `tests/test_cli_index_watchdog.py`
- Modify: `src/brainlayer/index_new.py`
- Modify: `src/brainlayer/cli/__init__.py`

1. Add a Typer CLI test using only `tmp_path` sources and a fake indexing adapter.
2. Set `BRAINLAYER_INDEX_MAX_RUNTIME_S`, capture the propagated deadline, simulate the typed deadline exception, and assert non-zero exit plus alarm code/context.
3. Add a second test where indexing finishes under the deadline and assert exit zero with no alarm.
4. Run both tests and confirm they fail because the CLI does not create or propagate a deadline.
5. Add the 1800-second default, positive env parsing, deadline propagation, file-boundary checks, and alarm-to-`typer.Exit` handling.
6. Re-run the CLI and transaction watchdog tests.

### Task 3: Verify and publish worker PR

**Files:**
- Create: `docs.local/handoffs/2026-07-10-brainlayerLead-w2-REPORT.md`

1. Run focused watchdog tests.
2. Run `ruff check src/` and `ruff format src/`, then re-run focused tests.
3. Run the safe pytest suite excluding `tests/test_vector_store.py` and `tests/test_engine.py`.
4. Review the complete diff and status.
5. Commit on `feat/index-txn-watchdog`, push, and open a ready-for-review PR.
6. Request `codex`, `cursor`, and `bugbot` reviews; inspect CI and actionable feedback without merging (lead owns merge).
7. Write the report with exact evidence and final `DONE_W2_INDEX_WATCHDOG` marker.
8. Append the required one-line buddy-channel status and store the milestone in BrainLayer.
