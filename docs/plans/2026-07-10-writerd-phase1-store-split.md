# Writerd Phase 1 Runtime/Migration Store Split Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make every indexed file use one cheap, schema-validated runtime writer connection while keeping all schema, FTS reconciliation, repair, and migration work in an explicit copy-only path.

**Architecture:** Keep `VectorStore` as the legacy migration-capable implementation for compatibility and rollback. Add three explicit entrypoints: `ReadonlyStore` for immutable readers, `WriterRuntimeStore` for an existing schema-compatible database with no open-time DDL or corpus scans, and `OfflineMigrator` for explicit schema/repair work on non-canonical copies. A runtime factory defaults to `WriterRuntimeStore` and honors `BRAINLAYER_RUNTIME_STORE=legacy`. Index opens the factory once around its file loop and passes the connection through the adapter. A deterministic schema contract fingerprint is computed from cheap `sqlite_schema` and `PRAGMA table_info` reads and is attached to Phase 0 `runtime_open` telemetry.

**Tech Stack:** Python 3.11+, APSW/sqlite-vec, Typer, pytest, Ruff, Phase 0 writer telemetry.

---

### Task 1: Prove the three-store contract and fail-closed fingerprint

**Files:**
- Create: `src/brainlayer/runtime_store.py`
- Create: `tests/test_runtime_store.py`

1. Write failing tests that bootstrap only `tmp_path` databases with legacy `VectorStore`, then require `ReadonlyStore`, `WriterRuntimeStore`, `OfflineMigrator`, and the schema fingerprint API.
2. Assert runtime open executes only an allowlist of connection/schema-probe statements: no CREATE/DROP/ALTER, INSERT/UPDATE/DELETE, FTS row traversal, COUNT, repair, reconciliation, or optimize.
3. Assert runtime open fails closed for a missing database and for a removed required trigger/column, releases the writer pidfile, and reports a stable expected/actual fingerprint diagnostic.
4. Assert `ReadonlyStore` opens `SQLITE_OPEN_READONLY` and never calls legacy initialization.
5. Implement the minimal shared instance-state setup, vec extension loading, capability flags, deterministic schema contract, and typed `SchemaFingerprintMismatch` error.
6. Run `pytest -q tests/test_runtime_store.py` until green.

### Task 2: Instrument runtime open and preserve rollback semantics

**Files:**
- Modify: `src/brainlayer/vector_store.py`
- Modify: `src/brainlayer/writer_telemetry.py` only if the existing metadata API is insufficient
- Modify: `tests/test_writer_telemetry_store_paths.py`
- Modify: `tests/test_runtime_store.py`

1. Add failing tests requiring one `runtime_open` writer-operation span with the schema fingerprint, completed outcome, duration, and zero corpus-scan/fullscan statements.
2. Ensure APSW's automatic best-practice connection hook cannot run `PRAGMA optimize` or persistent WAL mutation before runtime telemetry begins; retain explicit legacy initialization behavior.
3. Implement `open_writer_store()`: default new runtime path; exact rollback value `BRAINLAYER_RUNTIME_STORE=legacy`; reject unknown flag values fail closed.
4. Assert the legacy flag returns the old constructor and the default path never calls `_init_db_with_retry`.
5. Run focused runtime and telemetry tests until green.

### Task 3: Make migration and repair explicit and copy-only

**Files:**
- Modify: `src/brainlayer/cli/__init__.py`
- Modify: `src/brainlayer/cli_new.py`
- Modify: `src/brainlayer/mcp/_shared.py`
- Modify: `tests/test_cli_direct_sqlite.py`
- Modify: `tests/test_vector_store_readonly.py`
- Modify: `tests/test_runtime_store.py`

1. Write failing tests proving `OfflineMigrator` refuses the resolved canonical path unless the gated atomic-swap override is explicitly set.
2. Add an explicit copy-path schema migration command and route `repair-fts` through the same offline-only guard; never silently default either operation to canonical.
3. Replace readonly search pool/CLI constructors with `ReadonlyStore` and remove opportunistic search-time schema bootstrap. Missing/stale schemas must fail closed with an instruction to run the explicit copy migration/swap flow.
4. Route the cached MCP direct writer through `open_writer_store`; treat a schema fingerprint mismatch as a durable-queue reason so producers can still enqueue.
5. Preserve `VectorStore` only for legacy rollback and test/offline compatibility.
6. Run the focused CLI, readonly, MCP-store, and runtime tests until green.

### Task 4: Reuse one runtime writer connection for the complete index run

**Files:**
- Modify: `src/brainlayer/index_new.py`
- Modify: `src/brainlayer/cli/__init__.py`
- Modify: `tests/test_cli_index_watchdog.py`
- Modify: `tests/test_context_pipeline.py` only where adapter compatibility requires it

1. Write failing tests with two JSONL files that count runtime-store construction and assert exactly one open/close for the whole run.
2. Let `index_chunks_to_sqlite` accept an injected writer store while preserving standalone factory-open behavior.
3. Open one `open_writer_store()` around the CLI file loop and pass it to each adapter call.
4. Keep the watchdog's single run deadline, keyed APSW progress handler, rollback telemetry, committed-chunk accounting, and nonzero alarm exit unchanged under both new and legacy modes.
5. Run `pytest -q tests/test_cli_index_watchdog.py tests/test_vector_store_upsert_transactions.py tests/test_writer_telemetry_store_paths.py` until green.

### Task 5: Prove the production-size gate on a copy

**Files:**
- Create: `scripts/benchmark_runtime_store_open.py`
- Create: `tests/test_runtime_store_benchmark.py`

1. Add a safe benchmark CLI that refuses the canonical path, opens the supplied copy repeatedly, captures per-open duration and runtime-open telemetry, and reports p50/p95/p99/max plus scan/DDL/DML statement violations as JSON.
2. Unit-test canonical refusal and percentile/statement classification using `tmp_path` only.
3. Reuse an existing 17 GB copy if present; otherwise create a fresh `VACUUM INTO` snapshot from the canonical database using a read-only source connection and a distinct destination.
4. Run the benchmark only on that copy. Gate on p99 <100 ms and zero corpus-scan/DDL/DML statements. Measure the legacy constructor separately as the before receipt, with a bounded watchdog so a recurrence cannot wedge the worker.
5. Preserve the copy for reproducibility; never run the benchmark constructor against the live canonical database.

### Task 6: Verify, publish one PR, and hand off

**Files:**
- Create: `docs.local/handoffs/W3-PHASE1-REPORT.md`

1. Run focused tests, the handoff-required telemetry/watchdog/transaction suites, `ruff check src/ tests/`, `ruff format --check src/ tests/`, and `git diff --check`.
2. Review the complete diff and run bounded local CodeRabbit. Address only evidence-backed findings.
3. Commit intentional files, push with `BRAINLAYER_PREPUSH_SCOPE=changed-only`, and open one ready PR whose description includes incident `brainbar-c07ade22-ecc`, rollback instructions, copy-only safety, and gate measurements.
4. Post exactly one PR-open line to `~/Gits/orchestrator/docs.local/collab/driver-buddy-2026-07-07.md` under `## brainlayerLead-w3`.
5. Request Codex, Cursor, Bugbot, and CodeRabbit reviews, inspect CI/review feedback, and leave merging to the lead.
6. Write the report with branch, PR URL, changed files, commands/results, open p99 before/after, zero-scan telemetry proof, index single-open proof, rollback receipt, and exact final line `DONE_W3_PHASE1_STORE_SPLIT`.
7. Store the Phase 1 WHAT + WHY milestone in BrainLayer and verify it can be recalled.
