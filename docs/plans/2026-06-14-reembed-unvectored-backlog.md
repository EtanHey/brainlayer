# Reembed Unvectored Backlog Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clear the cold semantic backlog by embedding every active chunk that has no vector, regardless of source.

**Architecture:** Add a reusable pending-vector backfill module that counts/selects active unvectored chunks, batch-encodes content with the local BGE model, and writes both float and binary vector rows through the existing vector upsert path. Update the daemon backfill to use the same all-source pending selection and batch embedding adapter.

**Tech Stack:** Python, APSW/sqlite-vec, Typer CLI, sentence-transformers `BAAI/bge-large-en-v1.5`, pytest.

---

### Task 1: Daemon Backfill Widening

**Files:**
- Modify: `src/brainlayer/store.py`
- Test: `tests/test_deferred_embedding.py`

**Steps:**
1. Write a failing test proving `embed_pending_chunks()` embeds unvectored `claude_code` and `realtime_watcher` chunks, not just `manual`/`mcp`.
2. Write a failing test proving archived/superseded/aggregated chunks are skipped.
3. Write a failing test proving a batch-capable embedder receives one list call rather than per-row calls.
4. Update `embed_pending_chunks()` to select all active sources and use batch encoding when the supplied embedder supports it.
5. Run the focused deferred embedding tests.

### Task 2: One-Time Backfill CLI

**Files:**
- Create: `src/brainlayer/reembed_backfill.py`
- Modify: `src/brainlayer/cli/__init__.py`
- Test: `tests/test_reembed_backfill.py`

**Steps:**
1. Write failing tests for active unvectored chunk counting, batch selection, idempotent skip behavior, and both vector table writes.
2. Implement count/select/write helpers around `VectorStore`.
3. Implement `run_reembed_backfill()` with dry-run/test-limit support, progress logging, throughput, and resumability by querying only missing vectors.
4. Add `brainlayer reembed-backfill` CLI command.
5. Run focused backfill tests and CLI help/import checks.

### Task 3: Live Backfill Run

**Files:**
- No code changes expected.

**Steps:**
1. Verify the heavy-ML mutex by checking for `llama-server`, `ollama`, `whisper`, `mlx`, and enrichment processes.
2. Count active unvectored chunks before the run.
3. Run the backfill with conservative batch size on MPS when available.
4. Monitor RAM/swap and lower batch size if memory pressure rises.
5. Count active unvectored chunks after the run and spot-check semantic retrieval.

### Task 4: Completion

**Files:**
- Commit all changes on `fix/reembed-unvectored-backlog`.

**Steps:**
1. Run relevant focused tests, then `pytest` if feasible.
2. Store the implementation decision and measured outcome in BrainLayer.
3. Post DONE/BLOCKED evidence to gen-16 if cmux is reachable.
4. Commit and push with `--no-verify` only; do not merge.
