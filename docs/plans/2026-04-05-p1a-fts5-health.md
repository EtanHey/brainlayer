# FTS5 Health Monitoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 3-tier FTS5 health monitoring with append-only health events, hot-path count validation, periodic WAL checks, daily integrity checks, and emergency self-healing rebuilds.

**Architecture:** Extend `VectorStore` with health-oriented schema, cached count validation, WAL and integrity inspection methods, and a rebuild path that verifies post-conditions. Reuse those methods from MCP stats so operator-visible health stays on the same code path as automated monitoring.

**Tech Stack:** Python, APSW, SQLite FTS5, pytest, ruff

---

### Task 1: Red Tests For FTS Health API

**Files:**
- Create: `tests/test_fts5_health.py`
- Modify: `src/brainlayer/vector_store.py`

**Step 1: Write the failing tests**

Add isolated temp-DB tests for:
- `health_events` table creation
- synced counts
- warning/critical/emergency thresholds
- cache behavior
- WAL health shape
- deep integrity success
- rebuild success
- event logging

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_fts5_health.py -q`
Expected: FAIL with missing methods / missing table / missing health behavior

**Step 3: Write minimal implementation**

Add schema + methods on `VectorStore` to satisfy the first failing cases only.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_fts5_health.py -q`
Expected: PASS

### Task 2: Implement Tiered Health Logic

**Files:**
- Modify: `src/brainlayer/vector_store.py`
- Test: `tests/test_fts5_health.py`

**Step 1: Add cached count validation**

Implement `check_fts5_health(cache_ttl_seconds=60)` with:
- chunk/FTS counts
- percentage calculation
- severity mapping
- warning/critical/emergency event logging
- emergency rebuild + post-rebuild verification

**Step 2: Run targeted tests**

Run: `pytest tests/test_fts5_health.py -q`
Expected: threshold and cache tests move from FAIL to PASS

**Step 3: Add WAL and deep integrity checks**

Implement:
- `check_wal_health()`
- `deep_integrity_check()`
- `rebuild_fts5()`

Keep return payloads deterministic and append-only event logging explicit.

**Step 4: Run targeted tests**

Run: `pytest tests/test_fts5_health.py -q`
Expected: all health tests PASS

### Task 3: Expose Health In MCP Stats

**Files:**
- Modify: `src/brainlayer/mcp/search_handler.py`

**Step 1: Add lightweight health snapshot**

Enrich `_stats()` structured output with Tier 1 `fts5_health`.

**Step 2: Verify stats behavior**

Run an existing stats-focused test slice if needed, otherwise rely on targeted local sanity check.

### Task 4: Full Verification And Hygiene

**Files:**
- Modify: `src/brainlayer/vector_store.py`
- Modify: `src/brainlayer/mcp/search_handler.py`
- Create: `tests/test_fts5_health.py`

**Step 1: Run focused tests**

Run: `pytest tests/test_fts5_health.py -q`

**Step 2: Run broader safety checks**

Run:
- `pytest tests/test_3tool_aliases.py -q`
- `ruff check src/`
- `ruff format src/`

**Step 3: Run final regression confirmation**

Run the touched test files again.

### Task 5: Publish

**Files:**
- None

**Step 1: Review diff**

Run: `git diff -- src/brainlayer/vector_store.py src/brainlayer/mcp/search_handler.py tests/test_fts5_health.py docs/plans/2026-04-05-p1a-fts5-health.md`

**Step 2: Push branch**

Run: `git push -u origin feat/p1a-fts5-health`

**Step 3: Open PR**

Run: `gh pr create --title "feat: FTS5 3-tier health monitoring + self-healing (P1a)" --body "<summary>"`
