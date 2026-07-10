# Relevance Score Controls and Weighted RRF Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Finish the two settled relevance wins by exposing score boosts as opt-in environment controls and making warm hybrid search use configurable two-leg weighted RRF.

**Architecture:** Keep the existing explicit rerank kwargs and add small call-time environment helpers so current callers remain compatible while process-wide opt-ins work without module reloads. Keep fusion in `search_repo.py`, but isolate alpha parsing and deterministic two-leg RRF math so vector and FTS ranks are the only fusion inputs; the MCP layer continues to expose honest FTS fallback metadata when embeddings are unavailable.

**Tech Stack:** Python 3.11+, pytest, APSW/FTS5, Ruff, GitHub CLI.

---

### Task 1: Add opt-in environment controls for importance and recency boosts

**Files:**
- Modify: `src/brainlayer/search_repo.py`
- Test: `tests/test_hybrid_search.py`

**Step 1: Write the failing tests**

Add focused tests proving that unset environment controls leave equal base scores, `BRAINLAYER_SCORE_IMPORTANCE_BOOST=1` enables only the importance multiplier, `BRAINLAYER_SCORE_RECENCY_DECAY=1` enables only recency decay, and explicit kwargs still enable their existing behavior.

**Step 2: Run tests to verify RED**

Run: `pytest -q tests/test_hybrid_search.py -k 'score_boost or default_does_not_apply_recency'`

Expected: the environment-controlled tests fail because `hybrid_search` does not yet read either setting.

**Step 3: Write the minimal implementation**

Add a strict boolean environment helper, combine each environment control with its existing explicit kwarg at the start of `hybrid_search`, and use the effective values for the cache key and boost gates.

**Step 4: Run tests to verify GREEN**

Run the focused tests, then `pytest -q tests/test_hybrid_search.py tests/test_search_quality.py tests/test_agent_profiles.py tests/test_consumer_scoping.py`.

**Step 5: Verify and publish PR 1**

Run the brief-required Ruff commands and permitted pytest suite, commit only PR-1 files, push with `BRAINLAYER_PREPUSH_SCOPE=changed-only`, open the specified ready-for-review PR, and request Codex/Cursor/Bugbot review.

### Task 2: Make warm hybrid fusion a configurable two-leg weighted RRF

**Files:**
- Modify: `src/brainlayer/search_repo.py`
- Modify: `src/brainlayer/mcp/search_handler.py` only if the existing fallback response lacks the required explicit degraded marker
- Test: `tests/test_hybrid_search.py`
- Test: `tests/test_search_handler.py`

**Step 1: Write the failing tests**

Add deterministic rank-fixture tests for alpha `0.25`, a valid `BRAINLAYER_RRF_ALPHA` override, invalid/non-finite fallback to `0.25`, no trigram contribution, and MCP FTS-only fallback with explicit degraded honesty.

**Step 2: Run tests to verify RED**

Run: `pytest -q tests/test_hybrid_search.py tests/test_search_handler.py -k 'weighted_rrf or trigram or fts_fallback'`

Expected: math/override/trigram tests fail because fusion is inline, alpha is constant, and trigram is still a scored leg; the degraded marker test fails if the current MCP payload exposes only `search_mode` and `fallback_reason`.

**Step 3: Write the minimal implementation**

Add a bounded finite alpha parser and a pure two-leg weighted-RRF helper. Fuse only semantic and FTS candidate ranks, retain trigram diagnostics only if needed outside fusion, and add `degraded: true` to fallback payloads without changing hybrid success payloads.

**Step 4: Run tests to verify GREEN**

Run the focused tests, then all permitted search/MCP tests and the full suite excluding `tests/test_vector_store.py` and `tests/test_engine.py`.

**Step 5: Verify and publish PR 2**

Branch from PR 1, run Ruff and permitted pytest verification, commit only PR-2 files, push with the scoped pre-push gate, open the specified stacked PR, request reviewers, and record both URLs/evidence in `docs.local/handoffs/BL-A-REPORT.md`.
