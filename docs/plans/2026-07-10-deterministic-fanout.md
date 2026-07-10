# Deterministic Scoped Fan-Out Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an opt-in, deterministic `brain_search` fan-out with bounded candidates, stable dedupe, scope provenance, and fail-loud degradation metadata.

**Architecture:** The MCP dispatcher retains all existing routing and consumer-scope checks, then hands generic opt-in searches to a small fan-out helper. The helper plans no more than four sequential `_search` calls, caps each at 10 candidates, merges by `chunk_id` with stable RRF, and formats the combined structured/text response.

**Tech Stack:** Python 3.11+, asyncio, MCP `TextContent`/`CallToolResult`, pytest, Ruff.

---

### Task 1: Specify the pure fan-out contract

**Files:**
- Create: `tests/test_search_fanout.py`
- Create: `src/brainlayer/search_fanout.py`

**Step 1: Write failing tests**

Add deterministic unit tests for scope planning, tag-surface matching, the four-call/40-candidate bounds, RRF ordering, dedupe provenance, and degraded propagation. Use fake structured responses only; do not create a DB.

**Step 2: Run tests to verify RED**

Run: `python3 -m pytest tests/test_search_fanout.py -q`

Expected: collection failure because `brainlayer.search_fanout` does not exist.

**Step 3: Implement the minimum pure helper**

Create constants for four scopes, 10 candidates per scope, and RRF `k=60`; a deterministic taxonomy-tag matcher; a scope planner; and an async executor that accepts a search callable and base keyword arguments.

**Step 4: Run tests to verify GREEN**

Run: `python3 -m pytest tests/test_search_fanout.py -q`

Expected: all new helper tests pass.

### Task 2: Integrate the opt-in MCP parameter

**Files:**
- Modify: `src/brainlayer/mcp/__init__.py`
- Modify: `src/brainlayer/mcp/search_handler.py`
- Modify: `src/brainlayer/mcp/_format.py`
- Test: `tests/test_search_fanout.py`

**Step 1: Write failing integration tests**

Assert `brain_search` exposes a boolean `fan_out` schema property, `call_tool` forwards it through `_brain_recall`, and `_brain_search` bypasses the warm helper and invokes fan-out only when opted in.

**Step 2: Run tests to verify RED**

Run: `python3 -m pytest tests/test_search_fanout.py -q`

Expected: failures for absent schema/forwarding/dispatch behavior.

**Step 3: Implement minimal routing**

Thread `fan_out=False` through `call_tool`, `_brain_recall`, `_brain_search`, and `_brain_search_dispatch`. Skip the warm-helper route for fan-out calls. At the generic search seam, invoke the helper with the resolved consumer scope and existing filters. Add scope labels to formatted results and a visible degraded warning.

**Step 4: Run focused tests to verify GREEN**

Run: `python3 -m pytest tests/test_search_fanout.py tests/test_search_handler.py tests/test_search_filter_params.py -q`

Expected: all focused tests pass.

### Task 3: Verify, publish, and report

**Files:**
- Create: `docs.local/handoffs/BL-B-REPORT.md`
- Append: `/Users/etanheyman/Gits/orchestrator/docs.local/collab/driver-buddy-2026-07-07.md`

**Step 1: Run required verification**

Run:

```bash
python3 -m pytest
ruff check src/ tests/
ruff format src/ tests/
git diff --check
```

Re-run tests if formatting changes Python files.

**Step 2: Run the local review gate**

Run `coderabbit review --agent` with a bounded timeout. Address critical findings or record an unavailable/rate-limited reviewer.

**Step 3: Commit and push**

Commit only the planned source, test, and tracked docs. Push with:

```bash
BRAINLAYER_PREPUSH_SCOPE=changed-only git push -u origin feat/deterministic-fanout
```

**Step 4: Open the ready-for-review PR and invoke reviewers**

Create a structured PR, then request `@codex review` and `@cursor @bugbot review` as required by the repository brief. Read all returned feedback and address actionable findings; the lead owns merge.

**Step 5: Write and verify the worker report**

Record branch, PR URL, integration rationale, exact test evidence, and final line `DONE_BLB_FANOUT` in `docs.local/handoffs/BL-B-REPORT.md`. Append one timestamped PR-open line to the collab file and store the milestone in BrainLayer.
