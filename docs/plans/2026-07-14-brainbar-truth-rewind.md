# BrainBar Truth and Rewind Archive Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Preserve truthful BrainBar ingest metrics for local wall-clock timestamps and make watcher rewind archival commit safely through APSW.

**Architecture:** Keep BrainBar's existing normalized timestamp and watcher-liveness paths; add focused coverage that exercises a local wall-clock agent-store fixture instead of duplicating the already-landed truth fix. Change the rewind archive batcher to own an explicit APSW transaction (`BEGIN IMMEDIATE` / `COMMIT`, with `ROLLBACK` on failure), matching the repository's write convention.

**Tech Stack:** Swift/XCTest/SQLite3 for BrainBar dashboard coverage; Python/pytest/APSW for watcher archival.

---

### Task 1: Lock the existing local-time ingest truth

**Files:**
- Modify: `brain-bar/Tests/BrainBarTests/BrainDatabaseWindowedBucketsTests.swift`

**Step 1: Add the local wall-clock fixture test**

Add a test that inserts an agent-source chunk with a naive local `created_at` string and asserts `pipelineWindowBuckets` counts it in the current agent-store bucket.

**Step 2: Run the focused test**

Run: `swift test --filter BrainDatabaseWindowedBucketsTests`

Expected: PASS because current `normalizedUnixEpochSQL` already interprets naive `T` timestamps as local wall time. This is regression coverage for already-landed source behavior, not a reason to manufacture another production change.

**Step 3: Record truth-source evidence**

Document in the PR that the ad-hoc `created_at >= datetime('now', '-10 minutes')` comparison is invalid for mixed timestamp formats, while BrainBar uses normalized epochs plus `watcher_liveness_events`. Document that the installed 1.4.2 app at commit `a2fd98d1` predates the source truth fix and needs a BrainBar restart after a later build is installed.

### Task 2: Reproduce and fix APSW rewind archival

**Files:**
- Modify: `tests/test_rewind_batch_archival.py`
- Modify: `src/brainlayer/cli/__init__.py`

**Step 1: Write the real APSW regression**

Add a test that uses `_RewindArchiveBatcher` with its default `VectorStore`, archives one rewound session, closes the store, and verifies the two realtime watcher rows are durable.

**Step 2: Run the test to verify RED**

Run: `PYTHONPATH=src /Users/etanheyman/Gits/brainlayer/.venv/bin/python -m pytest tests/test_rewind_batch_archival.py::test_rewind_archiver_commits_with_real_apsw_connection -vv`

Expected: FAIL with `AttributeError: 'apsw.Connection' object has no attribute 'commit'`.

**Step 3: Implement the minimal APSW transaction**

In `_RewindArchiveBatcher.flush`, execute `BEGIN IMMEDIATE` before the archive update, execute `COMMIT` after reading `changes()`, and attempt `ROLLBACK` when the transaction was started and any operation fails. Preserve telemetry outcome and pending-session retry semantics.

**Step 4: Run the regression to verify GREEN**

Run the same focused pytest command.

Expected: PASS.

**Step 5: Run the rewind archive module**

Run: `PYTHONPATH=src /Users/etanheyman/Gits/brainlayer/.venv/bin/python -m pytest tests/test_rewind_batch_archival.py -vv`

Expected: all tests PASS.

### Task 3: Verify and publish an open PR

**Files:**
- Modify: `/Users/etanheyman/Gits/orchestrator/docs.local/collab/driver-buddy-2026-07-12.md` after PR completion

**Step 1: Run scoped quality gates**

Run:

```bash
swift test --package-path brain-bar --filter BrainDatabaseWindowedBucketsTests
PYTHONPATH=src /Users/etanheyman/Gits/brainlayer/.venv/bin/python -m pytest tests/test_rewind_batch_archival.py -vv
/Users/etanheyman/Gits/brainlayer/.venv/bin/ruff check src/brainlayer/cli/__init__.py tests/test_rewind_batch_archival.py
git diff --check
```

Expected: all commands exit 0.

**Step 2: Run repository-wide gates**

Run: `ulimit -n 4096 && PYTHONPATH=src /Users/etanheyman/Gits/brainlayer/.venv/bin/python -m pytest`

Run: `swift test --package-path brain-bar`

Expected: both suites complete with zero failures.

**Step 3: Review, commit, and push**

Run the bounded local CodeRabbit review. Fix any real issues, then create focused commits and push `fix/brainbar-truth-rewind`.

**Step 4: Open but never merge the PR**

Open a ready-for-review PR and request `@codex`, `@cursor`, and `@bugbot` review. The PR must remain OPEN. Do not run `gh pr merge`.

PR deploy note: restart `com.brainlayer.watch` for the Python rewind fix; rebuild/install and restart BrainBar for the dashboard truth code already present on current source.

Blocker: BrainLayer has no remote `develop` branch and no `feature/854-ux-batch-integration` branch. Resolve the requested base before PR creation; do not invent or publish a shared base branch.

**Step 5: Post the builder receipt**

Under the collab file's builders section, append exactly: `DONE: SLICE_<X> <card> PR: <url>` once the orchestrator supplies the slice/card mapping and the open PR URL exists.
