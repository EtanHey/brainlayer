# D2 Project Derivation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Derive watcher chunk projects only from trustworthy workspace signals and safely repair the existing numeric-project corruption.

**Architecture:** Keep source-path parsing only for Claude's explicit `projects/` workspace encoding.  For every other provider, obtain the project from session-record workspace metadata, normalize the workspace basename, and return `None` when no such signal is present.  A standalone, resumable migration will update only bare one- or two-digit projects, export a narrow rollback TSV first, batch updates, and record its own audit/progress state.

**Tech Stack:** Python 3, SQLite/APSW, pytest.

---

### Task 1: Define derivation regressions

**Files:**
- Modify: `tests/test_watcher_bridge.py`
- Modify: `src/brainlayer/watcher_bridge.py`

**Step 1: Write failing tests**

Add tests for a Codex date-partitioned source with a metadata `cwd`, a Claude `projects/` source, Cursor and Gemini sources with a metadata `cwd`, and a source with no derivable workspace. Assert no provider falls back to a numeric directory.

**Step 2: Run to verify failure**

Run: `pytest tests/test_watcher_bridge.py::TestProjectExtraction -v`

Expected: FAIL because extraction has no metadata input and returns the date segment.

**Step 3: Implement minimal derivation**

Change `_extract_project_from_source` to accept an optional entry/metadata workspace signal, preserve the Claude `projects/` parser, and reject pure numeric candidates in all paths. Pass each watcher entry to the helper.

**Step 4: Run to verify pass**

Run: `pytest tests/test_watcher_bridge.py::TestProjectExtraction -v`

Expected: PASS.

### Task 2: Implement the narrow, idempotent backfill

**Files:**
- Create: `src/brainlayer/project_backfill.py`
- Create: `tests/test_project_backfill.py`

**Step 1: Write failing migration tests**

Create a fixture DB containing numeric projects with recoverable and unrecoverable source sessions plus a non-numeric project. Test TSV rollback export, real-project/NULL/untouched counts, and a second migration run producing zero changes with identical counts.

**Step 2: Run to verify failure**

Run: `pytest tests/test_project_backfill.py -v`

Expected: FAIL because the migration module does not exist.

**Step 3: Implement minimal migration**

Create a migration that reads session metadata from source JSONL files, updates only `project GLOB '[0-9]' OR project GLOB '[0-9][0-9]'`, uses 5–10K transactions, checkpoints every three batches, stores a migration audit record, and offers a CLI with an explicit live-DB opt-in.

**Step 4: Run to verify pass**

Run: `pytest tests/test_project_backfill.py -v`

Expected: PASS.

### Task 3: Verify and execute the production pass

**Files:**
- Create: `REPORT.md`
- Create: `docs.local/tasks/d2-rollback-<timestamp>.tsv` (untracked rollback artifact)

**Step 1:** Run the focused and full pytest suites.

**Step 2:** Stop enrichment writers and claim the shared bulk-DB lock in `docs.local/.../collab.md`.

**Step 3:** Re-count numeric projects; export the rollback TSV; run `PRAGMA wal_checkpoint(FULL)`; execute the migration in batches; checkpoint every three batches and at completion; release the lock.

**Step 4:** Independently query the three-way split, remaining numeric projects, and distinct projects before/after. Write `REPORT.md` with command evidence and commit only repository changes (not the live DB artifact).
