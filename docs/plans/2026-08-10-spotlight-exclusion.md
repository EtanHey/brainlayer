# Spotlight Exclusion Setup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make BrainLayer setup create Spotlight-excluded runtime trees and make doctor warn about legacy unexcluded database directories.

**Architecture:** A small path helper recognizes `.metadata_never_index` on a directory or ancestor. Setup creates marker-bearing high-churn roots before children, and doctor performs a warning-only Darwin check against the active DB directory.

**Tech Stack:** Python 3.11+, pathlib, Typer, pytest, macOS Spotlight command-line tools.

---

## Task 1: Path exclusion contract

**Files:**
- Modify: `src/brainlayer/paths.py`
- Modify: `tests/test_paths.py`

1. Add failing tests for marker-on-self, marker-on-ancestor, and unmarked paths.
2. Run `pytest tests/test_paths.py -q` and confirm failure because the helper is absent.
3. Add `SPOTLIGHT_EXCLUSION_MARKER` and `is_spotlight_excluded(path)`.
4. Re-run `pytest tests/test_paths.py -q` and confirm green.

## Task 2: Setup layout

**Files:**
- Modify: `src/brainlayer/setup.py`
- Modify: `src/brainlayer/cli/__init__.py`
- Modify: `tests/test_installable_build.py`

1. Add failing tests that inject four roots, assert markers precede child creation, assert the full
   high-churn child set, assert idempotence, and assert CLI ordering.
2. Run the focused tests and confirm the missing layout function/order causes RED.
3. Implement `ensure_spotlight_excluded_layout()` and call it first from `brainlayer setup`.
4. Re-run the focused tests and confirm GREEN.

## Task 3: Doctor warning

**Files:**
- Modify: `src/brainlayer/doctor.py`
- Modify: `tests/test_doctor.py`

1. Add failing doctor tests for marked and unmarked DB directories with the check explicitly enabled.
2. Run those tests and confirm RED because the warning does not exist.
3. Add a Darwin-defaulted config flag and warning-only issue with remediation details.
4. Re-run the focused tests and confirm GREEN.

## Task 4: Migration runbook and evidence

**Files:**
- Create: `docs/operations/spotlight-exclusion-migration.md`
- Modify after merge: `docs.local/tasks/2026-08-10-spotlight-exclusion-setup.md`

1. Write the checkpoint/stop/move/mark/restore/restart/verify/rollback runbook.
2. Include the untruncated path inventory and disposable macOS probe commands.
3. Run the commands only against disposable probe directories; never touch canonical runtime data.
4. Mark the runbook READY in the source task, reserving the live-migration marker for its window.

## Task 5: Verification and PR loop

**Files:** all changed files.

1. Run focused pytest, Ruff, and the full repository test command.
2. Run local CodeRabbit review with a bounded timeout.
3. Commit with the required agent trailer, push, and open a ready PR with a signed body.
4. Append the PR and exact head to the collab, then route `READY_FOR_REVIEW_SPOTLIGHT` to the lead.
5. Address review findings, re-run verification, confirm CI and exact head, and merge.
6. Append merged evidence plus `DONE_SPOTLIGHT_EXCL` to the source task and a one-line DONE pointer to the lead.
