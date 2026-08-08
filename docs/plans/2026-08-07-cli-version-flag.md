# BrainLayer CLI Version Flag Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add fast, first-class `brainlayer --version` and `brainlayer -V` probes that print the installed package version and exit successfully.

**Architecture:** Register an eager global Typer option on the existing root app. Its callback reads only `brainlayer.__version__`, prints it, and exits before command execution, so the probe does not open a database or load a model.

**Tech Stack:** Python 3.11+, Typer, pytest

---

### Task 1: Specify the CLI contract

**Files:**
- Create: `tests/test_cli_version.py`

**Step 1: Write the failing test**

Add a parametrized `CliRunner` test for `--version` and `-V`. Assert exit code 0 and exact `__version__` output. Patch `brainlayer.cli.sqlite3.connect` to fail if the version path attempts to open SQLite.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_version.py -v`

Expected: both cases fail because the root app does not recognize either option.

### Task 2: Add the eager version option

**Files:**
- Modify: `src/brainlayer/cli/__init__.py`
- Test: `tests/test_cli_version.py`

**Step 1: Write the minimal implementation**

Import `__version__` from the package, add an eager option callback that prints it and raises `typer.Exit`, then register `--version` and `-V` on the root Typer callback.

**Step 2: Run focused tests to verify green**

Run: `pytest tests/test_cli_version.py tests/test_release_version_sync.py -v`

Expected: all version behavior and metadata consistency tests pass.

**Step 3: Run static and live CLI verification**

Run: `ruff check src/brainlayer/cli/__init__.py tests/test_cli_version.py`

Run: `brainlayer --version && brainlayer -V`

Expected: lint exits 0; both live commands print the package version and exit 0.

### Task 3: Complete the PR loop

**Files:**
- Modify only files listed above and this plan.

**Step 1:** Run the repository test suite required by `AGENTS.md` and inspect the complete output.

**Step 2:** Run bounded CodeRabbit pre-commit review, address actionable findings, and commit the scoped files.

**Step 3:** Push with `BRAINLAYER_PREPUSH_SCOPE=changed-only`, open a ready PR that closes #619, and request `@codex`, `@cursor`, and `@bugbot` review.

**Step 4:** Read CI and review feedback, address every material finding, and request re-review after fixes.

**Step 5:** Merge only after CI is green and at least one real review is present, verify the remote merge state and merged content, update tracking, and store the verified finding in BrainLayer.
