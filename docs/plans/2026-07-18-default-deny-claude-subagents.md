# Default Claude Subagent Denylist Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deny Claude subagent transcripts by default while preserving explicit environment overrides.

**Architecture:** Extend the existing immutable default pattern tuple rather than adding a second policy path. Verify behavior through the public `is_denylisted` API so the test covers home inference, glob expansion, and matching together.

**Tech Stack:** Python 3.11+, pytest, pathlib-based source policy.

---

### Task 1: Lock the default policy with a failing test

**Files:**
- Modify: `tests/test_ingest_denylist.py`

**Step 1: Change the ordinary-subagent policy test**

Rename `test_default_policy_allows_ordinary_claude_subagents` to
`test_default_policy_excludes_ordinary_claude_subagents` and assert that both
the `Explore` and `general-purpose` transcript paths are denylisted.

**Step 2: Run the focused test to verify RED**

Run: `pytest tests/test_ingest_denylist.py::test_default_policy_excludes_ordinary_claude_subagents -q`

Expected: FAIL because the old default allows both attributed workers.

### Task 2: Extend the default tuple

**Files:**
- Modify: `src/brainlayer/ingest_denylist.py`
- Test: `tests/test_ingest_denylist.py`

**Step 1: Add the exact default glob**

Add `~/.claude/projects/**/subagents/**` to `DEFAULT_INGEST_DENYLIST` without
changing override or attribution handling.

**Step 2: Verify GREEN**

Run: `pytest tests/test_ingest_denylist.py -q`

Expected: all denylist tests pass, including explicit-empty override coverage.

**Step 3: Run the repository gate**

Run the full pytest suite with the repository's normal exclusions and the
focused non-pytest checks used by the release gate.

**Step 4: Commit and publish**

Commit the code, tests, and design records with the required co-author trailer;
push `fix/default-deny-claude-subagents`; open a PR and request Codex, Cursor,
Bugbot, and CodeRabbit reviews.
