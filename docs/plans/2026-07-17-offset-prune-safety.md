# Offset Prune Safety Review Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the exact-head #597 review findings without allowing unavailable transcript subtrees or stale registry writers to lose durable offsets.

**Architecture:** Keep the existing first-poll pruning and lock/merge structure. Add fail-closed parent-subtree evidence before deletion, validate all persisted tombstone inputs, and make the atomic registry replacement crash-durable before clearing dirty state.

**Tech Stack:** Python 3.11-3.13, pathlib/os/stat/fcntl, pytest, Ruff.

---

### Task 1: Require live evidence in the candidate's containing subtree

**Files:**
- Modify: `tests/test_jsonl_watcher.py`
- Modify: `src/brainlayer/watcher.py`

**Step 1: Write failing safety tests**

Add tests that construct:

- an available watch root containing a live JSONL plus an empty nested directory whose tracked JSONL is absent;
- a tracked JSONL path that is itself a broken symlink;
- an accessible root and live sibling where the tracked candidate raises `PermissionError` during its final stat.

Assert that each offset remains present and that `last_prune_complete` is false.

**Step 2: Run the focused tests to verify RED**

Run:

```bash
uv run pytest -q \
  tests/test_jsonl_watcher.py::TestOffsetRegistry::test_prune_missing_files_preserves_offsets_under_empty_nested_subtree \
  tests/test_jsonl_watcher.py::TestOffsetRegistry::test_prune_missing_files_preserves_broken_symlink_file \
  tests/test_jsonl_watcher.py::TestOffsetRegistry::test_prune_missing_files_skips_stat_errors
```

Expected: at least the empty-subtree and direct-symlink tests fail because the current code authorizes pruning from a live file elsewhere in the root and starts symlink inspection at `candidate.parent`.

**Step 3: Implement the minimal evidence checks**

Update symlink traversal to inspect `candidate` as well as its ancestors. Treat an unavailable direct symlink target as unsafe; allow an available direct file target while continuing to require directory targets for symlink ancestors.

Before the final missing-file check, require the candidate's immediate parent to be an accessible directory and require at least one current `live_file` beneath that parent. On missing evidence or any `OSError`, set `_last_prune_complete = False` and preserve the offset.

**Step 4: Run the focused tests to verify GREEN**

Run the command from Step 2.

Expected: all three tests pass.

**Step 5: Run related pruning tests**

Run: `uv run pytest -q tests/test_jsonl_watcher.py -k 'prune or unavailable or symlink'`

Expected: all selected tests pass.

### Task 2: Validate tombstones and exercise a dirty stale writer

**Files:**
- Modify: `tests/test_jsonl_watcher.py`
- Modify: `src/brainlayer/watcher.py`

**Step 1: Write failing tombstone tests**

Add a malformed persisted tombstone map containing string, boolean, non-finite, non-string-key, and valid numeric values. Mark the deleted path dirty in `test_prune_tombstone_blocks_stale_registry_from_resurrecting_deleted_offset` before the pruning writer flushes.

Assert malformed values are discarded, the valid numeric tombstone remains usable, flush succeeds, and the stale dirty deleted path is not restored.

**Step 2: Run the tests to verify RED**

Run:

```bash
uv run pytest -q \
  tests/test_jsonl_watcher.py::TestOffsetRegistry::test_malformed_tombstones_are_sanitized \
  tests/test_jsonl_watcher.py::TestOffsetRegistry::test_prune_tombstone_blocks_stale_registry_from_resurrecting_deleted_offset
```

Expected: malformed values can currently reach numeric comparison and the strengthened stale-writer assertion lacks the required dirty setup.

**Step 3: Implement one tombstone sanitizer**

Add a static helper that returns only string keys whose timestamps are finite `int` or `float` values, excluding booleans. Use it in `_load()` and when reading the on-disk tombstone map inside `flush()`.

**Step 4: Run the tests to verify GREEN**

Run the command from Step 2.

Expected: both tests pass.

### Task 3: Make registry replacement crash-durable

**Files:**
- Modify: `tests/test_jsonl_watcher.py`
- Modify: `src/brainlayer/watcher.py`

**Step 1: Write the failing durability test**

Patch `os.fsync` with a recording wrapper, perform one registry flush, and assert it is called for both the temporary regular file and the registry parent directory before `flush()` returns true.

**Step 2: Run the test to verify RED**

Run: `uv run pytest -q tests/test_jsonl_watcher.py::TestOffsetRegistry::test_flush_fsyncs_file_and_parent_directory`

Expected: fail because the current implementation renames without either `fsync`.

**Step 3: Implement durable replacement**

After `json.dump`, flush and `os.fsync` the temporary file descriptor. Use `os.replace` for atomic replacement, open the parent directory read-only, `os.fsync` it, and close it in `finally`. Leave `_dirty` set when any `OSError` occurs.

**Step 4: Run the test to verify GREEN**

Run the command from Step 2.

Expected: pass with one file sync and one directory sync.

### Task 4: Verify, commit, and publish the exact head

**Files:**
- Modify: `src/brainlayer/watcher.py`
- Modify: `tests/test_jsonl_watcher.py`

**Step 1: Run focused and static gates**

Run:

```bash
uv run pytest -q tests/test_jsonl_watcher.py
uv run ruff check src/brainlayer/watcher.py tests/test_jsonl_watcher.py
uv run ruff format --check src/brainlayer/watcher.py tests/test_jsonl_watcher.py
git diff --check
```

Expected: all commands exit 0.

**Step 2: Run the complete pre-push gate**

Run: `ulimit -n 4096; CI=true BRAINLAYER_PREPUSH=1 BRAINLAYER_PREPUSH_SCOPE=full ./scripts/run_tests.sh`

Expected: unit, MCP registration, isolated eval/hooks, Bun, and shell regression gates all pass.

**Step 3: Commit the fix**

Run:

```bash
git add src/brainlayer/watcher.py tests/test_jsonl_watcher.py
git commit -m "fix: require subtree evidence before pruning offsets" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

**Step 4: Push and request exact-head review**

Push `fix/prune-stale-offsets`, reply to every new inline thread with the fixing SHA, and request `@codex`, `@coderabbitai`, `@cursor`, and `@bugbot` reviews. Do not merge.

**Step 5: Post merge-ready only after remote verification**

Confirm exact push/head parity, zero unresolved threads, all GitHub checks green, and clean exact-head Codex/CodeRabbit results before appending the layerSpec merge-ready receipt.
