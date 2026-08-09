# Hotlane MPS Stall Recovery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ensure hotlane backlog embedding cannot wedge in Metal initialization and resumes durable vector progress.

**Architecture:** Add an explicit device override to the lazy shared embedding wrapper without changing its default behavior. Make only the hotlane request CPU, preserving deterministic progress for its small background batches while other consumers retain automatic device selection.

**Tech Stack:** Python 3.13, sentence-transformers, PyTorch, pytest, Ruff.

---

### Task 1: Pin hotlane CPU selection

**Files:**
- Modify: `tests/test_hotlane_brainbar_daemon.py`
- Modify: `scripts/hotlane_brainbar_daemon.py`

1. Add a test whose device-aware model factory records the requested device and run zero daemon cycles.
2. Run the focused test and confirm it fails because the factory receives no device.
3. Add a small factory helper that passes `device="cpu"` when supported and preserves zero-argument test factories.
4. Run the focused hotlane tests and confirm they pass.

### Task 2: Support explicit embedding devices

**Files:**
- Create: `tests/test_embedding_device.py`
- Modify: `src/brainlayer/embeddings.py`

1. Add tests that explicit CPU selection is forwarded without probing MPS and that the wrapper cache separates automatic and explicit-device instances.
2. Run the focused tests and confirm they fail on the missing API.
3. Add the optional device field, explicit-device resolution, and device-aware singleton key.
4. Run the focused embedding and hotlane tests and confirm they pass.

### Task 3: Verify and integrate

**Files:**
- Verify all changed files and relevant runtime configuration.

1. Run Ruff, focused tests, and the full pytest suite.
2. Run the local CodeRabbit review with a bounded timeout and address real findings.
3. Commit with the required agent trailer, push with changed-only pre-push scope, open the signed PR, and request Codex plus lead-routed Claude review.
4. Address review findings, rerun gates, merge, and verify the merge contains the final head.
5. Update the executing checkout, kickstart `com.brainlayer.hotlane-brainbar`, and poll both vector row-id counts once per minute until each has increased for 15 consecutive samples.
6. Append the exact receipts to the wave25 collab and store the verified root cause and merged fix in BrainLayer.
