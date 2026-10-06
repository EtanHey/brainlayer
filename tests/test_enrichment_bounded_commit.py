import os
import subprocess
import sys
from unittest.mock import MagicMock

import pytest


def _candidate(chunk_id: str) -> dict:
    return {
        "id": chunk_id,
        "content": f"content for {chunk_id}",
        "project": "brainlayer",
        "content_type": "assistant_text",
        "source": "claude_code",
    }


def test_enrichment_batcher_flushes_overdue_single_pending_item(monkeypatch):
    from brainlayer import enrichment_controller as controller

    flushed_batches = []
    ticks = iter([10.0, 10.2, 10.2])
    monkeypatch.setattr(controller.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(controller, "_enqueue_enrichment_write_batch", lambda items: flushed_batches.append(items))

    batcher = controller._EnrichmentWriteBatcher(max_batch=25, max_interval_seconds=0.1)

    batcher.enqueue(_candidate("c0"), {"summary": "s0", "tags": []})
    batcher.enqueue(_candidate("c1"), {"summary": "s1", "tags": []})

    assert [[chunk["id"] for chunk, _, _ in batch] for batch in flushed_batches] == [["c0"]]

    batcher.flush()

    assert [[chunk["id"] for chunk, _, _ in batch] for batch in flushed_batches] == [["c0"], ["c1"]]


def test_enrichment_batcher_retains_pending_items_when_flush_fails(monkeypatch):
    from brainlayer import enrichment_controller as controller

    flushed_batches = []
    attempts = 0

    def flaky_enqueue(items):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("queue unavailable")
        flushed_batches.append(items)

    monkeypatch.setattr(controller, "_enqueue_enrichment_write_batch", flaky_enqueue)

    batcher = controller._EnrichmentWriteBatcher(max_batch=25, max_interval_seconds=10)
    batcher.enqueue(_candidate("c0"), {"summary": "s0", "tags": []})

    with pytest.raises(RuntimeError, match="queue unavailable"):
        batcher.flush()

    assert [chunk["id"] for chunk, _, _ in batcher._pending] == ["c0"]

    batcher.flush()

    assert batcher._pending == []
    assert [[chunk["id"] for chunk, _, _ in batch] for batch in flushed_batches] == [["c0"]]


def test_enrichment_batcher_retains_current_item_when_overdue_flush_fails(monkeypatch):
    from brainlayer import enrichment_controller as controller

    ticks = iter([10.0, 10.2])

    def fail_enqueue(items):
        raise RuntimeError("queue unavailable")

    monkeypatch.setattr(controller.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(controller, "_enqueue_enrichment_write_batch", fail_enqueue)

    batcher = controller._EnrichmentWriteBatcher(max_batch=25, max_interval_seconds=0.1)
    batcher.enqueue(_candidate("c0"), {"summary": "s0", "tags": []})

    batcher.enqueue(_candidate("c1"), {"summary": "s1", "tags": []})

    assert [chunk["id"] for chunk, _, _ in batcher._pending] == ["c0", "c1"]


def test_submit_write_yields_after_successful_write(monkeypatch):
    from brainlayer import enrichment_controller as controller

    sleeps = []

    class ImmediateQueue:
        def submit(self, name, callback):
            future = MagicMock()
            future.result.return_value = callback()
            return future

    monkeypatch.setattr(controller, "_get_store_write_queue", lambda store: ImmediateQueue())
    monkeypatch.setattr(controller, "_current_post_write_yield_seconds", lambda: 0.123)
    monkeypatch.setattr(controller, "_sleep", lambda seconds: sleeps.append(seconds))

    result = controller._submit_write(MagicMock(), "apply-enrichment:c0", lambda: "ok")

    assert result == "ok"
    assert sleeps == [0.123]


def test_invalid_commit_interval_env_does_not_crash_import():
    env = os.environ.copy()
    env["BRAINLAYER_MAX_COMMIT_INTERVAL_MS"] = "not-a-number"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from brainlayer import enrichment_controller as c; print(c.MAX_COMMIT_INTERVAL_SECONDS)",
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.25"


def test_invalid_commit_batch_env_does_not_crash_import():
    env = os.environ.copy()
    env["BRAINLAYER_MAX_COMMIT_BATCH"] = "not-a-number"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from brainlayer import enrichment_controller as c; print(c.MAX_COMMIT_BATCH)",
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "25"
