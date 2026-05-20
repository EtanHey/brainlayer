import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock


def _candidate(chunk_id: str) -> dict:
    return {
        "id": chunk_id,
        "content": f"content for {chunk_id}",
        "project": "brainlayer",
        "content_type": "assistant_text",
        "source": "claude_code",
    }


def test_enrich_batch_batches_arbitrated_enrichment_writes(monkeypatch, tmp_path):
    from brainlayer import enrichment_controller as controller
    from brainlayer import queue_io

    queue_dir = tmp_path / "queue"
    store = MagicMock()
    store.get_enrichment_candidates.return_value = [_candidate(f"c{i}") for i in range(5)]

    monkeypatch.setenv("BRAINLAYER_ARBITRATED", "1")
    monkeypatch.setenv("BRAINLAYER_MAX_COMMIT_BATCH", "2")
    monkeypatch.setattr(controller, "MAX_COMMIT_BATCH", 2, raising=False)
    monkeypatch.setattr(queue_io, "get_queue_dir", lambda: queue_dir)
    monkeypatch.setattr(controller, "_ensure_enrichment_columns", lambda store: None)
    monkeypatch.setattr(controller, "_is_duplicate_content", lambda store, content: False)
    monkeypatch.setattr(controller, "build_external_prompt", lambda chunk, sanitizer: ("prompt", SimpleNamespace()))
    monkeypatch.setattr(controller, "parse_enrichment", lambda text: {"summary": text, "tags": ["python"]})
    monkeypatch.setattr(controller, "Sanitizer", SimpleNamespace(from_env=lambda: SimpleNamespace()))
    monkeypatch.setattr(controller, "_get_gemini_client", lambda: SimpleNamespace())
    monkeypatch.setattr(controller, "_emit_enrichment_start", lambda *args, **kwargs: True)
    monkeypatch.setattr(controller, "_emit_enrichment_complete", lambda *args, **kwargs: True)
    monkeypatch.setattr(controller, "_emit_enrichment_error", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        controller,
        "_generate_content_with_rate_limit",
        lambda client, model, prompt, config, rate_limiter: SimpleNamespace(text="summary"),
    )

    result = controller.enrich_batch(store, limit=5)

    files = sorted(queue_dir.glob("enrichment-*.jsonl"))
    line_counts = [len(path.read_text(encoding="utf-8").splitlines()) for path in files]
    queued_chunk_ids = [
        json.loads(line)["chunk_id"]
        for path in files
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert result.enriched == 5
    assert sorted(line_counts, reverse=True) == [2, 2, 1]
    assert sorted(queued_chunk_ids) == ["c0", "c1", "c2", "c3", "c4"]


def test_enrichment_batcher_flushes_overdue_single_pending_item(monkeypatch):
    from brainlayer import enrichment_controller as controller

    flushed_batches = []
    ticks = iter([10.0, 10.2, 10.2])
    monkeypatch.setattr(controller.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(controller, "_enqueue_enrichment_write_batch", lambda items: flushed_batches.append(items))

    batcher = controller._EnrichmentWriteBatcher(max_batch=25, max_interval_seconds=0.1)

    batcher.enqueue(_candidate("c0"), {"summary": "s0", "tags": []})
    batcher.enqueue(_candidate("c1"), {"summary": "s1", "tags": []})

    assert [[chunk["id"] for chunk, _ in batch] for batch in flushed_batches] == [["c0"]]

    batcher.flush()

    assert [[chunk["id"] for chunk, _ in batch] for batch in flushed_batches] == [["c0"], ["c1"]]


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
