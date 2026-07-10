from __future__ import annotations

import json
from pathlib import Path

from brainlayer.store import store_memory
from brainlayer.vector_store import VectorStore


def _configure(monkeypatch, tmp_path: Path) -> Path:
    log_path = tmp_path / "writer-telemetry.jsonl"
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY", "1")
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY_PATH", str(log_path))
    monkeypatch.setenv("BRAINLAYER_WRITER_HEARTBEAT_DIR", str(tmp_path / "heartbeats"))
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(tmp_path / "pidfiles"))
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY_FTS_SAMPLE_TTL_SECONDS", "0")
    return log_path


def _finished(log_path: Path, operation: str) -> list[dict]:
    if not log_path.exists():
        return []
    return [
        event
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if (event := json.loads(line)).get("event") == "txn_finished" and event.get("operation") == operation
    ]


def _chunk(chunk_id: str) -> dict:
    content = f"Unique telemetry store path content {chunk_id}"
    return {
        "id": chunk_id,
        "content": content,
        "metadata": {"test": "writer-telemetry"},
        "source_file": "telemetry.jsonl",
        "project": "writer-telemetry",
        "content_type": "note",
        "char_count": len(content),
        "source": "test",
        "created_at": "2026-07-10T00:00:00Z",
    }


def _embedding(seed: int) -> list[float]:
    return [float(seed)] * 1024


def test_vector_store_init_emits_implicit_operation_span(tmp_path, monkeypatch):
    log_path = _configure(monkeypatch, tmp_path)

    store = VectorStore(tmp_path / "brainlayer.db")
    store.close()

    events = _finished(log_path, "init")
    assert len(events) == 1
    event = events[0]
    assert event["producer"] == "vector_store"
    assert event["lane"] == "maintenance"
    assert event["span_kind"] == "writer_operation"
    assert event["transaction_mode"] == "implicit_per_statement"
    assert event["outcome"] == "completed"
    assert any("chunks_fts" in statement["normalized_sql"] for statement in event["statements"])


def test_upsert_chunks_emits_one_span_per_pr570_sub_batch(tmp_path, monkeypatch):
    log_path = _configure(monkeypatch, tmp_path)
    monkeypatch.setenv("BRAINLAYER_INDEX_TXN_BATCH", "2")
    store = VectorStore(tmp_path / "brainlayer.db")
    log_path.unlink()

    assert (
        store.upsert_chunks(
            [_chunk(f"chunk-{index}") for index in range(3)],
            [_embedding(index) for index in range(3)],
        )
        == 3
    )
    store.close()

    events = _finished(log_path, "upsert_chunks")
    assert len(events) == 2
    assert [event["rows_planned"] for event in events] == [2, 1]
    assert all(event["producer"] == "index" for event in events)
    assert all(event["lane"] == "batch" for event in events)
    assert all(event["outcome"] == "commit" for event in events)
    assert all(event["duration_ms"] >= 0 for event in events)


def test_direct_store_memory_emits_interactive_transaction_span(tmp_path, monkeypatch):
    log_path = _configure(monkeypatch, tmp_path)
    store = VectorStore(tmp_path / "brainlayer.db")
    log_path.unlink()

    result = store_memory(
        store=store,
        embed_fn=None,
        content="A unique direct MCP telemetry store memory",
        memory_type="note",
        project="writer-telemetry",
        retry_on_busy=False,
    )
    store.close()

    assert result["id"]
    events = _finished(log_path, "store_memory")
    assert len(events) == 1
    assert events[0]["producer"] == "mcp"
    assert events[0]["lane"] == "interactive"
    assert events[0]["rows_planned"] == 1
    assert events[0]["outcome"] == "commit"
