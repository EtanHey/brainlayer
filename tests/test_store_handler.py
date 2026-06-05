"""Tests for MCP store handler responses."""

import json
from unittest.mock import patch

import apsw
import pytest


@pytest.mark.asyncio
async def test_busy_queue_fallback_returns_queued_chunk_id(tmp_path):
    """DB-busy fallback returns the durable queue chunk ID, not a sentinel."""
    from brainlayer.mcp.store_handler import _store

    queue_dir = tmp_path / "queue"

    with (
        patch("brainlayer.mcp.store_handler._get_vector_store"),
        patch("brainlayer.mcp.store_handler._normalize_project_name", return_value="test"),
        patch("brainlayer.store.store_memory", side_effect=apsw.BusyError("locked")),
        patch("brainlayer.queue_io.get_queue_dir", return_value=queue_dir),
    ):
        texts, structured = await _store(
            content="test memory",
            memory_type="note",
            project="test",
        )

    queued_files = list(queue_dir.glob("mcp-*.jsonl"))
    assert len(queued_files) == 1
    queued_event = json.loads(queued_files[0].read_text())
    expected_chunk_id = queued_event["chunk_id"]

    assert expected_chunk_id != "queued"
    assert structured["chunk_id"] == expected_chunk_id
    assert structured["queued"] is True
    assert any(expected_chunk_id in item.text for item in texts)


@pytest.mark.asyncio
async def test_store_preassigns_same_chunk_id_across_busy_retry(tmp_path, monkeypatch):
    """The MCP handler promises one chunk ID before the first write attempt."""
    from brainlayer.mcp.store_handler import _store

    queue_dir = tmp_path / "queue"
    seen_chunk_ids = []

    def flaky_store_memory(**kwargs):
        seen_chunk_ids.append(kwargs.get("chunk_id"))
        if len(seen_chunk_ids) == 1:
            raise apsw.BusyError("locked")
        return {"id": kwargs.get("chunk_id") or "store-generated-id", "related": []}

    with (
        patch("brainlayer.mcp.store_handler._get_vector_store"),
        patch("brainlayer.mcp.store_handler._normalize_project_name", return_value="test"),
        patch("brainlayer.store.store_memory", side_effect=flaky_store_memory),
        patch("brainlayer.queue_io.get_queue_dir", return_value=queue_dir),
    ):
        monkeypatch.setattr("brainlayer.mcp.store_handler._retry_delay", 0.001)
        texts, structured = await _store(
            content="retry should keep promised id",
            memory_type="note",
            project="test",
        )

    assert len(seen_chunk_ids) == 2
    assert seen_chunk_ids[0] is not None
    assert seen_chunk_ids == [structured["chunk_id"], structured["chunk_id"]]
    assert structured["chunk_id"].startswith("manual-")
    assert any(structured["chunk_id"] in item.text for item in texts)
    assert not list(queue_dir.glob("mcp-*.jsonl"))


@pytest.mark.asyncio
async def test_busy_queue_fallback_flushes_promised_chunk_id(tmp_path, monkeypatch):
    """DB-busy queue and drain replay persist the exact caller-visible chunk ID."""
    from brainlayer.drain import drain_once
    from brainlayer.mcp.store_handler import _store
    from brainlayer.vector_store import VectorStore

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    store = VectorStore(db_path)
    store.close()
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")

    with (
        patch("brainlayer.mcp.store_handler._get_vector_store"),
        patch("brainlayer.mcp.store_handler._normalize_project_name", return_value="test"),
        patch("brainlayer.store.store_memory", side_effect=apsw.BusyError("locked")),
        patch("brainlayer.queue_io.get_queue_dir", return_value=queue_dir),
    ):
        texts, structured = await _store(
            content="queued flush must preserve promised id",
            memory_type="note",
            project="test",
        )

    queued_files = list(queue_dir.glob("mcp-*.jsonl"))
    assert len(queued_files) == 1
    queued_event = json.loads(queued_files[0].read_text())
    promised_chunk_id = structured["chunk_id"]

    assert promised_chunk_id == queued_event["chunk_id"]
    assert promised_chunk_id.startswith("manual-")
    assert structured["queued"] is True
    assert any(promised_chunk_id in item.text for item in texts)

    assert drain_once(db_path=db_path, queue_dir=queue_dir, log_path=tmp_path / "drain.log") == 1

    conn = apsw.Connection(str(db_path))
    try:
        row = conn.cursor().execute("SELECT id, content FROM chunks WHERE id = ?", (promised_chunk_id,)).fetchone()
    finally:
        conn.close()

    assert row == (promised_chunk_id, "queued flush must preserve promised id")


@pytest.mark.asyncio
async def test_writer_in_use_error_queues_instead_of_erroring(tmp_path):
    """Writer pidfile contention queues the store instead of returning an MCP error."""
    from brainlayer.mcp.store_handler import _store
    from brainlayer.vector_store import WriterInUseError

    queue_dir = tmp_path / "queue"

    with (
        patch("brainlayer.mcp.store_handler._get_vector_store"),
        patch("brainlayer.mcp.store_handler._normalize_project_name", return_value="test"),
        patch(
            "brainlayer.store.store_memory",
            side_effect=WriterInUseError("another writer is using brainlayer.db (pid 123)"),
        ),
        patch("brainlayer.queue_io.get_queue_dir", return_value=queue_dir),
    ):
        texts, structured = await _store(
            content="queued under held writer pidfile",
            memory_type="note",
            project="test",
        )

    assert structured["queued"] is True
    assert structured["chunk_id"] != "queued"
    assert any("queued" in item.text.lower() for item in texts)
    assert len(list(queue_dir.glob("mcp-*.jsonl"))) == 1


@pytest.mark.asyncio
async def test_sqlite_prepare_lock_error_queues_instead_of_erroring(tmp_path):
    """Prepare-time transient SQLite lock failures are queueable store failures."""
    from brainlayer.mcp.store_handler import _store

    queue_dir = tmp_path / "queue"

    with (
        patch("brainlayer.mcp.store_handler._get_vector_store"),
        patch("brainlayer.mcp.store_handler._normalize_project_name", return_value="test"),
        patch(
            "brainlayer.store.store_memory",
            side_effect=RuntimeError("SQLite prepare failed: database schema is locked"),
        ),
        patch("brainlayer.queue_io.get_queue_dir", return_value=queue_dir),
    ):
        texts, structured = await _store(
            content="queued under prepare-time sqlite lock",
            memory_type="note",
            project="test",
        )

    assert structured["queued"] is True
    assert structured["chunk_id"] != "queued"
    assert any("queued" in item.text.lower() for item in texts)
    assert len(list(queue_dir.glob("mcp-*.jsonl"))) == 1


@pytest.mark.asyncio
async def test_busy_queue_fallback_preserves_supersedes(tmp_path):
    """DB-busy fallback queues supersedes so the deferred write keeps lifecycle intent."""
    from brainlayer.mcp.store_handler import _store

    queue_dir = tmp_path / "queue"

    with (
        patch("brainlayer.mcp.store_handler._get_vector_store"),
        patch("brainlayer.mcp.store_handler._normalize_project_name", return_value="test"),
        patch("brainlayer.store.store_memory", side_effect=apsw.BusyError("locked")),
        patch("brainlayer.queue_io.get_queue_dir", return_value=queue_dir),
    ):
        await _store(
            content="replacement memory",
            memory_type="note",
            project="test",
            supersedes="manual-old1234",
        )

    queued_files = list(queue_dir.glob("mcp-*.jsonl"))
    assert len(queued_files) == 1
    queued_event = json.loads(queued_files[0].read_text())
    assert queued_event["supersedes"] == "manual-old1234"


@pytest.mark.asyncio
async def test_arbitrated_queue_fallback_returns_queued_chunk_id(tmp_path, monkeypatch):
    """Arbitrated fallback also reports the queued event's real chunk ID."""
    from brainlayer.mcp.store_handler import _store

    monkeypatch.setenv("BRAINLAYER_ARBITRATED", "1")
    with (
        patch("brainlayer.queue_io.get_queue_dir", return_value=tmp_path),
        patch("brainlayer.search_repo.clear_hybrid_search_cache"),
    ):
        texts, structured = await _store(content="arbitrated queued memory", memory_type="note", project="test")

    queued_files = list(tmp_path.glob("mcp-*.jsonl"))
    assert len(queued_files) == 1
    queued_event = json.loads(queued_files[0].read_text())
    expected_chunk_id = queued_event["chunk_id"]

    assert expected_chunk_id != "queued"
    assert structured["chunk_id"] == expected_chunk_id
    assert structured["queued"] is True
    assert any(expected_chunk_id in item.text for item in texts)
