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
