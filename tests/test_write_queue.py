"""Tests for MCP-level write queue and DB lock resilience.

Tests that brain_store, brain_update, and brain_search handle DB lock
gracefully — retry with backoff, queue to JSONL on failure, flush on success.

All tests use tmp_path fixtures (no real DB contention).
"""

import json
from unittest.mock import MagicMock, patch

import apsw
import pytest

from brainlayer.mcp.store_handler import (
    _flush_pending_stores,
    _queue_store,
)

# ---------------------------------------------------------------------------
# _queue_store / _flush_pending_stores unit tests
# ---------------------------------------------------------------------------


class TestQueueStore:
    """JSONL queue for buffering writes when DB is locked."""

    def test_queue_store_writes_jsonl(self, tmp_path):
        """_queue_store writes a JSONL line to the pending file."""
        with patch(
            "brainlayer.mcp.store_handler._get_pending_store_path",
            return_value=tmp_path / "pending-stores.jsonl",
        ):
            _queue_store({"content": "test memory", "memory_type": "note"})

        path = tmp_path / "pending-stores.jsonl"
        assert path.exists()
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        item = json.loads(lines[0])
        assert item["content"] == "test memory"
        assert item["memory_type"] == "note"

    def test_queue_store_appends(self, tmp_path):
        """Multiple queue calls append, not overwrite."""
        with patch(
            "brainlayer.mcp.store_handler._get_pending_store_path",
            return_value=tmp_path / "pending-stores.jsonl",
        ):
            _queue_store({"content": "first", "memory_type": "note"})
            _queue_store({"content": "second", "memory_type": "learning"})

        lines = (tmp_path / "pending-stores.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_queue_max_size_drops_oldest(self, tmp_path):
        """Queue respects max size limit and drops oldest items."""
        pending_path = tmp_path / "pending-stores.jsonl"
        with patch(
            "brainlayer.mcp.store_handler._get_pending_store_path",
            return_value=pending_path,
        ):
            # Queue 105 items (over the 100 limit)
            for i in range(105):
                _queue_store({"content": f"item-{i}", "memory_type": "note"})

        lines = pending_path.read_text().strip().splitlines()
        # Should be exactly 100, with oldest 5 dropped
        assert len(lines) == 100
        # First retained item should be item-5 (oldest 5 dropped)
        first_item = json.loads(lines[0])
        assert first_item["content"] == "item-5"
        # Last item should be the newest
        last_item = json.loads(lines[-1])
        assert last_item["content"] == "item-104"


class TestFlushPendingStores:
    """Flush pending JSONL queue back to DB."""

    def test_flush_drains_queue(self, tmp_path):
        """Successful flush empties the JSONL file."""
        pending_path = tmp_path / "pending-stores.jsonl"
        pending_path.write_text(json.dumps({"content": "queued item", "memory_type": "note"}) + "\n")

        mock_store = MagicMock()
        mock_embed = MagicMock(return_value=[0.1] * 1024)

        with patch(
            "brainlayer.mcp.store_handler._get_pending_store_path",
            return_value=pending_path,
        ):
            # store_memory is imported inside _flush_pending_stores via `from ..store import store_memory`
            with patch("brainlayer.store.store_memory") as mock_store_memory:
                mock_store_memory.return_value = {"id": "test-123", "related": []}
                flushed = _flush_pending_stores(mock_store, mock_embed)

        assert flushed == 1
        assert not pending_path.exists()  # File deleted after full flush

    def test_flush_keeps_failed_items(self, tmp_path):
        """Items that fail to flush are kept in the queue."""
        pending_path = tmp_path / "pending-stores.jsonl"
        pending_path.write_text(
            json.dumps({"content": "good", "memory_type": "note"})
            + "\n"
            + json.dumps({"content": "bad", "memory_type": "note"})
            + "\n"
        )

        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise apsw.BusyError("still locked")
            return {"id": "test-123", "related": []}

        with patch(
            "brainlayer.mcp.store_handler._get_pending_store_path",
            return_value=pending_path,
        ):
            with patch("brainlayer.store.store_memory", side_effect=side_effect):
                flushed = _flush_pending_stores(MagicMock(), MagicMock())

        assert flushed == 1
        remaining = pending_path.read_text().strip().splitlines()
        assert len(remaining) == 1
        assert json.loads(remaining[0])["content"] == "bad"

    def test_flush_noop_when_no_file(self, tmp_path):
        """Flush returns 0 when no pending file exists."""
        with patch(
            "brainlayer.mcp.store_handler._get_pending_store_path",
            return_value=tmp_path / "nonexistent.jsonl",
        ):
            flushed = _flush_pending_stores(MagicMock(), MagicMock())

        assert flushed == 0


# ---------------------------------------------------------------------------
# MCP handler resilience tests
# ---------------------------------------------------------------------------


class TestStoreRetryOnLock:
    """brain_store should queue to JSONL when DB is locked."""

    @pytest.mark.asyncio
    async def test_store_queues_on_busy_error(self, tmp_path):
        """When store_memory raises BusyError, the item is queued to JSONL."""
        from brainlayer.mcp.store_handler import _store

        pending_path = tmp_path / "pending-stores.jsonl"

        with (
            patch("brainlayer.mcp.store_handler._get_vector_store") as mock_vs,
            patch("brainlayer.mcp.store_handler._get_embedding_model") as mock_em,
            patch("brainlayer.mcp.store_handler._normalize_project_name", return_value="test"),
            # store_memory is imported inside _store via `from ..store import store_memory`
            patch("brainlayer.store.store_memory", side_effect=apsw.BusyError("locked")),
            patch(
                "brainlayer.mcp.store_handler._get_pending_store_path",
                return_value=pending_path,
            ),
        ):
            mock_em.return_value.embed_query.return_value = [0.1] * 1024

            result = await _store(
                content="test memory",
                memory_type="note",
                project="test",
            )

        # Should return queued response, not error
        texts, structured = result
        assert structured["chunk_id"] == "queued"
        assert any("queued" in t.text.lower() for t in texts)

        # Should have written to JSONL
        assert pending_path.exists()
        item = json.loads(pending_path.read_text().strip())
        assert item["content"] == "test memory"


class TestBrainUpdateRetryOnLock:
    """brain_update should retry on BusyError before failing."""

    @pytest.mark.asyncio
    async def test_update_retries_before_failing(self):
        """brain_update retries up to 3 times with backoff on BusyError."""
        from brainlayer.mcp.store_handler import _brain_update

        call_count = 0

        def archive_side_effect(chunk_id):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise apsw.BusyError("database is locked")
            return True

        mock_store = MagicMock()
        mock_store.archive_chunk = archive_side_effect

        with (
            patch("brainlayer.mcp.store_handler._get_vector_store", return_value=mock_store),
            patch("brainlayer.mcp.store_handler._retry_delay", 0.001),  # fast tests
        ):
            result = await _brain_update(action="archive", chunk_id="test-chunk-123")

        # Should succeed after retries (2 BusyError + 1 success)
        assert call_count == 3
        # Result should be a list of TextContent (success), not a CallToolResult error
        assert isinstance(result, list)
        assert len(result) > 0
        text = result[0].text
        assert "archived" in text

    @pytest.mark.asyncio
    async def test_update_fails_after_max_retries(self):
        """brain_update returns error after exhausting retries."""
        from brainlayer.mcp.store_handler import _brain_update

        mock_store = MagicMock()
        mock_store.archive_chunk = MagicMock(side_effect=apsw.BusyError("locked"))

        with (
            patch("brainlayer.mcp.store_handler._get_vector_store", return_value=mock_store),
            patch("brainlayer.mcp.store_handler._retry_delay", 0.001),
        ):
            result = await _brain_update(action="archive", chunk_id="test-chunk-123")

        # Should be an error result after retries exhausted
        from mcp.types import CallToolResult

        assert isinstance(result, CallToolResult)
        assert result.isError is True


class TestBrainSearchRetryOnLock:
    """brain_search (reads) should retry on BusyError in WAL mode."""

    @pytest.mark.asyncio
    async def test_search_retries_on_busy(self):
        """Search retries when DB is temporarily locked."""
        from brainlayer.mcp.search_handler import _search

        call_count = 0
        original_search_results = {
            "ids": [["c1"]],
            "documents": [["test doc"]],
            "metadatas": [[{"project": "test", "content_type": "note", "source_file": "s.jsonl"}]],
            "distances": [[0.1]],
        }

        def hybrid_search_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise apsw.BusyError("database is locked")
            return original_search_results

        mock_store = MagicMock()
        mock_store.count.return_value = 100
        mock_store.hybrid_search = hybrid_search_side_effect
        mock_store.enrich_results_with_session_context = lambda r: r

        mock_model = MagicMock()
        mock_model.embed_query.return_value = [0.1] * 1024

        with (
            patch("brainlayer.mcp.search_handler._get_vector_store", return_value=mock_store),
            patch("brainlayer.mcp.search_handler._get_embedding_model", return_value=mock_model),
            patch("brainlayer.mcp.search_handler._normalize_project_name", return_value=None),
            patch("brainlayer.mcp.search_handler._retry_delay", 0.001),
        ):
            result = await _search(query="test query")

        assert call_count == 3  # 2 failures + 1 success
        # Should return results, not error
        texts, structured = result
        assert structured["total"] > 0
