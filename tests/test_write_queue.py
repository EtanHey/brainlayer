"""Tests for MCP-level write queue and DB lock resilience.

Tests that brain_store, brain_update, and brain_search handle DB lock
gracefully — retry with backoff, queue to JSONL on failure, flush on success.

All tests use tmp_path fixtures (no real DB contention).
"""

import json
import threading
from unittest.mock import MagicMock, patch

import apsw
import pytest

from brainlayer.mcp.store_handler import (
    _flush_pending_stores,
    _queue_store,
)
from brainlayer.queue_io import enqueue_hook_chunk

# ---------------------------------------------------------------------------
# _queue_store / _flush_pending_stores unit tests
# ---------------------------------------------------------------------------


class TestQueueStore:
    """JSONL queue for buffering writes when DB is locked."""

    def test_queue_store_writes_jsonl(self, tmp_path):
        """_queue_store writes to the unified arbitration queue."""
        with patch("brainlayer.queue_io.get_queue_dir", return_value=tmp_path):
            _queue_store({"content": "test memory", "memory_type": "note"})

        files = list(tmp_path.glob("mcp-*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text().strip().splitlines()
        assert len(lines) == 1
        item = json.loads(lines[0])
        assert item["kind"] == "store_memory"
        assert item["content"] == "test memory"
        assert item["memory_type"] == "note"

    def test_queue_store_appends(self, tmp_path):
        """Multiple queue calls create independent durable files."""
        with patch("brainlayer.queue_io.get_queue_dir", return_value=tmp_path):
            _queue_store({"content": "first", "memory_type": "note"})
            _queue_store({"content": "second", "memory_type": "learning"})

        files = sorted(tmp_path.glob("mcp-*.jsonl"))
        assert len(files) == 2
        contents = {json.loads(path.read_text())["content"] for path in files}
        assert contents == {"first", "second"}

    def test_queue_store_keeps_burst_items(self, tmp_path):
        """Unified queue keeps burst items as separate files."""
        with patch("brainlayer.queue_io.get_queue_dir", return_value=tmp_path):
            for i in range(105):
                _queue_store({"content": f"item-{i}", "memory_type": "note"})

        assert len(list(tmp_path.glob("mcp-*.jsonl"))) == 105

    def test_enqueue_hook_chunk_preserves_zero_timestamp(self, tmp_path):
        path = enqueue_hook_chunk(
            session_id="session-zero",
            content="hook memory with explicit epoch timestamp",
            timestamp=0.0,
            queue_dir=tmp_path,
        )

        item = json.loads(path.read_text())

        assert item["timestamp"] == 0.0


class TestSingleWriterQueue:
    def test_single_worker_serializes_writes(self):
        from brainlayer.pipeline.write_queue import WriteQueue

        write_queue = WriteQueue(maxsize=32)
        write_queue.start()

        submission_order = []
        persisted_order = []
        lock = threading.Lock()
        barrier = threading.Barrier(6)
        futures = [None] * 5

        def submit_value(index: int) -> None:
            barrier.wait()
            with lock:
                submission_order.append(index)
                futures[index] = write_queue.submit(
                    f"write-{index}",
                    lambda idx=index: persisted_order.append(idx),
                )

        threads = [threading.Thread(target=submit_value, args=(i,)) for i in range(5)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()
        for future in futures:
            future.result(timeout=2)

        write_queue.stop()

        assert persisted_order == submission_order

    def test_queue_durability_on_crash(self):
        from brainlayer.pipeline.write_queue import WriteQueue

        write_queue = WriteQueue(maxsize=32)
        write_queue.start()

        persisted = []

        def raise_error() -> None:
            raise RuntimeError("boom")

        crash_future = write_queue.submit(
            "crash-write",
            raise_error,
            crash_on_error=True,
        )
        futures = [write_queue.submit(f"write-{index}", lambda idx=index: persisted.append(idx)) for index in range(9)]

        with pytest.raises(RuntimeError, match="boom"):
            crash_future.result(timeout=2)

        write_queue.start()
        for future in futures:
            future.result(timeout=2)

        write_queue.stop()

        assert persisted == list(range(9))

    def test_submit_raises_when_queue_is_full(self):
        from brainlayer.pipeline.write_queue import WriteQueue, WriteQueueFullError

        write_queue = WriteQueue(maxsize=1)
        write_queue.start = lambda: None
        first_future = write_queue.submit("write-0", lambda: None)

        with pytest.raises(WriteQueueFullError, match="write queue is full"):
            write_queue.submit("write-1", lambda: None, timeout=0.01)
        assert not first_future.done()


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

    def test_flush_preserves_legacy_pending_chunk_id(self, tmp_path):
        """Legacy pending-stores flush must persist the caller-visible queued ID."""
        pending_path = tmp_path / "pending-stores.jsonl"
        pending_path.write_text(
            json.dumps(
                {
                    "chunk_id": "manual-promised1234",
                    "content": "queued item",
                    "memory_type": "note",
                }
            )
            + "\n"
        )

        with patch(
            "brainlayer.mcp.store_handler._get_pending_store_path",
            return_value=pending_path,
        ):
            with patch("brainlayer.store.store_memory") as mock_store_memory:
                mock_store_memory.return_value = {"id": "manual-promised1234", "related": []}
                flushed = _flush_pending_stores(MagicMock(), MagicMock())

        assert flushed == 1
        assert mock_store_memory.call_args.kwargs["chunk_id"] == "manual-promised1234"

    def test_flush_preserves_legacy_pending_supersedes(self, tmp_path):
        """Legacy pending-stores flush must apply queued supersedes metadata."""
        pending_path = tmp_path / "pending-stores.jsonl"
        pending_path.write_text(
            json.dumps(
                {
                    "chunk_id": "manual-new1234",
                    "content": "replacement item",
                    "memory_type": "note",
                    "supersedes": "manual-old1234",
                }
            )
            + "\n"
        )

        mock_store = MagicMock()
        with patch(
            "brainlayer.mcp.store_handler._get_pending_store_path",
            return_value=pending_path,
        ):
            with patch("brainlayer.store.store_memory") as mock_store_memory:
                mock_store_memory.return_value = {"id": "manual-new1234", "related": []}
                flushed = _flush_pending_stores(mock_store, MagicMock())

        assert flushed == 1
        mock_store.supersede_chunk.assert_called_once_with("manual-old1234", "manual-new1234")

    def test_flush_keeps_item_when_legacy_supersedes_fails(self, tmp_path):
        """A failed queued supersedes transition must not drop the pending store item."""
        pending_path = tmp_path / "pending-stores.jsonl"
        pending_path.write_text(
            json.dumps(
                {
                    "chunk_id": "manual-new1234",
                    "content": "replacement item",
                    "memory_type": "note",
                    "supersedes": "manual-old1234",
                }
            )
            + "\n"
        )

        mock_store = MagicMock()
        mock_store.supersede_chunk.return_value = False
        with patch(
            "brainlayer.mcp.store_handler._get_pending_store_path",
            return_value=pending_path,
        ):
            with patch("brainlayer.store.store_memory") as mock_store_memory:
                mock_store_memory.return_value = {"id": "manual-new1234", "related": []}
                flushed = _flush_pending_stores(mock_store, MagicMock())

        assert flushed == 0
        remaining = pending_path.read_text().strip().splitlines()
        assert len(remaining) == 1
        assert json.loads(remaining[0])["chunk_id"] == "manual-new1234"
        mock_store.supersede_chunk.assert_called_once_with("manual-old1234", "manual-new1234")

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

        queue_dir = tmp_path / "queue"

        with (
            patch("brainlayer.mcp.store_handler._get_vector_store") as mock_vs,
            patch("brainlayer.mcp.store_handler._get_embedding_model") as mock_em,
            patch("brainlayer.mcp.store_handler._normalize_project_name", return_value="test"),
            # store_memory is imported inside _store via `from ..store import store_memory`
            patch("brainlayer.store.store_memory", side_effect=apsw.BusyError("locked")),
            patch("brainlayer.queue_io.get_queue_dir", return_value=queue_dir),
        ):
            mock_em.return_value.embed_query.return_value = [0.1] * 1024

            result = await _store(
                content="test memory",
                memory_type="note",
                project="test",
            )

        # Should return queued response, not error
        texts, structured = result
        assert structured["chunk_id"].startswith("manual-")
        assert structured["queued"] is True
        assert any("queued" in t.text.lower() for t in texts)

        # Should have written to the unified queue
        files = list(queue_dir.glob("mcp-*.jsonl"))
        assert len(files) == 1
        item = json.loads(files[0].read_text().strip())
        assert item["kind"] == "store_memory"
        assert item["chunk_id"] == structured["chunk_id"]
        assert item["content"] == "test memory"

    @pytest.mark.asyncio
    async def test_arbitrated_store_validates_before_queueing(self, tmp_path, monkeypatch):
        """Arbitrated store should not report queued success for invalid content."""
        from brainlayer.mcp.store_handler import _store

        monkeypatch.setenv("BRAINLAYER_ARBITRATED", "1")
        with patch("brainlayer.queue_io.get_queue_dir", return_value=tmp_path):
            result = await _store(content="  ", memory_type="note", project="test")

        assert result.isError is True
        assert "content must be non-empty" in result.content[0].text
        assert not list(tmp_path.glob("mcp-*.jsonl"))

    @pytest.mark.asyncio
    async def test_arbitrated_store_clears_search_cache(self, tmp_path, monkeypatch):
        """Queued stores invalidate local search cache even before the drain writes."""
        from brainlayer.mcp.store_handler import _store

        monkeypatch.setenv("BRAINLAYER_ARBITRATED", "1")
        with (
            patch("brainlayer.queue_io.get_queue_dir", return_value=tmp_path),
            patch("brainlayer.search_repo.clear_hybrid_search_cache") as clear_cache,
        ):
            texts, structured = await _store(content="queued cache invalidation", memory_type="note", project="test")

        assert structured["chunk_id"].startswith("manual-")
        assert structured["queued"] is True
        assert any("queued" in item.text.lower() for item in texts)
        clear_cache.assert_called_once_with()


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
