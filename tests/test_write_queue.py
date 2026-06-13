"""Tests for MCP-level write queue and DB lock resilience.

Tests that brain_store, brain_update, and brain_search handle DB lock
gracefully — retry with backoff, queue to JSONL on failure, flush on success.

All tests use tmp_path fixtures (no real DB contention).
"""

import json
import sqlite3
import threading
import time
from unittest.mock import MagicMock, patch

import apsw
import pytest

from brainlayer.drain import _open_connection, drain_once
from brainlayer.mcp.store_handler import (
    _flush_pending_stores,
    _queue_store,
)
from brainlayer.queue_io import enqueue_enrichment_updates, enqueue_hook_chunk

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

    def test_drain_preserves_hook_event_created_at_and_project(self, tmp_path, monkeypatch):
        """Hook/realtime queue events must replay reservation metadata, not flush time."""
        from brainlayer.vector_store import VectorStore

        db_path = tmp_path / "hook-created-at.db"
        queue_dir = tmp_path / "queue"
        VectorStore(db_path).close()
        monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")

        event = {
            "kind": "hook_chunk",
            "session_id": "session-created-at",
            "chunk_id": "rt-session-created-at",
            "content": "hook queue created_at reservation timestamp should survive drain",
            "content_hash": "created-at-hash",
            "project": "brainlayer",
            "source_file": "/Users/test/Gits/brainlayer/session.jsonl",
            "created_at": "2026-06-06T20:04:14Z",
        }
        queue_dir.mkdir()
        (queue_dir / "hook-created-at.jsonl").write_text(json.dumps(event) + "\n")

        drained = drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=1, log_path=tmp_path / "drain.log")

        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT created_at, project, source FROM chunks WHERE id = ?", (event["chunk_id"],)
            ).fetchone()

        assert drained == 1
        assert row == ("2026-06-06T20:04:14Z", "brainlayer", "realtime")

    def test_drain_embeds_hook_event_chunk(self, tmp_path, monkeypatch):
        """Hook queue events must return chunk IDs so drain can embed them."""
        from brainlayer.vector_store import VectorStore

        db_path = tmp_path / "hook-embedded.db"
        queue_dir = tmp_path / "queue"
        VectorStore(db_path).close()
        monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "1")
        event = {
            "kind": "hook_chunk",
            "session_id": "session-embedded",
            "chunk_id": "rt-session-embedded",
            "content": "hook queue event should receive a vector row from drain",
            "content_hash": "embedded-hash",
            "project": "brainlayer",
            "source_file": "/Users/test/Gits/brainlayer/session.jsonl",
            "created_at": "2026-06-06T20:04:14Z",
        }
        queue_dir.mkdir()
        (queue_dir / "hook-embedded.jsonl").write_text(json.dumps(event) + "\n")

        drained = drain_once(
            db_path=db_path,
            queue_dir=queue_dir,
            batch_size=1,
            log_path=tmp_path / "drain.log",
            embed_fn=lambda _content: [0.0] * 1024,
        )

        conn = _open_connection(db_path)
        try:
            row = conn.execute("SELECT chunk_id FROM chunk_vectors WHERE chunk_id = ?", (event["chunk_id"],)).fetchone()
        finally:
            conn.close()

        assert drained == 1
        assert row == (event["chunk_id"],)

    def test_apply_hook_returns_canonical_chunk_id_for_embedding(self, monkeypatch):
        """Merged hook events should embed the canonical row, not the duplicate ID."""
        from brainlayer import drain

        def fake_insert_or_merge(_conn, _row):
            return "rt-canonical"

        monkeypatch.setattr(drain, "_insert_or_merge_chunk", fake_insert_or_merge)

        result = drain._apply_hook(
            None,
            {
                "session_id": "session-duplicate",
                "chunk_id": "rt-duplicate",
                "content": "queued realtime duplicate should report canonical id",
                "content_hash": "duplicate-hash",
                "source_file": "/Users/test/Gits/brainlayer/session.jsonl",
                "created_at": "2026-06-06T20:04:14Z",
            },
        )

        assert result.chunk_id == "rt-canonical"


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

    def test_flush_preserves_legacy_pending_created_at_and_project(self, tmp_path):
        """Legacy pending-stores flush must replay reservation metadata."""
        pending_path = tmp_path / "pending-stores.jsonl"
        pending_path.write_text(
            json.dumps(
                {
                    "chunk_id": "manual-promised5678",
                    "content": "queued item",
                    "memory_type": "note",
                    "project": "brainlayer",
                    "created_at": "2026-06-06T18:45:12Z",
                }
            )
            + "\n"
        )

        with patch(
            "brainlayer.mcp.store_handler._get_pending_store_path",
            return_value=pending_path,
        ):
            with patch("brainlayer.store.store_memory") as mock_store_memory:
                mock_store_memory.return_value = {"id": "manual-promised5678", "related": []}
                flushed = _flush_pending_stores(MagicMock(), MagicMock())

        assert flushed == 1
        assert mock_store_memory.call_args.kwargs["chunk_id"] == "manual-promised5678"
        assert mock_store_memory.call_args.kwargs["project"] == "brainlayer"
        assert mock_store_memory.call_args.kwargs["created_at"] == "2026-06-06T18:45:12Z"

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

    def test_flush_does_not_requeue_after_durable_store_when_legacy_supersedes_fails(self, tmp_path):
        """A failed supersede must not replay an already stored replacement."""
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

        assert flushed == 1
        assert not pending_path.exists()
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

    def test_legacy_pending_store_receipt_reports_replay_not_drain(self, tmp_path):
        from brainlayer.mcp.store_handler import _deferred_store_receipt

        pending_path = tmp_path / "pending-stores.jsonl"

        receipt = _deferred_store_receipt("manual-promised", pending_path)

        assert receipt["deferred"]["action"] == "queued_for_replay"

    def test_legacy_pending_store_enqueue_and_flush_use_shared_file_lock(self, tmp_path):
        pending_path = tmp_path / "pending-stores.jsonl"

        with (
            patch("brainlayer.queue_io.enqueue_store", side_effect=RuntimeError("queue unavailable")),
            patch("brainlayer.mcp.store_handler._get_pending_store_path", return_value=pending_path),
            patch("brainlayer.mcp.store_handler.fcntl.flock") as flock,
        ):
            _queue_store({"content": "queued item", "memory_type": "note"})
            pending_path.write_text(json.dumps({"content": "queued item", "memory_type": "note"}) + "\n")
            with patch("brainlayer.store.store_memory", return_value={"id": "test-123", "related": []}):
                _flush_pending_stores(MagicMock(), MagicMock())

        assert flock.call_count >= 4

    def test_legacy_pending_store_lock_path_matches_brainbar_dotfile(self, tmp_path):
        from brainlayer.mcp.store_handler import _pending_store_lock_path

        pending_path = tmp_path / "pending-stores.jsonl"

        assert _pending_store_lock_path(pending_path) == tmp_path / ".pending-stores.jsonl.lock"

    def test_legacy_pending_store_path_follows_active_db_env(self, tmp_path, monkeypatch):
        from brainlayer.mcp.store_handler import _get_pending_store_path

        active_db = tmp_path / "active" / "brainlayer.db"
        monkeypatch.setenv("BRAINLAYER_DB", str(active_db))

        assert _get_pending_store_path() == active_db.parent / "pending-stores.jsonl"

    def test_flush_rewrites_remaining_legacy_queue_via_atomic_replace(self, tmp_path, monkeypatch):
        pending_path = tmp_path / "pending-stores.jsonl"
        pending_path.write_text(
            json.dumps({"content": "good", "memory_type": "note"})
            + "\n"
            + json.dumps({"content": "bad", "memory_type": "note"})
            + "\n"
        )
        replaced = {}
        original_replace = type(pending_path).replace

        def track_replace(self, target):
            replaced["source"] = self
            replaced["target"] = target
            return original_replace(self, target)

        def side_effect(**kwargs):
            if kwargs["content"] == "bad":
                raise apsw.BusyError("still locked")
            return {"id": "test-123", "related": []}

        monkeypatch.setattr(type(pending_path), "replace", track_replace)
        with patch("brainlayer.mcp.store_handler._get_pending_store_path", return_value=pending_path):
            with patch("brainlayer.store.store_memory", side_effect=side_effect):
                flushed = _flush_pending_stores(MagicMock(), MagicMock())

        assert flushed == 1
        assert replaced["source"] == pending_path.with_suffix(".jsonl.tmp")
        assert replaced["target"] == pending_path
        assert json.loads(pending_path.read_text().strip())["content"] == "bad"


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
    async def test_store_retries_busy_error_before_queueing(self, tmp_path, monkeypatch):
        """brain_store should wait for short lock bursts instead of queueing immediately."""
        from brainlayer.mcp.store_handler import _store

        queue_dir = tmp_path / "queue"
        attempts = 0

        def flaky_store_memory(**kwargs):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise apsw.BusyError("database is locked")
            return {"id": "manual-landed", "related": []}

        with (
            patch("brainlayer.mcp.store_handler._get_vector_store", return_value=MagicMock()),
            patch("brainlayer.mcp.store_handler._normalize_project_name", return_value="test"),
            patch("brainlayer.store.store_memory", side_effect=flaky_store_memory),
            patch("brainlayer.queue_io.get_queue_dir", return_value=queue_dir),
        ):
            monkeypatch.setattr("brainlayer.mcp.store_handler._retry_delay", 0.001)
            texts, structured = await _store(
                content="test memory",
                memory_type="note",
                project="test",
            )

        assert attempts == 3
        assert structured == {"chunk_id": "manual-landed", "related": []}
        assert any("manual-landed" in item.text for item in texts)
        assert not list(queue_dir.glob("mcp-*.jsonl"))

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


def test_drain_limits_enrichment_events_per_transaction(tmp_path, monkeypatch):
    """Large enrichment queue files should be split so MCP store files can interleave."""
    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    log_path = tmp_path / "drain.log"

    conn = apsw.Connection(str(db_path))
    conn.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT,
            summary TEXT,
            enriched_at TEXT,
            enrich_status TEXT,
            content_hash TEXT
        )
        """
    )
    for idx in range(4):
        conn.execute(
            "INSERT INTO chunks (id, content, content_hash) VALUES (?, ?, ?)",
            (f"c{idx}", f"content {idx}", f"h{idx}"),
        )
    conn.close()

    queue_file = enqueue_enrichment_updates(
        [{"chunk_id": f"c{idx}", "content_hash": f"h{idx}", "enrichment": {"summary": f"s{idx}"}} for idx in range(4)],
        queue_dir=queue_dir,
    )

    monkeypatch.setenv("BRAINLAYER_DRAIN_MAX_EVENTS_PER_TRANSACTION", "2")

    assert drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=1, log_path=log_path) == 2
    assert queue_file.exists()
    assert len(queue_file.read_text(encoding="utf-8").splitlines()) == 2

    conn = apsw.Connection(str(db_path))
    summaries = dict(conn.execute("SELECT id, summary FROM chunks ORDER BY id"))
    conn.close()

    assert summaries == {"c0": "s0", "c1": "s1", "c2": None, "c3": None}


def _create_burn_drain_db(path):
    conn = apsw.Connection(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT,
            summary TEXT,
            enriched_at TEXT,
            enrich_status TEXT,
            content_hash TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO chunks (id, content, summary, enriched_at, enrich_status, content_hash)
        VALUES
            ('already-done', 'content 1', 'old summary', '2026-05-30T00:00:00Z', 'success', 'h1'),
            ('needs-update', 'content 2', NULL, NULL, NULL, 'h2')
        """
    )
    conn.close()


def test_burn_drain_skips_verified_stale_enrichment_and_applies_real_update(tmp_path):
    from brainlayer.drain import burn_drain_once

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    log_path = tmp_path / "burn.log"
    _create_burn_drain_db(db_path)
    stale = enqueue_enrichment_updates(
        [
            {
                "chunk_id": "already-done",
                "content_hash": "h1",
                "enrichment": {"summary": "redundant summary"},
            }
        ],
        queue_dir=queue_dir,
    )
    real = enqueue_enrichment_updates(
        [
            {
                "chunk_id": "needs-update",
                "content_hash": "h2",
                "enrichment": {"summary": "real summary"},
            }
        ],
        queue_dir=queue_dir,
    )

    result = burn_drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=10, log_path=log_path)

    assert result.applied_events == 1
    assert result.skipped_verified_stale == 1
    assert result.files_deleted == 2
    assert result.checkpoints == 1
    assert not stale.exists()
    assert not real.exists()
    conn = apsw.Connection(str(db_path))
    try:
        rows = dict(conn.execute("SELECT id, summary FROM chunks ORDER BY id"))
    finally:
        conn.close()
    assert rows == {"already-done": "old summary", "needs-update": "real summary"}


def test_burn_drain_applies_same_hash_event_when_provenance_state_missing(tmp_path):
    from brainlayer.drain import burn_drain_once

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    log_path = tmp_path / "burn.log"
    conn = apsw.Connection(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT,
            summary TEXT,
            enriched_at TEXT,
            enrich_status TEXT,
            content_hash TEXT,
            provenance_class TEXT,
            raw_entities_json TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO chunks (id, content, summary, enriched_at, enrich_status, content_hash, provenance_class)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("already-done", "content 1", "old summary", "2026-05-30T00:00:00Z", "success", "h1", None),
    )
    conn.close()
    queued = enqueue_enrichment_updates(
        [
            {
                "chunk_id": "already-done",
                "content_hash": "h1",
                "enrichment": {"summary": "redundant summary"},
                "entities": [{"name": "controlLayer"}],
                "provenance_class": "RAW-ETAN-DIRECT",
            }
        ],
        queue_dir=queue_dir,
    )

    result = burn_drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=10, log_path=log_path)

    assert result.applied_events == 1
    assert result.skipped_verified_stale == 0
    assert not queued.exists()
    conn = apsw.Connection(str(db_path))
    try:
        row = conn.execute(
            "SELECT summary, provenance_class, raw_entities_json FROM chunks WHERE id = ?",
            ("already-done",),
        ).fetchone()
        queued_entity = conn.execute("SELECT entity, chunk_id, reason FROM provenance_resolve_queue").fetchone()
    finally:
        conn.close()
    assert row[0] == "redundant summary"
    assert row[1] == "RAW-ETAN-DIRECT"
    assert json.loads(row[2]) == [{"name": "controlLayer"}]
    assert queued_entity == ("controlLayer", "already-done", "enrichment")


def test_burn_drain_redundant_enrichment_preserves_provenance_enqueue(tmp_path):
    from brainlayer.drain import burn_drain_once

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    log_path = tmp_path / "burn.log"
    entities = [{"name": "controlLayer"}]
    conn = apsw.Connection(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT,
            summary TEXT,
            enriched_at TEXT,
            enrich_status TEXT,
            content_hash TEXT,
            provenance_class TEXT,
            raw_entities_json TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO chunks (
            id, content, summary, enriched_at, enrich_status,
            content_hash, provenance_class, raw_entities_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "already-done",
            "content 1",
            "redundant summary",
            "2026-05-30T00:00:00Z",
            "success",
            "h1",
            "RAW-ETAN-DIRECT",
            json.dumps(entities),
        ),
    )
    conn.close()
    queued = enqueue_enrichment_updates(
        [
            {
                "chunk_id": "already-done",
                "content_hash": "h1",
                "enrichment": {"summary": "redundant summary"},
                "entities": entities,
                "provenance_class": "RAW-ETAN-DIRECT",
            }
        ],
        queue_dir=queue_dir,
    )

    result = burn_drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=10, log_path=log_path)

    assert result.applied_events == 0
    assert result.skipped_verified_stale == 1
    assert not queued.exists()
    conn = apsw.Connection(str(db_path))
    try:
        queued_entity = conn.execute("SELECT entity, chunk_id, reason FROM provenance_resolve_queue").fetchone()
    finally:
        conn.close()
    assert queued_entity == ("controlLayer", "already-done", "enrichment")


def test_burn_drain_default_write_batch_is_bounded(tmp_path):
    from brainlayer.drain import burn_drain_once

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    log_path = tmp_path / "burn.log"
    _create_burn_drain_db(db_path)

    conn = apsw.Connection(str(db_path))
    try:
        with conn:
            for idx in range(150):
                conn.execute(
                    "INSERT INTO chunks (id, content, content_hash) VALUES (?, ?, ?)",
                    (f"burst-{idx}", f"content {idx}", f"hash-{idx}"),
                )
    finally:
        conn.close()

    for idx in range(150):
        enqueue_enrichment_updates(
            [
                {
                    "chunk_id": f"burst-{idx}",
                    "content_hash": f"hash-{idx}",
                    "enrichment": {"summary": f"summary {idx}"},
                }
            ],
            queue_dir=queue_dir,
        )

    result = burn_drain_once(db_path=db_path, queue_dir=queue_dir, log_path=log_path)

    assert result.files_deleted <= 100
    assert result.applied_events <= 100
    assert len(list(queue_dir.glob("*.jsonl"))) >= 50


def test_readonly_vector_store_reads_during_burn_drain_writer_burst(tmp_path, monkeypatch):
    from brainlayer import drain
    from brainlayer.vector_store import VectorStore

    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    monkeypatch.setenv("BRAINLAYER_READ_BUSY_TIMEOUT_MS", "250")

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    log_path = tmp_path / "burn.log"
    _create_burn_drain_db(db_path)

    conn = apsw.Connection(str(db_path))
    try:
        with conn:
            for idx in range(25):
                conn.execute(
                    "INSERT INTO chunks (id, content, content_hash) VALUES (?, ?, ?)",
                    (f"burst-{idx}", f"content {idx}", f"hash-{idx}"),
                )
    finally:
        conn.close()

    for idx in range(25):
        enqueue_enrichment_updates(
            [
                {
                    "chunk_id": f"burst-{idx}",
                    "content_hash": f"hash-{idx}",
                    "enrichment": {"summary": f"summary {idx}"},
                }
            ],
            queue_dir=queue_dir,
        )

    entered_transaction = threading.Event()
    release_transaction = threading.Event()
    drain_result = []
    drain_errors = []
    real_apply_event = drain._apply_event

    def wait_inside_writer_transaction(conn, event):
        entered_transaction.set()
        release_transaction.wait(timeout=2)
        return real_apply_event(conn, event)

    monkeypatch.setattr(drain, "_apply_event", wait_inside_writer_transaction)

    def run_drain():
        try:
            drain_result.append(
                drain.burn_drain_once(
                    db_path=db_path,
                    queue_dir=queue_dir,
                    log_path=log_path,
                    max_events_per_transaction=25,
                )
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            drain_errors.append(exc)

    thread = threading.Thread(target=run_drain)
    thread.start()
    try:
        assert entered_transaction.wait(timeout=2), "drain did not enter its write transaction"

        started_at = time.monotonic()
        store = VectorStore(db_path, readonly=True)
        try:
            assert store.count() == 27
        finally:
            store.close()
        elapsed = time.monotonic() - started_at

        assert elapsed < 0.5
    finally:
        release_transaction.set()
        thread.join(timeout=3)

    assert not thread.is_alive()
    if drain_errors:
        raise drain_errors[0]
    assert drain_result[0].applied_events == 25


def test_burn_drain_preserves_queue_file_when_batch_rolls_back(tmp_path, monkeypatch):
    from brainlayer import drain

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    log_path = tmp_path / "burn.log"
    _create_burn_drain_db(db_path)
    queued = enqueue_enrichment_updates(
        [
            {
                "chunk_id": "needs-update",
                "content_hash": "h2",
                "enrichment": {"summary": "must not commit"},
            }
        ],
        queue_dir=queue_dir,
    )

    def fail_apply(*_args, **_kwargs):
        raise RuntimeError("synthetic batch failure")

    monkeypatch.setattr(drain, "_apply_event", fail_apply)

    result = drain.burn_drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=10, log_path=log_path)

    assert result.failed_files == 1
    assert result.files_deleted == 0
    assert queued.exists()
    conn = apsw.Connection(str(db_path))
    try:
        summary = conn.execute("SELECT summary FROM chunks WHERE id = 'needs-update'").fetchone()[0]
    finally:
        conn.close()
    assert summary is None


def test_burn_drain_rolls_back_provenance_enqueue_with_batch(tmp_path, monkeypatch):
    from brainlayer import drain

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    log_path = tmp_path / "burn.log"
    _create_burn_drain_db(db_path)
    queued = enqueue_enrichment_updates(
        [
            {
                "chunk_id": "needs-update",
                "content_hash": "h2",
                "enrichment": {"summary": "must roll back"},
                "entities": [{"name": "controlLayer"}],
            }
        ],
        queue_dir=queue_dir,
    )

    monkeypatch.setattr(drain, "_embedding_enabled", lambda: True)

    def fail_after_events(*_args, **_kwargs):
        raise RuntimeError("synthetic post-event batch failure")

    monkeypatch.setattr(drain, "_embed_store_chunks", fail_after_events)

    result = drain.burn_drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=10, log_path=log_path)

    assert result.failed_files == 1
    assert result.files_deleted == 0
    assert queued.exists()
    conn = apsw.Connection(str(db_path))
    try:
        summary = conn.execute("SELECT summary FROM chunks WHERE id = 'needs-update'").fetchone()[0]
        has_queue_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'provenance_resolve_queue'"
        ).fetchone()
        queue_count = (
            conn.execute("SELECT COUNT(*) FROM provenance_resolve_queue").fetchone()[0] if has_queue_table else 0
        )
    finally:
        conn.close()
    assert summary is None
    assert queue_count == 0


def test_burn_drain_enqueues_provenance_without_helper_commit(tmp_path, monkeypatch):
    from brainlayer import drain

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    log_path = tmp_path / "burn.log"
    _create_burn_drain_db(db_path)
    queued = enqueue_enrichment_updates(
        [
            {
                "chunk_id": "needs-update",
                "content_hash": "h2",
                "enrichment": {"summary": "committed by drain only"},
                "entities": [{"name": "controlLayer"}],
            }
        ],
        queue_dir=queue_dir,
    )
    calls = []

    def record_enqueue(*_args, **kwargs):
        calls.append(kwargs)
        return 1

    monkeypatch.setattr(drain, "enqueue_provenance_resolution_for_entities", record_enqueue)

    result = drain.burn_drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=10, log_path=log_path)

    assert result.applied_events == 1
    assert not queued.exists()
    assert calls == [{"chunk_id": "needs-update", "commit": False}]


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
