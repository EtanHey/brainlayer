"""Tests for deferred/async embedding in brain_store.

TDD: Tests written BEFORE implementation.

Goal: brain_store returns immediately without blocking on embedding generation.
Embeddings are generated in the background and backfilled.
"""

import threading
import time

import pytest

from brainlayer.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    """Create a fresh VectorStore for testing."""
    db_path = tmp_path / "test.db"
    s = VectorStore(db_path)
    yield s
    s.close()


@pytest.fixture
def slow_embed():
    """Embedding function that takes 2 seconds — simulates real model."""

    def _embed(text: str) -> list[float]:
        time.sleep(2)
        seed = sum(ord(c) for c in text[:50]) % 100
        return [float(seed + i) / 1000.0 for i in range(1024)]

    return _embed


@pytest.fixture
def fast_embed():
    """Instant embedding function for background processing tests."""

    def _embed(text: str) -> list[float]:
        seed = sum(ord(c) for c in text[:50]) % 100
        return [float(seed + i) / 1000.0 for i in range(1024)]

    return _embed


def _insert_pending_chunk(
    store: VectorStore,
    *,
    chunk_id: str,
    source: str,
    content: str | None = None,
    created_at: str = "2026-06-14T00:00:00+00:00",
) -> None:
    cursor = store.conn.cursor()
    text = content or f"Pending content for {chunk_id}"
    cursor.execute(
        """INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type,
            value_type, char_count, source, created_at, enriched_at, enrich_status,
            summary, tags, importance, chunk_origin, seen_count, last_seen_at,
            content_class
        ) VALUES (?, ?, '{}', 'test.jsonl', 'test', 'note',
            'HIGH', ?, ?, ?, NULL, NULL,
            NULL, NULL, NULL, 'raw', 1, ?,
            'human-authored')""",
        (chunk_id, text, len(text), source, created_at, created_at),
    )


def _has_vector(store: VectorStore, chunk_id: str) -> bool:
    cursor = store.conn.cursor()
    return (
        cursor.execute(
            "SELECT COUNT(*) FROM chunk_vectors WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()[0]
        == 1
    )


def _pending_active_count(store: VectorStore) -> int:
    cursor = store.conn.cursor()
    return cursor.execute(
        """
        SELECT COUNT(*)
        FROM chunks c
        LEFT JOIN chunk_vectors v ON c.id = v.chunk_id
        WHERE v.chunk_id IS NULL
          AND c.content IS NOT NULL
          AND c.content != ''
          AND c.archived_at IS NULL
          AND c.superseded_by IS NULL
          AND c.aggregated_into IS NULL
          AND COALESCE(c.archived, 0) = 0
          AND COALESCE(c.status, 'active') = 'active'
        """
    ).fetchone()[0]


class TestDeferredStore:
    """store_memory with embed_fn=None stores without embedding."""

    def test_store_without_embed_fn_returns_id(self, store):
        """Storing with embed_fn=None still returns a valid chunk ID."""
        from brainlayer.store import store_memory

        result = store_memory(
            store=store,
            embed_fn=None,
            content="Deferred embedding test",
            memory_type="note",
            project="test",
        )
        assert result["id"] is not None
        assert result["id"].startswith("manual-")

    def test_store_without_embed_fn_persists_content(self, store):
        """Content is persisted even without embedding."""
        from brainlayer.store import store_memory

        result = store_memory(
            store=store,
            embed_fn=None,
            content="This content should be in the DB",
            memory_type="learning",
            project="test",
        )
        cursor = store.conn.cursor()
        rows = list(cursor.execute("SELECT content FROM chunks WHERE id = ?", (result["id"],)))
        assert rows[0][0] == "This content should be in the DB"

    def test_store_without_embed_fn_has_no_vector(self, store):
        """Chunk stored without embed_fn has no entry in chunk_vectors."""
        from brainlayer.store import store_memory

        result = store_memory(
            store=store,
            embed_fn=None,
            content="No vector yet",
            memory_type="note",
            project="test",
        )
        cursor = store.conn.cursor()
        rows = list(cursor.execute("SELECT COUNT(*) FROM chunk_vectors WHERE chunk_id = ?", (result["id"],)))
        assert rows[0][0] == 0

    def test_store_without_embed_fn_is_fast(self, store, slow_embed):
        """Storing without embed_fn is much faster than with embed_fn."""
        from brainlayer.store import store_memory

        start = time.monotonic()
        store_memory(
            store=store,
            embed_fn=None,
            content="This should be instant",
            memory_type="note",
            project="test",
        )
        elapsed = time.monotonic() - start
        assert elapsed < 0.1, f"store_memory without embed_fn took {elapsed:.3f}s, expected <0.1s"

    def test_store_with_embed_fn_still_works(self, store, fast_embed):
        """Backward compat: passing embed_fn still embeds synchronously."""
        from brainlayer.store import store_memory

        result = store_memory(
            store=store,
            embed_fn=fast_embed,
            content="Sync embedding still works",
            memory_type="decision",
            project="test",
        )
        cursor = store.conn.cursor()
        rows = list(cursor.execute("SELECT COUNT(*) FROM chunk_vectors WHERE chunk_id = ?", (result["id"],)))
        assert rows[0][0] == 1


class TestBackgroundEmbedder:
    """Background embedder backfills NULL embeddings."""

    def test_embed_pending_chunks(self, store, fast_embed):
        """embed_pending_chunks fills in embeddings for un-embedded chunks."""
        from brainlayer.store import embed_pending_chunks, store_memory

        # Store 3 chunks without embeddings
        ids = []
        for i in range(3):
            result = store_memory(
                store=store,
                embed_fn=None,
                content=f"Pending chunk {i}",
                memory_type="note",
                project="test",
            )
            ids.append(result["id"])

        # Verify none have embeddings
        cursor = store.conn.cursor()
        for cid in ids:
            rows = list(cursor.execute("SELECT COUNT(*) FROM chunk_vectors WHERE chunk_id = ?", (cid,)))
            assert rows[0][0] == 0

        # Run background embedder
        count = embed_pending_chunks(store=store, embed_fn=fast_embed)
        assert count == 3

        # Verify all now have embeddings
        for cid in ids:
            rows = list(cursor.execute("SELECT COUNT(*) FROM chunk_vectors WHERE chunk_id = ?", (cid,)))
            assert rows[0][0] == 1, f"Chunk {cid} still missing embedding"

    def test_embed_pending_skips_already_embedded(self, store, fast_embed):
        """embed_pending_chunks doesn't re-embed chunks that already have vectors."""
        from brainlayer.store import embed_pending_chunks, store_memory

        # Store with embedding
        store_memory(
            store=store,
            embed_fn=fast_embed,
            content="Already embedded",
            memory_type="note",
            project="test",
        )
        # Store without embedding
        store_memory(
            store=store,
            embed_fn=None,
            content="Needs embedding",
            memory_type="note",
            project="test",
        )

        count = embed_pending_chunks(store=store, embed_fn=fast_embed)
        assert count == 1  # Only the un-embedded one

    def test_embed_pending_idempotent(self, store, fast_embed):
        """Running embed_pending_chunks twice doesn't double-embed."""
        from brainlayer.store import embed_pending_chunks, store_memory

        store_memory(
            store=store,
            embed_fn=None,
            content="Idempotent test",
            memory_type="note",
            project="test",
        )

        first_count = embed_pending_chunks(store=store, embed_fn=fast_embed)
        assert first_count == 1

        second_count = embed_pending_chunks(store=store, embed_fn=fast_embed)
        assert second_count == 0

    def test_embed_pending_respects_batch_size(self, store, fast_embed):
        """embed_pending_chunks processes in batches."""
        from brainlayer.store import embed_pending_chunks, store_memory

        for i in range(10):
            store_memory(
                store=store,
                embed_fn=None,
                content=f"Batch test chunk {i}",
                memory_type="note",
                project="test",
            )

        count = embed_pending_chunks(store=store, embed_fn=fast_embed, batch_size=3)
        assert count == 3  # Only processes batch_size items

    def test_embed_pending_chunks_embeds_all_active_sources(self, store, fast_embed):
        """embed_pending_chunks self-heals active unvectored chunks regardless of source."""
        from brainlayer.store import embed_pending_chunks

        sources = ["manual", "mcp", "claude_code", "realtime_watcher", "youtube", "whatsapp"]
        for source in sources:
            _insert_pending_chunk(store, chunk_id=f"{source}-pending", source=source)

        count = embed_pending_chunks(store=store, embed_fn=fast_embed, batch_size=20)

        assert count == len(sources)
        cursor = store.conn.cursor()
        for source in sources:
            vector_count = cursor.execute(
                "SELECT COUNT(*) FROM chunk_vectors WHERE chunk_id = ?",
                (f"{source}-pending",),
            ).fetchone()[0]
            assert vector_count == 1, f"{source} chunk was not embedded"

    def test_embed_pending_chunks_skips_inactive_lifecycle_rows(self, store, fast_embed):
        """Archived, superseded, and aggregated chunks are intentional exclusions."""
        from brainlayer.store import embed_pending_chunks

        _insert_pending_chunk(store, chunk_id="active-pending", source="claude_code")
        _insert_pending_chunk(store, chunk_id="archived-pending", source="claude_code")
        _insert_pending_chunk(store, chunk_id="superseded-pending", source="claude_code")
        _insert_pending_chunk(store, chunk_id="aggregated-pending", source="claude_code")
        _insert_pending_chunk(store, chunk_id="archived-flag-pending", source="claude_code")
        _insert_pending_chunk(store, chunk_id="inactive-pending", source="claude_code")

        cursor = store.conn.cursor()
        cursor.execute("UPDATE chunks SET archived_at = '2026-06-14T00:00:00+00:00' WHERE id = 'archived-pending'")
        cursor.execute("UPDATE chunks SET superseded_by = 'replacement' WHERE id = 'superseded-pending'")
        cursor.execute("UPDATE chunks SET aggregated_into = 'aggregate' WHERE id = 'aggregated-pending'")
        cursor.execute("UPDATE chunks SET archived = 1 WHERE id = 'archived-flag-pending'")
        cursor.execute("UPDATE chunks SET status = 'inactive' WHERE id = 'inactive-pending'")

        count = embed_pending_chunks(store=store, embed_fn=fast_embed, batch_size=20)

        assert count == 1
        embedded_ids = {
            row[0]
            for row in cursor.execute(
                "SELECT chunk_id FROM chunk_vectors WHERE chunk_id LIKE '%-pending' ORDER BY chunk_id"
            )
        }
        assert embedded_ids == {"active-pending"}

    def test_embed_pending_chunks_uses_batch_embedder_when_available(self, store):
        """Batch-capable callers can avoid per-row model.encode calls."""
        from brainlayer.store import embed_pending_chunks

        for i in range(4):
            _insert_pending_chunk(store, chunk_id=f"batch-pending-{i}", source="realtime_watcher")

        single_calls = []
        batch_calls = []

        def single_embed(text: str) -> list[float]:
            single_calls.append(text)
            return [0.0] * 1024

        def batch_embed(texts: list[str]) -> list[list[float]]:
            batch_calls.append(list(texts))
            return [[float(i + 1)] * 1024 for i, _ in enumerate(texts)]

        count = embed_pending_chunks(
            store=store,
            embed_fn=single_embed,
            batch_size=4,
            embed_batch_fn=batch_embed,
        )

        assert count == 4
        assert single_calls == []
        assert len(batch_calls) == 1
        assert len(batch_calls[0]) == 4

    def test_embed_pending_chunks_selects_pending_rows_from_rowid_support_table(self):
        """Backlog selection must avoid scanning the vec0 virtual table."""
        from brainlayer.store import embed_pending_chunks

        class FakeCursor:
            def __init__(self):
                self.queries = []
                self.params = []

            def execute(self, sql, params=()):
                self.queries.append(sql)
                self.params.append(params)
                if "SELECT c.id, c.content FROM chunks c" in sql:
                    return [("pending-rowid", "Needs vector")]
                raise AssertionError(f"unexpected SQL: {sql}")

        class FakeConn:
            def __init__(self, cursor):
                self._cursor = cursor

            def cursor(self):
                return self._cursor

        class FakeStore:
            db_path = None

            def __init__(self):
                self.cursor = FakeCursor()
                self.conn = FakeConn(self.cursor)
                self.upserts = []

            def _upsert_chunk_vector(self, cursor, chunk_id, embedding):
                self.upserts.append((cursor, chunk_id, embedding))

        fake_store = FakeStore()
        count = embed_pending_chunks(
            store=fake_store,
            embed_fn=lambda _text: [1.0] * 1024,
            batch_size=7,
        )

        selection_sql = fake_store.cursor.queries[0]
        assert count == 1
        assert "chunk_vectors_rowids r ON c.id = r.id" in selection_sql
        assert "LEFT JOIN chunk_vectors v" not in selection_sql
        assert fake_store.cursor.params[0] == (7,)
        assert fake_store.upserts == [(fake_store.cursor, "pending-rowid", [1.0] * 1024)]

    def test_g1_fifo_reproduction_new_chunk_waits_behind_old_backlog(self, store, fast_embed):
        """G1 RED proof: oldest-first drain alone starves a just-stored chunk for one pass."""
        from brainlayer.store import embed_pending_chunks, store_memory

        for i in range(60):
            _insert_pending_chunk(
                store,
                chunk_id=f"old-backlog-{i:03d}",
                source="claude_code",
                created_at=f"2026-06-13T00:{i:02d}:00+00:00",
            )

        result = store_memory(
            store=store,
            embed_fn=None,
            content="fresh hot lane candidate should not wait",
            memory_type="note",
            project="test",
            created_at="2026-06-14T00:00:00+00:00",
        )

        count = embed_pending_chunks(store=store, embed_fn=fast_embed, batch_size=50)

        assert count == 50
        assert not _has_vector(store, result["id"])
        assert all(_has_vector(store, f"old-backlog-{i:03d}") for i in range(50))

    def test_embed_hot_chunk_embeds_new_chunk_before_oldest_drain(self, store, fast_embed):
        """G2: hot lane fast-tracks the just-stored chunk without consuming drain slots."""
        from brainlayer.store import embed_hot_chunk, embed_pending_chunks, store_memory

        def lane_embed(text: str) -> list[float]:
            if "g2 hot chunk" in text:
                return [1.0] + [0.0] * 1023
            return [0.0, 1.0] + [0.0] * 1022

        for i in range(60):
            _insert_pending_chunk(
                store,
                chunk_id=f"g2-old-backlog-{i:03d}",
                source="claude_code",
                created_at=f"2026-06-13T00:{i:02d}:00+00:00",
            )

        result = store_memory(
            store=store,
            embed_fn=None,
            content="g2 hot chunk vector-only proof",
            memory_type="note",
            project="test",
            created_at="2026-06-14T00:00:00+00:00",
        )

        assert embed_hot_chunk(store=store, embed_fn=lane_embed, chunk_id=result["id"]) is True
        assert _has_vector(store, result["id"])

        drain_count = embed_pending_chunks(store=store, embed_fn=lane_embed, batch_size=50)

        assert drain_count == 50
        assert _has_vector(store, result["id"])
        assert all(_has_vector(store, f"g2-old-backlog-{i:03d}") for i in range(50))

        results = store.hybrid_search(
            query_embedding=lane_embed("g2 hot chunk vector-only proof"),
            query_text="no_keyword_match_for_g2_hot_lane",
            n_results=1,
            project_filter="test",
        )
        assert results["ids"][0][0] == result["id"]

    def test_embed_hot_chunk_is_idempotent_and_does_not_double_embed(self, store, fast_embed):
        """G2/G3: already-vectorized chunks are skipped instead of embedded again."""
        from brainlayer.store import embed_hot_chunk, store_memory

        calls = []

        def recording_embed(text: str) -> list[float]:
            calls.append(text)
            return fast_embed(text)

        result = store_memory(
            store=store,
            embed_fn=None,
            content="hot idempotency proof",
            memory_type="note",
            project="test",
        )

        assert embed_hot_chunk(store=store, embed_fn=recording_embed, chunk_id=result["id"]) is True
        assert embed_hot_chunk(store=store, embed_fn=recording_embed, chunk_id=result["id"]) is False

        duplicate_count = (
            store.conn.cursor()
            .execute(
                """
            SELECT COUNT(*)
            FROM (
                SELECT chunk_id
                FROM chunk_vectors
                GROUP BY chunk_id
                HAVING COUNT(*) > 1
            )
            """
            )
            .fetchone()[0]
        )
        assert calls == ["hot idempotency proof"]
        assert duplicate_count == 0

    def test_g3_hot_lane_preserves_monotonic_shrink_and_oldest_tail(self, store, fast_embed):
        """G3: hot lane plus FIFO drain shrinks old backlog and clears the oldest tail."""
        from brainlayer.store import embed_hot_chunk, embed_pending_chunks, store_memory

        def lane_embed(text: str) -> list[float]:
            if "g3 fresh hot lane candidate" in text:
                return [1.0] + [0.0] * 1023
            return [0.0, 1.0] + [0.0] * 1022

        for i in range(120):
            _insert_pending_chunk(
                store,
                chunk_id=f"g3-old-backlog-{i:03d}",
                source="claude_code",
                created_at=f"2026-06-13T{i // 60:02d}:{i % 60:02d}:00+00:00",
            )

        oldest_50 = [f"g3-old-backlog-{i:03d}" for i in range(50)]
        pending_counts = [_pending_active_count(store)]

        for i in range(4):
            content = f"g3 fresh hot lane candidate {i}"
            result = store_memory(
                store=store,
                embed_fn=None,
                content=content,
                memory_type="note",
                project="test",
                created_at=f"2026-06-14T00:0{i}:00+00:00",
            )

            assert embed_hot_chunk(store=store, embed_fn=lane_embed, chunk_id=result["id"]) is True
            assert _has_vector(store, result["id"])

            results = store.hybrid_search(
                query_embedding=lane_embed(content),
                query_text=f"no_keyword_match_for_g3_hot_lane_{i}",
                n_results=1,
                project_filter="test",
            )
            assert results["ids"][0][0] == result["id"]

            embed_pending_chunks(store=store, embed_fn=lane_embed, batch_size=30)
            pending_counts.append(_pending_active_count(store))

        assert all(after < before for before, after in zip(pending_counts, pending_counts[1:]))
        assert all(_has_vector(store, chunk_id) for chunk_id in oldest_50)

        duplicate_count = (
            store.conn.cursor()
            .execute(
                """
            SELECT COUNT(*)
            FROM (
                SELECT chunk_id
                FROM chunk_vectors
                GROUP BY chunk_id
                HAVING COUNT(*) > 1
            )
            """
            )
            .fetchone()[0]
        )
        assert duplicate_count == 0

    def test_embed_hot_chunk_failure_is_deferred_safe(self, store):
        """G4: hot embed failure does not remove or strand the stored row."""
        from brainlayer.store import embed_hot_chunk, store_memory

        result = store_memory(
            store=store,
            embed_fn=None,
            content="hot lane fallback remains text searchable",
            memory_type="note",
            project="test",
        )

        def failing_embed(_text: str) -> list[float]:
            raise RuntimeError("model unavailable")

        assert embed_hot_chunk(store=store, embed_fn=failing_embed, chunk_id=result["id"]) is False
        assert not _has_vector(store, result["id"])
        fts_results = store.hybrid_search(
            query_embedding=[0.0] * 1024,
            query_text="hot lane fallback remains text searchable",
            n_results=1,
            project_filter="test",
        )
        assert fts_results["ids"][0][0] == result["id"]

    def test_embed_pending_batch_write_failure_continues_remainder(self, store):
        """CodeRabbit #483: one failed vector write must not abort the rest of the batch."""
        from brainlayer.store import embed_pending_chunks

        for i in range(3):
            _insert_pending_chunk(store, chunk_id=f"batch-write-{i}", source="claude_code")

        original_upsert = store._upsert_chunk_vector

        def flaky_upsert(cursor, chunk_id, embedding):
            if chunk_id == "batch-write-1":
                raise RuntimeError("simulated write failure")
            original_upsert(cursor, chunk_id, embedding)

        store._upsert_chunk_vector = flaky_upsert

        count = embed_pending_chunks(
            store=store,
            embed_fn=lambda _text: [0.0] * 1024,
            batch_size=3,
            embed_batch_fn=lambda texts: [[float(i)] * 1024 for i, _ in enumerate(texts)],
        )

        assert count == 2
        assert _has_vector(store, "batch-write-0")
        assert not _has_vector(store, "batch-write-1")
        assert _has_vector(store, "batch-write-2")

    def test_embed_pending_batch_embed_failure_falls_back_per_row(self, store):
        """Batch embed failure should not let one bad row block newer pending chunks."""
        from brainlayer.store import embed_pending_chunks

        _insert_pending_chunk(store, chunk_id="batch-fallback-good-0", source="claude_code", content="good zero")
        _insert_pending_chunk(store, chunk_id="batch-fallback-bad-1", source="claude_code", content="bad one")
        _insert_pending_chunk(store, chunk_id="batch-fallback-good-2", source="claude_code", content="good two")

        def row_embed(text: str) -> list[float]:
            if "bad" in text:
                raise RuntimeError("simulated bad row")
            return [0.25] * 1024

        count = embed_pending_chunks(
            store=store,
            embed_fn=row_embed,
            batch_size=3,
            embed_batch_fn=lambda _texts: (_ for _ in ()).throw(RuntimeError("simulated batch failure")),
        )

        assert count == 2
        assert _has_vector(store, "batch-fallback-good-0")
        assert not _has_vector(store, "batch-fallback-bad-1")
        assert _has_vector(store, "batch-fallback-good-2")


class TestMCPStoreDeferred:
    """MCP _store handler uses deferred embedding."""

    def test_mcp_store_returns_fast(self):
        """_store_new returns in <200ms even though embedding model is slow."""
        import asyncio
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from brainlayer.mcp.store_handler import _store_new

        class SlowModel:
            def embed_query(self, text):
                time.sleep(2)
                return [0.1] * 1024

        tmp_dir = tempfile.mkdtemp()
        tmp_store = VectorStore(Path(tmp_dir) / "test.db")

        try:
            # Patch at the store_handler module level (where the names are bound)
            with (
                patch("brainlayer.mcp.store_handler._get_vector_store", return_value=tmp_store),
                patch("brainlayer.mcp.store_handler._get_embedding_model", return_value=SlowModel()),
                patch(
                    "brainlayer.mcp.store_handler._get_pending_store_path", return_value=Path(tmp_dir) / "pending.jsonl"
                ),
                patch("brainlayer.mcp.store_handler._normalize_project_name", return_value="test"),
            ):
                loop = asyncio.new_event_loop()
                try:
                    start = time.monotonic()
                    loop.run_until_complete(_store_new(content="Fast store test", memory_type="note", project="test"))
                    elapsed = time.monotonic() - start

                    assert elapsed < 0.5, f"_store_new took {elapsed:.3f}s, expected <0.5s (deferred embedding)"
                finally:
                    loop.close()
        finally:
            # Daemon thread may still be using the connection; give it a moment
            time.sleep(0.1)
            try:
                tmp_store.close()
            except Exception:
                pass  # Daemon thread may still hold the connection

    @pytest.mark.asyncio
    async def test_mcp_background_hot_embeds_new_chunk_and_drains_oldest_backlog(self, tmp_path, monkeypatch):
        """G7: real MCP background thread hot-embeds the stored chunk before FIFO drain."""
        from unittest.mock import patch

        from brainlayer.mcp.store_handler import _store_new

        monkeypatch.delenv("BRAINLAYER_ARBITRATED", raising=False)

        db_path = tmp_path / "test.db"
        tmp_store = VectorStore(db_path)
        for i in range(70):
            _insert_pending_chunk(
                tmp_store,
                chunk_id=f"g7-old-backlog-{i:03d}",
                source="claude_code",
                created_at=f"2026-06-13T00:{i:02d}:00+00:00",
            )

        class FastModel:
            def embed_query(self, text):
                seed = sum(ord(c) for c in text[:50]) % 100
                return [float(seed + i) / 1000.0 for i in range(1024)]

            def embed_texts(self, texts, batch_size=64):
                return [self.embed_query(text) for text in texts]

        try:
            with (
                patch("brainlayer.mcp.store_handler._get_vector_store", return_value=tmp_store),
                patch("brainlayer.mcp.store_handler._get_embedding_model", return_value=FastModel()),
                patch("brainlayer.mcp.store_handler._normalize_project_name", return_value="test"),
                patch("brainlayer.enrichment_controller.enrich_single", return_value={"summary": "skipped"}),
            ):
                _content, structured = await _store_new(
                    content="g7 hot mcp lane searchable",
                    memory_type="note",
                    project="test",
                )

                chunk_id = structured["chunk_id"]
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    drained_old = sum(_has_vector(tmp_store, f"g7-old-backlog-{i:03d}") for i in range(64))
                    if _has_vector(tmp_store, chunk_id) and drained_old == 64:
                        break
                    time.sleep(0.05)

            assert _has_vector(tmp_store, chunk_id)
            assert all(_has_vector(tmp_store, f"g7-old-backlog-{i:03d}") for i in range(64))
            assert not _has_vector(tmp_store, "g7-old-backlog-064")
        finally:
            for thread in threading.enumerate():
                if thread.daemon and thread.name != "MainThread":
                    thread.join(timeout=0.1)
            tmp_store.close()
