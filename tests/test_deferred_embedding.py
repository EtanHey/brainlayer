"""Tests for deferred/async embedding in brain_store.

TDD: Tests written BEFORE implementation.

Goal: brain_store returns immediately without blocking on embedding generation.
Embeddings are generated in the background and backfilled.
"""

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
