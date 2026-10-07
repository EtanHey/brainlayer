"""Local embedding and queue flushing survive retirement of auto-enrichment."""

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from brainlayer.vector_store import VectorStore

# ── enrich_single unit tests ────────────────────────────────────


# ── Integration: _store triggers auto-enrichment ────────────────


class TestStoreAutoEnrich:
    """Local embedding and queue flushing survive retirement of auto-enrichment."""

    @pytest.fixture(autouse=True)
    def _isolate_store_paths(self, tmp_path, monkeypatch):
        """Keep shared queue pressure and legacy flushes out of these store tests."""
        monkeypatch.setenv("GOOGLE_API_KEY", "synthetic-retirement-key")
        monkeypatch.setenv("GROQ_API_KEY", "synthetic-retirement-key")
        monkeypatch.setenv("BRAINLAYER_AUTO_ENRICH", "1")
        queue_dir = tmp_path / "queue"
        queue_dir.mkdir()
        monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue_dir))
        monkeypatch.setenv("BRAINLAYER_DB", str(tmp_path / "test.db"))

    @pytest.mark.asyncio
    async def test_store_embeds_and_flushes_without_enrich_single(self, tmp_path, monkeypatch):
        """The owned thread embeds and flushes without invoking a model producer."""
        from brainlayer.mcp import store_handler

        enriched_ids = []
        flushed = []
        original_flush = store_handler._flush_pending_stores

        def tracking_flush(bg_store, embed_fn):
            flushed.append(bg_store.db_path)
            return original_flush(bg_store, embed_fn)

        monkeypatch.setattr(store_handler, "_flush_pending_stores", tracking_flush)

        def mock_enrich_single(bg_store, cid):
            enriched_ids.append(cid)
            return {"summary": "enriched"}

        monkeypatch.setattr(
            "brainlayer.enrichment_controller.enrich_single",
            mock_enrich_single,
        )

        db_path = tmp_path / "test.db"
        test_store = VectorStore(db_path)

        monkeypatch.setattr(store_handler, "_get_vector_store", lambda: test_store)

        mock_model = MagicMock()
        mock_model.embed_query = lambda text: [0.1] * 1024
        monkeypatch.setattr(store_handler, "_get_embedding_model", lambda: mock_model)

        started_threads = []
        real_thread = threading.Thread

        def tracking_thread(*args, **kwargs):
            thread = real_thread(*args, **kwargs)
            started_threads.append(thread)
            return thread

        monkeypatch.setattr(store_handler, "threading", SimpleNamespace(Thread=tracking_thread))

        result = await store_handler._store_new(
            content="Test auto-enrichment integration",
            memory_type="learning",
            project="test",
        )

        content_items, structured = result
        chunk_id = structured["chunk_id"]

        assert len(started_threads) == 1
        started_threads[0].join(timeout=5.0)
        assert not started_threads[0].is_alive()

        assert enriched_ids == []
        assert len(flushed) == 1
        assert Path(flushed[0]) == db_path
        assert test_store.conn.execute("SELECT 1 FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,)).fetchone()
        test_store.close()

    @pytest.mark.asyncio
    async def test_store_receipt_and_embedding_do_not_require_enrichment(self, tmp_path, monkeypatch):
        """The stored receipt and local vector do not depend on a cloud producer."""
        from brainlayer.mcp import store_handler

        calls = []

        def mock_enrich_single(bg_store, cid):
            calls.append(cid)
            raise RuntimeError("Gemini exploded")

        monkeypatch.setattr(
            "brainlayer.enrichment_controller.enrich_single",
            mock_enrich_single,
        )

        db_path = tmp_path / "test.db"
        test_store = VectorStore(db_path)

        monkeypatch.setattr(store_handler, "_get_vector_store", lambda: test_store)

        mock_model = MagicMock()
        mock_model.embed_query = lambda text: [0.1] * 1024
        monkeypatch.setattr(store_handler, "_get_embedding_model", lambda: mock_model)

        started_threads = []
        real_thread = threading.Thread

        def tracking_thread(*args, **kwargs):
            thread = real_thread(*args, **kwargs)
            started_threads.append(thread)
            return thread

        monkeypatch.setattr(store_handler, "threading", SimpleNamespace(Thread=tracking_thread))

        result = await store_handler._store_new(
            content="Store should succeed regardless of enrichment",
            memory_type="note",
        )

        content_items, structured = result
        assert structured["chunk_id"] != "queued"
        assert "STORED" in content_items[0].text
        assert structured["chunk_id"] in content_items[0].text

        assert len(started_threads) == 1
        started_threads[0].join(timeout=5.0)
        assert not started_threads[0].is_alive()

        assert calls == []
        assert test_store.conn.execute(
            "SELECT 1 FROM chunk_vectors WHERE chunk_id = ?", (structured["chunk_id"],)
        ).fetchone()

        test_store.close()


# ── Environment variable opt-out ────────────────────────────────
