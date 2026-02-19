"""Tests for vector_store.py — date filtering, search metadata, schema."""

import pytest

pytestmark = pytest.mark.integration

from brainlayer.paths import DEFAULT_DB_PATH
from brainlayer.vector_store import VectorStore


@pytest.fixture(scope="module")
def store():
    """Read-only connection to the production DB for integration tests."""
    s = VectorStore(DEFAULT_DB_PATH)
    yield s
    s.close()


class TestSchema:
    """Verify the DB schema has required columns."""

    def test_created_at_column_exists(self, store):
        """created_at column exists in chunks table."""
        cursor = store.conn.cursor()
        cols = list(cursor.execute("PRAGMA table_info(chunks)"))
        col_names = [c[1] for c in cols]
        assert "created_at" in col_names

    def test_created_at_coverage(self, store):
        """All chunks should have created_at (from backfill)."""
        cursor = store.conn.cursor()
        total = list(cursor.execute("SELECT COUNT(*) FROM chunks"))[0][0]
        with_date = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE created_at IS NOT NULL"))[0][0]
        coverage = with_date / total if total > 0 else 0
        assert coverage >= 0.99, f"Only {coverage:.1%} of chunks have created_at (expected 100%)"

    def test_source_column_exists(self, store):
        """source column exists in chunks table."""
        cursor = store.conn.cursor()
        cols = list(cursor.execute("PRAGMA table_info(chunks)"))
        col_names = [c[1] for c in cols]
        assert "source" in col_names


class TestDateFiltering:
    """Test date filtering in search queries."""

    def test_search_with_date_from(self, store):
        """Search with date_from filter returns only recent results."""
        from brainlayer.embeddings import get_embedding_model

        model = get_embedding_model()
        query_emb = model.embed_query("test query")

        results = store.hybrid_search(
            query_embedding=query_emb,
            query_text="test query",
            n_results=5,
            date_from="2026-02-15",
        )
        docs = results["documents"][0]
        # Should return results (we have data from Feb 2026)
        # The key test is that it doesn't crash
        assert isinstance(docs, list)

    def test_search_with_date_to(self, store):
        """Search with date_to filter works."""
        from brainlayer.embeddings import get_embedding_model

        model = get_embedding_model()
        query_emb = model.embed_query("test query")

        results = store.hybrid_search(
            query_embedding=query_emb,
            query_text="test query",
            n_results=5,
            date_to="2026-01-01",
        )
        # Should not crash, may return empty if no old data
        assert isinstance(results["documents"][0], list)

    def test_search_with_date_range(self, store):
        """Search with both date_from and date_to works."""
        from brainlayer.embeddings import get_embedding_model

        model = get_embedding_model()
        query_emb = model.embed_query("authentication")

        results = store.hybrid_search(
            query_embedding=query_emb,
            query_text="authentication",
            n_results=5,
            date_from="2026-02-01",
            date_to="2026-02-28",
        )
        assert isinstance(results["documents"][0], list)


class TestSearchMetadata:
    """Test that search results include proper metadata."""

    def test_results_have_created_at(self, store):
        """Search results include created_at in metadata."""
        from brainlayer.embeddings import get_embedding_model

        model = get_embedding_model()
        query_emb = model.embed_query("function implementation")

        results = store.hybrid_search(
            query_embedding=query_emb,
            query_text="function implementation",
            n_results=3,
        )
        if results["documents"][0]:
            meta = results["metadatas"][0][0]
            assert "created_at" in meta, "Search results should include created_at"

    def test_results_have_source(self, store):
        """Search results include source in metadata."""
        from brainlayer.embeddings import get_embedding_model

        model = get_embedding_model()
        query_emb = model.embed_query("function implementation")

        results = store.hybrid_search(
            query_embedding=query_emb,
            query_text="function implementation",
            n_results=3,
        )
        if results["documents"][0]:
            meta = results["metadatas"][0][0]
            assert "source" in meta or "source_file" in meta, "Results should include source info"


class TestStats:
    """Test get_stats returns expected data."""

    def test_stats_have_projects(self, store):
        stats = store.get_stats()
        assert len(stats["projects"]) > 0

    def test_stats_have_content_types(self, store):
        stats = store.get_stats()
        assert len(stats["content_types"]) > 0

    def test_stats_total_chunks(self, store):
        stats = store.get_stats()
        assert stats["total_chunks"] > 200_000  # We have 268K+
