"""Test dashboard functionality."""

from unittest.mock import Mock, patch

from brainlayer.dashboard.search import BM25, HybridSearchEngine


def test_bm25_basic():
    """Test basic BM25 functionality."""
    bm25 = BM25()
    documents = [
        "python programming language",
        "javascript web development",
        "machine learning python",
    ]

    bm25.fit(documents)

    # Test search
    results = bm25.search("python", n_results=2)
    assert len(results) <= 2
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


def test_bm25_empty():
    """BM25 with no documents returns empty."""
    bm25 = BM25()
    bm25.fit([])
    results = bm25.search("python")
    assert results == []


def test_hybrid_engine_unfitted_fallback():
    """Unfitted engine falls back to semantic-only search."""
    engine = HybridSearchEngine()
    mock_store = Mock()

    # Mock semantic search response from VectorStore
    mock_store.search.return_value = {
        "documents": [["test document"]],
        "metadatas": [[{"project": "test", "content_type": "ai_code"}]],
        "distances": [[0.5]],
    }
    mock_store.get_all_chunks.return_value = []  # Empty = can't fit BM25

    with patch.object(engine, "_embedding_model") as mock_model:
        mock_model.embed_query.return_value = [0.1] * 1024
        engine._embedding_model = mock_model

        results = engine.search(mock_store, "test query")

        assert "documents" in results
        assert "metadatas" in results
        assert "distances" in results


def test_hybrid_engine_none_store():
    """Engine with None store returns empty results."""
    engine = HybridSearchEngine()
    results = engine.search(None, "test query")
    assert results == {"documents": [[]], "metadatas": [[]], "distances": [[]]}
