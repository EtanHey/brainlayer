"""Test dashboard functionality."""

import pytest
from unittest.mock import Mock, patch
from zikaron.dashboard.search import BM25, HybridSearchEngine
from zikaron.dashboard.views import HomeView, MemoryView


def test_bm25_basic():
    """Test basic BM25 functionality."""
    bm25 = BM25()
    documents = [
        "python programming language",
        "javascript web development", 
        "machine learning python"
    ]
    
    bm25.fit(documents)
    
    # Test search
    results = bm25.search("python", n_results=2)
    assert len(results) <= 2
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


def test_home_view():
    """Test home view rendering."""
    stats = {
        "total_chunks": 1000,
        "projects": ["project1", "project2"],
        "content_types": ["code", "chat"]
    }
    
    view = HomeView(stats)
    panel = view.render()
    
    assert panel is not None
    assert "Home" in str(panel)


def test_memory_view():
    """Test memory view rendering."""
    mock_engine = Mock(spec=HybridSearchEngine)
    mock_collection = Mock()
    stats = {"total_chunks": 100, "projects": ["test"], "content_types": ["code"]}
    
    view = MemoryView(mock_engine, mock_collection, stats)
    panel = view.render()
    
    assert panel is not None
    assert "Memory Search" in str(panel)


def test_hybrid_search_fallback():
    """Test hybrid search fallback to semantic search."""
    engine = HybridSearchEngine()
    mock_collection = Mock()
    
    # Mock semantic search response
    mock_collection.query.return_value = {
        "documents": [["test document"]],
        "metadatas": [["test": "meta"]],
        "distances": [[0.5]]
    }
    
    with patch('zikaron.dashboard.search.embed_query') as mock_embed:
        mock_embed.return_value = [0.1] * 384  # Mock embedding
        
        results = engine.search(mock_collection, "test query")
        
        assert "documents" in results
        assert "metadatas" in results
        assert "distances" in results
