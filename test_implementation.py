"""Test the new sqlite-vec implementation."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

from zikaron.vector_store import VectorStore
from zikaron.embeddings import EmbeddingModel
from zikaron.pipeline.chunk import Chunk, ContentType, Value


def test_vector_store():
    """Test basic vector store operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        # Create test data
        chunks = [
            {
                "id": "test:0",
                "content": "This is a test document about Python programming.",
                "metadata": {"test": "value"},
                "source_file": "test.jsonl",
                "project": "test_project",
                "content_type": "code",
                "value_type": "high",
                "char_count": 45
            }
        ]
        
        embeddings = [[0.1] * 384]  # Mock 384-dim embedding
        
        # Test upsert
        with VectorStore(db_path) as store:
            count = store.upsert_chunks(chunks, embeddings)
            assert count == 1
            
            # Test count
            assert store.count() == 1
            
            # Test search by embedding
            results = store.search(
                query_embedding=[0.1] * 384,
                n_results=1
            )
            
            assert len(results["documents"][0]) == 1
            assert "Python programming" in results["documents"][0][0]
            
            # Test text search
            results = store.search(
                query_text="Python",
                n_results=1
            )
            
            assert len(results["documents"][0]) == 1
            assert "Python programming" in results["documents"][0][0]
            
            # Test stats
            stats = store.get_stats()
            assert stats["total_chunks"] == 1
            assert "test_project" in stats["projects"]


def test_embeddings():
    """Test embedding model."""
    # Create mock chunks
    chunk = Chunk(
        content="Test content for embedding",
        content_type=ContentType.CODE,
        value=Value.HIGH,
        char_count=27,
        metadata={}
    )
    
    model = EmbeddingModel()
    
    # Test chunk embedding
    embedded = model.embed_chunks([chunk])
    assert len(embedded) == 1
    assert len(embedded[0].embedding) == 384  # bge-small-en-v1.5 dimension
    
    # Test query embedding
    query_emb = model.embed_query("test query")
    assert len(query_emb) == 384


if __name__ == "__main__":
    print("Testing vector store...")
    test_vector_store()
    print("✓ Vector store test passed")
    
    print("Testing embeddings...")
    test_embeddings()
    print("✓ Embeddings test passed")
    
    print("All tests passed!")
