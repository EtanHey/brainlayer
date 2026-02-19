"""Test vector store functionality (FTS5, hybrid search, context view)."""

import json
import tempfile
from pathlib import Path

import pytest

from brainlayer.vector_store import VectorStore, serialize_f32


@pytest.fixture
def store(tmp_path):
    """Create a temporary vector store."""
    db_path = tmp_path / "test.db"
    return VectorStore(db_path)


@pytest.fixture
def populated_store(store):
    """Store with sample data for testing."""
    chunks = [
        {
            "id": "chunk-1",
            "content": "How to implement OTP authentication in Python",
            "metadata": {"role": "user"},
            "source_file": "/session/conv1.jsonl",
            "project": "my-project",
            "content_type": "user_message",
            "char_count": 50,
            "source": "claude_code",
        },
        {
            "id": "chunk-2",
            "content": "Here is the OTP implementation using pyotp library",
            "metadata": {"role": "assistant"},
            "source_file": "/session/conv1.jsonl",
            "project": "my-project",
            "content_type": "ai_code",
            "char_count": 55,
            "source": "claude_code",
        },
        {
            "id": "chunk-3",
            "content": "React useEffect cleanup function for websockets",
            "metadata": {"role": "user"},
            "source_file": "/session/conv2.jsonl",
            "project": "app-a",
            "content_type": "user_message",
            "char_count": 50,
            "source": "claude_code",
        },
    ]
    # Use 1024-dim fake embeddings
    embeddings = [[float(i) / 1024] * 1024 for i in range(3)]
    store.upsert_chunks(chunks, embeddings)
    return store


def test_fts5_table_created(store):
    """FTS5 virtual table should exist after init."""
    cursor = store.conn.cursor()
    tables = [row[0] for row in cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )]
    assert "chunks_fts" in tables


def test_fts5_auto_populated_on_insert(populated_store):
    """FTS5 should be populated via triggers when chunks are inserted."""
    cursor = populated_store.conn.cursor()
    fts_count = list(cursor.execute("SELECT COUNT(*) FROM chunks_fts"))[0][0]
    chunk_count = list(cursor.execute("SELECT COUNT(*) FROM chunks"))[0][0]
    assert fts_count == chunk_count
    assert fts_count == 3


def test_fts5_keyword_search(populated_store):
    """FTS5 should find exact keyword matches."""
    cursor = populated_store.conn.cursor()
    results = list(cursor.execute(
        "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH 'OTP' ORDER BY rank"
    ))
    # Both chunk-1 and chunk-2 contain "OTP"
    ids = [r[0] for r in results]
    assert "chunk-1" in ids
    assert "chunk-2" in ids
    assert "chunk-3" not in ids


def test_hybrid_search_returns_results(populated_store):
    """Hybrid search should return results combining semantic + keyword."""
    query_embedding = [0.001] * 1024
    results = populated_store.hybrid_search(
        query_embedding=query_embedding,
        query_text="OTP",
        n_results=5,
    )
    assert "documents" in results
    assert "metadatas" in results
    assert "distances" in results
    assert len(results["documents"][0]) > 0


def test_hybrid_search_respects_project_filter(populated_store):
    """Hybrid search should filter by project."""
    query_embedding = [0.001] * 1024
    results = populated_store.hybrid_search(
        query_embedding=query_embedding,
        query_text="implementation",
        n_results=5,
        project_filter="app-a",
    )
    for meta in results["metadatas"][0]:
        assert meta.get("project") == "app-a"


def test_get_context_with_conversation(populated_store):
    """Context view should return surrounding chunks."""
    cursor = populated_store.conn.cursor()
    # Manually set conversation_id and position
    cursor.execute(
        "UPDATE chunks SET conversation_id = '/session/conv1.jsonl', position = 0 WHERE id = 'chunk-1'"
    )
    cursor.execute(
        "UPDATE chunks SET conversation_id = '/session/conv1.jsonl', position = 1 WHERE id = 'chunk-2'"
    )

    result = populated_store.get_context("chunk-1", before=0, after=5)
    assert result["target"] is not None
    assert result["target"]["id"] == "chunk-1"
    assert len(result["context"]) == 2  # chunk-1 and chunk-2
    # chunk-1 should be marked as target
    target_chunks = [c for c in result["context"] if c.get("is_target")]
    assert len(target_chunks) == 1


def test_get_context_missing_chunk(populated_store):
    """Context view should handle missing chunk gracefully."""
    result = populated_store.get_context("nonexistent-id")
    assert result.get("error") == "Chunk not found"


def test_get_context_no_conversation_id(populated_store):
    """Context view should handle chunks without conversation_id."""
    result = populated_store.get_context("chunk-3")
    assert "error" in result
    assert "no conversation context" in result["error"].lower()


def test_serialize_f32():
    """serialize_f32 should produce correct byte length."""
    vec = [1.0, 2.0, 3.0]
    data = serialize_f32(vec)
    assert len(data) == 12  # 3 floats * 4 bytes
