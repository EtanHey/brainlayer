"""Tests for search gaps F-J.

F: content_type + language indexes exist
G: tag_filter uses chunk_tags index (not json_each)
H: FTS5 AND mode returns intersection for multi-word queries
I: backward-compat brainlayer_search alias passes all brain_search params
J: compact results include importance field
"""

import json
import struct
import uuid
from pathlib import Path

import pytest


def _make_store(tmp_path: Path):
    """Create a fresh VectorStore with test data."""
    from brainlayer.vector_store import VectorStore

    db_path = tmp_path / "test.db"
    store = VectorStore(db_path)
    return store


def _insert_chunk(store, content: str, **kwargs):
    """Insert a test chunk with optional metadata."""
    chunk_id = kwargs.pop("chunk_id", str(uuid.uuid4()))
    metadata = kwargs.pop("metadata", {})
    source_file = kwargs.pop("source_file", "test.jsonl")
    project = kwargs.pop("project", "test-project")
    content_type = kwargs.pop("content_type", "ai_response")
    tags = kwargs.pop("tags", None)
    importance = kwargs.pop("importance", None)
    language = kwargs.pop("language", None)

    cursor = store.conn.cursor()
    cursor.execute(
        """INSERT INTO chunks (id, content, metadata, source_file, project, content_type, tags, importance, language)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            chunk_id,
            content,
            json.dumps(metadata),
            source_file,
            project,
            content_type,
            json.dumps(tags) if tags else None,
            importance,
            language,
        ),
    )

    # Insert a dummy embedding vector (1024 dims of zeros)
    embedding = struct.pack("1024f", *([0.0] * 1024))
    cursor.execute(
        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, embedding),
    )

    return chunk_id


# ── Gap F: content_type + language indexes ──────────────────────────────────


class TestContentTypeIndex:
    """Gap F: Verify content_type and language indexes exist on the chunks table."""

    def test_content_type_index_exists(self, tmp_path):
        store = _make_store(tmp_path)
        cursor = store.conn.cursor()
        indexes = {row[1] for row in cursor.execute("PRAGMA index_list(chunks)")}
        assert "idx_chunks_content_type" in indexes

    def test_language_index_exists(self, tmp_path):
        store = _make_store(tmp_path)
        cursor = store.conn.cursor()
        indexes = {row[1] for row in cursor.execute("PRAGMA index_list(chunks)")}
        assert "idx_chunks_language" in indexes


# ── Gap G: tag_filter uses chunk_tags index ─────────────────────────────────


class TestTagFilterUsesIndex:
    """Gap G: Tag queries should use the chunk_tags junction table (indexed), not json_each()."""

    def test_chunk_tags_table_exists(self, tmp_path):
        store = _make_store(tmp_path)
        cursor = store.conn.cursor()
        tables = {row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "chunk_tags" in tables

    def test_chunk_tags_index_exists(self, tmp_path):
        store = _make_store(tmp_path)
        cursor = store.conn.cursor()
        indexes = {row[1] for row in cursor.execute("PRAGMA index_list(chunk_tags)")}
        assert "idx_chunk_tags_tag" in indexes

    def test_tag_trigger_populates_junction_table(self, tmp_path):
        store = _make_store(tmp_path)
        chunk_id = _insert_chunk(
            store,
            "test chunk with tags",
            tags=["python", "search", "mcp"],
        )
        cursor = store.conn.cursor()
        tags = [row[0] for row in cursor.execute("SELECT tag FROM chunk_tags WHERE chunk_id = ?", (chunk_id,))]
        assert sorted(tags) == ["mcp", "python", "search"]

    def test_search_tag_filter_uses_chunk_tags(self, tmp_path):
        """Verify that tag_filter in search queries chunk_tags, not json_each."""
        store = _make_store(tmp_path)
        _insert_chunk(
            store,
            "tagged with python",
            tags=["python"],
            content_type="ai_response",
        )
        _insert_chunk(
            store,
            "tagged with rust",
            tags=["rust"],
            content_type="ai_response",
        )

        # Search with tag filter — should only return python-tagged chunk
        results = store.search(
            query_text="tagged",
            tag_filter="python",
        )
        assert len(results["documents"][0]) == 1
        assert "python" in results["documents"][0][0]


# ── Gap H: FTS5 AND search returns intersection ────────────────────────────


class TestFts5AutoMode:
    """Gap H: FTS5 auto mode should favor recall for long multi-word queries."""

    def test_escape_fts5_short_query_uses_and(self):
        """≤3 terms should use AND (space = implicit AND in FTS5)."""
        from brainlayer._helpers import _escape_fts5_query

        result = _escape_fts5_query("authentication middleware")
        # Space-separated quoted terms = implicit AND in FTS5
        assert result == '"authentication" "middleware"'

    def test_escape_fts5_long_query_uses_or(self):
        """4+ terms should use OR to avoid zero-result over-constrained MATCH queries."""
        from brainlayer._helpers import _escape_fts5_query

        result = _escape_fts5_query("authentication middleware session tokens")
        assert result == '"authentication" OR "middleware" OR "session" OR "tokens"'

    def test_fts5_and_returns_intersection(self, tmp_path):
        """Search for 'alpha beta' should only return chunks containing BOTH words."""
        store = _make_store(tmp_path)
        # Chunk with both terms
        _insert_chunk(store, "alpha and beta are both here")
        # Chunk with only one term
        _insert_chunk(store, "only alpha is here")
        _insert_chunk(store, "only beta is here")

        # FTS5 search — should only return the chunk with both terms
        cursor = store.conn.cursor()
        from brainlayer._helpers import _escape_fts5_query

        fts_query = _escape_fts5_query("alpha beta")
        results = list(
            cursor.execute(
                """SELECT f.chunk_id, c.content
                FROM chunks_fts f
                JOIN chunks c ON f.chunk_id = c.id
                WHERE chunks_fts MATCH ?""",
                (fts_query,),
            )
        )
        assert len(results) == 1
        assert "both" in results[0][1]

    def test_fts5_or_mode_still_works(self):
        """match_mode='or' should still use OR for entity search."""
        from brainlayer._helpers import _escape_fts5_query

        result = _escape_fts5_query("Avi Simon", match_mode="or")
        assert "OR" in result

    def test_hybrid_search_long_query_uses_fts_or_and_returns_results(self, tmp_path, monkeypatch):
        """Long multi-keyword searches should still surface partial lexical matches."""
        store = _make_store(tmp_path)
        expected = "owner profile and career work history"
        _insert_chunk(store, expected)
        _insert_chunk(store, "gardening notes and grocery list")

        monkeypatch.setattr(
            store,
            "search",
            lambda **kwargs: {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]},
        )

        results = store.hybrid_search(
            query_embedding=[0.0] * 1024,
            query_text="owner profile career work history years experience",
            n_results=5,
        )

        assert expected in results["documents"][0]

    def test_empty_fts5_query_returns_no_match_expression(self):
        """Blank input should skip FTS instead of expanding to a match-all wildcard."""
        from brainlayer._helpers import _escape_fts5_query

        assert _escape_fts5_query("   ") == ""

    def test_search_entities_returns_empty_for_blank_query(self, tmp_path):
        """Entity FTS search should short-circuit on blank input."""
        store = _make_store(tmp_path)
        assert store.search_entities("   ") == []


# ── Gap I: backward-compat alias passes all params ──────────────────────────


class TestBackwardCompatAlias:
    """Gap I: brainlayer_search alias must pass all parameters that brain_search supports."""

    @pytest.mark.asyncio
    async def test_brainlayer_search_passes_file_path(self, monkeypatch):
        """brainlayer_search should forward file_path to _brain_search."""
        captured = {}

        async def mock_brain_search(**kwargs):
            captured.update(kwargs)
            from mcp.types import TextContent

            return [TextContent(type="text", text="mock")]

        import brainlayer.mcp as mcp_pkg

        monkeypatch.setattr(mcp_pkg, "_brain_search", mock_brain_search)

        # Bypass timeout wrapper
        async def no_timeout(coro, timeout=15):
            return await coro

        monkeypatch.setattr(mcp_pkg, "_with_timeout", no_timeout)

        await mcp_pkg.call_tool(
            "brainlayer_search",
            {
                "query": "test",
                "file_path": "src/main.py",
                "chunk_id": "abc123",
                "entity_id": "ent-1",
                "before": 5,
                "after": 5,
                "max_results": 20,
            },
        )

        assert captured.get("file_path") == "src/main.py"
        assert captured.get("chunk_id") == "abc123"
        assert captured.get("entity_id") == "ent-1"
        assert captured.get("before") == 5
        assert captured.get("after") == 5
        assert captured.get("max_results") == 20


# ── Gap J: compact results include importance ───────────────────────────────


class TestCompactIncludesImportance:
    """Gap J: _build_compact_result must include importance when present."""

    def test_compact_result_includes_importance(self):
        from brainlayer.mcp._shared import _build_compact_result

        item = {
            "chunk_id": "abc123",
            "score": 0.85,
            "project": "brainlayer",
            "date": "2026-03-18",
            "content": "A very important decision was made",
            "summary": "Important decision",
            "importance": 9,
        }
        result = _build_compact_result(item)
        assert result["importance"] == 9

    def test_compact_result_omits_none_importance(self):
        from brainlayer.mcp._shared import _build_compact_result

        item = {
            "chunk_id": "abc123",
            "score": 0.85,
            "project": "brainlayer",
            "content": "No importance set",
        }
        result = _build_compact_result(item)
        assert "importance" not in result

    def test_compact_result_has_snippet(self):
        from brainlayer.mcp._shared import _build_compact_result

        item = {
            "content": "x" * 200,
        }
        result = _build_compact_result(item)
        assert len(result["snippet"]) == 150
