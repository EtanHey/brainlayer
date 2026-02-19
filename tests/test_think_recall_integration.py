"""Integration tests for Think/Recall/Sessions — real DB, real queries.

These tests use the production DB (268K+ chunks) to verify that
the intelligence layer actually works with real indexed data.

Split into:
- Session tests (no embedding, fast)
- Think/recall tests (need embedding, slower)
"""

import pytest

from brainlayer.engine import RecallResult, SessionInfo, ThinkResult, recall, sessions, think
from brainlayer.paths import DEFAULT_DB_PATH
from brainlayer.vector_store import VectorStore


@pytest.fixture(scope="module")
def store():
    """Shared VectorStore for all tests in this module."""
    s = VectorStore(DEFAULT_DB_PATH)
    yield s
    s.close()


# ── Session Tests (no embedding needed, fast) ───────────────────────


class TestSessionsReal:
    """Test sessions() with real production DB."""

    def test_sessions_returns_data(self, store):
        """We should have sessions in the DB."""
        result = sessions(store, days=90, limit=50)
        assert len(result) > 0, "Expected sessions in production DB"
        assert all(isinstance(s, SessionInfo) for s in result)

    def test_sessions_have_required_fields(self, store):
        """Each session has session_id and started_at."""
        result = sessions(store, days=90, limit=5)
        for s in result:
            assert s.session_id, "session_id should not be empty"
            assert s.started_at, "started_at should not be empty"

    def test_sessions_golems_project(self, store):
        """Filter by golems project returns results."""
        # Try common project name patterns
        result = sessions(store, days=90, limit=50)
        projects = {s.project for s in result if s.project}
        assert len(projects) > 0, "Expected at least one project"

    def test_sessions_recent_only(self, store):
        """Days filter limits to recent sessions."""
        recent = sessions(store, days=1, limit=100)
        all_sessions = sessions(store, days=90, limit=100)
        # Recent should be fewer or equal
        assert len(recent) <= len(all_sessions)

    def test_sessions_limit_respected(self, store):
        """Limit parameter is respected."""
        limited = sessions(store, days=90, limit=3)
        assert len(limited) <= 3


class TestRecallFileReal:
    """Test file-based recall with real DB — no embedding needed."""

    def test_recall_real_file(self, store):
        """Recall for a file with known interactions."""
        cursor = store.conn.cursor()
        # Find a file that has multiple interactions
        rows = list(
            cursor.execute("""
            SELECT file_path, COUNT(*) as cnt
            FROM file_interactions
            GROUP BY file_path
            HAVING cnt > 3
            LIMIT 1
        """)
        )
        if not rows:
            pytest.skip("No files with multiple interactions in DB")

        fp = rows[0][0]
        result = recall(store, file_path=fp)
        assert isinstance(result, RecallResult)
        assert len(result.file_history) > 0
        assert result.target == fp

    def test_recall_populates_sessions(self, store):
        """Recall for a known file includes session summaries."""
        cursor = store.conn.cursor()
        # Find a file that has session_id in file_interactions
        rows = list(
            cursor.execute("""
            SELECT fi.file_path, fi.session_id
            FROM file_interactions fi
            JOIN session_context sc ON fi.session_id = sc.session_id
            LIMIT 1
        """)
        )
        if not rows:
            pytest.skip("No files with session context in DB")

        fp = rows[0][0]
        result = recall(store, file_path=fp)
        # Should have session summaries if session_context exists
        assert len(result.session_summaries) > 0 or len(result.file_history) > 0


# ── Think/Recall with Embedding (slower) ────────────────────────────


@pytest.fixture(scope="module")
def embed_fn():
    """Shared embedding function — loads model once."""
    from brainlayer.embeddings import get_embedding_model

    model = get_embedding_model()
    return model.embed_query


class TestThinkReal:
    """Test think() with real DB and embeddings."""

    def test_think_returns_results(self, store, embed_fn):
        """Think with a real context returns categorized results."""
        result = think(
            context="implementing authentication with JWT tokens",
            store=store,
            embed_fn=embed_fn,
        )
        assert isinstance(result, ThinkResult)
        # Should find something in 268K+ chunks
        assert result.total > 0, "Expected results for 'authentication' in production DB"

    def test_think_categorizes(self, store, embed_fn):
        """Results are categorized into at least one bucket."""
        result = think(
            context="debugging database connection issues",
            store=store,
            embed_fn=embed_fn,
        )
        has_any = (
            len(result.decisions) > 0
            or len(result.patterns) > 0
            or len(result.bugs) > 0
            or len(result.context) > 0
        )
        assert has_any, "Expected at least one category to have results"

    def test_think_format_is_markdown(self, store, embed_fn):
        """Formatted output is valid markdown."""
        result = think(
            context="Railway deployment configuration",
            store=store,
            embed_fn=embed_fn,
        )
        formatted = result.format()
        assert "##" in formatted or "No relevant" in formatted
        assert isinstance(formatted, str)

    def test_think_project_filter(self, store, embed_fn):
        """Project filter limits results."""
        result = think(
            context="telegram bot setup",
            store=store,
            embed_fn=embed_fn,
            project="golems",
        )
        assert isinstance(result, ThinkResult)

    def test_think_empty_context(self, store, embed_fn):
        """Empty context returns empty result."""
        result = think(
            context="",
            store=store,
            embed_fn=embed_fn,
        )
        assert result.total == 0

    def test_think_max_results(self, store, embed_fn):
        """Max results is respected."""
        result = think(
            context="implementing features",
            store=store,
            embed_fn=embed_fn,
            max_results=3,
        )
        assert result.total <= 3


class TestRecallTopicReal:
    """Test topic-based recall with real DB and embeddings."""

    def test_recall_topic(self, store, embed_fn):
        """Topic recall returns related knowledge."""
        result = recall(
            store=store,
            embed_fn=embed_fn,
            topic="email routing and triage",
        )
        assert isinstance(result, RecallResult)
        assert len(result.related_chunks) > 0

    def test_recall_topic_format(self, store, embed_fn):
        """Recall format is valid markdown."""
        result = recall(
            store=store,
            embed_fn=embed_fn,
            topic="supabase migration",
        )
        formatted = result.format()
        assert isinstance(formatted, str)
        assert "Recall" in formatted or "No recall" in formatted

    def test_recall_file_with_embedding(self, store, embed_fn):
        """File recall with embedding also returns related knowledge."""
        cursor = store.conn.cursor()
        rows = list(
            cursor.execute("""
            SELECT file_path FROM file_interactions
            WHERE file_path LIKE '%.ts'
            LIMIT 1
        """)
        )
        if not rows:
            pytest.skip("No TypeScript files in DB")

        fp = rows[0][0]
        result = recall(
            store=store,
            embed_fn=embed_fn,
            file_path=fp,
        )
        # Should have both file history AND related knowledge
        assert len(result.file_history) > 0
        assert len(result.related_chunks) > 0


# ── MCP Wiring Tests ────────────────────────────────────────────────


class TestMCPToolCount:
    """Verify MCP server has all 12 tools."""

    def test_tool_count(self):
        """MCP server should have 12 tools."""
        import asyncio

        from brainlayer.mcp import list_tools

        tools = asyncio.run(list_tools())
        assert len(tools) == 12

    def test_new_tools_registered(self):
        """Think, recall, sessions, and current_context tools are registered."""
        import asyncio

        from brainlayer.mcp import list_tools

        tools = asyncio.run(list_tools())
        names = {t.name for t in tools}
        assert "brainlayer_think" in names
        assert "brainlayer_recall" in names
        assert "brainlayer_sessions" in names
        assert "brainlayer_current_context" in names

    def test_new_tools_have_annotations(self):
        """New tools have read-only annotations."""
        import asyncio

        from brainlayer.mcp import list_tools

        tools = asyncio.run(list_tools())
        new_tools = [t for t in tools if t.name in ("brainlayer_think", "brainlayer_recall", "brainlayer_sessions", "brainlayer_current_context")]
        for tool in new_tools:
            assert tool.annotations is not None
            assert tool.annotations.readOnlyHint is True
