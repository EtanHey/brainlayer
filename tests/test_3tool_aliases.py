"""Tests for Phase 1: 3-Tool Simplification (Aliases).

Covers:
- Smart mode detection from query text
- brain_recall mode=search routing
- brain_recall mode=entity routing
- brain_search → brain_recall(mode=search) delegation
- brain_entity → brain_recall(mode=entity) delegation
- brain_update / brain_expand / brain_tags deprecation (isError: true)
- Backward compatibility: all old modes still work
- Edge cases: empty query, None query, mixed signals
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from brainlayer.mcp.search_handler import _brain_recall, _smart_detect_mode

# ── Smart Mode Detection ─────────────────────────────────────────────────────


class TestSmartDetectMode:
    """_smart_detect_mode auto-routes queries to the right mode."""

    def test_explicit_mode_passthrough(self):
        """Explicit mode is always respected, regardless of query."""
        assert _smart_detect_mode("BrainLayer", "search") == "search"
        assert _smart_detect_mode("stats", "entity") == "entity"
        assert _smart_detect_mode("anything", "context") == "context"

    def test_stats_signals(self):
        """Queries with stats keywords route to stats mode."""
        assert _smart_detect_mode("how many chunks do I have", None) == "stats"
        assert _smart_detect_mode("show me stats", None) == "stats"
        assert _smart_detect_mode("total chunks in the knowledge base", None) == "stats"
        assert _smart_detect_mode("count of memories", None) == "stats"
        assert _smart_detect_mode("statistics overview", None) == "stats"

    def test_context_signals(self):
        """Queries with context keywords route to context mode."""
        assert _smart_detect_mode("what am I working on", None) == "context"
        assert _smart_detect_mode("right now", None) == "context"
        assert _smart_detect_mode("current context", None) == "context"
        assert _smart_detect_mode("what's active", None) == "context"

    def test_entity_proper_noun(self):
        """Capitalized proper nouns route to entity mode."""
        assert _smart_detect_mode("BrainLayer", None) == "entity"
        assert _smart_detect_mode("Etan Heyman", None) == "entity"
        assert _smart_detect_mode("Cantaloupe AI", None) == "entity"

    def test_entity_not_triggered_by_common_words(self):
        """Common English words starting with uppercase should NOT trigger entity mode."""
        assert _smart_detect_mode("How", None) == "search"
        assert _smart_detect_mode("What", None) == "search"
        assert _smart_detect_mode("Find", None) == "search"
        assert _smart_detect_mode("Search", None) == "search"

    def test_default_search(self):
        """Most queries default to search mode."""
        assert _smart_detect_mode("how did I implement authentication", None) == "search"
        assert _smart_detect_mode("debugging the API", None) == "search"
        assert _smart_detect_mode("convex mutation patterns", None) == "search"

    def test_none_query_defaults_context(self):
        """No query and no mode defaults to context."""
        assert _smart_detect_mode(None, None) == "context"

    def test_empty_query_defaults_context(self):
        """Empty query defaults to context."""
        assert _smart_detect_mode("", None) == "context"

    def test_long_capitalized_phrase_not_entity(self):
        """4+ capitalized words should not trigger entity mode (too long for entity name)."""
        assert _smart_detect_mode("How To Fix Authentication Bugs", None) == "search"


# ── brain_recall mode=search ─────────────────────────────────────────────────


class TestBrainRecallSearchMode:
    """brain_recall(mode='search') delegates to _brain_search."""

    def test_search_mode_delegates_to_brain_search(self):
        """mode=search routes to _brain_search with all params."""
        with patch(
            "brainlayer.mcp.search_handler._brain_search",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ) as mock_search:
            asyncio.run(_brain_recall(mode="search", query="test query", project="test-proj"))

        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["query"] == "test query"
        assert call_kwargs["project"] == "test-proj"

    def test_search_mode_passes_filters(self):
        """mode=search passes all filter params through."""
        with patch(
            "brainlayer.mcp.search_handler._brain_search",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ) as mock_search:
            asyncio.run(
                _brain_recall(
                    mode="search",
                    query="auth bug",
                    tag="bug-fix",
                    content_type="stack_trace",
                    importance_min=7,
                    num_results=10,
                    detail="full",
                )
            )

        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["tag"] == "bug-fix"
        assert call_kwargs["content_type"] == "stack_trace"
        assert call_kwargs["importance_min"] == 7
        assert call_kwargs["num_results"] == 10
        assert call_kwargs["detail"] == "full"

    def test_search_mode_requires_query(self):
        """mode=search without query returns error."""
        result = asyncio.run(_brain_recall(mode="search", query=None))
        assert result.isError is True
        assert "query is required" in result.content[0].text


# ── brain_recall mode=entity ─────────────────────────────────────────────────


class TestBrainRecallEntityMode:
    """brain_recall(mode='entity') delegates to entity handler."""

    def test_entity_mode_delegates(self):
        """mode=entity routes to _brain_entity handler."""
        with patch(
            "brainlayer.mcp.entity_handler._brain_entity",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ) as mock_entity:
            asyncio.run(_brain_recall(mode="entity", query="BrainLayer"))

        mock_entity.assert_called_once_with(query="BrainLayer", entity_type=None)

    def test_entity_mode_passes_entity_type(self):
        """mode=entity passes entity_type filter."""
        with patch(
            "brainlayer.mcp.entity_handler._brain_entity",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ) as mock_entity:
            asyncio.run(_brain_recall(mode="entity", query="Etan", entity_type="person"))

        mock_entity.assert_called_once_with(query="Etan", entity_type="person")

    def test_entity_mode_requires_query(self):
        """mode=entity without query returns error."""
        result = asyncio.run(_brain_recall(mode="entity", query=None))
        assert result.isError is True
        assert "query is required" in result.content[0].text


# ── Smart routing through brain_recall ────────────────────────────────────────


class TestSmartRouting:
    """brain_recall with query but no mode uses smart detection."""

    def test_proper_noun_auto_routes_to_entity(self):
        """Capitalized proper noun query auto-routes to entity mode."""
        with patch(
            "brainlayer.mcp.entity_handler._brain_entity",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ) as mock_entity:
            asyncio.run(_brain_recall(query="BrainLayer"))

        mock_entity.assert_called_once()

    def test_stats_query_auto_routes_to_stats(self):
        """Stats keyword query auto-routes to stats mode."""
        with (
            patch(
                "brainlayer.mcp.search_handler._stats",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "brainlayer.mcp.search_handler._list_projects",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            # Should NOT raise — routes to stats, not search
            asyncio.run(_brain_recall(query="how many chunks"))

    def test_regular_query_auto_routes_to_search(self):
        """Regular query auto-routes to search mode."""
        with patch(
            "brainlayer.mcp.search_handler._brain_search",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ) as mock_search:
            asyncio.run(_brain_recall(query="authentication patterns"))

        mock_search.assert_called_once()


# ── Backward compat: old modes still work ─────────────────────────────────────


class TestBackwardCompatRecallModes:
    """Old brain_recall modes (context, sessions, etc.) remain functional."""

    def test_context_mode_still_works(self):
        """mode=context still routes to _current_context."""
        with patch(
            "brainlayer.mcp.search_handler._current_context",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ) as mock_ctx:
            asyncio.run(_brain_recall(mode="context"))

        mock_ctx.assert_called_once_with(hours=24)

    def test_sessions_mode_still_works(self):
        """mode=sessions still routes to _sessions."""
        with patch(
            "brainlayer.mcp.search_handler._sessions",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ) as mock_sessions:
            asyncio.run(_brain_recall(mode="sessions", days=14, limit=50))

        mock_sessions.assert_called_once_with(project=None, days=14, limit=50)

    def test_stats_mode_still_works(self):
        """mode=stats still routes to _stats + _list_projects."""
        with (
            patch(
                "brainlayer.mcp.search_handler._stats",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_stats,
            patch(
                "brainlayer.mcp.search_handler._list_projects",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_projects,
        ):
            asyncio.run(_brain_recall(mode="stats"))

        mock_stats.assert_called_once()
        mock_projects.assert_called_once()

    def test_no_args_defaults_to_context(self):
        """No arguments at all defaults to context mode."""
        with patch(
            "brainlayer.mcp.search_handler._current_context",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ) as mock_ctx:
            asyncio.run(_brain_recall())

        mock_ctx.assert_called_once()


# ── Deprecation stubs (isError: true) ─────────────────────────────────────────


class TestDeprecationStubs:
    """brain_update, brain_expand, brain_tags return isError: true."""

    def test_brain_update_returns_deprecation_error(self):
        """brain_update returns isError with deprecation message."""
        from brainlayer.mcp import call_tool

        result = asyncio.run(call_tool("brain_update", {"action": "update", "chunk_id": "abc123"}))
        assert result.isError is True
        assert "deprecated" in result.content[0].text.lower()
        assert "brain_store" in result.content[0].text or "brain_supersede" in result.content[0].text

    def test_brain_expand_returns_deprecation_error(self):
        """brain_expand returns isError with deprecation message."""
        from brainlayer.mcp import call_tool

        result = asyncio.run(call_tool("brain_expand", {"chunk_id": "abc123", "context": 3}))
        assert result.isError is True
        assert "deprecated" in result.content[0].text.lower()
        assert "brain_recall" in result.content[0].text

    def test_brain_tags_returns_deprecation_error(self):
        """brain_tags returns isError with deprecation message."""
        from brainlayer.mcp import call_tool

        result = asyncio.run(call_tool("brain_tags", {"action": "list"}))
        assert result.isError is True
        assert "deprecated" in result.content[0].text.lower()
        assert "brain_recall" in result.content[0].text or "brain_store" in result.content[0].text


# ── call_tool delegation ──────────────────────────────────────────────────────


class TestCallToolDelegation:
    """call_tool routes brain_search and brain_entity to brain_recall."""

    def test_brain_search_delegates_to_recall_search(self):
        """call_tool('brain_search') delegates to _brain_recall(mode='search')."""
        with patch(
            "brainlayer.mcp._brain_recall",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ) as mock_recall:
            from brainlayer.mcp import call_tool

            asyncio.run(call_tool("brain_search", {"query": "test"}))

        mock_recall.assert_called_once()
        call_kwargs = mock_recall.call_args[1]
        assert call_kwargs["mode"] == "search"
        assert call_kwargs["query"] == "test"

    def test_brain_entity_delegates_to_recall_entity(self):
        """call_tool('brain_entity') delegates to _brain_recall(mode='entity')."""
        with patch(
            "brainlayer.mcp._brain_recall",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ) as mock_recall:
            from brainlayer.mcp import call_tool

            asyncio.run(call_tool("brain_entity", {"query": "BrainLayer"}))

        mock_recall.assert_called_once()
        call_kwargs = mock_recall.call_args[1]
        assert call_kwargs["mode"] == "entity"
        assert call_kwargs["query"] == "BrainLayer"

    def test_brain_recall_with_query_passes_all_new_params(self):
        """call_tool('brain_recall') passes new Phase 1 params."""
        with patch(
            "brainlayer.mcp._brain_recall",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ) as mock_recall:
            from brainlayer.mcp import call_tool

            asyncio.run(
                call_tool(
                    "brain_recall",
                    {
                        "query": "auth patterns",
                        "mode": "search",
                        "tag": "authentication",
                        "entity_type": "concept",
                    },
                )
            )

        call_kwargs = mock_recall.call_args[1]
        assert call_kwargs["query"] == "auth patterns"
        assert call_kwargs["mode"] == "search"
        assert call_kwargs["tag"] == "authentication"
        assert call_kwargs["entity_type"] == "concept"


# ── MCP tools/list backward compat ───────────────────────────────────────────


class TestToolsListBackwardCompat:
    """MCP tools/list still exposes all tools for backward compatibility."""

    def test_all_working_tools_listed(self):
        """All 5+ working tools must appear in list_tools."""
        from brainlayer.mcp import list_tools

        tools = asyncio.run(list_tools())
        tool_names = {t.name for t in tools}

        # Core 5 working tools
        assert "brain_search" in tool_names
        assert "brain_store" in tool_names
        assert "brain_recall" in tool_names
        assert "brain_digest" in tool_names
        assert "brain_entity" in tool_names

    def test_deprecated_tools_still_listed(self):
        """Deprecated tools still appear in list_tools (backward compat)."""
        from brainlayer.mcp import list_tools

        tools = asyncio.run(list_tools())
        tool_names = {t.name for t in tools}

        assert "brain_update" in tool_names
        assert "brain_expand" in tool_names
        assert "brain_tags" in tool_names

    def test_brain_recall_has_search_and_entity_modes(self):
        """brain_recall tool schema includes search and entity in mode enum."""
        from brainlayer.mcp import list_tools

        tools = asyncio.run(list_tools())
        recall_tool = next(t for t in tools if t.name == "brain_recall")
        mode_enum = recall_tool.inputSchema["properties"]["mode"]["enum"]

        assert "search" in mode_enum
        assert "entity" in mode_enum
        # Old modes still present
        assert "context" in mode_enum
        assert "sessions" in mode_enum
        assert "stats" in mode_enum

    def test_brain_recall_has_query_param(self):
        """brain_recall tool schema includes query parameter."""
        from brainlayer.mcp import list_tools

        tools = asyncio.run(list_tools())
        recall_tool = next(t for t in tools if t.name == "brain_recall")
        assert "query" in recall_tool.inputSchema["properties"]


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases for mode detection and routing."""

    def test_unknown_mode_returns_error(self):
        """Explicit unknown mode returns error."""
        result = asyncio.run(_brain_recall(mode="nonexistent"))
        assert result.isError is True
        assert "Unknown recall mode" in result.content[0].text

    def test_operations_mode_requires_session_id(self):
        """mode=operations without session_id returns error."""
        result = asyncio.run(_brain_recall(mode="operations"))
        assert result.isError is True
        assert "session_id required" in result.content[0].text

    def test_summary_mode_requires_session_id(self):
        """mode=summary without session_id returns error."""
        result = asyncio.run(_brain_recall(mode="summary"))
        assert result.isError is True
        assert "session_id required" in result.content[0].text
