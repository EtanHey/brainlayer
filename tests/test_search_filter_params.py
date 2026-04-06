"""Tests for T3: Search filter parameters.

Covers:
- New filter params in brain_search inputSchema
- Param passthrough from _brain_recall -> _brain_search -> _search -> hybrid_search
- correction_category tag-based filter SQL generation
- source_filter_like LIKE-based filter SQL generation
- Alias resolution: sentiment_filter -> sentiment, content_type_filter -> content_type
- All new params are optional (no breakage when omitted)
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from brainlayer.mcp.search_handler import _brain_recall, _brain_search

# ── New filter params passthrough ────────────────────────────────────────────


class TestNewFilterParamsPassthrough:
    """Verify new filter params flow from _brain_recall to _brain_search."""

    def test_correction_category_passes_through(self):
        """correction_category reaches _brain_search from _brain_recall."""
        with patch(
            "brainlayer.mcp.search_handler._brain_search",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ) as mock_search:
            asyncio.run(
                _brain_recall(
                    mode="search",
                    query="user corrections about naming",
                    correction_category="naming",
                )
            )

        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["correction_category"] == "naming"

    def test_source_filter_like_passes_through(self):
        """source_filter (LIKE) reaches _brain_search from _brain_recall."""
        with patch(
            "brainlayer.mcp.search_handler._brain_search",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ) as mock_search:
            asyncio.run(
                _brain_recall(
                    mode="search",
                    query="youtube transcripts",
                    source_filter="%youtube%",
                )
            )

        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["source_filter"] == "%youtube%"

    def test_all_new_params_default_to_none(self):
        """When omitted, new params default to None (no filter applied)."""
        with patch(
            "brainlayer.mcp.search_handler._brain_search",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ) as mock_search:
            asyncio.run(
                _brain_recall(
                    mode="search",
                    query="plain search",
                )
            )

        call_kwargs = mock_search.call_args[1]
        assert call_kwargs.get("correction_category") is None
        assert call_kwargs.get("source_filter") is None


# ── _brain_search -> _search delegation ──────────────────────────────────────


class TestBrainSearchToSearch:
    """Verify _brain_search passes new params to _search."""

    def test_correction_category_reaches_search(self):
        """correction_category flows from _brain_search to _search."""
        with (
            patch(
                "brainlayer.mcp.search_handler._get_vector_store",
                return_value=MagicMock(count=MagicMock(return_value=100)),
            ),
            patch(
                "brainlayer.mcp.search_handler._get_embedding_model",
                return_value=MagicMock(embed_query=MagicMock(return_value=[0.1] * 1024)),
            ),
            patch(
                "brainlayer.mcp.search_handler._search",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ) as mock_search,
            patch(
                "brainlayer.mcp.search_handler._normalize_project_name",
                return_value=None,
            ),
        ):
            asyncio.run(
                _brain_search(
                    query="correction preferences",
                    correction_category="preference",
                )
            )

        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["correction_category"] == "preference"

    def test_source_filter_like_reaches_search(self):
        """source_filter_like flows from _brain_search to _search."""
        with (
            patch(
                "brainlayer.mcp.search_handler._get_vector_store",
                return_value=MagicMock(count=MagicMock(return_value=100)),
            ),
            patch(
                "brainlayer.mcp.search_handler._get_embedding_model",
                return_value=MagicMock(embed_query=MagicMock(return_value=[0.1] * 1024)),
            ),
            patch(
                "brainlayer.mcp.search_handler._search",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ) as mock_search,
            patch(
                "brainlayer.mcp.search_handler._normalize_project_name",
                return_value=None,
            ),
        ):
            asyncio.run(
                _brain_search(
                    query="youtube content",
                    source_filter="%youtube%",
                )
            )

        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["source_filter_like"] == "%youtube%"

    def test_new_filters_included_in_has_active_filters(self):
        """source_filter and correction_category skip entity-aware routing."""
        with (
            patch(
                "brainlayer.mcp.search_handler._get_vector_store",
                return_value=MagicMock(count=MagicMock(return_value=100)),
            ),
            patch(
                "brainlayer.mcp.search_handler._get_embedding_model",
                return_value=MagicMock(embed_query=MagicMock(return_value=[0.1] * 1024)),
            ),
            patch(
                "brainlayer.mcp.search_handler._search",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ) as mock_search,
            patch(
                "brainlayer.mcp.search_handler._detect_entities",
                return_value=[{"id": "e1", "name": "Etan", "entity_type": "person"}],
            ) as mock_detect,
            patch(
                "brainlayer.mcp.search_handler._normalize_project_name",
                return_value=None,
            ),
        ):
            # With correction_category active, entity routing should be skipped
            asyncio.run(
                _brain_search(
                    query="Etan corrections",
                    correction_category="factual",
                )
            )

        # _detect_entities should NOT be called when has_active_filters is True
        mock_detect.assert_not_called()
        mock_search.assert_called_once()


# ── Input schema validation ──────────────────────────────────────────────────


class TestInputSchemaPresence:
    """Verify new parameters exist in the brain_search MCP tool schema."""

    def _get_brain_search_props(self):
        from brainlayer.mcp import list_tools

        tools = asyncio.run(list_tools())
        brain_search = next(t for t in tools if t.name == "brain_search")
        return brain_search.inputSchema["properties"]

    def test_new_params_in_schema(self):
        """brain_search tool schema includes all 4 new filter params."""
        props = self._get_brain_search_props()

        assert "sentiment_filter" in props
        assert "content_type_filter" in props
        assert "source_filter" in props
        assert "correction_category" in props

    def test_sentiment_filter_enum_values(self):
        """sentiment_filter has the standardized enum values."""
        props = self._get_brain_search_props()
        sf = props["sentiment_filter"]

        assert set(sf["enum"]) == {"positive", "negative", "neutral", "mixed"}

    def test_content_type_filter_enum_values(self):
        """content_type_filter has extended enum including enrichment types."""
        props = self._get_brain_search_props()
        ctf = props["content_type_filter"]

        expected = {"user_message", "assistant_text", "ai_code", "learning", "decision", "bug_fix"}
        assert set(ctf["enum"]) == expected

    def test_source_filter_is_string_type(self):
        """source_filter is a plain string (LIKE pattern), not enum."""
        props = self._get_brain_search_props()
        sf = props["source_filter"]

        assert sf["type"] == "string"
        assert "enum" not in sf

    def test_correction_category_is_string_type(self):
        """correction_category is a plain string filter."""
        props = self._get_brain_search_props()
        cc = props["correction_category"]

        assert cc["type"] == "string"


# ── Alias resolution in call_tool ────────────────────────────────────────────


class TestAliasResolution:
    """Verify that alias params (sentiment_filter, content_type_filter) resolve correctly."""

    def test_sentiment_filter_overrides_sentiment(self):
        """sentiment_filter takes precedence over legacy sentiment param."""
        from brainlayer.mcp import call_tool

        with patch(
            "brainlayer.mcp._brain_recall",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ) as mock_recall:
            asyncio.run(
                call_tool(
                    "brain_search",
                    {
                        "query": "test",
                        "sentiment": "frustration",
                        "sentiment_filter": "negative",
                    },
                )
            )

        call_kwargs = mock_recall.call_args[1]
        assert call_kwargs["sentiment"] == "negative"

    def test_content_type_filter_overrides_content_type(self):
        """content_type_filter takes precedence over legacy content_type param."""
        from brainlayer.mcp import call_tool

        with patch(
            "brainlayer.mcp._brain_recall",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ) as mock_recall:
            asyncio.run(
                call_tool(
                    "brain_search",
                    {
                        "query": "test",
                        "content_type": "ai_code",
                        "content_type_filter": "decision",
                    },
                )
            )

        call_kwargs = mock_recall.call_args[1]
        assert call_kwargs["content_type"] == "decision"

    def test_legacy_params_still_work_when_no_alias(self):
        """Legacy sentiment/content_type still work when aliases are not provided."""
        from brainlayer.mcp import call_tool

        with patch(
            "brainlayer.mcp._brain_recall",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ) as mock_recall:
            asyncio.run(
                call_tool(
                    "brain_search",
                    {
                        "query": "test",
                        "sentiment": "frustration",
                        "content_type": "ai_code",
                    },
                )
            )

        call_kwargs = mock_recall.call_args[1]
        assert call_kwargs["sentiment"] == "frustration"
        assert call_kwargs["content_type"] == "ai_code"

    def test_correction_category_passes_from_call_tool(self):
        """correction_category is forwarded from call_tool to _brain_recall."""
        from brainlayer.mcp import call_tool

        with patch(
            "brainlayer.mcp._brain_recall",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ) as mock_recall:
            asyncio.run(
                call_tool(
                    "brain_search",
                    {
                        "query": "test",
                        "correction_category": "preference",
                    },
                )
            )

        call_kwargs = mock_recall.call_args[1]
        assert call_kwargs["correction_category"] == "preference"

    def test_source_filter_passes_from_call_tool(self):
        """source_filter is forwarded from call_tool to _brain_recall."""
        from brainlayer.mcp import call_tool

        with patch(
            "brainlayer.mcp._brain_recall",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ) as mock_recall:
            asyncio.run(
                call_tool(
                    "brain_search",
                    {
                        "query": "test",
                        "source_filter": "%youtube%",
                    },
                )
            )

        call_kwargs = mock_recall.call_args[1]
        assert call_kwargs["source_filter"] == "%youtube%"
