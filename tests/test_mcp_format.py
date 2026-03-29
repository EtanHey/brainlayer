"""Tests for MCP output formatting (_format.py).

Covers: format_search_results, format_store_result, format_entity_card,
        format_entity_simple, format_stats, format_digest_result, format_kg_search.
Also tests _truncate, _pad, _importance_bar helpers.
"""

from brainlayer.mcp._format import (
    _pad,
    _truncate,
    format_digest_result,
    format_entity_card,
    format_entity_simple,
    format_kg_search,
    format_search_results,
    format_stats,
    format_store_result,
)

# --- Helper tests ---


class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("hello", 80) == "hello"

    def test_exact_length_unchanged(self):
        assert _truncate("a" * 80, 80) == "a" * 80

    def test_long_text_truncated_with_ellipsis(self):
        result = _truncate("a" * 100, 80)
        assert len(result) == 80
        assert result.endswith("\u2026")

    def test_newlines_replaced(self):
        assert "\n" not in _truncate("line1\nline2\nline3", 80)

    def test_empty_string(self):
        assert _truncate("", 80) == ""

    def test_none_returns_empty(self):
        assert _truncate(None, 80) == ""


class TestPad:
    def test_left_align_default(self):
        result = _pad("hi", 10)
        assert result == "hi        "
        assert len(result) == 10

    def test_right_align(self):
        result = _pad("hi", 10, "right")
        assert result == "        hi"

    def test_center_align(self):
        result = _pad("hi", 10, "center")
        assert len(result) == 10
        assert result.strip() == "hi"

    def test_truncates_when_too_long(self):
        result = _pad("very long text here", 10)
        assert len(result) == 10
        assert result.endswith("\u2026")

    def test_none_becomes_empty(self):
        result = _pad(None, 10)
        assert len(result) == 10


# --- format_search_results ---


class TestFormatSearchResults:
    def _make_result(self, **overrides):
        base = {
            "chunk_id": "abc123def456",
            "score": 0.85,
            "project": "brainlayer",
            "date": "2026-03-29",
            "importance": 7,
            "summary": "Test summary for this chunk",
            "snippet": "Raw snippet content here",
            "tags": ["test", "memory"],
        }
        base.update(overrides)
        return base

    def test_zero_results(self):
        result = format_search_results("test query", [], 0)
        assert "No results found" in result
        assert "test query" in result
        # Box-drawing chars
        assert "\u250c" in result  # top-left
        assert "\u2514" in result  # bottom-left

    def test_single_result_structure(self):
        results = [self._make_result()]
        output = format_search_results("test", results, 1)
        assert "1 result" in output
        assert "abc123def456" in output[:200]  # chunk_id near top
        assert "0.85" in output  # score
        assert "brainlayer" in output  # project
        assert "2026-03-29" in output  # date
        assert "Test summary" in output  # summary used over snippet

    def test_multiple_results(self):
        results = [
            self._make_result(chunk_id="aaa111", score=0.95),
            self._make_result(chunk_id="bbb222", score=0.80),
            self._make_result(chunk_id="ccc333", score=0.65),
        ]
        output = format_search_results("multi", results, 3)
        assert "3 results" in output
        assert "[1]" in output
        assert "[2]" in output
        assert "[3]" in output

    def test_tags_displayed(self):
        results = [self._make_result(tags=["decision", "architecture"])]
        output = format_search_results("tagged", results, 1)
        assert "decision" in output
        assert "architecture" in output

    def test_no_tags_no_tag_line(self):
        results = [self._make_result(tags=None)]
        output = format_search_results("no tags", results, 1)
        assert "tags:" not in output

    def test_uses_summary_over_snippet(self):
        results = [self._make_result(summary="The summary", snippet="The snippet")]
        output = format_search_results("q", results, 1)
        assert "The summary" in output

    def test_falls_back_to_snippet(self):
        results = [self._make_result(summary="", snippet="Fallback snippet")]
        output = format_search_results("q", results, 1)
        assert "Fallback snippet" in output

    def test_box_drawing_chars(self):
        results = [self._make_result()]
        output = format_search_results("q", results, 1)
        assert "\u250c" in output  # ┌
        assert "\u2502" in output  # │
        assert "\u251c" in output  # ├
        assert "\u2514" in output  # └

    def test_missing_importance(self):
        results = [self._make_result(importance=None)]
        output = format_search_results("q", results, 1)
        # Should not crash, should show dash
        assert "\u2500" in output

    def test_long_query_truncated(self):
        long_query = "a" * 200
        output = format_search_results(long_query, [], 0)
        # Query should be truncated in header
        assert len(output.split("\n")[0]) < 250


# --- format_store_result ---


class TestFormatStoreResult:
    def test_basic_store(self):
        result = format_store_result("chunk_abc123")
        assert "chunk_abc123" in result
        assert "\u2714" in result  # checkmark

    def test_with_supersede(self):
        result = format_store_result("new_id", superseded="old_id")
        assert "new_id" in result
        assert "old_id" in result
        assert "superseded" in result

    def test_queued(self):
        result = format_store_result("ignored", queued=True)
        assert "queued" in result.lower() or "\u23f3" in result
        assert "DB busy" in result or "busy" in result.lower()


# --- format_entity_card ---


class TestFormatEntityCard:
    def test_minimal_entity(self):
        entity = {"name": "Etan Heyman", "entity_id": "ent_123", "profile": {}}
        result = format_entity_card(entity)
        assert "Etan Heyman" in result
        assert "ent_123" in result
        assert "\u250c" in result
        assert "\u2514" in result

    def test_with_relations(self):
        entity = {
            "name": "Test Person",
            "entity_id": "ent_1",
            "profile": {},
            "relations": [
                {"relation_type": "works_with", "target": {"name": "Other Person"}},
            ],
        }
        result = format_entity_card(entity)
        assert "works_with" in result
        assert "Other Person" in result

    def test_with_memories(self):
        entity = {
            "name": "Test",
            "entity_id": "e1",
            "profile": {},
            "memories": [
                {"type": "decision", "date": "2026-03-29", "content": "Chose X over Y"},
            ],
            "memory_count": 1,
        }
        result = format_entity_card(entity)
        assert "Memories" in result
        assert "decision" in result

    def test_with_profile_fields(self):
        entity = {
            "name": "Dev",
            "entity_id": "e2",
            "profile": {"role": "Engineer", "company": "Acme"},
        }
        result = format_entity_card(entity)
        assert "Engineer" in result
        assert "Acme" in result


# --- format_entity_simple ---


class TestFormatEntitySimple:
    def test_empty_returns_empty(self):
        assert format_entity_simple({}) == ""
        assert format_entity_simple(None) == ""

    def test_basic_entity(self):
        entity = {"name": "BrainLayer", "id": "ent_bl", "entity_type": "project"}
        result = format_entity_simple(entity)
        assert "BrainLayer" in result
        assert "project" in result

    def test_with_relations(self):
        entity = {
            "name": "X",
            "id": "1",
            "entity_type": "person",
            "relations": [{"relation_type": "knows", "target_name": "Y"}],
        }
        result = format_entity_simple(entity)
        assert "knows" in result
        assert "Y" in result

    def test_with_chunks(self):
        entity = {
            "name": "X",
            "id": "1",
            "entity_type": "tool",
            "chunks": [{"content": "Some memory about X"}],
        }
        result = format_entity_simple(entity)
        assert "Some memory" in result


# --- format_stats ---


class TestFormatStats:
    def test_basic_stats(self):
        stats = {
            "total_chunks": 297000,
            "projects": ["brainlayer", "voicelayer", "cmuxlayer"],
            "content_types": ["ai_code", "user_message"],
        }
        result = format_stats(stats)
        assert "297,000" in result
        assert "brainlayer" in result
        assert "ai_code" in result
        assert "\u250c" in result
        assert "\u2514" in result

    def test_many_projects_truncated(self):
        stats = {
            "total_chunks": 100,
            "projects": [f"proj{i}" for i in range(20)],
            "content_types": ["note"],
        }
        result = format_stats(stats)
        assert "..." in result


# --- format_digest_result ---


class TestFormatDigestResult:
    def test_enrich_mode(self):
        result = format_digest_result({
            "mode": "enrich",
            "attempted": 50,
            "enriched": 45,
            "skipped": 3,
            "failed": 2,
        })
        assert "enrich" in result
        assert "50" in result
        assert "45" in result

    def test_digest_mode(self):
        result = format_digest_result({
            "mode": "digest",
            "chunks_created": 12,
            "entities_created": 3,
            "relations_created": 5,
        })
        assert "digest" in result
        assert "12" in result
        assert "3" in result

    def test_with_action_items(self):
        result = format_digest_result({
            "mode": "digest",
            "chunks_created": 1,
            "entities_created": 0,
            "relations_created": 0,
            "action_items": [{"description": "Review the auth module"}],
        })
        assert "Review the auth module" in result

    def test_connect_mode(self):
        result = format_digest_result({
            "mode": "connect",
            "chunks_created": 5,
            "entities_created": 2,
            "relations_created": 8,
        })
        assert "connect" in result


# --- format_kg_search ---


class TestFormatKgSearch:
    def test_basic_kg_search(self):
        results = [
            {"chunk_id": "c1", "score": 0.9, "snippet": "Memory about Etan"},
        ]
        facts = [
            {"source": "Etan", "relation": "works_on", "target": "BrainLayer"},
        ]
        output = format_kg_search("Etan", results, facts, "Etan work")
        assert "Etan" in output
        assert "works_on" in output
        assert "BrainLayer" in output
        assert "Memory about Etan" in output

    def test_no_facts(self):
        results = [{"chunk_id": "c1", "score": 0.5, "snippet": "stuff"}]
        output = format_kg_search("X", results, [], "X query")
        assert "Knowledge Graph" not in output

    def test_no_results(self):
        output = format_kg_search("Nobody", [], [], "Nobody query")
        assert "0 result" in output


# --- _build_compact_result tags passthrough ---


class TestBuildCompactResultTags:
    """Verify that _build_compact_result passes through tags."""

    def test_tags_included(self):
        from brainlayer.mcp._shared import _build_compact_result

        item = {
            "chunk_id": "abc",
            "score": 0.9,
            "project": "test",
            "content": "hello world",
            "date": "2026-01-01",
            "importance": 5,
            "summary": "test summary",
            "tags": ["tag1", "tag2"],
        }
        result = _build_compact_result(item)
        assert "tags" in result
        assert result["tags"] == ["tag1", "tag2"]

    def test_no_tags_no_key(self):
        from brainlayer.mcp._shared import _build_compact_result

        item = {
            "chunk_id": "abc",
            "score": 0.9,
            "project": "test",
            "content": "hello",
        }
        result = _build_compact_result(item)
        assert "tags" not in result
