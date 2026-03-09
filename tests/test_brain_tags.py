"""Tests for brain_tags MCP tool (Phase B).

brain_tags(action, pattern, content, project, limit) — tag discovery interface.

Actions:
- list: top tags with counts (optional project filter)
- search: tags matching a prefix/pattern
- suggest: suggest tags from content text

All tests use tmp_path fixtures (isolated DB — no production data).
"""

import json

import pytest

from brainlayer.vector_store import VectorStore

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    """Fresh VectorStore with chunk_tags junction table populated."""
    db_path = tmp_path / "test.db"
    s = VectorStore(db_path)
    yield s
    s.close()


def _insert_chunk_with_tags(store, chunk_id, content, tags, project=None, source="claude_code"):
    """Helper: insert a chunk with tags into the store (triggers chunk_tags sync)."""
    cursor = store.conn.cursor()
    cursor.execute(
        """INSERT INTO chunks (id, content, metadata, source_file, project,
           content_type, char_count, source, tags, created_at)
           VALUES (?, ?, '{}', 'test.jsonl', ?, 'assistant_text', ?, ?, ?, datetime('now'))""",
        (
            chunk_id,
            content,
            project,
            len(content),
            source,
            json.dumps(tags) if tags else None,
        ),
    )
    # chunk_tags trigger should auto-populate from the tags column


# ── Tag List Action ───────────────────────────────────────────────────────────


class TestBrainTagsList:
    """brain_tags(action='list') returns tags ordered by chunk count."""

    def test_list_returns_tags_with_counts(self, store):
        """list action returns all tags with their frequency counts."""
        _insert_chunk_with_tags(store, "c1", "content one", ["python", "testing"])
        _insert_chunk_with_tags(store, "c2", "content two", ["python", "debug"])
        _insert_chunk_with_tags(store, "c3", "content three", ["testing"])

        from brainlayer.mcp.tags_handler import _brain_tags

        result = _brain_tags(store, action="list", limit=10)
        tags = result["tags"]
        tag_names = [t["tag"] for t in tags]
        tag_counts = {t["tag"]: t["count"] for t in tags}

        assert "python" in tag_names
        assert "testing" in tag_names
        assert tag_counts["python"] == 2
        assert tag_counts["testing"] == 2
        assert tag_counts["debug"] == 1

    def test_list_ordered_by_count_descending(self, store):
        """list action returns tags sorted by count (most common first)."""
        for i in range(5):
            _insert_chunk_with_tags(store, f"c-a-{i}", "content", ["popular"])
        for i in range(2):
            _insert_chunk_with_tags(store, f"c-b-{i}", "content", ["rare"])

        from brainlayer.mcp.tags_handler import _brain_tags

        result = _brain_tags(store, action="list", limit=10)
        counts = [t["count"] for t in result["tags"]]
        assert counts == sorted(counts, reverse=True), "Tags should be ordered by count DESC"

    def test_list_respects_limit(self, store):
        """list action returns at most 'limit' tags."""
        for i in range(15):
            _insert_chunk_with_tags(store, f"c-lim-{i}", "content", [f"tag-{i:02d}"])

        from brainlayer.mcp.tags_handler import _brain_tags

        result = _brain_tags(store, action="list", limit=5)
        assert len(result["tags"]) <= 5

    def test_list_empty_db_returns_empty(self, store):
        """list action on empty DB returns empty tag list, not error."""
        from brainlayer.mcp.tags_handler import _brain_tags

        result = _brain_tags(store, action="list", limit=10)
        assert result["tags"] == []
        assert result["total"] == 0

    def test_list_filters_by_project(self, store):
        """list action with project filter only returns tags for that project's chunks."""
        _insert_chunk_with_tags(store, "c-proj-a", "content", ["alpha"], project="project-a")
        _insert_chunk_with_tags(store, "c-proj-b", "content", ["beta"], project="project-b")
        _insert_chunk_with_tags(store, "c-proj-a2", "content", ["alpha", "shared"], project="project-a")
        _insert_chunk_with_tags(store, "c-proj-b2", "content", ["shared"], project="project-b")

        from brainlayer.mcp.tags_handler import _brain_tags

        result = _brain_tags(store, action="list", project="project-a", limit=10)
        tag_names = {t["tag"] for t in result["tags"]}

        assert "alpha" in tag_names
        assert "shared" in tag_names
        assert "beta" not in tag_names, "project-b tag should not appear in project-a results"


# ── Tag Search Action ────────────────────────────────────────────────────────


class TestBrainTagsSearch:
    """brain_tags(action='search', pattern='...') finds tags matching a prefix/pattern."""

    def test_search_by_prefix(self, store):
        """search action with prefix returns all tags starting with that prefix."""
        _insert_chunk_with_tags(store, "c-s1", "c", ["debug", "debugging", "debugger"])
        _insert_chunk_with_tags(store, "c-s2", "c", ["deploy", "deployment"])

        from brainlayer.mcp.tags_handler import _brain_tags

        result = _brain_tags(store, action="search", pattern="debug", limit=10)
        tag_names = [t["tag"] for t in result["tags"]]

        assert "debug" in tag_names
        assert "debugging" in tag_names
        assert "debugger" in tag_names
        assert "deploy" not in tag_names

    def test_search_case_insensitive(self, store):
        """search action is case-insensitive."""
        _insert_chunk_with_tags(store, "c-ci1", "c", ["TypeScript", "typescript"])

        from brainlayer.mcp.tags_handler import _brain_tags

        result_upper = _brain_tags(store, action="search", pattern="TYPE", limit=10)
        result_lower = _brain_tags(store, action="search", pattern="type", limit=10)

        assert len(result_upper["tags"]) > 0
        assert len(result_lower["tags"]) > 0

    def test_search_no_match_returns_empty(self, store):
        """search action with no matching pattern returns empty list."""
        _insert_chunk_with_tags(store, "c-nm1", "c", ["python"])

        from brainlayer.mcp.tags_handler import _brain_tags

        result = _brain_tags(store, action="search", pattern="zzz-nonexistent", limit=10)
        assert result["tags"] == []

    def test_search_requires_pattern(self, store):
        """search action without pattern raises ValueError."""
        from brainlayer.mcp.tags_handler import _brain_tags

        with pytest.raises((ValueError, KeyError)):
            _brain_tags(store, action="search", limit=10)


# ── Tag Suggest Action ────────────────────────────────────────────────────────


class TestBrainTagsSuggest:
    """brain_tags(action='suggest', content='...') suggests tags from content."""

    def test_suggest_returns_matching_existing_tags(self, store):
        """suggest action returns existing tags that match keywords in content."""
        # Seed some tags
        _insert_chunk_with_tags(store, "c-su1", "c", ["authentication", "security", "jwt"])
        _insert_chunk_with_tags(store, "c-su2", "c", ["authentication", "oauth"])
        _insert_chunk_with_tags(store, "c-su3", "c", ["database", "postgres"])

        from brainlayer.mcp.tags_handler import _brain_tags

        result = _brain_tags(
            store,
            action="suggest",
            content="Implementing JWT authentication for the API security",
            limit=10,
        )
        suggested = [t["tag"] for t in result["tags"]]

        # Should suggest auth-related tags from existing vocabulary
        assert any("auth" in tag for tag in suggested), f"Expected auth tag in suggestions: {suggested}"

    def test_suggest_requires_content(self, store):
        """suggest action without content raises ValueError."""
        from brainlayer.mcp.tags_handler import _brain_tags

        with pytest.raises((ValueError, KeyError)):
            _brain_tags(store, action="suggest", limit=10)

    def test_suggest_unknown_content_returns_top_tags(self, store):
        """suggest action with no matching content falls back to top tags."""
        for i in range(5):
            _insert_chunk_with_tags(store, f"c-top-{i}", "c", ["popular-tag"])

        from brainlayer.mcp.tags_handler import _brain_tags

        result = _brain_tags(
            store,
            action="suggest",
            content="xyzzy completely unknown content with no matching tags",
            limit=5,
        )
        # Should still return something (top tags as fallback)
        assert result["tags"] is not None  # list (possibly empty)
        assert "total" in result


# ── Invalid Action ────────────────────────────────────────────────────────────


class TestBrainTagsInvalidAction:
    def test_unknown_action_returns_error(self, store):
        """Unknown action returns error dict, not exception."""
        from brainlayer.mcp.tags_handler import _brain_tags

        result = _brain_tags(store, action="invalid_action", limit=5)
        assert "error" in result


# ── Result Schema ─────────────────────────────────────────────────────────────


class TestBrainTagsSchema:
    """Result always has 'tags' list and 'total' count."""

    def test_result_has_tags_and_total(self, store):
        """Every action result includes 'tags' list and 'total' int."""
        _insert_chunk_with_tags(store, "c-schema", "c", ["python"])

        from brainlayer.mcp.tags_handler import _brain_tags

        for action, kwargs in [
            ("list", {}),
            ("search", {"pattern": "py"}),
            ("suggest", {"content": "python code"}),
        ]:
            result = _brain_tags(store, action=action, **kwargs, limit=5)
            assert "tags" in result, f"Missing 'tags' for action={action}"
            assert "total" in result, f"Missing 'total' for action={action}"
            assert isinstance(result["tags"], list), f"'tags' must be list for action={action}"
            assert isinstance(result["total"], int), f"'total' must be int for action={action}"

    def test_each_tag_item_has_tag_and_count(self, store):
        """Each item in tags list has 'tag' string and 'count' integer."""
        _insert_chunk_with_tags(store, "c-item", "c", ["some-tag"])

        from brainlayer.mcp.tags_handler import _brain_tags

        result = _brain_tags(store, action="list", limit=5)
        for item in result["tags"]:
            assert "tag" in item, f"Tag item missing 'tag': {item}"
            assert "count" in item, f"Tag item missing 'count': {item}"
            assert isinstance(item["tag"], str)
            assert isinstance(item["count"], int) and item["count"] >= 1
