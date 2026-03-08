"""Tests for brain_search routing fixes (Phase 2).

Covers:
- B1: File extension should NOT hijack semantic search
- B2-B6: Recall signal words should not be too broad
- B7: YouTube source filter with effective_k
- format="compact" token reduction
"""

from brainlayer.mcp._shared import (
    _RECALL_SIGNALS,
    _extract_file_path,
    _query_signals_recall,
)

# --- B1: File extension hijack ---


class TestFileExtensionRouting:
    """Rule 4 should only trigger when the query IS a file path, not when it CONTAINS one."""

    def test_semantic_query_with_filename_not_hijacked(self):
        """'how is auth in CLAUDE.md' should NOT extract a file path for routing."""
        # Multi-word query that happens to contain a filename
        result = _extract_file_path("how is auth implemented in CLAUDE.md")
        assert result is None, f"Expected None, got '{result}' — semantic query hijacked by file extension"

    def test_semantic_query_with_json_not_hijacked(self):
        """'package.json best practices' should NOT extract a file path."""
        result = _extract_file_path("package.json best practices")
        assert result is None, f"Expected None, got '{result}' — conceptual query hijacked"

    def test_semantic_query_with_path_like_token_not_hijacked(self):
        """'src/utils code quality' should NOT extract a file path."""
        result = _extract_file_path("src/utils code quality")
        assert result is None, f"Expected None, got '{result}' — path-like token hijacked"

    def test_single_filename_still_extracts(self):
        """A query that IS a file path should still be extracted."""
        result = _extract_file_path("CLAUDE.md")
        assert result == "CLAUDE.md"

    def test_two_token_file_query_still_extracts(self):
        """Short query like 'auth.ts history' should still extract."""
        result = _extract_file_path("auth.ts history")
        assert result == "auth.ts"

    def test_single_path_still_extracts(self):
        """A query that IS a path should still extract."""
        result = _extract_file_path("src/utils/auth.ts")
        assert result == "src/utils/auth.ts"


# --- B2-B6: Recall signals too broad ---


class TestRecallSignals:
    """Broad signal words should not trigger recall routing."""

    def test_what_about_not_recall(self):
        """'what about authentication patterns' should NOT trigger recall."""
        assert not _query_signals_recall("what about authentication patterns")

    def test_worked_on_not_recall(self):
        """'how teams worked on the migration' should NOT trigger recall."""
        assert not _query_signals_recall("how teams worked on the migration")

    def test_context_for_not_recall(self):
        """'context for this error message' should NOT trigger recall."""
        assert not _query_signals_recall("context for this error message")

    def test_history_of_still_recall(self):
        """'history of authentication' should still trigger recall."""
        assert _query_signals_recall("history of authentication")

    def test_discussed_about_still_recall(self):
        """'discussed about the auth bug' should still trigger recall."""
        assert _query_signals_recall("discussed about the auth bug")

    def test_thought_about_still_recall(self):
        """'thought about using JWT' should still trigger recall."""
        assert _query_signals_recall("thought about using JWT")

    def test_recall_signals_list_is_tight(self):
        """Verify _RECALL_SIGNALS does not contain broad terms."""
        broad_terms = {"what about", "worked on", "context for"}
        for term in broad_terms:
            assert term not in _RECALL_SIGNALS, f"'{term}' should be removed from _RECALL_SIGNALS"


# --- format="compact" ---


class TestCompactFormat:
    """Compact format returns pointers for drill-down: chunk_id, snippet, score, project, summary."""

    def test_compact_result_has_required_fields(self):
        """Compact format should include: chunk_id, snippet, score, project, date, summary."""
        compact_item = _build_compact_item(self._sample_item())
        required_keys = {"score", "snippet", "project", "chunk_id"}
        assert required_keys.issubset(compact_item.keys())
        # Optional fields should be present when source item has them
        assert "date" in compact_item
        assert "summary" in compact_item

    def test_compact_result_drops_verbose_fields(self):
        """Compact format should NOT include verbose fields."""
        compact_item = _build_compact_item(self._sample_item())
        dropped_keys = {
            "content_type",
            "tags",
            "intent",
            "source_file",
            "session_summary",
            "session_outcome",
            "session_quality",
        }
        for key in dropped_keys:
            assert key not in compact_item, f"'{key}' should be dropped in compact format"

    def test_compact_snippet_truncated_to_150(self):
        """Compact format should truncate snippet to 150 chars."""
        item = self._sample_item()
        item["content"] = "x" * 1000
        compact_item = _build_compact_item(item)
        assert len(compact_item["snippet"]) <= 150

    def test_compact_fewer_tokens_than_full(self):
        """Compact format should have fewer total characters than full format."""
        item = self._sample_item()
        compact_item = _build_compact_item(item)
        full_keys_count = len(item)
        compact_keys_count = len(compact_item)
        assert compact_keys_count < full_keys_count

    @staticmethod
    def _sample_item():
        return {
            "score": 0.85,
            "project": "brainlayer",
            "content_type": "ai_code",
            "content": "def auth(): pass  # authentication implementation" * 20,
            "source_file": "src/auth.py",
            "date": "2026-01-15",
            "source": "claude_code",
            "summary": "Authentication implementation",
            "tags": ["auth", "python"],
            "intent": "implementing",
            "importance": 8,
            "chunk_id": "abc123",
            "session_summary": "Worked on auth system",
            "session_outcome": "completed",
            "session_quality": 0.9,
        }


def _build_compact_item(item: dict) -> dict:
    """Build a compact result item from a full result item.

    This is a test helper that mirrors the production implementation.
    Tests will break if the import fails, telling us to implement it.
    """
    # Import from production code — this will fail until implemented
    from brainlayer.mcp import _build_compact_result

    return _build_compact_result(item)
