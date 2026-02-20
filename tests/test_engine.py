"""Tests for engine.py — Think/Recall/Sessions intelligence layer.

Split into:
- Unit tests: categorization, formatting (no DB/embedding needed)
- Integration tests: real DB queries (marked slow, need DB)
"""

import pytest

from brainlayer.engine import (
    CurrentContext,
    RecallResult,
    SessionInfo,
    ThinkResult,
    _format_memory_item,
    categorize_by_intent,
    format_sessions,
)

# ── Unit Tests: Categorization ──────────────────────────────────────


class TestCategorizeByIntent:
    """Test intent-based categorization of search results."""

    def test_empty_items(self):
        """Empty input returns empty ThinkResult."""
        result = categorize_by_intent([])
        assert result.total == 0
        assert result.decisions == []
        assert result.patterns == []
        assert result.bugs == []
        assert result.context == []

    def test_decisions_categorized(self):
        """Items with deciding/designing intent go to decisions."""
        items = [
            {"content": "Decided to use JWT", "intent": "deciding", "importance": 8},
            {"content": "Designed auth flow", "intent": "designing", "importance": 7},
        ]
        result = categorize_by_intent(items)
        assert len(result.decisions) == 2
        assert result.total == 2

    def test_bugs_categorized(self):
        """Items with debugging intent go to bugs."""
        items = [
            {"content": "Fixed EADDRINUSE crash", "intent": "debugging", "importance": 9},
        ]
        result = categorize_by_intent(items)
        assert len(result.bugs) == 1
        assert result.patterns == []

    def test_patterns_categorized(self):
        """Items with implementing/configuring intent go to patterns."""
        items = [
            {"content": "Implemented retry logic", "intent": "implementing"},
            {"content": "Configured Railway env", "intent": "configuring"},
        ]
        result = categorize_by_intent(items)
        assert len(result.patterns) == 2

    def test_unknown_intent_goes_to_context(self):
        """Items with unknown or empty intent go to context."""
        items = [
            {"content": "Some discussion", "intent": "discussing"},
            {"content": "No intent", "intent": ""},
            {"content": "None intent"},
        ]
        result = categorize_by_intent(items)
        assert len(result.context) == 3

    def test_mixed_intents(self):
        """Mixed intents are categorized correctly."""
        items = [
            {"content": "Decision A", "intent": "deciding"},
            {"content": "Bug fix B", "intent": "debugging"},
            {"content": "Pattern C", "intent": "implementing"},
            {"content": "Context D", "intent": "discussing"},
            {"content": "Decision E", "intent": "designing"},
        ]
        result = categorize_by_intent(items)
        assert len(result.decisions) == 2
        assert len(result.bugs) == 1
        assert len(result.patterns) == 1
        assert len(result.context) == 1
        assert result.total == 5


# ── Unit Tests: Formatting ──────────────────────────────────────────


class TestFormatMemoryItem:
    """Test individual memory item formatting."""

    def test_basic_format(self):
        """Basic item with content only."""
        item = {"content": "Some memory content"}
        result = _format_memory_item(item)
        assert "Some memory content" in result
        assert result.startswith("- ")

    def test_with_summary(self):
        """Summary is used instead of content when available."""
        item = {"content": "Long content...", "summary": "Short summary"}
        result = _format_memory_item(item)
        assert "Short summary" in result
        assert "Long content" not in result

    def test_with_date(self):
        """Date is included when available."""
        item = {"content": "Memory", "created_at": "2026-02-15T10:30:00"}
        result = _format_memory_item(item)
        assert "[2026-02-15]" in result

    def test_with_project(self):
        """Project is included when available."""
        item = {"content": "Memory", "project": "golems"}
        result = _format_memory_item(item)
        assert "(golems)" in result

    def test_high_importance_bold(self):
        """High importance items (>=7) are bolded."""
        item = {"content": "Critical decision", "importance": 8}
        result = _format_memory_item(item)
        assert "**" in result

    def test_low_importance_not_bold(self):
        """Low importance items are not bolded."""
        item = {"content": "Minor note", "importance": 3}
        result = _format_memory_item(item)
        assert "**" not in result

    def test_long_content_truncated(self):
        """Content longer than 200 chars is truncated."""
        item = {"content": "x" * 500}
        result = _format_memory_item(item)
        assert "..." in result
        assert len(result) < 300


class TestThinkResultFormat:
    """Test ThinkResult markdown formatting."""

    def test_empty_result(self):
        """Empty result returns simple message."""
        result = ThinkResult(query="test")
        assert result.format() == "No relevant memories found."

    def test_with_decisions(self):
        """Results with decisions include decisions header."""
        result = ThinkResult(
            query="authentication",
            decisions=[{"content": "Use JWT", "intent": "deciding"}],
            total=1,
        )
        formatted = result.format()
        assert "## Relevant Memories" in formatted
        assert "### Decisions & Design" in formatted
        assert "Use JWT" in formatted

    def test_with_all_categories(self):
        """All category headers appear when populated."""
        result = ThinkResult(
            query="test",
            decisions=[{"content": "D"}],
            patterns=[{"content": "P"}],
            bugs=[{"content": "B"}],
            context=[{"content": "C"}],
            total=4,
        )
        formatted = result.format()
        assert "Decisions & Design" in formatted
        assert "Patterns & Implementations" in formatted
        assert "Related Bugs & Fixes" in formatted
        assert "Related Context" in formatted
        assert "4 memories retrieved" in formatted

    def test_empty_categories_hidden(self):
        """Empty categories don't show headers."""
        result = ThinkResult(
            query="test",
            decisions=[{"content": "D"}],
            total=1,
        )
        formatted = result.format()
        assert "Decisions & Design" in formatted
        assert "Patterns" not in formatted
        assert "Bugs" not in formatted


class TestRecallResultFormat:
    """Test RecallResult markdown formatting."""

    def test_empty_result(self):
        """Empty result returns message with target."""
        result = RecallResult(target="auth.ts")
        assert "auth.ts" in result.format()
        assert "No recall data" in result.format()

    def test_with_file_history(self):
        """File history is formatted correctly."""
        result = RecallResult(
            target="auth.ts",
            file_history=[
                {"action": "Read", "timestamp": "2026-02-15T10:00:00", "session_id": "abc12345xyz"},
                {"action": "Edit", "timestamp": "2026-02-15T10:05:00", "session_id": "abc12345xyz"},
            ],
        )
        formatted = result.format()
        assert "## Recall: auth.ts" in formatted
        assert "### File History" in formatted
        assert "**Read**" in formatted
        assert "**Edit**" in formatted

    def test_with_session_summaries(self):
        """Session summaries are included."""
        result = RecallResult(
            target="auth.ts",
            file_history=[{"action": "Read", "timestamp": "2026-02-15", "session_id": "abc12345"}],
            session_summaries=[
                {"session_id": "abc12345xyz", "branch": "feat/auth", "started_at": "2026-02-15T09:00:00"},
            ],
        )
        formatted = result.format()
        assert "Sessions That Touched This" in formatted
        assert "feat/auth" in formatted


class TestCurrentContextFormat:
    """Test CurrentContext formatting."""

    def test_empty_context(self):
        """Fully empty context returns message."""
        ctx = CurrentContext()
        assert "No recent session" in ctx.format()

    def test_projects_without_sessions_still_shown(self):
        """Projects from chunks fallback are shown even without sessions."""
        ctx = CurrentContext(
            active_projects=["golems", "brainlayer"],
            recent_files=["src/auth.py"],
        )
        formatted = ctx.format()
        assert "golems" in formatted
        assert "No recent session" not in formatted

    def test_with_projects(self):
        """Active projects are shown."""
        ctx = CurrentContext(
            recent_sessions=[SessionInfo(session_id="abc12345xyz", started_at="2026-02-15")],
            active_projects=["golems", "brainlayer"],
        )
        formatted = ctx.format()
        assert "golems" in formatted
        assert "brainlayer" in formatted

    def test_with_branches(self):
        """Active branches are shown."""
        ctx = CurrentContext(
            recent_sessions=[SessionInfo(session_id="abc12345xyz", started_at="2026-02-15")],
            active_branches=["feat/think-recall", "main"],
        )
        formatted = ctx.format()
        assert "feat/think-recall" in formatted

    def test_with_plan(self):
        """Active plan is shown."""
        ctx = CurrentContext(
            recent_sessions=[SessionInfo(session_id="abc12345xyz", started_at="2026-02-15")],
            active_plan="brainlayer-launch",
        )
        formatted = ctx.format()
        assert "brainlayer-launch" in formatted

    def test_recent_files_truncated(self):
        """Recent files show filename only, limited to 10."""
        ctx = CurrentContext(
            recent_sessions=[SessionInfo(session_id="abc12345xyz", started_at="2026-02-15")],
            recent_files=[f"/path/to/file{i}.ts" for i in range(15)],
        )
        formatted = ctx.format()
        # Should show filenames not full paths
        assert "file0.ts" in formatted
        # Should be limited to 10
        assert "file10.ts" not in formatted


class TestFormatSessions:
    """Test sessions list formatting."""

    def test_empty_sessions(self):
        """Empty list returns message."""
        result = format_sessions([], days=7)
        assert "No sessions found" in result

    def test_basic_sessions(self):
        """Sessions are formatted with key info."""
        sessions_list = [
            SessionInfo(
                session_id="abc12345xyz",
                project="golems",
                branch="feat/auth",
                started_at="2026-02-15T10:00:00",
            ),
        ]
        result = format_sessions(sessions_list, days=7)
        assert "## Recent Sessions" in result
        assert "abc12345" in result
        assert "golems" in result
        assert "feat/auth" in result

    def test_session_with_plan(self):
        """Sessions with plan info show it."""
        sessions_list = [
            SessionInfo(
                session_id="abc12345xyz",
                project="golems",
                branch="main",
                started_at="2026-02-15T10:00:00",
                plan_name="brainlayer-launch",
                plan_phase="phase-3",
            ),
        ]
        result = format_sessions(sessions_list)
        assert "brainlayer-launch" in result
        assert "phase-3" in result

    def test_session_count(self):
        """Total session count is shown."""
        sessions_list = [SessionInfo(session_id=f"id{i}", project="p", started_at="2026-02-15") for i in range(3)]
        result = format_sessions(sessions_list)
        assert "3 sessions" in result


# ── Integration Tests: DB Queries ────────────────────────────────────


@pytest.mark.integration
class TestSessionsIntegration:
    """Test sessions() with real DB. Requires production DB."""

    @pytest.fixture(scope="class")
    def store(self):
        from brainlayer.paths import DEFAULT_DB_PATH
        from brainlayer.vector_store import VectorStore

        s = VectorStore(DEFAULT_DB_PATH)
        yield s
        s.close()

    def test_sessions_returns_list(self, store):
        """sessions() returns a list of SessionInfo objects."""
        from brainlayer.engine import sessions

        result = sessions(store, days=30, limit=5)
        assert isinstance(result, list)
        for s in result:
            assert isinstance(s, SessionInfo)
            assert s.session_id

    def test_sessions_ordered_by_date(self, store):
        """Sessions are ordered most recent first."""
        from brainlayer.engine import sessions

        result = sessions(store, days=90, limit=10)
        if len(result) >= 2:
            # Most recent first
            assert result[0].started_at >= result[1].started_at

    def test_sessions_project_filter(self, store):
        """Project filter limits results."""
        from brainlayer.engine import sessions

        all_sessions = sessions(store, days=90, limit=100)
        if all_sessions:
            project = all_sessions[0].project
            if project:
                filtered = sessions(store, project=project, days=90, limit=100)
                for s in filtered:
                    assert s.project == project


@pytest.mark.integration
class TestRecallFileIntegration:
    """Test recall() with real DB for file-based recall (no embedding needed)."""

    @pytest.fixture(scope="class")
    def store(self):
        from brainlayer.paths import DEFAULT_DB_PATH
        from brainlayer.vector_store import VectorStore

        s = VectorStore(DEFAULT_DB_PATH)
        yield s
        s.close()

    def test_recall_unknown_file(self, store):
        """Recall for unknown file returns empty result."""
        from brainlayer.engine import recall

        result = recall(store, file_path="nonexistent/file/path.xyz")
        assert isinstance(result, RecallResult)
        assert result.file_history == []

    def test_recall_known_file(self, store):
        """Recall for a known file returns history."""
        from brainlayer.engine import recall

        # Find a file that has interactions
        cursor = store.conn.cursor()
        rows = list(cursor.execute("SELECT file_path FROM file_interactions LIMIT 1"))
        if rows:
            fp = rows[0][0]
            result = recall(store, file_path=fp)
            assert len(result.file_history) > 0
            assert result.target == fp
