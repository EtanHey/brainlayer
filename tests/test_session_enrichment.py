"""Tests for Phase 7 — Session-level enrichment pipeline.

Tests cover:
- session_enrichments table creation and CRUD
- Conversation reconstruction from chunks
- LLM response parsing and validation
- Session listing and enrichment orchestration
- CLI command integration
"""

import json

import pytest

from brainlayer.vector_store import VectorStore

# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    """Create a fresh VectorStore for testing."""
    db_path = tmp_path / "test.db"
    s = VectorStore(db_path)
    yield s
    s.close()


@pytest.fixture
def mock_embed():
    """Mock embedding function that returns a fixed 1024-dim vector."""

    def _embed(text: str) -> list[float]:
        seed = sum(ord(c) for c in text[:50]) % 100
        return [float(seed + i) / 1000.0 for i in range(1024)]

    return _embed


@pytest.fixture
def populated_store(store):
    """Store with sample session chunks inserted."""
    cursor = store.conn.cursor()
    session_id = "abc123def456"
    source_file = f"/home/user/.claude/projects/test/{session_id}.jsonl"

    chunks = [
        ("c1", "How do I set up authentication?", "user_message", "2026-02-20T10:00:00Z"),
        (
            "c2",
            "I'll help you set up JWT authentication. First, install the jsonwebtoken package.",
            "assistant_text",
            "2026-02-20T10:00:05Z",
        ),
        ("c3", "npm install jsonwebtoken bcrypt", "ai_code", "2026-02-20T10:00:10Z"),
        ("c4", "Actually, use bun instead of npm", "user_message", "2026-02-20T10:00:30Z"),
        ("c5", "Good point! Let me use bun. `bun add jsonwebtoken bcrypt`", "assistant_text", "2026-02-20T10:00:35Z"),
        ("c6", "Now let's create the auth middleware.", "assistant_text", "2026-02-20T10:01:00Z"),
        ("c7", "export function authMiddleware(req, res, next) { ... }", "ai_code", "2026-02-20T10:01:10Z"),
        ("c8", "Great, that works! Let's add tests.", "user_message", "2026-02-20T10:02:00Z"),
    ]

    from brainlayer.vector_store import serialize_f32

    dummy_embedding = [0.1] * 1024

    for cid, content, ctype, created_at in chunks:
        full_id = f"{session_id}:{cid}"
        cursor.execute(
            """INSERT INTO chunks
               (id, content, metadata, source_file, project, content_type,
                value_type, char_count, source, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                full_id,
                content,
                "{}",
                source_file,
                "test-project",
                ctype,
                "HIGH",
                len(content),
                "claude_code",
                created_at,
            ),
        )
        cursor.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (full_id,))
        cursor.execute(
            "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
            (full_id, serialize_f32(dummy_embedding)),
        )

    store._test_session_id = session_id
    store._test_source_file = source_file
    return store


# ── Table Schema Tests ──────────────────────────────────────────


class TestSessionEnrichmentsTable:
    """Test session_enrichments table exists and has correct schema."""

    def test_table_exists(self, store):
        """session_enrichments table is created on DB init."""
        cursor = store.conn.cursor()
        tables = [row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        assert "session_enrichments" in tables

    def test_fts_table_exists(self, store):
        """session_enrichments_fts virtual table is created."""
        cursor = store.conn.cursor()
        tables = [row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        assert "session_enrichments_fts" in tables

    def test_table_columns(self, store):
        """session_enrichments has all expected columns."""
        cursor = store.conn.cursor()
        cols = {row[1] for row in cursor.execute("PRAGMA table_info(session_enrichments)")}
        expected = {
            "id",
            "session_id",
            "file_path",
            "enrichment_version",
            "enrichment_model",
            "enrichment_timestamp",
            "session_start_time",
            "session_end_time",
            "duration_seconds",
            "message_count",
            "user_message_count",
            "assistant_message_count",
            "tool_call_count",
            "session_summary",
            "primary_intent",
            "outcome",
            "complexity_score",
            "session_quality_score",
            "decisions_made",
            "corrections",
            "learnings",
            "mistakes",
            "patterns",
            "topic_tags",
            "tool_usage_stats",
            "what_worked",
            "what_failed",
            "summary_embedding",
        }
        assert expected.issubset(cols)

    def test_unique_session_id_constraint(self, store):
        """session_id is unique — second insert for same session updates."""
        store.upsert_session_enrichment(
            {
                "session_id": "test-session-1",
                "session_summary": "First version",
                "message_count": 10,
            }
        )
        store.upsert_session_enrichment(
            {
                "session_id": "test-session-1",
                "session_summary": "Updated version",
                "message_count": 20,
            }
        )
        result = store.get_session_enrichment("test-session-1")
        assert result["session_summary"] == "Updated version"
        assert result["message_count"] == 20


# ── CRUD Tests ──────────────────────────────────────────────────


class TestSessionEnrichmentCRUD:
    """Test insert/read/update operations."""

    def test_upsert_and_get(self, store):
        """Basic upsert and retrieval."""
        enrichment = {
            "session_id": "session-abc",
            "session_summary": "Fixed authentication bug in JWT middleware",
            "primary_intent": "debugging",
            "outcome": "success",
            "complexity_score": 5,
            "session_quality_score": 7,
            "message_count": 42,
            "user_message_count": 15,
            "assistant_message_count": 20,
            "tool_call_count": 7,
            "decisions_made": [{"decision": "Use RS256", "rationale": "More secure"}],
            "corrections": [{"what_was_wrong": "Used npm", "what_user_wanted": "Use bun"}],
            "learnings": ["JWT tokens need refresh logic"],
            "mistakes": ["Forgot to handle token expiry"],
            "patterns": [],
            "topic_tags": ["jwt", "authentication", "debugging"],
            "tool_usage_stats": [{"tool": "Read", "count": 12}],
            "what_worked": "Step-by-step debugging approach",
            "what_failed": "Initial token validation logic was wrong",
        }
        store.upsert_session_enrichment(enrichment)

        result = store.get_session_enrichment("session-abc")
        assert result is not None
        assert result["session_summary"] == "Fixed authentication bug in JWT middleware"
        assert result["primary_intent"] == "debugging"
        assert result["outcome"] == "success"
        assert result["complexity_score"] == 5
        assert result["session_quality_score"] == 7
        assert result["message_count"] == 42
        assert isinstance(result["decisions_made"], list)
        assert len(result["decisions_made"]) == 1
        assert result["decisions_made"][0]["decision"] == "Use RS256"
        assert isinstance(result["corrections"], list)
        assert isinstance(result["learnings"], list)
        assert isinstance(result["topic_tags"], list)
        assert "jwt" in result["topic_tags"]

    def test_get_nonexistent_returns_none(self, store):
        """Getting nonexistent session returns None."""
        result = store.get_session_enrichment("nonexistent")
        assert result is None

    def test_list_enriched_sessions(self, store):
        """list_enriched_sessions returns session IDs."""
        store.upsert_session_enrichment(
            {
                "session_id": "s1",
                "session_summary": "First",
                "message_count": 5,
            }
        )
        store.upsert_session_enrichment(
            {
                "session_id": "s2",
                "session_summary": "Second",
                "message_count": 3,
            }
        )
        enriched = store.list_enriched_sessions()
        assert "s1" in enriched
        assert "s2" in enriched
        assert len(enriched) == 2

    def test_enrichment_stats(self, store):
        """get_session_enrichment_stats returns aggregate data."""
        store.upsert_session_enrichment(
            {
                "session_id": "s1",
                "session_summary": "Debug",
                "message_count": 5,
                "primary_intent": "debugging",
                "outcome": "success",
                "session_quality_score": 8,
            }
        )
        store.upsert_session_enrichment(
            {
                "session_id": "s2",
                "session_summary": "Implement",
                "message_count": 10,
                "primary_intent": "implementing",
                "outcome": "success",
                "session_quality_score": 6,
            }
        )
        stats = store.get_session_enrichment_stats()
        assert stats["total_enriched_sessions"] == 2
        assert stats["by_outcome"]["success"] == 2
        assert stats["avg_quality_score"] == 7.0

    def test_json_fields_serialized(self, store):
        """JSON array fields are properly serialized and deserialized."""
        store.upsert_session_enrichment(
            {
                "session_id": "json-test",
                "session_summary": "Testing JSON",
                "message_count": 1,
                "decisions_made": [{"decision": "A"}, {"decision": "B"}],
                "topic_tags": ["tag1", "tag2", "tag3"],
            }
        )
        result = store.get_session_enrichment("json-test")
        assert isinstance(result["decisions_made"], list)
        assert len(result["decisions_made"]) == 2
        assert isinstance(result["topic_tags"], list)
        assert len(result["topic_tags"]) == 3

    def test_fts_search_on_summary(self, store):
        """FTS5 search works on session summaries."""
        store.upsert_session_enrichment(
            {
                "session_id": "fts-test",
                "session_summary": "Implemented authentication with JWT tokens",
                "message_count": 1,
                "what_worked": "Clean separation of concerns",
                "what_failed": "Initial CORS configuration was wrong",
            }
        )
        cursor = store.conn.cursor()
        rows = list(
            cursor.execute(
                "SELECT session_id FROM session_enrichments_fts WHERE session_enrichments_fts MATCH 'authentication'"
            )
        )
        assert len(rows) == 1
        assert rows[0][0] == "fts-test"


# ── Conversation Reconstruction Tests ───────────────────────────


class TestConversationReconstruction:
    """Test reassembling chunks into conversations."""

    def test_reconstruct_orders_by_time(self, populated_store):
        """Chunks are ordered by created_at timestamp."""
        from brainlayer.pipeline.session_enrichment import reconstruct_session

        session_id = populated_store._test_session_id
        result = reconstruct_session(populated_store, session_id)

        assert result["message_count"] == 8
        assert "How do I set up authentication" in result["conversation"]
        # The correction should come after the npm install
        npm_pos = result["conversation"].find("npm install")
        bun_pos = result["conversation"].find("use bun")
        assert npm_pos < bun_pos

    def test_reconstruct_counts_messages(self, populated_store):
        """Message counts are correct."""
        from brainlayer.pipeline.session_enrichment import reconstruct_session

        session_id = populated_store._test_session_id
        result = reconstruct_session(populated_store, session_id)

        assert result["user_message_count"] == 3  # c1, c4, c8
        assert result["assistant_message_count"] == 5  # c2(assistant_text), c3(ai_code), c5, c6, c7
        assert result["tool_call_count"] == 0  # no tool calls in this session

    def test_reconstruct_calculates_timing(self, populated_store):
        """Duration is calculated from first to last chunk."""
        from brainlayer.pipeline.session_enrichment import reconstruct_session

        session_id = populated_store._test_session_id
        result = reconstruct_session(populated_store, session_id)

        assert result["session_start_time"] == "2026-02-20T10:00:00Z"
        assert result["session_end_time"] == "2026-02-20T10:02:00Z"
        assert result["duration_seconds"] == 120  # 2 minutes

    def test_reconstruct_empty_session(self, store):
        """Nonexistent session returns empty result."""
        from brainlayer.pipeline.session_enrichment import reconstruct_session

        result = reconstruct_session(store, "nonexistent-session")
        assert result["chunks"] == []
        assert result["conversation"] == ""
        assert result["message_count"] == 0

    def test_reconstruct_formats_roles(self, populated_store):
        """Conversation text includes role prefixes."""
        from brainlayer.pipeline.session_enrichment import reconstruct_session

        session_id = populated_store._test_session_id
        result = reconstruct_session(populated_store, session_id)

        assert "USER:" in result["conversation"]
        assert "ASSISTANT:" in result["conversation"]


# ── LLM Response Parsing Tests ──────────────────────────────────


class TestParseSessionEnrichment:
    """Test parsing LLM JSON responses."""

    def test_parse_valid_response(self):
        """Valid JSON response is parsed correctly."""
        from brainlayer.pipeline.session_enrichment import parse_session_enrichment

        response = json.dumps(
            {
                "session_summary": "Fixed authentication bug in JWT middleware by adding proper token validation",
                "primary_intent": "debugging",
                "outcome": "success",
                "complexity_score": 5,
                "session_quality_score": 7,
                "decisions_made": [{"decision": "Use RS256", "rationale": "More secure"}],
                "corrections": [],
                "learnings": ["JWT needs refresh logic"],
                "mistakes": [],
                "patterns": [],
                "topic_tags": ["jwt", "auth"],
                "tool_usage_stats": [{"tool": "Read", "count": 5}],
                "what_worked": "Step-by-step approach",
                "what_failed": "Initial logic wrong",
            }
        )

        result = parse_session_enrichment(response)
        assert result is not None
        assert result["session_summary"].startswith("Fixed authentication")
        assert result["primary_intent"] == "debugging"
        assert result["outcome"] == "success"
        assert result["complexity_score"] == 5
        assert len(result["decisions_made"]) == 1

    def test_parse_response_with_markdown_wrapper(self):
        """JSON wrapped in markdown code block is still parsed."""
        from brainlayer.pipeline.session_enrichment import parse_session_enrichment

        response = '```json\n{"session_summary": "This is a valid session about deployment", "primary_intent": "deploying", "outcome": "success"}\n```'
        result = parse_session_enrichment(response)
        assert result is not None
        assert result["primary_intent"] == "deploying"

    def test_parse_empty_response(self):
        """Empty/None response returns None."""
        from brainlayer.pipeline.session_enrichment import parse_session_enrichment

        assert parse_session_enrichment(None) is None
        assert parse_session_enrichment("") is None

    def test_parse_invalid_json(self):
        """Invalid JSON returns None."""
        from brainlayer.pipeline.session_enrichment import parse_session_enrichment

        assert parse_session_enrichment("not json at all") is None

    def test_parse_missing_summary(self):
        """Response without session_summary returns None (required field)."""
        from brainlayer.pipeline.session_enrichment import parse_session_enrichment

        response = json.dumps({"primary_intent": "debugging"})
        assert parse_session_enrichment(response) is None

    def test_parse_clamps_scores(self):
        """Scores are clamped to 1-10 range."""
        from brainlayer.pipeline.session_enrichment import parse_session_enrichment

        response = json.dumps(
            {
                "session_summary": "Valid session with out-of-range scores that need clamping",
                "complexity_score": 15,
                "session_quality_score": -3,
            }
        )
        result = parse_session_enrichment(response)
        assert result is not None
        assert result["complexity_score"] == 10
        assert result["session_quality_score"] == 1

    def test_parse_normalizes_intent(self):
        """Intent values are lowercased and trimmed."""
        from brainlayer.pipeline.session_enrichment import parse_session_enrichment

        response = json.dumps(
            {
                "session_summary": "Valid session with uppercase intent value that needs normalizing",
                "primary_intent": "  Debugging  ",
            }
        )
        result = parse_session_enrichment(response)
        assert result["primary_intent"] == "debugging"

    def test_parse_invalid_intent_ignored(self):
        """Invalid intent value is not included."""
        from brainlayer.pipeline.session_enrichment import parse_session_enrichment

        response = json.dumps(
            {
                "session_summary": "Valid session but with an invalid intent that should be ignored",
                "primary_intent": "invalid_intent",
            }
        )
        result = parse_session_enrichment(response)
        assert "primary_intent" not in result

    def test_parse_caps_arrays(self):
        """Arrays are capped at reasonable limits."""
        from brainlayer.pipeline.session_enrichment import parse_session_enrichment

        response = json.dumps(
            {
                "session_summary": "Valid session with many learnings that should be capped at twenty max",
                "learnings": [f"Learning #{i}" for i in range(50)],
            }
        )
        result = parse_session_enrichment(response)
        assert len(result["learnings"]) == 20


# ── Session Listing Tests ───────────────────────────────────────


class TestListSessionsForEnrichment:
    """Test discovering sessions that need enrichment."""

    def test_list_from_chunks(self, populated_store):
        """Sessions are discovered from chunks source_file."""
        from brainlayer.pipeline.session_enrichment import list_sessions_for_enrichment

        sessions = list_sessions_for_enrichment(populated_store)
        # Should find at least the test session
        session_ids = [s[0] for s in sessions]
        assert populated_store._test_session_id in session_ids

    def test_already_enriched_excluded(self, populated_store):
        """Sessions that are already enriched are excluded."""
        from brainlayer.pipeline.session_enrichment import list_sessions_for_enrichment

        sid = populated_store._test_session_id

        # Enrich the session first
        populated_store.upsert_session_enrichment(
            {
                "session_id": sid,
                "session_summary": "Already enriched",
                "message_count": 8,
            }
        )

        sessions = list_sessions_for_enrichment(populated_store)
        session_ids = [s[0] for s in sessions]
        assert sid not in session_ids


# ── End-to-End Enrichment Tests ─────────────────────────────────


class TestEnrichSession:
    """Test the full enrichment pipeline."""

    def test_enrich_with_mock_llm(self, populated_store):
        """Full enrichment pipeline with mocked LLM."""
        from brainlayer.pipeline.session_enrichment import enrich_session

        # Mock LLM that returns valid JSON
        def mock_llm(prompt: str) -> str:
            return json.dumps(
                {
                    "session_summary": "Set up JWT authentication with bun package manager instead of npm",
                    "primary_intent": "implementing",
                    "outcome": "success",
                    "complexity_score": 4,
                    "session_quality_score": 7,
                    "decisions_made": [{"decision": "Use bun over npm", "rationale": "User preference"}],
                    "corrections": [{"what_was_wrong": "Used npm", "what_user_wanted": "Use bun"}],
                    "learnings": ["User prefers bun package manager"],
                    "mistakes": [],
                    "patterns": [],
                    "topic_tags": ["jwt", "authentication", "bun"],
                    "tool_usage_stats": [],
                    "what_worked": "Responsive to user corrections",
                    "what_failed": "Initial package manager choice",
                }
            )

        sid = populated_store._test_session_id
        result = enrich_session(
            store=populated_store,
            session_id=sid,
            call_llm_fn=mock_llm,
            project="test-project",
        )

        assert result is not None
        assert result["session_summary"].startswith("Set up JWT")
        assert result["primary_intent"] == "implementing"
        assert result["outcome"] == "success"
        assert len(result["corrections"]) == 1
        assert result["message_count"] == 8

        # Verify it was persisted
        stored = populated_store.get_session_enrichment(sid)
        assert stored is not None
        assert stored["session_summary"] == result["session_summary"]

    def test_enrich_empty_session_returns_none(self, store):
        """Enriching nonexistent session returns None."""
        from brainlayer.pipeline.session_enrichment import enrich_session

        result = enrich_session(
            store=store,
            session_id="nonexistent",
            call_llm_fn=lambda p: "{}",
        )
        assert result is None

    def test_enrich_with_llm_failure(self, populated_store):
        """LLM returning None doesn't crash."""
        from brainlayer.pipeline.session_enrichment import enrich_session

        result = enrich_session(
            store=populated_store,
            session_id=populated_store._test_session_id,
            call_llm_fn=lambda p: None,
        )
        assert result is None

    def test_enrich_with_bad_llm_response(self, populated_store):
        """LLM returning invalid JSON doesn't crash."""
        from brainlayer.pipeline.session_enrichment import enrich_session

        result = enrich_session(
            store=populated_store,
            session_id=populated_store._test_session_id,
            call_llm_fn=lambda p: "not valid json response",
        )
        assert result is None


# ── Prompt Building Tests ───────────────────────────────────────


class TestBuildSessionPrompt:
    """Test prompt construction."""

    def test_prompt_includes_conversation(self):
        """Prompt includes the conversation text."""
        from brainlayer.pipeline.session_enrichment import build_session_prompt

        prompt = build_session_prompt("USER: Hello\nASSISTANT: Hi!", "test-project")
        assert "USER: Hello" in prompt
        assert "ASSISTANT: Hi!" in prompt

    def test_prompt_includes_project(self):
        """Prompt includes the project name."""
        from brainlayer.pipeline.session_enrichment import build_session_prompt

        prompt = build_session_prompt("test conversation", "my-project")
        assert "my-project" in prompt

    def test_prompt_escapes_braces(self):
        """Braces in conversation are escaped for str.format()."""
        from brainlayer.pipeline.session_enrichment import build_session_prompt

        # This would crash if braces aren't escaped
        prompt = build_session_prompt("code: function() { return {}; }", "test")
        assert "function()" in prompt
