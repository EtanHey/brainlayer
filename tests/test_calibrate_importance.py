"""Tests for Phase 1 importance calibration — heuristic SQL fix.

TDD tests for scripts/calibrate_importance.py.
Uses tmp_path fixtures for full isolation from production DB.
"""

import json
import uuid

import pytest

from brainlayer.calibrate import calibrate_importance
from brainlayer.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    """Fresh VectorStore for testing."""
    db_path = tmp_path / "test.db"
    s = VectorStore(db_path)
    yield s
    s.close()


def _insert_chunk(store, content="test", content_type=None, source=None, importance=None, source_file="test.jsonl"):
    """Helper to insert a chunk with specific attributes."""
    chunk_id = f"test-{uuid.uuid4().hex[:12]}"
    cursor = store.conn.cursor()
    cursor.execute(
        """INSERT INTO chunks (id, content, metadata, source_file,
           content_type, source, importance)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (chunk_id, content, json.dumps({}), source_file, content_type, source, importance),
    )
    return chunk_id


class TestNullImportanceDefaulted:
    """NULL importance chunks get sensible defaults by content type."""

    def test_manual_source_gets_8(self, store):
        cid = _insert_chunk(
            store, content="Always use retries", content_type="decision", source="manual", importance=None
        )
        calibrate_importance(store)
        row = store.conn.cursor().execute("SELECT importance FROM chunks WHERE id = ?", (cid,)).fetchone()
        assert row[0] == 8

    def test_conversation_gets_3(self, store):
        cid = _insert_chunk(
            store,
            content="Let me check that file",
            content_type="assistant_text",
            source="claude_code",
            importance=None,
        )
        calibrate_importance(store)
        row = store.conn.cursor().execute("SELECT importance FROM chunks WHERE id = ?", (cid,)).fetchone()
        assert row[0] == 3

    def test_media_gets_4(self, store):
        cid = _insert_chunk(
            store, content="Huberman discusses sleep", content_type="assistant_text", source="youtube", importance=None
        )
        calibrate_importance(store)
        row = store.conn.cursor().execute("SELECT importance FROM chunks WHERE id = ?", (cid,)).fetchone()
        assert row[0] == 4

    def test_ai_code_gets_5(self, store):
        cid = _insert_chunk(
            store, content="def hello(): pass", content_type="ai_code", source="claude_code", importance=None
        )
        calibrate_importance(store)
        row = store.conn.cursor().execute("SELECT importance FROM chunks WHERE id = ?", (cid,)).fetchone()
        assert row[0] == 5

    def test_curated_knowledge_gets_7(self, store):
        cid = _insert_chunk(
            store, content="Always chunk at 512 tokens", content_type="learning", source="claude_code", importance=None
        )
        calibrate_importance(store)
        row = store.conn.cursor().execute("SELECT importance FROM chunks WHERE id = ?", (cid,)).fetchone()
        assert row[0] == 7


class TestDecisionKeywordsExemptFromCap:
    """Chunks containing decision/learning keywords are NOT capped even if
    their content_type would normally trigger a cap."""

    def test_decided_keyword_keeps_high_score(self, store):
        cid = _insert_chunk(
            store,
            content="We decided to use PostgreSQL instead of MySQL",
            content_type="assistant_text",
            source="claude_code",
            importance=8,
        )
        calibrate_importance(store)
        row = store.conn.cursor().execute("SELECT importance FROM chunks WHERE id = ?", (cid,)).fetchone()
        # Should NOT be capped to 5 — decision keyword exempts it
        assert row[0] == 8

    def test_learned_keyword_keeps_high_score(self, store):
        cid = _insert_chunk(
            store,
            content="I learned that WAL mode prevents corruption",
            content_type="user_message",
            source="claude_code",
            importance=7,
        )
        calibrate_importance(store)
        row = store.conn.cursor().execute("SELECT importance FROM chunks WHERE id = ?", (cid,)).fetchone()
        assert row[0] == 7

    def test_mistake_keyword_keeps_high_score(self, store):
        cid = _insert_chunk(
            store,
            content="The mistake was not checking NULL before indexing",
            content_type="assistant_text",
            source="claude_code",
            importance=9,
        )
        calibrate_importance(store)
        row = store.conn.cursor().execute("SELECT importance FROM chunks WHERE id = ?", (cid,)).fetchone()
        assert row[0] == 9

    def test_critical_keyword_keeps_high_score(self, store):
        cid = _insert_chunk(
            store,
            content="CRITICAL: never run bulk deletes with FTS trigger active",
            content_type="assistant_text",
            source="claude_code",
            importance=9,
        )
        calibrate_importance(store)
        row = store.conn.cursor().execute("SELECT importance FROM chunks WHERE id = ?", (cid,)).fetchone()
        assert row[0] == 9

    def test_no_keyword_gets_capped(self, store):
        """Conversation chunk WITHOUT decision keywords IS capped."""
        cid = _insert_chunk(
            store,
            content="The weather is nice today and I like Python",
            content_type="assistant_text",
            source="claude_code",
            importance=8,
        )
        calibrate_importance(store)
        row = store.conn.cursor().execute("SELECT importance FROM chunks WHERE id = ?", (cid,)).fetchone()
        assert row[0] == 5  # Capped: min(8-2, 5) = 5


class TestDryRunNoModification:
    """--dry-run reports distribution but does NOT modify any rows."""

    def test_dry_run_preserves_importance(self, store):
        cid = _insert_chunk(
            store,
            content="Random conversation text without value",
            content_type="assistant_text",
            source="claude_code",
            importance=9,
        )
        result = calibrate_importance(store, dry_run=True)
        row = store.conn.cursor().execute("SELECT importance FROM chunks WHERE id = ?", (cid,)).fetchone()
        # Must be unchanged
        assert row[0] == 9

    def test_dry_run_returns_distribution(self, store):
        _insert_chunk(store, content="a", content_type="assistant_text", importance=2)
        _insert_chunk(store, content="b", content_type="assistant_text", importance=7)
        _insert_chunk(store, content="c", content_type="assistant_text", importance=9)
        result = calibrate_importance(store, dry_run=True)
        assert "before" in result
        assert "after_simulated" in result
        assert result["before"]["total"] == 3

    def test_dry_run_null_stays_null(self, store):
        cid = _insert_chunk(
            store,
            content="Should stay null in dry run",
            content_type="assistant_text",
            source="claude_code",
            importance=None,
        )
        calibrate_importance(store, dry_run=True)
        row = store.conn.cursor().execute("SELECT importance FROM chunks WHERE id = ?", (cid,)).fetchone()
        assert row[0] is None


class TestConversationCapping:
    """Conversation content types get capped at 5 when inflated."""

    def test_assistant_text_capped(self, store):
        cid = _insert_chunk(store, content="Here is the file content", content_type="assistant_text", importance=8)
        calibrate_importance(store)
        row = store.conn.cursor().execute("SELECT importance FROM chunks WHERE id = ?", (cid,)).fetchone()
        assert row[0] == 5

    def test_file_read_capped(self, store):
        cid = _insert_chunk(store, content="import os\nimport sys", content_type="file_read", importance=7)
        calibrate_importance(store)
        row = store.conn.cursor().execute("SELECT importance FROM chunks WHERE id = ?", (cid,)).fetchone()
        assert row[0] == 5

    def test_low_importance_conversation_untouched(self, store):
        cid = _insert_chunk(store, content="Just a note", content_type="assistant_text", importance=4)
        calibrate_importance(store)
        row = store.conn.cursor().execute("SELECT importance FROM chunks WHERE id = ?", (cid,)).fetchone()
        assert row[0] == 4


class TestBrainStoreFloor:
    """brain_store entries (source=manual) get floored at 8."""

    def test_low_manual_raised(self, store):
        cid = _insert_chunk(store, content="Important decision", content_type="decision", source="manual", importance=5)
        calibrate_importance(store)
        row = store.conn.cursor().execute("SELECT importance FROM chunks WHERE id = ?", (cid,)).fetchone()
        assert row[0] == 8

    def test_high_manual_preserved(self, store):
        cid = _insert_chunk(
            store, content="Critical architecture note", content_type="decision", source="manual", importance=10
        )
        calibrate_importance(store)
        row = store.conn.cursor().execute("SELECT importance FROM chunks WHERE id = ?", (cid,)).fetchone()
        assert row[0] == 10


class TestMediaCapping:
    """Media transcripts (youtube/podcast) capped at 5."""

    def test_youtube_capped(self, store):
        cid = _insert_chunk(store, content="Huberman on dopamine", source="youtube", importance=8)
        calibrate_importance(store)
        row = store.conn.cursor().execute("SELECT importance FROM chunks WHERE id = ?", (cid,)).fetchone()
        assert row[0] == 5

    def test_youtube_under_cap_untouched(self, store):
        cid = _insert_chunk(store, content="Podcast intro", source="youtube", importance=3)
        calibrate_importance(store)
        row = store.conn.cursor().execute("SELECT importance FROM chunks WHERE id = ?", (cid,)).fetchone()
        assert row[0] == 3


class TestGeneralDeflation:
    """Chunks not covered by specific rules get general deflation."""

    def test_importance_8_minus_2(self, store):
        cid = _insert_chunk(store, content="Some ai code block", content_type="ai_code", importance=8)
        calibrate_importance(store)
        row = store.conn.cursor().execute("SELECT importance FROM chunks WHERE id = ?", (cid,)).fetchone()
        assert row[0] == 6

    def test_importance_6_minus_1(self, store):
        cid = _insert_chunk(store, content="Config file content", content_type="ai_code", importance=6)
        calibrate_importance(store)
        row = store.conn.cursor().execute("SELECT importance FROM chunks WHERE id = ?", (cid,)).fetchone()
        assert row[0] == 5

    def test_importance_5_untouched(self, store):
        cid = _insert_chunk(store, content="Normal content", content_type="ai_code", importance=5)
        calibrate_importance(store)
        row = store.conn.cursor().execute("SELECT importance FROM chunks WHERE id = ?", (cid,)).fetchone()
        assert row[0] == 5
