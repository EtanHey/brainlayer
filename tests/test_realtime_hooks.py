"""Tests for real-time indexing hooks — prompt/response pairing and storage."""

import sqlite3

import pytest

from brainlayer.hooks.indexer import RealtimeIndexer


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary SQLite DB with the required schema."""
    db_path = tmp_path / "test.db"
    indexer = RealtimeIndexer(db_path=str(db_path))
    return indexer, db_path


class TestPromptCapture:
    """UserPromptSubmit hook captures the user prompt."""

    def test_stores_pending_prompt(self, tmp_db):
        indexer, _ = tmp_db
        indexer.capture_prompt(
            session_id="sess-001",
            prompt_text="How do I implement authentication?",
            cwd="/Users/test/project",
        )
        assert "sess-001" in indexer.pending_prompts
        assert indexer.pending_prompts["sess-001"]["text"] == "How do I implement authentication?"

    def test_overwrites_stale_prompt(self, tmp_db):
        indexer, _ = tmp_db
        indexer.capture_prompt("sess-001", "first prompt", "/cwd")
        indexer.capture_prompt("sess-001", "second prompt", "/cwd")
        assert indexer.pending_prompts["sess-001"]["text"] == "second prompt"

    def test_multiple_sessions(self, tmp_db):
        indexer, _ = tmp_db
        indexer.capture_prompt("sess-001", "prompt A", "/cwd-a")
        indexer.capture_prompt("sess-002", "prompt B", "/cwd-b")
        assert len(indexer.pending_prompts) == 2


class TestResponsePairing:
    """Stop hook pairs response with pending prompt and stores."""

    def test_pairs_prompt_and_response(self, tmp_db):
        indexer, db_path = tmp_db
        indexer.capture_prompt("sess-001", "How do auth?", "/project")
        chunk_id = indexer.index_response(
            session_id="sess-001",
            response_text="Use JWT tokens with refresh rotation.",
            cwd="/project",
        )
        assert chunk_id is not None

        # Verify stored in DB
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT content, session_id, project FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        conn.close()
        assert row is not None
        assert "How do auth?" in row[0]
        assert "JWT tokens" in row[0]
        assert row[1] == "sess-001"

    def test_clears_pending_after_pairing(self, tmp_db):
        indexer, _ = tmp_db
        indexer.capture_prompt("sess-001", "prompt", "/cwd")
        indexer.index_response("sess-001", "response", "/cwd")
        assert "sess-001" not in indexer.pending_prompts

    def test_response_without_prompt_still_indexes(self, tmp_db):
        """Stop can fire without a UserPromptSubmit (e.g., resumed session)."""
        indexer, db_path = tmp_db
        chunk_id = indexer.index_response(
            session_id="sess-002",
            response_text="Here's the implementation...",
            cwd="/project",
        )
        assert chunk_id is not None

        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT content FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        conn.close()
        assert "implementation" in row[0]


class TestContentHashDedup:
    """session_id + content_hash prevents duplicate indexing."""

    def test_duplicate_content_skipped(self, tmp_db):
        indexer, db_path = tmp_db
        id1 = indexer.index_response("sess-001", "Same response text", "/cwd")
        id2 = indexer.index_response("sess-001", "Same response text", "/cwd")
        assert id1 is not None
        assert id2 is None  # Dedup — same content in same session

    def test_same_content_different_session_allowed(self, tmp_db):
        indexer, _ = tmp_db
        id1 = indexer.index_response("sess-001", "Same response text", "/cwd")
        id2 = indexer.index_response("sess-002", "Same response text", "/cwd")
        assert id1 is not None
        assert id2 is not None  # Different sessions — both stored


class TestChapterBoundaries:
    """PostCompact creates chapter markers."""

    def test_creates_chapter_on_compact(self, tmp_db):
        indexer, db_path = tmp_db
        chapter_id = indexer.record_chapter(
            session_id="sess-001",
            compact_summary="Discussion about auth patterns and JWT implementation.",
            trigger="auto",
        )
        assert chapter_id is not None

        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT session_id, compact_summary, trigger FROM chapters WHERE id = ?",
            (chapter_id,),
        ).fetchone()
        conn.close()
        assert row[0] == "sess-001"
        assert "auth patterns" in row[1]
        assert row[2] == "auto"

    def test_chapter_index_increments(self, tmp_db):
        indexer, db_path = tmp_db
        indexer.record_chapter("sess-001", "Chapter 0 summary", "auto")
        indexer.record_chapter("sess-001", "Chapter 1 summary", "auto")

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT chapter_index FROM chapters WHERE session_id = ? ORDER BY chapter_index",
            ("sess-001",),
        ).fetchall()
        conn.close()
        assert [r[0] for r in rows] == [0, 1]


class TestStalePromptCleanup:
    """Prompts older than 5 minutes are evicted."""

    def test_cleanup_removes_old_prompts(self, tmp_db):
        import time

        indexer, _ = tmp_db
        indexer.capture_prompt("sess-old", "old prompt", "/cwd")
        # Manually backdate the timestamp
        indexer.pending_prompts["sess-old"]["timestamp"] = time.time() - 400
        indexer.capture_prompt("sess-new", "new prompt", "/cwd")

        indexer.cleanup_stale_prompts(max_age_seconds=300)
        assert "sess-old" not in indexer.pending_prompts
        assert "sess-new" in indexer.pending_prompts


class TestProjectExtraction:
    """Project name extracted from cwd."""

    def test_extracts_project_from_cwd(self, tmp_db):
        indexer, _ = tmp_db
        assert indexer._extract_project("/Users/test/Gits/brainlayer") == "brainlayer"
        assert indexer._extract_project("/Users/test/Gits/voicelayer") == "voicelayer"

    def test_handles_trailing_slash(self, tmp_db):
        indexer, _ = tmp_db
        assert indexer._extract_project("/Users/test/Gits/brainlayer/") == "brainlayer"


class TestFallbackQueue:
    """When socket/DB is unavailable, queue to file."""

    def test_writes_to_queue_on_db_error(self, tmp_path):
        queue_dir = tmp_path / "queue"
        queue_dir.mkdir()
        indexer = RealtimeIndexer(db_path="/nonexistent/path.db", queue_dir=str(queue_dir))

        # This should fail to write to DB and fall back to queue
        indexer.index_response("sess-001", "response text", "/cwd")

        queue_files = list(queue_dir.iterdir())
        assert len(queue_files) >= 1
