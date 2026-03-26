"""Tests for the watcher → BrainLayer bridge (end-to-end).

Covers:
- JSONL entry → classify → chunk → DB insert
- Dedup via content hash
- Project name extraction
- FTS5 searchability of inserted chunks
"""

import json
import sqlite3

import pytest

from brainlayer.vector_store import VectorStore
from brainlayer.watcher import JSONLWatcher
from brainlayer.watcher_bridge import (
    _extract_project_from_source,
    _normalize_project_name,
    create_flush_callback,
)


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test.db"
    s = VectorStore(db_path)
    yield s
    s.close()


def _make_jsonl_entry(role="assistant", text="Test response", **extra):
    """Create a JSONL entry matching Claude Code's format."""
    entry = {
        "type": "assistant",
        "message": {
            "role": role,
            "content": [{"type": "text", "text": text}],
        },
        "timestamp": "2026-03-26T12:00:00Z",
    }
    entry.update(extra)
    return entry


# ── Project Name Extraction ──────────────────────────────────────────────────


class TestProjectExtraction:
    def test_normalize_encoded_path(self):
        assert _normalize_project_name("-Users-etanheyman-Gits-brainlayer") == "brainlayer"

    def test_simple_name(self):
        assert _normalize_project_name("my-project") == "project"

    def test_extract_from_source_file(self):
        path = "/Users/etanheyman/.claude/projects/-Users-etanheyman-Gits-brainlayer/abc123.jsonl"
        assert _extract_project_from_source(path) == "brainlayer"


# ── Flush Callback ───────────────────────────────────────────────────────────


class TestFlushCallback:
    def test_inserts_valid_entry(self, tmp_path):
        db_path = tmp_path / "test.db"
        VectorStore(db_path).close()  # init schema

        flush = create_flush_callback(db_path)
        entry = _make_jsonl_entry(
            text="This is a substantive test response with more than enough content to reliably pass the classification and chunking pipeline thresholds"
        )
        entry["_source_file"] = str(tmp_path / "projects" / "-Users-test-Gits-myproject" / "session.jsonl")

        flush([entry])

        # Verify chunk in DB
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT id, content, source FROM chunks WHERE source = 'realtime_watcher'").fetchall()
        conn.close()
        assert len(rows) >= 1
        assert "realtime_watcher" in rows[0][2]

    def test_skips_noise_entries(self, tmp_path):
        db_path = tmp_path / "test.db"
        VectorStore(db_path).close()

        flush = create_flush_callback(db_path)
        # Very short content → classified as noise → skipped
        entry = _make_jsonl_entry(text="ok")
        entry["_source_file"] = "/tmp/test.jsonl"

        flush([entry])

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT COUNT(*) FROM chunks WHERE source = 'realtime_watcher'").fetchone()
        conn.close()
        assert rows[0] == 0

    def test_dedup_same_content(self, tmp_path):
        db_path = tmp_path / "test.db"
        VectorStore(db_path).close()

        flush = create_flush_callback(db_path)
        entry = _make_jsonl_entry(
            text="This exact same content should only appear once in the database even when flushed twice"
        )
        entry["_source_file"] = "/tmp/projects/test-project/session.jsonl"

        flush([entry])
        flush([entry])

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT COUNT(*) FROM chunks WHERE source = 'realtime_watcher'").fetchone()
        conn.close()
        assert rows[0] == 1


# ── Full Pipeline Integration ────────────────────────────────────────────────


class TestFullPipeline:
    def test_watcher_to_db(self, tmp_path):
        """Write JSONL → watcher polls → chunks appear in DB."""
        db_path = tmp_path / "test.db"
        VectorStore(db_path).close()

        # Set up project directory with a JSONL file
        project_dir = tmp_path / "projects" / "test-project"
        project_dir.mkdir(parents=True)
        jsonl_file = project_dir / "session1.jsonl"
        entry = _make_jsonl_entry(
            text="Integration test: the watcher should pick this up and insert it into brainlayer database for real-time search"
        )
        jsonl_file.write_text(json.dumps(entry) + "\n")

        flush = create_flush_callback(db_path)
        watcher = JSONLWatcher(
            watch_dir=tmp_path / "projects",
            registry_path=tmp_path / "offsets.json",
            on_flush=flush,
            batch_size=1,
        )

        # One poll cycle
        count = watcher.poll_once()
        watcher.indexer.flush()

        # Verify in DB
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT content FROM chunks WHERE source = 'realtime_watcher'").fetchall()
        conn.close()
        assert len(rows) >= 1

    def test_fts5_searchable_after_insert(self, tmp_path):
        """Inserted chunks should be findable via FTS5 immediately."""
        db_path = tmp_path / "test.db"
        VectorStore(db_path).close()

        project_dir = tmp_path / "projects" / "test-project"
        project_dir.mkdir(parents=True)
        jsonl_file = project_dir / "session1.jsonl"
        entry = _make_jsonl_entry(
            text="UniqueSearchableToken: the watcher bridge should make this findable via FTS5 search"
        )
        jsonl_file.write_text(json.dumps(entry) + "\n")

        flush = create_flush_callback(db_path)
        watcher = JSONLWatcher(
            watch_dir=tmp_path / "projects",
            registry_path=tmp_path / "offsets.json",
            on_flush=flush,
            batch_size=1,
        )
        watcher.poll_once()
        watcher.indexer.flush()

        # Search via FTS5
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            """SELECT c.content FROM chunks_fts f
               JOIN chunks c ON c.id = f.chunk_id
               WHERE chunks_fts MATCH '"UniqueSearchableToken"'"""
        ).fetchall()
        conn.close()
        assert len(rows) >= 1
        assert "UniqueSearchableToken" in rows[0][0]
