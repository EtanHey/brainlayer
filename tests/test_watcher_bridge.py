"""Tests for the watcher → BrainLayer bridge (end-to-end).

Covers:
- Pre-classify filters (entry type whitelist, min length, system-reminder)
- Post-chunk filters (system-reminder stripping, pure deletion diffs)
- JSONL entry → classify → chunk → DB insert
- Dedup via content hash
- Project name extraction
- FTS5 searchability of inserted chunks
- Rewind detection (file shrink = checkpoint restore)
"""

import json
import sqlite3

from brainlayer.vector_store import VectorStore
from brainlayer.watcher import JSONLTailer, JSONLWatcher
from brainlayer.watcher_bridge import (
    _extract_project_from_source,
    _normalize_project_name,
    _strip_system_reminders,
    create_flush_callback,
    should_skip_chunk_content,
    should_skip_entry,
)


def _make_jsonl_entry(role="assistant", text="Test response", entry_type=None, **extra):
    """Create a JSONL entry matching Claude Code's format."""
    entry = {
        "type": entry_type or role,
        "message": {
            "role": role,
            "content": [{"type": "text", "text": text}],
        },
        "timestamp": "2026-03-26T12:00:00Z",
    }
    entry.update(extra)
    return entry


_LONG_TEXT = "This is a substantive test response with more than enough content to reliably pass the classification and chunking pipeline thresholds"


# ── Pre-Classify Filters ─────────────────────────────────────────────────────


class TestPreClassifyFilters:
    def test_skip_progress_type(self):
        assert should_skip_entry({"type": "progress"}) is not None

    def test_skip_file_history_snapshot(self):
        assert should_skip_entry({"type": "file-history-snapshot"}) is not None

    def test_skip_system_type(self):
        assert should_skip_entry({"type": "system"}) is not None

    def test_skip_pr_link(self):
        assert should_skip_entry({"type": "pr-link"}) is not None

    def test_keep_user_type(self):
        entry = _make_jsonl_entry(role="user", text="What is the architecture of this project?", entry_type="user")
        assert should_skip_entry(entry) is None

    def test_keep_assistant_type(self):
        entry = _make_jsonl_entry(text=_LONG_TEXT, entry_type="assistant")
        assert should_skip_entry(entry) is None

    def test_skip_short_content(self):
        entry = _make_jsonl_entry(text="ok", entry_type="assistant")
        assert should_skip_entry(entry) == "too_short"

    def test_skip_system_reminder_only(self):
        entry = _make_jsonl_entry(
            text="<system-reminder>Hook output that's already in BrainLayer</system-reminder>",
            entry_type="assistant",
        )
        assert should_skip_entry(entry) == "system_reminder_only"

    def test_keep_content_with_system_reminder_plus_real_text(self):
        entry = _make_jsonl_entry(
            text="<system-reminder>hook output</system-reminder>\nHere is my actual substantive response about the architecture",
            entry_type="assistant",
        )
        assert should_skip_entry(entry) is None

    def test_skip_unknown_type(self):
        assert should_skip_entry({"type": "unknown-new-type"}) is not None


# ── Post-Chunk Filters ───────────────────────────────────────────────────────


class TestPostChunkFilters:
    def test_strip_system_reminders(self):
        text = "Real content <system-reminder>noise</system-reminder> more real"
        cleaned = _strip_system_reminders(text)
        assert "system-reminder" not in cleaned
        assert "Real content" in cleaned

    def test_skip_system_reminder_residue(self):
        assert should_skip_chunk_content("<system-reminder>all noise</system-reminder>") is not None

    def test_keep_real_content(self):
        assert (
            should_skip_chunk_content("This is a substantial piece of real content about architecture decisions")
            is None
        )


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
        VectorStore(db_path).close()

        flush = create_flush_callback(db_path)
        entry = _make_jsonl_entry(text=_LONG_TEXT, entry_type="assistant")
        entry["_source_file"] = str(tmp_path / "projects" / "-Users-test-Gits-myproject" / "session.jsonl")

        flush([entry])

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT id, content, source FROM chunks WHERE source = 'realtime_watcher'").fetchall()
        conn.close()
        assert len(rows) >= 1

    def test_skips_noise_entries(self, tmp_path):
        db_path = tmp_path / "test.db"
        VectorStore(db_path).close()

        flush = create_flush_callback(db_path)
        entry = _make_jsonl_entry(text="ok", entry_type="assistant")
        entry["_source_file"] = "/tmp/test.jsonl"

        flush([entry])

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT COUNT(*) FROM chunks WHERE source = 'realtime_watcher'").fetchone()
        conn.close()
        assert rows[0] == 0

    def test_skips_progress_type(self, tmp_path):
        db_path = tmp_path / "test.db"
        VectorStore(db_path).close()

        flush = create_flush_callback(db_path)
        flush([{"type": "progress", "_source_file": "/tmp/test.jsonl", "data": {}}])

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT COUNT(*) FROM chunks WHERE source = 'realtime_watcher'").fetchone()
        conn.close()
        assert rows[0] == 0

    def test_dedup_same_content(self, tmp_path):
        db_path = tmp_path / "test.db"
        VectorStore(db_path).close()

        flush = create_flush_callback(db_path)
        entry = _make_jsonl_entry(
            text="This exact same content should only appear once in the database even when flushed twice",
            entry_type="assistant",
        )
        entry["_source_file"] = "/tmp/projects/test-project/session.jsonl"

        flush([entry])
        flush([entry])

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT COUNT(*) FROM chunks WHERE source = 'realtime_watcher'").fetchone()
        conn.close()
        assert rows[0] == 1

    def test_auto_tags_detected_correction_user_message(self, tmp_path):
        db_path = tmp_path / "test.db"
        VectorStore(db_path).close()

        flush = create_flush_callback(db_path)
        entry = _make_jsonl_entry(
            role="user",
            text="No, that's wrong. Avi works at Lightricks.",
            entry_type="user",
        )
        entry["_source_file"] = "/tmp/projects/test-project/session.jsonl"

        flush([entry])

        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT tags FROM chunks WHERE source = 'realtime_watcher' AND content_type = 'user_message'"
        ).fetchone()
        tag_rows = conn.execute(
            "SELECT tag FROM chunk_tags WHERE chunk_id IN (SELECT id FROM chunks WHERE source = 'realtime_watcher')"
        ).fetchall()
        conn.close()

        assert row is not None
        tags = json.loads(row[0])
        assert "correction" in tags
        assert "correction:factual" in tags
        assert "auto-detected" in tags
        assert {tag for (tag,) in tag_rows} >= {"correction", "correction:factual", "auto-detected"}


# ── Full Pipeline Integration ────────────────────────────────────────────────


class TestFullPipeline:
    def test_watcher_to_db(self, tmp_path):
        db_path = tmp_path / "test.db"
        VectorStore(db_path).close()

        project_dir = tmp_path / "projects" / "test-project"
        project_dir.mkdir(parents=True)
        jsonl_file = project_dir / "session1.jsonl"
        entry = _make_jsonl_entry(
            text="Integration test: the watcher should pick this up and insert it into brainlayer database for real-time search",
            entry_type="assistant",
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

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT content FROM chunks WHERE source = 'realtime_watcher'").fetchall()
        conn.close()
        assert len(rows) >= 1

    def test_fts5_searchable_after_insert(self, tmp_path):
        db_path = tmp_path / "test.db"
        VectorStore(db_path).close()

        project_dir = tmp_path / "projects" / "test-project"
        project_dir.mkdir(parents=True)
        jsonl_file = project_dir / "session1.jsonl"
        entry = _make_jsonl_entry(
            text="UniqueSearchableToken: the watcher bridge should make this findable via FTS5 search",
            entry_type="assistant",
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

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            """SELECT c.content FROM chunks_fts f
               JOIN chunks c ON c.id = f.chunk_id
               WHERE chunks_fts MATCH '"UniqueSearchableToken"'"""
        ).fetchall()
        conn.close()
        assert len(rows) >= 1
        assert "UniqueSearchableToken" in rows[0][0]


# ── Rewind Detection ────────────────────────────────────────────────────────


class TestRewindDetection:
    def test_tailer_detects_file_shrink(self, tmp_path):
        f = tmp_path / "session.jsonl"
        f.write_text('{"id":"1"}\n{"id":"2"}\n{"id":"3"}\n')

        tailer = JSONLTailer(str(f))
        tailer.read_new_lines()
        old_offset = tailer.offset
        assert old_offset > 0

        # Simulate checkpoint restore — file shrinks
        f.write_text('{"id":"1"}\n')

        assert tailer.check_rewind() is True
        assert tailer.rewound is True
        assert tailer.rewind_old_offset == old_offset
        assert tailer.offset == 0

    def test_tailer_no_rewind_on_append(self, tmp_path):
        f = tmp_path / "session.jsonl"
        f.write_text('{"id":"1"}\n')

        tailer = JSONLTailer(str(f))
        tailer.read_new_lines()

        with open(f, "a") as fh:
            fh.write('{"id":"2"}\n')

        assert tailer.check_rewind() is False
        assert tailer.rewound is False

    def test_watcher_calls_rewind_callback(self, tmp_path):
        project_dir = tmp_path / "projects" / "test-project"
        project_dir.mkdir(parents=True)
        f = project_dir / "abc12345.jsonl"
        f.write_text('{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"first"}]}}\n')

        rewind_calls = []

        def on_rewind(filepath, session_id, old_offset, new_offset):
            rewind_calls.append((session_id, old_offset, new_offset))

        watcher = JSONLWatcher(
            watch_dir=tmp_path / "projects",
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda x: None,
            on_rewind=on_rewind,
            batch_size=1,
        )

        watcher.poll_once()

        # Simulate rewind
        f.write_text('{"type":"user"}\n')
        watcher.poll_once()

        assert len(rewind_calls) == 1
        assert rewind_calls[0][0] == "abc12345"
