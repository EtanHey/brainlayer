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
import time

import apsw

from brainlayer.vector_store import VectorStore
from brainlayer.watcher import JSONLTailer, JSONLWatcher, default_watch_roots
from brainlayer.watcher_bridge import (
    _extract_claude_conversation_id,
    _extract_project_from_source,
    _extract_raw_text,
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

    def test_extract_raw_text_accepts_bare_string_messages_for_user_and_assistant(self):
        long_text = "Bare string messages should not crash the watcher bridge extraction path."

        assert _extract_raw_text({"type": "user", "message": long_text}) == long_text
        assert _extract_raw_text({"type": "assistant", "message": long_text}) == long_text
        assert _extract_raw_text({"type": "user", "message": {"content": long_text}}) == long_text
        assert (
            _extract_raw_text({"type": "assistant", "message": {"content": [{"type": "text", "text": long_text}]}})
            == long_text
        )


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
        assert _normalize_project_name("my-project") == "my-project"

    def test_extract_from_source_file(self):
        path = "/Users/etanheyman/.claude/projects/-Users-etanheyman-Gits-brainlayer/abc123.jsonl"
        assert _extract_project_from_source(path) == "brainlayer"

    def test_extract_from_nested_subagent_source_file(self):
        path = (
            "/Users/etanheyman/.claude/projects/"
            "-Users-etanheyman-Gits-brainlayer-grill/"
            "6ff50d4a-1c98-41f4-aa55-541080c1076f/subagents/agent-acompact-123.jsonl"
        )
        assert _extract_project_from_source(path) == "brainlayer-grill"

    def test_extract_from_brainlayer_grill_top_level_file(self):
        path = "/Users/etanheyman/.claude/projects/-Users-etanheyman-Gits-brainlayer-grill/abc123.jsonl"
        assert _extract_project_from_source(path) == "brainlayer-grill"

    def test_extract_claude_conversation_id_from_top_level_jsonl(self):
        path = (
            "/Users/etanheyman/.claude/projects/-Users-etanheyman-Gits-brainlayer/"
            "3679128a-f371-445f-82ba-b3946e2f20b6.jsonl"
        )
        assert _extract_claude_conversation_id(path) == "3679128a-f371-445f-82ba-b3946e2f20b6"

    def test_extract_claude_conversation_id_from_nested_subagent_jsonl(self):
        path = (
            "/Users/etanheyman/.claude/projects/-Users-etanheyman-Gits-brainlayer/"
            "3679128a-f371-445f-82ba-b3946e2f20b6/subagents/agent-acompact-123.jsonl"
        )
        assert _extract_claude_conversation_id(path) == "3679128a-f371-445f-82ba-b3946e2f20b6"


# ── Flush Callback ───────────────────────────────────────────────────────────


class TestFlushCallback:
    def test_arbitrated_flush_enqueues_watcher_chunks_without_db_write(self, tmp_path, monkeypatch):
        db_path = tmp_path / "test.db"
        queue_dir = tmp_path / "queue"
        VectorStore(db_path).close()
        monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue_dir))

        flush = create_flush_callback(db_path, arbitrated=True)
        entry = _make_jsonl_entry(text=_LONG_TEXT, entry_type="assistant")
        entry["_source_file"] = str(tmp_path / "projects" / "-Users-test-Gits-myproject" / "session.jsonl")

        inserted = flush([entry])

        queued_files = list(queue_dir.glob("watcher-*.jsonl"))
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT COUNT(*) FROM chunks WHERE source = 'realtime_watcher'").fetchone()
        conn.close()
        assert inserted == 1
        assert len(queued_files) == 1
        assert rows == (0,)

    def test_direct_write_busy_spills_to_queue_instead_of_silently_dropping(self, tmp_path, monkeypatch):
        import brainlayer.watcher_bridge as bridge

        db_path = tmp_path / "test.db"
        queue_dir = tmp_path / "queue"
        VectorStore(db_path).close()
        monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue_dir))
        monkeypatch.setenv("BRAINLAYER_WATCHER_WRITE_DEADLINE_S", "0")
        monkeypatch.setattr(bridge, "find_duplicate", lambda *_args, **_kwargs: (_ for _ in ()).throw(apsw.BusyError()))

        flush = create_flush_callback(db_path, arbitrated=False)
        entry = _make_jsonl_entry(text=_LONG_TEXT, entry_type="assistant")
        entry["_source_file"] = str(tmp_path / "projects" / "test-project" / "session.jsonl")
        entry["_line_end_offset"] = 123

        result = flush([entry])

        assert result[str(entry["_source_file"])] == 123
        assert len(list(queue_dir.glob("watcher-*.jsonl"))) == 1
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT COUNT(*) FROM chunks WHERE source = 'realtime_watcher'").fetchone()
        conn.close()
        assert rows == (0,)

    def test_classification_poison_entry_confirms_its_offset_and_does_not_block_later_entries(
        self, tmp_path, monkeypatch
    ):
        import brainlayer.watcher_bridge as bridge

        db_path = tmp_path / "test.db"
        VectorStore(db_path).close()
        original_classify = bridge.classify_content

        def classify_or_poison(entry):
            if entry.get("poison"):
                raise ValueError("deterministic bad entry")
            return original_classify(entry)

        monkeypatch.setattr(bridge, "classify_content", classify_or_poison)
        flush = create_flush_callback(db_path, arbitrated=False)
        source_file = str(tmp_path / "projects" / "test-project" / "session.jsonl")
        poison = _make_jsonl_entry(text=_LONG_TEXT, entry_type="assistant", poison=True)
        poison["_source_file"] = source_file
        poison["_line_end_offset"] = 10
        healthy = _make_jsonl_entry(text=_LONG_TEXT + " healthy", entry_type="assistant")
        healthy["_source_file"] = source_file
        healthy["_line_end_offset"] = 30

        result = flush([poison, healthy])

        assert result[source_file] == 30
        assert result.skipped == 1
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT COUNT(*) FROM chunks WHERE source = 'realtime_watcher'").fetchone()
        conn.close()
        assert rows[0] >= 1

    def test_inserts_valid_entry(self, tmp_path):
        db_path = tmp_path / "test.db"
        VectorStore(db_path).close()

        flush = create_flush_callback(db_path, arbitrated=False)
        entry = _make_jsonl_entry(text=_LONG_TEXT, entry_type="assistant")
        entry["_source_file"] = str(tmp_path / "projects" / "-Users-test-Gits-myproject" / "session.jsonl")

        flush([entry])

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT id, content, source FROM chunks WHERE source = 'realtime_watcher'").fetchall()
        conn.close()
        assert len(rows) >= 1

    def test_inserts_claude_conversation_id_metadata(self, tmp_path):
        db_path = tmp_path / "test.db"
        VectorStore(db_path).close()

        flush = create_flush_callback(db_path, arbitrated=False)
        conversation_id = "3679128a-f371-445f-82ba-b3946e2f20b6"
        entry = _make_jsonl_entry(text=_LONG_TEXT, entry_type="assistant", sessionId="brainlayer-session-9")
        entry["_source_file"] = str(tmp_path / "projects" / "-Users-test-Gits-myproject" / f"{conversation_id}.jsonl")

        flush([entry])

        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT metadata FROM chunks WHERE source = 'realtime_watcher'").fetchone()
        conn.close()

        assert row is not None
        metadata = json.loads(row[0])
        assert metadata["session_id"] == "brainlayer-session-9"
        assert metadata["claude_conversation_id"] == conversation_id

    def test_insert_sets_ingested_at(self, tmp_path):
        db_path = tmp_path / "test.db"
        VectorStore(db_path).close()

        flush = create_flush_callback(db_path, arbitrated=False)
        entry = _make_jsonl_entry(
            text=(
                "Watcher ingested_at regression coverage should store a fresh unix timestamp "
                "for a substantive assistant response with enough detail to pass the classifier "
                "and persist through the realtime watcher insert path."
            ),
            entry_type="assistant",
        )
        entry["_source_file"] = str(tmp_path / "projects" / "test-project" / "session.jsonl")

        before = int(time.time())
        flush([entry])
        after = int(time.time())

        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT ingested_at FROM chunks WHERE source = 'realtime_watcher'").fetchone()
        conn.close()

        assert row is not None
        assert row[0] is not None
        assert before - 5 <= row[0] <= after + 5

    def test_skips_noise_entries(self, tmp_path):
        db_path = tmp_path / "test.db"
        VectorStore(db_path).close()

        flush = create_flush_callback(db_path, arbitrated=False)
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

        flush = create_flush_callback(db_path, arbitrated=False)
        flush([{"type": "progress", "_source_file": "/tmp/test.jsonl", "data": {}}])

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT COUNT(*) FROM chunks WHERE source = 'realtime_watcher'").fetchone()
        conn.close()
        assert rows[0] == 0

    def test_dedup_same_content(self, tmp_path):
        db_path = tmp_path / "test.db"
        VectorStore(db_path).close()

        flush = create_flush_callback(db_path, arbitrated=False)
        entry = _make_jsonl_entry(
            text="This exact same content should only appear once in the database even when flushed twice",
            entry_type="assistant",
        )
        entry["_source_file"] = "/tmp/projects/test-project/session.jsonl"

        flush([entry])
        flush([entry])

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT COUNT(*), MAX(seen_count) FROM chunks WHERE source = 'realtime_watcher'").fetchone()
        audit_count = conn.execute("SELECT COUNT(*) FROM dedupe_audit WHERE mechanism = 'sha256_same_id'").fetchone()[0]
        conn.close()
        assert rows[0] == 1
        assert rows[1] == 2
        assert audit_count == 1

    def test_dedup_merges_same_content_across_source_files(self, tmp_path):
        db_path = tmp_path / "test.db"
        VectorStore(db_path).close()

        flush = create_flush_callback(db_path, arbitrated=False)
        content = "This exact same content should merge even when it appears in a different session file"
        first = _make_jsonl_entry(text=content, entry_type="assistant", timestamp="2026-05-16T09:00:00Z")
        second = _make_jsonl_entry(text=content, entry_type="assistant", timestamp="2026-05-16T10:00:00Z")
        first["_source_file"] = "/tmp/projects/test-project/alpha-session.jsonl"
        second["_source_file"] = "/tmp/projects/test-project/beta-session.jsonl"

        flush([first, second])

        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT id, seen_count, last_seen_at FROM chunks WHERE source = 'realtime_watcher'"
        ).fetchone()
        aliases = conn.execute("SELECT old_chunk_id, canonical_chunk_id FROM chunk_id_alias").fetchall()
        audit_count = conn.execute("SELECT COUNT(*) FROM dedupe_audit").fetchone()[0]
        conn.close()

        assert row[1] == 2
        assert row[2] == "2026-05-16T10:00:00Z"
        assert len(aliases) == 1
        assert aliases[0][1] == row[0]
        assert audit_count == 1

    def test_direct_dedup_into_non_realtime_canonical_records_liveness(self, tmp_path):
        db_path = tmp_path / "test.db"
        VectorStore(db_path).close()

        flush = create_flush_callback(db_path, arbitrated=False)
        content = "Direct watcher dedupe liveness should count durable merges into manual canonical chunks"
        first = _make_jsonl_entry(text=content, entry_type="assistant", timestamp="2026-05-16T09:00:00Z")
        second = _make_jsonl_entry(text=content, entry_type="assistant", timestamp="2026-05-16T10:00:00Z")
        first["_source_file"] = "/tmp/projects/test-project/alpha-session.jsonl"
        second["_source_file"] = "/tmp/projects/test-project/beta-session.jsonl"

        flush([first])
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("UPDATE chunks SET source = 'manual' WHERE source = 'realtime_watcher'")
            conn.commit()

        flush([second])

        conn = sqlite3.connect(str(db_path))
        source, seen_count = conn.execute("SELECT source, seen_count FROM chunks").fetchone()
        realtime_rows = conn.execute("SELECT COUNT(*) FROM chunks WHERE source = 'realtime_watcher'").fetchone()[0]
        liveness_count = conn.execute("SELECT COUNT(*) FROM watcher_liveness_events").fetchone()[0]
        conn.close()

        assert source == "manual"
        assert seen_count == 2
        assert realtime_rows == 0
        assert liveness_count == 2

    def test_auto_tags_detected_correction_user_message(self, tmp_path):
        db_path = tmp_path / "test.db"
        VectorStore(db_path).close()

        flush = create_flush_callback(db_path, arbitrated=False)
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
    def test_watcher_denylist_blocks_agent_transcript_roots_before_db_insert(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)
        db_path = tmp_path / "test.db"
        VectorStore(db_path).close()

        denylisted = {
            "claude": tmp_path / ".claude" / "projects" / "proj" / "session-123" / "subagents" / "agent-a111.jsonl",
            "codex": tmp_path / ".codex" / "sessions" / "2026" / "07" / "02" / "worker.jsonl",
            "cursor": tmp_path
            / ".cursor"
            / "projects"
            / "repo"
            / "agent-transcripts"
            / "agent-session"
            / "agent-session.jsonl",
            "gemini": tmp_path / ".gemini" / "sessions" / "worker.jsonl",
        }
        for provider, path in denylisted.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    _make_jsonl_entry(
                        text=(
                            f"BL10SHOULDNOTINDEX{provider.upper()} source denylist sentinel. "
                            "This assistant response explains durable watcher ingestion and contains enough "
                            "substantial technical detail for classification and chunking."
                        ),
                        entry_type="assistant",
                    )
                )
                + "\n"
            )

        control = tmp_path / ".claude" / "projects" / "proj" / "direct-session.jsonl"
        control.write_text(
            json.dumps(
                _make_jsonl_entry(
                    text=(
                        "BL10CONTROLINDEXES genuine direct Claude session sentinel. "
                        "This assistant response explains durable watcher ingestion and contains enough "
                        "substantial technical detail for classification and chunking."
                    ),
                    entry_type="assistant",
                )
            )
            + "\n"
        )

        flush = create_flush_callback(db_path, arbitrated=False)
        watcher = JSONLWatcher(
            watch_roots=default_watch_roots(home=tmp_path),
            registry_path=tmp_path / "offsets.json",
            on_flush=flush,
            batch_size=1,
        )
        watcher.poll_once()
        watcher.indexer.flush()

        conn = sqlite3.connect(str(db_path))
        try:
            denylist_globs = [
                str(tmp_path / ".claude" / "projects") + "/*/*/subagents/*",
                str(tmp_path / ".codex" / "sessions") + "/*",
                str(tmp_path / ".cursor") + "/*/agent-transcripts/*",
                str(tmp_path / ".gemini" / "sessions") + "/*",
            ]
            for glob in denylist_globs:
                assert conn.execute("SELECT count(*) FROM chunks WHERE source_file GLOB ?", (glob,)).fetchone()[0] == 0
            for provider in denylisted:
                assert (
                    conn.execute(
                        "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH ?",
                        (f'"BL10SHOULDNOTINDEX{provider.upper()}"',),
                    ).fetchone()[0]
                    == 0
                )
            assert (
                conn.execute(
                    "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH ?",
                    ('"BL10CONTROLINDEXES"',),
                ).fetchone()[0]
                >= 1
            )
        finally:
            conn.close()

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

        flush = create_flush_callback(db_path, arbitrated=False)
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

        flush = create_flush_callback(db_path, arbitrated=False)
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

    def test_watcher_backfills_existing_nested_subagent_file_with_project(self, tmp_path):
        db_path = tmp_path / "test.db"
        VectorStore(db_path).close()

        project_dir = tmp_path / "projects" / "-Users-test-Gits-brainlayer-grill"
        subagent_dir = project_dir / "session-123" / "subagents"
        subagent_dir.mkdir(parents=True)
        jsonl_file = subagent_dir / "agent-acompact-123.jsonl"
        entry = _make_jsonl_entry(
            text="Nested subagent transcript should be discovered on startup and stored under brainlayer-grill",
            entry_type="assistant",
        )
        jsonl_file.write_text(json.dumps(entry) + "\n")

        flush = create_flush_callback(db_path, arbitrated=False)
        watcher = JSONLWatcher(
            watch_dir=tmp_path / "projects",
            registry_path=tmp_path / "offsets.json",
            on_flush=flush,
            batch_size=1,
        )

        watcher.poll_once()
        watcher.indexer.flush()

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT project, source_file FROM chunks WHERE source = 'realtime_watcher'").fetchall()
        conn.close()
        assert rows
        assert rows[0][0] == "brainlayer-grill"
        assert rows[0][1].endswith("subagents/agent-acompact-123.jsonl")


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
