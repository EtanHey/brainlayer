"""Tests for real-time JSONL file watcher prototype.

Covers:
- OffsetRegistry: persist/restore offsets, atomic writes, inode tracking
- JSONLTailer: tail-follow, partial line buffering, corrupt line handling
- BatchIndexer: batching, flush interval, thread safety
- JSONLWatcher: file discovery, poll cycle, end-to-end integration
"""

import json
import os
import sqlite3
import threading
import time

import pytest

from brainlayer.alarm import BrainLayerAlarm
from brainlayer.watcher import BatchIndexer, CoverageWatchdog, JSONLTailer, JSONLWatcher, OffsetRegistry, WatchRoot
from brainlayer.watcher_bridge import FlushWatermarks

# ── OffsetRegistry Tests ─────────────────────────────────────────────────────


class TestOffsetRegistry:
    def test_set_and_get(self, tmp_path):
        reg = OffsetRegistry(tmp_path / "offsets.json")
        reg.set("/path/to/file.jsonl", 1024, 12345)
        offset, inode = reg.get("/path/to/file.jsonl")
        assert offset == 1024
        assert inode == 12345

    def test_get_unknown_returns_zero(self, tmp_path):
        reg = OffsetRegistry(tmp_path / "offsets.json")
        offset, inode = reg.get("/nonexistent")
        assert offset == 0
        assert inode == 0

    def test_flush_and_reload(self, tmp_path):
        path = tmp_path / "offsets.json"
        reg = OffsetRegistry(path)
        reg.set("/a.jsonl", 500, 111)
        reg.flush()

        reg2 = OffsetRegistry(path)
        offset, inode = reg2.get("/a.jsonl")
        assert offset == 500
        assert inode == 111

    def test_flush_noop_when_clean(self, tmp_path):
        reg = OffsetRegistry(tmp_path / "offsets.json")
        assert reg.flush() is True  # no-op, still True

    def test_remove_entry(self, tmp_path):
        reg = OffsetRegistry(tmp_path / "offsets.json")
        reg.set("/a.jsonl", 100, 1)
        reg.remove("/a.jsonl")
        offset, inode = reg.get("/a.jsonl")
        assert offset == 0

    def test_load_corrupt_file(self, tmp_path):
        path = tmp_path / "offsets.json"
        path.write_text("not json{{{")
        reg = OffsetRegistry(path)
        offset, inode = reg.get("/anything")
        assert offset == 0


# ── JSONLTailer Tests ────────────────────────────────────────────────────────


class TestJSONLTailer:
    def test_read_complete_lines(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_text('{"type":"msg","id":"1"}\n{"type":"msg","id":"2"}\n')

        tailer = JSONLTailer(str(f))
        lines = tailer.read_new_lines()
        assert len(lines) == 2
        assert lines[0]["id"] == "1"
        assert lines[1]["id"] == "2"

    def test_partial_line_buffered(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_bytes(b'{"type":"msg","id":"1"}\n{"partial":')

        tailer = JSONLTailer(str(f))
        lines = tailer.read_new_lines()
        assert len(lines) == 1  # Only the complete line

        # Append the rest
        with open(f, "ab") as fh:
            fh.write(b'"value"}\n')
        lines = tailer.read_new_lines()
        assert len(lines) == 1
        assert lines[0]["partial"] == "value"

    def test_corrupt_line_skipped(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_text('{"good":"line"}\nnot json at all\n{"also":"good"}\n')

        tailer = JSONLTailer(str(f))
        lines = tailer.read_new_lines()
        assert len(lines) == 2
        assert lines[0]["good"] == "line"
        assert lines[1]["also"] == "good"

    def test_resume_from_offset(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_text('{"id":"1"}\n{"id":"2"}\n')

        # First line is 13 bytes + newline = 14
        first_line_bytes = len(b'{"id":"1"}\n')
        tailer = JSONLTailer(str(f), offset=first_line_bytes)
        lines = tailer.read_new_lines()
        assert len(lines) == 1
        assert lines[0]["id"] == "2"

    def test_empty_file(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_text("")
        tailer = JSONLTailer(str(f))
        assert tailer.read_new_lines() == []

    def test_nonexistent_file(self, tmp_path):
        tailer = JSONLTailer(str(tmp_path / "nope.jsonl"))
        assert tailer.read_new_lines() == []

    def test_incremental_append(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_text('{"id":"1"}\n')

        tailer = JSONLTailer(str(f))
        lines1 = tailer.read_new_lines()
        assert len(lines1) == 1

        # Append more
        with open(f, "a") as fh:
            fh.write('{"id":"2"}\n{"id":"3"}\n')
        lines2 = tailer.read_new_lines()
        assert len(lines2) == 2

        # No new data
        lines3 = tailer.read_new_lines()
        assert len(lines3) == 0

    def test_non_dict_json_skipped(self, tmp_path):
        """JSON arrays and strings should be skipped — only dicts."""
        f = tmp_path / "test.jsonl"
        f.write_text('[1,2,3]\n"just a string"\n{"valid":"dict"}\n')

        tailer = JSONLTailer(str(f))
        lines = tailer.read_new_lines()
        assert len(lines) == 1
        assert lines[0]["valid"] == "dict"

    def test_get_inode(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_text("{}\n")
        tailer = JSONLTailer(str(f))
        assert tailer.get_inode() > 0

    def test_get_inode_missing_file(self, tmp_path):
        tailer = JSONLTailer(str(tmp_path / "nope.jsonl"))
        assert tailer.get_inode() == 0


# ── BatchIndexer Tests ───────────────────────────────────────────────────────


class TestBatchIndexer:
    def test_flush_on_batch_size(self):
        flushed = []
        indexer = BatchIndexer(on_flush=lambda items: flushed.extend(items), batch_size=3)
        indexer.add([{"a": 1}, {"b": 2}, {"c": 3}])
        assert len(flushed) == 3

    def test_no_flush_under_batch_size(self):
        flushed = []
        indexer = BatchIndexer(on_flush=lambda items: flushed.extend(items), batch_size=5)
        indexer.add([{"a": 1}, {"b": 2}])
        assert len(flushed) == 0

    def test_tick_flushes_on_interval(self):
        flushed = []
        indexer = BatchIndexer(
            on_flush=lambda items: flushed.extend(items),
            batch_size=100,
            flush_interval_ms=0,  # immediate
        )
        indexer.add([{"a": 1}])
        indexer.tick()
        assert len(flushed) == 1

    def test_manual_flush(self):
        flushed = []
        indexer = BatchIndexer(on_flush=lambda items: flushed.extend(items), batch_size=100)
        indexer.add([{"a": 1}, {"b": 2}])
        indexer.flush()
        assert len(flushed) == 2

    def test_total_flushed_counter(self):
        indexer = BatchIndexer(on_flush=lambda items: None, batch_size=2)
        indexer.add([{"a": 1}, {"b": 2}])
        indexer.add([{"c": 3}, {"d": 4}])
        assert indexer.total_flushed == 4

    def test_flush_callback_watermark_is_forwarded_to_offset_callback(self):
        confirmed = []
        indexer = BatchIndexer(
            on_flush=lambda items: {"/tmp/source.jsonl": items[-1]["_line_end_offset"]},
            batch_size=2,
            on_confirm_offsets=confirmed.append,
        )

        indexer.add(
            [
                {"_source_file": "/tmp/source.jsonl", "_line_end_offset": 10},
                {"_source_file": "/tmp/source.jsonl", "_line_end_offset": 20},
            ]
        )

        assert confirmed == [{"/tmp/source.jsonl": 20}]

    def test_flush_error_retains_buffer(self):
        def bad_flush(items):
            raise RuntimeError("flush failed")

        indexer = BatchIndexer(on_flush=bad_flush, batch_size=1)
        # Should not raise, and buffer should be retained for retry
        indexer.add([{"a": 1}])
        assert indexer.total_flushed == 0
        assert len(indexer._buffer) == 1  # Retained for retry


# ── JSONLWatcher Integration Tests ──────────────────────────────────────────


class TestJSONLWatcher:
    def _make_project_dir(self, tmp_path, name="test-project"):
        project = tmp_path / "projects" / name
        project.mkdir(parents=True)
        return project

    def test_discover_jsonl_files(self, tmp_path):
        project = self._make_project_dir(tmp_path)
        (project / "session1.jsonl").write_text('{"id":"1"}\n')
        (project / "session2.jsonl").write_text('{"id":"2"}\n')
        (project / "notes.txt").write_text("not jsonl")

        watcher = JSONLWatcher(
            watch_dir=tmp_path / "projects",
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda x: None,
        )
        files = watcher._discover_jsonl_files()
        assert len(files) == 2
        assert all(f.endswith(".jsonl") for f in files)

    def test_discover_jsonl_files_includes_nested_subagents(self, tmp_path):
        project = self._make_project_dir(tmp_path, "-Users-test-Gits-brainlayer-grill")
        session_dir = project / "session-123"
        subagents_dir = session_dir / "subagents"
        subagents_dir.mkdir(parents=True)
        nested_jsonl = subagents_dir / "agent-acompact-1.jsonl"
        nested_jsonl.write_text('{"id":"1"}\n')

        watcher = JSONLWatcher(
            watch_dir=tmp_path / "projects",
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda x: None,
        )

        files = watcher._discover_jsonl_files()

        assert str(nested_jsonl) in files

    def test_poll_once_reads_new_lines(self, tmp_path):
        project = self._make_project_dir(tmp_path)
        (project / "s1.jsonl").write_text('{"type":"msg","text":"hello"}\n')

        flushed = []
        watcher = JSONLWatcher(
            watch_dir=tmp_path / "projects",
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda items: flushed.extend(items),
            batch_size=1,
        )
        count = watcher.poll_once()
        assert count == 1
        assert len(flushed) == 1
        assert flushed[0]["text"] == "hello"

    def test_poll_twice_no_duplicates(self, tmp_path):
        project = self._make_project_dir(tmp_path)
        (project / "s1.jsonl").write_text('{"id":"1"}\n')

        flushed = []
        watcher = JSONLWatcher(
            watch_dir=tmp_path / "projects",
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda items: flushed.extend(items),
            batch_size=1,
        )
        watcher.poll_once()
        watcher.poll_once()
        assert len(flushed) == 1  # No duplicate

    def test_incremental_append_between_polls(self, tmp_path):
        project = self._make_project_dir(tmp_path)
        f = project / "s1.jsonl"
        f.write_text('{"id":"1"}\n')

        flushed = []
        watcher = JSONLWatcher(
            watch_dir=tmp_path / "projects",
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda items: flushed.extend(items),
            batch_size=1,
        )
        watcher.poll_once()
        assert len(flushed) == 1

        with open(f, "a") as fh:
            fh.write('{"id":"2"}\n')
        watcher.poll_once()
        assert len(flushed) == 2

    def test_source_file_tagged(self, tmp_path):
        project = self._make_project_dir(tmp_path)
        f = project / "s1.jsonl"
        f.write_text('{"id":"1"}\n')

        flushed = []
        watcher = JSONLWatcher(
            watch_dir=tmp_path / "projects",
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda items: flushed.extend(items),
            batch_size=1,
        )
        watcher.poll_once()
        assert "_source_file" in flushed[0]
        assert flushed[0]["_source_file"].endswith("s1.jsonl")
        assert flushed[0]["_line_end_offset"] == len(b'{"id":"1"}\n')

    def test_offsets_advance_only_to_flush_confirmed_watermark(self, tmp_path):
        project = self._make_project_dir(tmp_path)
        f = project / "s1.jsonl"
        first_line = b'{"id":"1"}\n'
        second_line = b'{"id":"2"}\n'
        f.write_bytes(first_line + second_line)

        def partial_flush(items):
            return {items[0]["_source_file"]: items[0]["_line_end_offset"]}

        watcher = JSONLWatcher(
            watch_dir=tmp_path / "projects",
            registry_path=tmp_path / "offsets.json",
            on_flush=partial_flush,
            batch_size=100,
            flush_interval_ms=0,
        )

        assert watcher.poll_once() == 2

        offset, _inode = watcher.registry.get(str(f))
        assert offset == len(first_line)
        assert watcher._tailers[str(f)].offset == len(first_line + second_line)

    def test_poll_once_isolates_file_crashes_and_still_flushes_health(self, tmp_path):
        project = self._make_project_dir(tmp_path)
        poison = project / "poison.jsonl"
        healthy = project / "healthy.jsonl"
        poison.write_text('{"id":"poison"}\n')
        healthy.write_text('{"id":"healthy"}\n')
        flushed = []
        health_path = tmp_path / "watcher-health.json"

        watcher = JSONLWatcher(
            watch_dir=tmp_path / "projects",
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda items: (
                flushed.extend(items) or {str(healthy): len(b'{"id":"healthy"}\n')}
                if items and items[0].get("id") == "healthy"
                else {}
            ),
            batch_size=1,
            health_path=health_path,
        )
        watcher._discover_jsonl_files = lambda: [str(poison), str(healthy)]
        original_normalize = watcher._normalize_lines

        def crash_one_file(filepath, new_lines):
            if filepath == str(poison):
                raise AttributeError("poison parse failure")
            return original_normalize(filepath, new_lines)

        watcher._normalize_lines = crash_one_file

        assert watcher.poll_once() == 1
        assert [item["id"] for item in flushed] == ["healthy"]
        assert watcher.registry.get(str(poison))[0] == 0
        assert watcher.registry.get(str(healthy))[0] == len(b'{"id":"healthy"}\n')
        payload = json.loads(health_path.read_text())
        assert payload["poll_count"] == 1

    def test_offset_survives_restart(self, tmp_path):
        project = self._make_project_dir(tmp_path)
        f = project / "s1.jsonl"
        f.write_text('{"id":"1"}\n{"id":"2"}\n')

        flushed1 = []
        w1 = JSONLWatcher(
            watch_dir=tmp_path / "projects",
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda items: flushed1.extend(items) or {str(f): max(item["_line_end_offset"] for item in items)},
            batch_size=1,
        )
        w1.poll_once()
        w1.registry.flush()
        assert len(flushed1) == 2

        # Append more, create new watcher (simulates restart)
        with open(f, "a") as fh:
            fh.write('{"id":"3"}\n')

        flushed2 = []
        w2 = JSONLWatcher(
            watch_dir=tmp_path / "projects",
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda items: flushed2.extend(items) or {str(f): max(item["_line_end_offset"] for item in items)},
            batch_size=1,
        )
        w2.poll_once()
        assert len(flushed2) == 1
        assert flushed2[0]["id"] == "3"

    def test_start_stop_threading(self, tmp_path):
        project = self._make_project_dir(tmp_path)
        (project / "s1.jsonl").write_text('{"id":"1"}\n')

        flushed = []
        watcher = JSONLWatcher(
            watch_dir=tmp_path / "projects",
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda items: flushed.extend(items),
            batch_size=1,
            poll_interval_s=0.05,
        )
        t = threading.Thread(target=watcher.start)
        t.start()
        time.sleep(0.2)
        watcher.stop()
        t.join(timeout=2)
        assert not t.is_alive()
        assert len(flushed) >= 1

    def test_multiple_projects(self, tmp_path):
        p1 = self._make_project_dir(tmp_path, "project-a")
        p2 = self._make_project_dir(tmp_path, "project-b")
        (p1 / "s1.jsonl").write_text('{"project":"a"}\n')
        (p2 / "s2.jsonl").write_text('{"project":"b"}\n')

        flushed = []
        watcher = JSONLWatcher(
            watch_dir=tmp_path / "projects",
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda items: flushed.extend(items),
            batch_size=1,
        )
        watcher.poll_once()
        projects = {f["project"] for f in flushed}
        assert projects == {"a", "b"}

    def test_multi_root_discovers_claude_codex_cursor_and_gemini_files(self, tmp_path):
        claude_project = tmp_path / "claude" / "projects" / "proj"
        codex_sessions = tmp_path / "codex" / "sessions"
        cursor_sessions = tmp_path / "cursor" / "sessions"
        gemini_sessions = tmp_path / "gemini" / "sessions"
        for root in (claude_project, codex_sessions, cursor_sessions, gemini_sessions):
            root.mkdir(parents=True)
            (root / f"{root.parent.name}.jsonl").write_text('{"id":"1"}\n')

        watcher = JSONLWatcher(
            watch_roots=[
                WatchRoot("claude", tmp_path / "claude" / "projects"),
                WatchRoot("codex", codex_sessions),
                WatchRoot("cursor", cursor_sessions),
                WatchRoot("gemini", gemini_sessions),
            ],
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda x: None,
        )

        files = watcher._discover_jsonl_files()

        assert len(files) == 4
        assert {watcher.provider_for_file(path) for path in files} == {"claude", "codex", "cursor", "gemini"}

    def test_multi_root_discovers_newest_jsonl_files_first(self, tmp_path):
        codex_sessions = tmp_path / "codex" / "sessions"
        cursor_sessions = tmp_path / "cursor" / "sessions"
        codex_sessions.mkdir(parents=True)
        cursor_sessions.mkdir(parents=True)
        old_codex = codex_sessions / "old.jsonl"
        fresh_cursor = cursor_sessions / "fresh.jsonl"
        old_codex.write_text('{"id":"old"}\n')
        fresh_cursor.write_text('{"id":"fresh"}\n')
        os.utime(old_codex, (1000, 1000))
        os.utime(fresh_cursor, (2000, 2000))

        watcher = JSONLWatcher(
            watch_roots=[
                WatchRoot("codex", codex_sessions),
                WatchRoot("cursor", cursor_sessions),
            ],
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda x: None,
        )

        files = watcher._discover_jsonl_files()

        assert files == [str(fresh_cursor), str(old_codex)]
        assert watcher.provider_for_file(str(fresh_cursor)) == "cursor"
        assert watcher.provider_for_file(str(old_codex)) == "codex"

    def test_poll_once_limits_each_file_so_active_roots_do_not_starve(self, tmp_path):
        codex_sessions = tmp_path / "codex" / "sessions"
        cursor_sessions = tmp_path / "cursor" / "sessions"
        codex_sessions.mkdir(parents=True)
        cursor_sessions.mkdir(parents=True)
        hot_codex = codex_sessions / "hot.jsonl"
        fresh_cursor = cursor_sessions / "fresh.jsonl"
        hot_codex.write_text(
            "\n".join(
                json.dumps({"role": "user", "content": f"codex active line {idx} with enough content"})
                for idx in range(3)
            )
            + "\n"
        )
        fresh_cursor.write_text(
            json.dumps(
                {"type": "message", "payload": {"role": "user", "content": "cursor active line with enough content"}}
            )
            + "\n"
        )
        os.utime(hot_codex, (3000, 3000))
        os.utime(fresh_cursor, (2000, 2000))

        flushed = []
        watcher = JSONLWatcher(
            watch_roots=[
                WatchRoot("codex", codex_sessions),
                WatchRoot("cursor", cursor_sessions),
            ],
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda items: flushed.extend(items),
            batch_size=1,
            max_lines_per_file=1,
        )

        assert watcher.poll_once() == 2
        assert [item["_provider"] for item in flushed] == ["codex", "cursor"]
        assert watcher._tailers[str(hot_codex)].offset < hot_codex.stat().st_size

        flushed.clear()
        assert watcher.poll_once() == 1
        assert [item["_provider"] for item in flushed] == ["codex"]

    def test_codex_root_normalizes_role_content_entries(self, tmp_path):
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        (sessions / "session.jsonl").write_text(
            json.dumps(
                {
                    "role": "user",
                    "content": "Explain the watcher arbitration design with enough detail to index.",
                    "timestamp": "2026-06-17T10:00:00Z",
                }
            )
            + "\n"
        )

        flushed = []
        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda items: flushed.extend(items),
            batch_size=1,
        )

        assert watcher.poll_once() == 1
        assert flushed[0]["type"] == "user"
        assert flushed[0]["message"]["content"][0]["text"].startswith("Explain the watcher")
        assert flushed[0]["_provider"] == "codex"

    def test_codex_root_normalizes_real_response_item_payload_entries(self, tmp_path):
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        (sessions / "session.jsonl").write_text(
            json.dumps(
                {
                    "timestamp": "2026-06-17T10:00:00Z",
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Explain the watcher arbitration design with enough detail to index.",
                            }
                        ],
                    },
                }
            )
            + "\n"
        )

        flushed = []
        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda items: flushed.extend(items),
            batch_size=1,
        )

        assert watcher.poll_once() == 1
        assert flushed[0]["type"] == "user"
        assert flushed[0]["message"]["content"][0]["text"].startswith("Explain the watcher")
        assert flushed[0]["_provider"] == "codex"

    def test_normalizer_ignores_string_author_without_poll_failure(self, tmp_path):
        sessions = tmp_path / "cursor" / "sessions"
        sessions.mkdir(parents=True)
        (sessions / "session.jsonl").write_text(
            json.dumps(
                {
                    "timestamp": "2026-06-17T10:00:00Z",
                    "author": "user",
                    "content": "This row lacks a structured role and should be skipped without crashing.",
                }
            )
            + "\n"
        )

        flushed = []
        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("cursor", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda items: flushed.extend(items),
            batch_size=1,
        )

        assert watcher.poll_once() == 0
        assert flushed == []

    def test_normalizer_uses_text_message_without_dict_role_lookup_failure(self, tmp_path):
        sessions = tmp_path / "gemini" / "sessions"
        sessions.mkdir(parents=True)
        (sessions / "session.jsonl").write_text(
            json.dumps(
                {
                    "timestamp": "2026-06-17T10:00:00Z",
                    "role": "model",
                    "message": "Gemini live adapter verification text should normalize cleanly.",
                }
            )
            + "\n"
        )

        flushed = []
        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("gemini", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda items: flushed.extend(items),
            batch_size=1,
        )

        assert watcher.poll_once() == 1
        assert flushed[0]["type"] == "assistant"
        assert flushed[0]["message"]["content"][0]["text"].startswith("Gemini live adapter")

    def test_health_snapshot_uses_db_realtime_insert_rate_when_db_path_is_available(self, tmp_path):
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        transcript = sessions / "session.jsonl"
        transcript.write_text(
            json.dumps(
                {
                    "role": "assistant",
                    "content": "A substantive assistant response that should be observed by the watcher.",
                }
            )
            + "\n"
        )
        db_path = tmp_path / "brainlayer.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE chunks (source TEXT, ingested_at INTEGER, created_at TEXT)")
        conn.commit()
        conn.close()

        health_path = tmp_path / "watcher-health.json"
        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda items: len(items),
            batch_size=1,
            health_path=health_path,
            db_path=db_path,
        )

        watcher.poll_once()

        payload = json.loads(health_path.read_text())
        assert payload["active_jsonl_entries_per_minute"] > 0
        assert payload["watcher_chunks_output_per_minute"] > 0
        assert payload["db_realtime_inserts_per_minute"] == 0

    def test_health_snapshot_raises_alarm_on_zero_db_writes_while_active(self, tmp_path):
        now = [0.0]
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        transcript = sessions / "session.jsonl"
        transcript.write_text(
            json.dumps(
                {
                    "role": "assistant",
                    "content": "A substantive assistant response that should be observed by the watcher.",
                }
            )
            + "\n"
        )
        db_path = tmp_path / "brainlayer.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE chunks (source TEXT, ingested_at INTEGER, created_at TEXT)")
        conn.commit()
        conn.close()
        health_path = tmp_path / "watcher-health.json"
        watchdog = CoverageWatchdog(
            lag_threshold_bytes=1_000_000,
            alert_after_s=5,
            now_fn=lambda: now[0],
        )

        def flush_without_durable_write(items):
            return {str(transcript): max(item["_line_end_offset"] for item in items)}

        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=flush_without_durable_write,
            batch_size=1,
            health_path=health_path,
            db_path=db_path,
            coverage_watchdog=watchdog,
        )

        watcher.poll_once()
        now[0] = 6.0
        with pytest.raises(BrainLayerAlarm) as raised:
            watcher.poll_once()

        payload = json.loads(health_path.read_text())
        assert payload["providers"] == ["codex"]
        assert payload["alerting"] is True
        assert "coverage_drop" in payload["alert_reasons"]
        assert raised.value.code == "watcher_zero_writes_while_active"
        assert raised.value.details["active_jsonl_entries_per_minute"] > 0
        assert raised.value.details["db_realtime_inserts_per_minute"] == 0

    def test_health_snapshot_holds_zero_write_alarm_across_quiet_burst(self, tmp_path):
        now = [0.0]
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        transcript = sessions / "session.jsonl"
        transcript.write_text(
            json.dumps(
                {
                    "role": "assistant",
                    "content": "A finite accepted burst should still alarm if the drain never writes it.",
                }
            )
            + "\n"
        )
        db_path = tmp_path / "brainlayer.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE chunks (source TEXT, ingested_at INTEGER, created_at TEXT)")
        conn.commit()
        conn.close()
        health_path = tmp_path / "watcher-health.json"

        def flush_without_durable_write(items):
            return {str(transcript): max(item["_line_end_offset"] for item in items)}

        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=flush_without_durable_write,
            batch_size=1,
            health_path=health_path,
            db_path=db_path,
            coverage_watchdog=CoverageWatchdog(
                lag_threshold_bytes=1_000_000,
                alert_after_s=120,
                now_fn=lambda: now[0],
            ),
        )

        watcher.poll_once()
        now[0] = 61.0
        watcher._health_window_started = time.monotonic() - 61
        assert watcher.poll_once() == 0
        assert json.loads(health_path.read_text())["active_jsonl_entries_per_minute"] > 0

        now[0] = 121.0
        watcher._health_window_started = time.monotonic() - 121
        with pytest.raises(BrainLayerAlarm) as raised:
            watcher.poll_once()

        assert raised.value.code == "watcher_zero_writes_while_active"
        assert raised.value.details["watcher_chunks_output_per_minute"] > 0

    def test_health_snapshot_resets_partial_durable_window_before_later_zero_write_alarm(self, tmp_path, monkeypatch):
        now = [0.0]
        monkeypatch.setattr("brainlayer.watcher.time.time", lambda: now[0])
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        transcript = sessions / "session.jsonl"
        transcript.write_text(
            json.dumps(
                {
                    "role": "assistant",
                    "content": "The first watcher line gets a durable liveness row.",
                }
            )
            + "\n"
        )
        db_path = tmp_path / "brainlayer.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE chunks (source TEXT, ingested_at INTEGER, created_at TEXT)")
        conn.execute("CREATE TABLE watcher_liveness_events (chunk_id TEXT, ingested_at INTEGER)")
        conn.commit()
        conn.close()
        health_path = tmp_path / "watcher-health.json"
        flush_calls = [0]

        def flush_first_call_only(items):
            flush_calls[0] += 1
            if flush_calls[0] == 1:
                with sqlite3.connect(db_path) as write_conn:
                    write_conn.execute(
                        "INSERT INTO watcher_liveness_events (chunk_id, ingested_at) VALUES (?, ?)",
                        ("first-durable", int(now[0])),
                    )
                    write_conn.commit()
            return {str(transcript): max(item["_line_end_offset"] for item in items)}

        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=flush_first_call_only,
            batch_size=1,
            health_path=health_path,
            db_path=db_path,
            coverage_watchdog=CoverageWatchdog(
                coverage_ratio_threshold=0.75,
                lag_threshold_bytes=1_000_000,
                alert_after_s=60,
                now_fn=lambda: now[0],
            ),
        )

        watcher.poll_once()
        with transcript.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "role": "assistant",
                        "content": "The second watcher line is accepted while durable writes are already behind.",
                    }
                )
                + "\n"
            )
        now[0] = 61.0
        watcher._health_window_started = time.monotonic() - 61
        assert watcher.poll_once() == 1
        assert watcher._health_window_started_epoch == 61.0

        with transcript.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "role": "assistant",
                        "content": "The third watcher line should alarm because no durable writes followed reset.",
                    }
                )
                + "\n"
            )
        now[0] = 122.0
        watcher._health_window_started = time.monotonic() - 61
        with pytest.raises(BrainLayerAlarm) as raised:
            watcher.poll_once()

        assert raised.value.code == "watcher_zero_writes_while_active"
        assert raised.value.details["durable_writes_per_minute"] == 0

    def test_health_snapshot_raises_alarm_when_db_probe_fails_while_active(self, tmp_path):
        now = [0.0]
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        transcript = sessions / "session.jsonl"
        transcript.write_text(
            json.dumps(
                {
                    "role": "assistant",
                    "content": "A substantive assistant response that should be observed by the watcher.",
                }
            )
            + "\n"
        )
        db_path = tmp_path / "brainlayer.db"
        sqlite3.connect(db_path).close()
        health_path = tmp_path / "watcher-health.json"
        watchdog = CoverageWatchdog(
            lag_threshold_bytes=1_000_000,
            alert_after_s=5,
            now_fn=lambda: now[0],
        )

        def flush_without_probeable_db(items):
            return {str(transcript): max(item["_line_end_offset"] for item in items)}

        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=flush_without_probeable_db,
            batch_size=1,
            health_path=health_path,
            db_path=db_path,
            coverage_watchdog=watchdog,
        )

        watcher.poll_once()
        now[0] = 6.0
        with pytest.raises(BrainLayerAlarm) as raised:
            watcher.poll_once()

        payload = json.loads(health_path.read_text())
        assert payload["db_realtime_inserts_per_minute"] is None
        assert payload["db_probe_failed"] is True
        assert raised.value.code == "watcher_zero_writes_while_active"
        assert raised.value.details["db_probe_failed"] is True

    def test_health_snapshot_raises_alarm_when_flush_fails_while_active(self, tmp_path):
        now = [0.0]
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        transcript = sessions / "session.jsonl"
        transcript.write_text(
            json.dumps(
                {
                    "role": "assistant",
                    "content": "A substantive assistant response that should be flushed but the writer fails.",
                }
            )
            + "\n"
        )
        db_path = tmp_path / "brainlayer.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE chunks (source TEXT, ingested_at INTEGER, created_at TEXT)")
        conn.commit()
        conn.close()
        health_path = tmp_path / "watcher-health.json"
        watchdog = CoverageWatchdog(
            lag_threshold_bytes=1_000_000,
            alert_after_s=5,
            now_fn=lambda: now[0],
        )

        def fail_flush(_items):
            raise RuntimeError("queue unavailable")

        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=fail_flush,
            batch_size=1,
            health_path=health_path,
            db_path=db_path,
            coverage_watchdog=watchdog,
        )

        watcher.poll_once()
        now[0] = 6.0
        with pytest.raises(BrainLayerAlarm) as raised:
            watcher.poll_once()

        payload = json.loads(health_path.read_text())
        assert payload["failed_flush_inputs_per_minute"] > 0
        assert payload["active_jsonl_entries_per_minute"] > 0
        assert payload["watcher_chunks_output_per_minute"] == 0
        assert raised.value.code == "watcher_zero_writes_while_active"

    def test_health_snapshot_does_not_treat_quarantined_retry_as_active_input(self, tmp_path, monkeypatch):
        now = [0.0]
        monkeypatch.setenv("BRAINLAYER_WATCHER_FLUSH_RETAIN_LIMIT", "2")
        monkeypatch.setenv("BRAINLAYER_WATCHER_QUARANTINE_DIR", str(tmp_path / "quarantine"))
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        transcript = sessions / "session.jsonl"
        transcript.write_text(
            json.dumps(
                {
                    "role": "assistant",
                    "content": "A poison watcher line should not remain active after quarantine.",
                }
            )
            + "\n"
        )
        db_path = tmp_path / "brainlayer.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE chunks (source TEXT, ingested_at INTEGER, created_at TEXT)")
        conn.commit()
        conn.close()
        health_path = tmp_path / "watcher-health.json"
        watchdog = CoverageWatchdog(
            lag_threshold_bytes=1_000_000,
            alert_after_s=120,
            now_fn=lambda: now[0],
        )

        def fail_flush(_items):
            raise RuntimeError("poison batch")

        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=fail_flush,
            batch_size=1,
            health_path=health_path,
            db_path=db_path,
            coverage_watchdog=watchdog,
        )

        watcher.poll_once()
        now[0] = 61.0
        watcher._health_window_started = time.monotonic() - 61
        watcher.indexer._last_flush = time.monotonic() - 1
        assert watcher.poll_once() == 0

        payload = json.loads(health_path.read_text())
        assert payload["failed_flush_inputs_per_minute"] == 0
        assert payload["active_jsonl_entries_per_minute"] == 0

        now[0] = 121.0
        watcher._health_window_started = time.monotonic() - 60
        assert watcher.poll_once() == 0

    def test_health_snapshot_does_not_alarm_when_active_input_is_intentionally_skipped(self, tmp_path):
        now = [0.0]
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        transcript = sessions / "session.jsonl"
        transcript.write_text(
            json.dumps(
                {
                    "role": "assistant",
                    "content": "A normalized assistant response that the flush classifier intentionally skips.",
                }
            )
            + "\n"
        )
        db_path = tmp_path / "brainlayer.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE chunks (source TEXT, ingested_at INTEGER, created_at TEXT)")
        conn.commit()
        conn.close()
        health_path = tmp_path / "watcher-health.json"
        watchdog = CoverageWatchdog(
            lag_threshold_bytes=1_000_000,
            alert_after_s=5,
            now_fn=lambda: now[0],
        )

        def flush_all_skipped(items):
            return FlushWatermarks(
                {str(transcript): max(item["_line_end_offset"] for item in items)},
                inserted=0,
                skipped=len(items),
            )

        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=flush_all_skipped,
            batch_size=1,
            health_path=health_path,
            db_path=db_path,
            coverage_watchdog=watchdog,
        )

        watcher.poll_once()
        now[0] = 6.0
        assert watcher.poll_once() == 0

        payload = json.loads(health_path.read_text())
        assert payload["normalized_jsonl_entries_per_minute"] > 0
        assert payload["active_jsonl_entries_per_minute"] == 0
        assert payload["watcher_chunks_output_per_minute"] == 0
        assert payload["alerting"] is False

    def test_health_snapshot_counts_drain_watcher_liveness_events(self, tmp_path):
        now = [0.0]
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        transcript = sessions / "session.jsonl"
        transcript.write_text(
            json.dumps(
                {
                    "role": "assistant",
                    "content": "A watcher chunk that merges into a non realtime canonical row still writes liveness.",
                }
            )
            + "\n"
        )
        db_path = tmp_path / "brainlayer.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE chunks (source TEXT, ingested_at INTEGER, created_at TEXT)")
        conn.execute("CREATE TABLE watcher_liveness_events (chunk_id TEXT, ingested_at INTEGER)")
        conn.commit()
        conn.close()
        health_path = tmp_path / "watcher-health.json"

        def flush_with_liveness(items):
            with sqlite3.connect(db_path) as write_conn:
                write_conn.execute(
                    "INSERT INTO watcher_liveness_events (chunk_id, ingested_at) VALUES (?, ?)",
                    ("manual-canonical", int(time.time())),
                )
                write_conn.commit()
            return FlushWatermarks(
                {str(transcript): max(item["_line_end_offset"] for item in items)},
                inserted=len(items),
                skipped=0,
            )

        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=flush_with_liveness,
            batch_size=1,
            health_path=health_path,
            db_path=db_path,
            coverage_watchdog=CoverageWatchdog(alert_after_s=0, now_fn=lambda: now[0]),
        )

        watcher.poll_once()

        payload = json.loads(health_path.read_text())
        assert payload["active_jsonl_entries_per_minute"] > 0
        assert payload["db_realtime_inserts_per_minute"] > 0
        assert payload["alerting"] is False

    def test_db_realtime_insert_probe_casts_created_at_fallback_to_epoch(self, tmp_path):
        db_path = tmp_path / "brainlayer.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE chunks (source TEXT, ingested_at INTEGER, created_at TEXT)")
            conn.execute(
                "INSERT INTO chunks (source, ingested_at, created_at) VALUES (?, ?, ?)",
                ("realtime_watcher", None, "2020-01-01T00:00:00Z"),
            )
            conn.commit()

        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda items: None,
            batch_size=1,
            db_path=db_path,
        )
        watcher._health_window_started_epoch = 2_000_000_000

        assert watcher._db_realtime_inserts_since_window_start() == 0

    def test_start_propagates_brainlayer_alarm_from_poll_once(self, tmp_path, monkeypatch):
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda items: None,
            batch_size=1,
            poll_interval_s=0,
        )
        alarm = BrainLayerAlarm("watcher_zero_writes_while_active", "fatal write-side degradation")

        def raise_from_poll_once():
            raise alarm

        monkeypatch.setattr(watcher, "poll_once", raise_from_poll_once)

        with pytest.raises(BrainLayerAlarm) as raised:
            watcher.start()

        assert raised.value is alarm

    def test_health_snapshot_does_not_alarm_when_legitimately_idle(self, tmp_path):
        health_path = tmp_path / "watcher-health.json"
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda items: None,
            batch_size=1,
            health_path=health_path,
            coverage_watchdog=CoverageWatchdog(alert_after_s=0),
        )

        watcher.poll_once()

        payload = json.loads(health_path.read_text())
        assert payload["active_jsonl_entries_per_minute"] == 0
        assert payload["alerting"] is False
