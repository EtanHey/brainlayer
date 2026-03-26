"""Tests for real-time JSONL file watcher prototype.

Covers:
- OffsetRegistry: persist/restore offsets, atomic writes, inode tracking
- JSONLTailer: tail-follow, partial line buffering, corrupt line handling
- BatchIndexer: batching, flush interval, thread safety
- JSONLWatcher: file discovery, poll cycle, end-to-end integration
"""

import threading
import time

from brainlayer.watcher import BatchIndexer, JSONLTailer, JSONLWatcher, OffsetRegistry

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

    def test_flush_error_doesnt_crash(self):
        def bad_flush(items):
            raise RuntimeError("flush failed")

        indexer = BatchIndexer(on_flush=bad_flush, batch_size=1)
        # Should not raise
        indexer.add([{"a": 1}])
        assert indexer.total_flushed == 0


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

    def test_offset_survives_restart(self, tmp_path):
        project = self._make_project_dir(tmp_path)
        f = project / "s1.jsonl"
        f.write_text('{"id":"1"}\n{"id":"2"}\n')

        flushed1 = []
        w1 = JSONLWatcher(
            watch_dir=tmp_path / "projects",
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda items: flushed1.extend(items),
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
            on_flush=lambda items: flushed2.extend(items),
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
