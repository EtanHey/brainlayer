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
import stat
import threading
import time
from pathlib import Path

import pytest

from brainlayer.alarm import BrainLayerAlarm
from brainlayer.watcher import (
    BatchIndexer,
    CoverageWatchdog,
    JSONLTailer,
    JSONLWatcher,
    OffsetRegistry,
    WatchRoot,
    default_watch_roots,
)
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

    def test_flush_fsyncs_file_and_parent_directory(self, monkeypatch, tmp_path):
        synced_modes: list[int] = []
        real_fsync = os.fsync

        def recording_fsync(fd):
            synced_modes.append(os.fstat(fd).st_mode)
            real_fsync(fd)

        monkeypatch.setattr(os, "fsync", recording_fsync)
        reg = OffsetRegistry(tmp_path / "offsets.json")
        reg.set("/session.jsonl", 100, 1)

        assert reg.flush() is True
        assert any(stat.S_ISREG(mode) for mode in synced_modes)
        assert any(stat.S_ISDIR(mode) for mode in synced_modes)

    def test_flush_uses_windows_byte_lock_when_fcntl_is_unavailable(self, monkeypatch, tmp_path):
        from brainlayer import watcher as watcher_module

        lock_operations: list[int] = []

        class FakeMsvcrt:
            LK_NBLCK = 1
            LK_UNLCK = 2

            @staticmethod
            def locking(_fd, operation, _byte_count):
                lock_operations.append(operation)

        monkeypatch.setattr(watcher_module, "fcntl", None)
        monkeypatch.setattr(watcher_module, "msvcrt", FakeMsvcrt, raising=False)
        reg = OffsetRegistry(tmp_path / "offsets.json")
        reg.set("/session.jsonl", 100, 1)

        assert reg.flush() is True
        assert lock_operations == [FakeMsvcrt.LK_NBLCK, FakeMsvcrt.LK_UNLCK]

    def test_flush_does_not_overwrite_corrupt_existing_registry(self, tmp_path):
        registry_path = tmp_path / "offsets.json"
        reg = OffsetRegistry(registry_path)
        reg.set("/session.jsonl", 100, 1)
        assert reg.flush() is True
        reg.set("/session.jsonl", 200, 1)
        registry_path.write_text("{corrupt")

        assert reg.flush() is False
        assert registry_path.read_text() == "{corrupt"

    def test_remove_entry(self, tmp_path):
        reg = OffsetRegistry(tmp_path / "offsets.json")
        reg.set("/a.jsonl", 100, 1)
        reg.remove("/a.jsonl")
        offset, inode = reg.get("/a.jsonl")
        assert offset == 0

    def test_prune_missing_files_removes_only_deleted_paths_and_persists(self, tmp_path):
        registry_path = tmp_path / "offsets.json"
        existing = tmp_path / "existing.jsonl"
        deleted = tmp_path / "deleted.jsonl"
        existing.write_text('{"id":"kept"}\n')

        reg = OffsetRegistry(registry_path)
        reg.set(str(existing), 100, 1)
        reg.set(str(deleted), 200, 2)
        reg.flush()

        assert reg.prune_missing_files([tmp_path]) == 1
        assert reg.get(str(existing)) == (100, 1)
        assert reg.get(str(deleted)) == (0, 0)
        assert reg.flush() is True

        reloaded = OffsetRegistry(registry_path)
        assert reloaded.get(str(existing)) == (100, 1)
        assert reloaded.get(str(deleted)) == (0, 0)

    def test_prune_live_parent_evidence_uses_linear_ancestry_checks(self, monkeypatch, tmp_path):
        tracked_count = 80
        registry = OffsetRegistry(tmp_path / "offsets.json")
        live_files = []
        for index in range(tracked_count):
            session_dir = tmp_path / f"session-{index}"
            session_dir.mkdir()
            live_file = session_dir / "live.jsonl"
            live_file.write_text('{"id":"live"}\n')
            live_files.append(live_file)
            registry.set(str(session_dir / "deleted.jsonl"), 100, index + 1)

        real_is_relative_to = Path.is_relative_to
        ancestry_checks = 0

        def counting_is_relative_to(path, other):
            nonlocal ancestry_checks
            ancestry_checks += 1
            return real_is_relative_to(path, other)

        monkeypatch.setattr(Path, "is_relative_to", counting_is_relative_to)

        assert registry.prune_missing_files([tmp_path], live_files) == tracked_count
        assert ancestry_checks <= tracked_count * 6

    def test_prune_flush_preserves_newer_offsets_from_concurrent_registry(self, tmp_path):
        registry_path = tmp_path / "offsets.json"
        existing = tmp_path / "existing.jsonl"
        deleted = tmp_path / "deleted.jsonl"
        existing.write_text('{"id":"kept"}\n')

        initial = OffsetRegistry(registry_path)
        initial.set(str(existing), 100, 1)
        initial.set(str(deleted), 200, 2)
        assert initial.flush() is True

        pruning_registry = OffsetRegistry(registry_path)
        live_registry = OffsetRegistry(registry_path)
        live_registry.set(str(existing), 300, 1)
        assert live_registry.flush() is True

        assert pruning_registry.prune_missing_files([tmp_path], [existing]) == 1
        assert pruning_registry.flush() is True

        reloaded = OffsetRegistry(registry_path)
        assert reloaded.get(str(existing)) == (300, 1)
        assert reloaded.get(str(deleted)) == (0, 0)

    def test_stale_registry_cannot_regress_offset_for_same_inode(self, tmp_path):
        registry_path = tmp_path / "offsets.json"
        initial = OffsetRegistry(registry_path)
        initial.set("/session.jsonl", 50, 1)
        assert initial.flush() is True

        advancing_registry = OffsetRegistry(registry_path)
        stale_registry = OffsetRegistry(registry_path)
        advancing_registry.set("/session.jsonl", 300, 1)
        assert advancing_registry.flush() is True
        stale_registry.set("/session.jsonl", 100, 1)
        assert stale_registry.flush() is True

        assert OffsetRegistry(registry_path).get("/session.jsonl") == (300, 1)

    def test_newer_rewind_generation_allows_lower_offset_and_blocks_stale_writer(self, tmp_path):
        registry_path = tmp_path / "offsets.json"
        initial = OffsetRegistry(registry_path)
        initial.set("/session.jsonl", 300, 1)
        assert initial.flush() is True

        rewinding_registry = OffsetRegistry(registry_path)
        stale_registry = OffsetRegistry(registry_path)
        generation = rewinding_registry.mark_rewind("/session.jsonl", 1)
        rewinding_registry.set("/session.jsonl", 100, 1)
        assert rewinding_registry.flush() is True

        stale_registry.set("/session.jsonl", 400, 1)
        assert stale_registry.flush() is True

        reloaded = OffsetRegistry(registry_path)
        assert reloaded.get("/session.jsonl") == (100, 1)
        assert reloaded._data["/session.jsonl"]["generation"] == generation

    def test_prune_tombstone_blocks_stale_registry_from_resurrecting_deleted_offset(self, tmp_path):
        registry_path = tmp_path / "offsets.json"
        existing = tmp_path / "existing.jsonl"
        deleted = tmp_path / "deleted.jsonl"
        existing.write_text('{"id":"kept"}\n')

        initial = OffsetRegistry(registry_path)
        initial.set(str(existing), 100, 1)
        initial.set(str(deleted), 200, 2)
        assert initial.flush() is True

        pruning_registry = OffsetRegistry(registry_path)
        stale_registry = OffsetRegistry(registry_path)
        stale_registry.set(str(deleted), 250, 2)
        assert pruning_registry.prune_missing_files([tmp_path], [existing]) == 1
        assert pruning_registry.flush() is True

        stale_registry.set(str(existing), 300, 1)
        assert stale_registry.flush() is True

        reloaded = OffsetRegistry(registry_path)
        assert reloaded.get(str(existing)) == (300, 1)
        assert reloaded.get(str(deleted)) == (0, 0)

    def test_prune_tombstone_blocks_delayed_set_after_prune_flush(self, tmp_path):
        registry_path = tmp_path / "offsets.json"
        existing = tmp_path / "existing.jsonl"
        deleted = tmp_path / "deleted.jsonl"
        existing.write_text('{"id":"kept"}\n')

        initial = OffsetRegistry(registry_path)
        initial.set(str(existing), 100, 1)
        initial.set(str(deleted), 200, 2)
        assert initial.flush() is True

        pruning_registry = OffsetRegistry(registry_path)
        delayed_registry = OffsetRegistry(registry_path)
        assert pruning_registry.prune_missing_files([tmp_path], [existing]) == 1
        assert pruning_registry.flush() is True

        delayed_registry.set(str(deleted), 250, 2)
        assert delayed_registry.flush() is True

        assert OffsetRegistry(registry_path).get(str(deleted)) == (0, 0)

    def test_prune_tombstone_allows_recreated_file_with_new_inode(self, tmp_path):
        registry_path = tmp_path / "offsets.json"
        existing = tmp_path / "existing.jsonl"
        deleted = tmp_path / "deleted.jsonl"
        existing.write_text('{"id":"kept"}\n')

        initial = OffsetRegistry(registry_path)
        initial.set(str(existing), 100, 1)
        initial.set(str(deleted), 200, 2)
        assert initial.flush() is True

        pruning_registry = OffsetRegistry(registry_path)
        recreated_registry = OffsetRegistry(registry_path)
        assert pruning_registry.prune_missing_files([tmp_path], [existing]) == 1
        assert pruning_registry.flush() is True

        recreated_registry.set(str(deleted), 50, 3)
        assert recreated_registry.flush() is True
        assert OffsetRegistry(registry_path).get(str(deleted)) == (0, 0)

        recreated_registry.set(str(deleted), 50, 3)
        assert recreated_registry.flush() is True

        assert OffsetRegistry(registry_path).get(str(deleted)) == (50, 3)

    def test_prune_tombstone_blocks_older_different_inode_writer(self, tmp_path):
        registry_path = tmp_path / "offsets.json"
        existing = tmp_path / "existing.jsonl"
        deleted = tmp_path / "deleted.jsonl"
        existing.write_text('{"id":"kept"}\n')

        initial = OffsetRegistry(registry_path)
        initial.set(str(existing), 100, 1)
        initial.set(str(deleted), 200, 2)
        assert initial.flush() is True

        older_inode_registry = OffsetRegistry(registry_path)
        replacement_registry = OffsetRegistry(registry_path)
        replacement_registry.set(str(deleted), 25, 3)
        assert replacement_registry.flush() is True

        pruning_registry = OffsetRegistry(registry_path)
        assert pruning_registry.prune_missing_files([tmp_path], [existing]) == 1
        assert pruning_registry.flush() is True

        older_inode_registry.set(str(deleted), 250, 2)
        assert older_inode_registry.flush() is True

        assert OffsetRegistry(registry_path).get(str(deleted)) == (0, 0)

    def test_registry_loaded_after_tombstone_can_reuse_same_inode(self, tmp_path):
        registry_path = tmp_path / "offsets.json"
        existing = tmp_path / "existing.jsonl"
        deleted = tmp_path / "deleted.jsonl"
        existing.write_text('{"id":"kept"}\n')

        initial = OffsetRegistry(registry_path)
        initial.set(str(existing), 100, 1)
        initial.set(str(deleted), 200, 2)
        assert initial.flush() is True

        pruning_registry = OffsetRegistry(registry_path)
        assert pruning_registry.prune_missing_files([tmp_path], [existing]) == 1
        assert pruning_registry.flush() is True

        recreated_registry = OffsetRegistry(registry_path)
        recreated_registry.set(str(deleted), 50, 2)
        assert recreated_registry.flush() is True

        assert OffsetRegistry(registry_path).get(str(deleted)) == (50, 2)

    def test_registry_loaded_after_tombstone_rejects_missing_inode(self, tmp_path):
        registry_path = tmp_path / "offsets.json"
        existing = tmp_path / "existing.jsonl"
        deleted = tmp_path / "deleted.jsonl"
        existing.write_text('{"id":"kept"}\n')

        initial = OffsetRegistry(registry_path)
        initial.set(str(existing), 100, 1)
        initial.set(str(deleted), 200, 2)
        assert initial.flush() is True

        pruning_registry = OffsetRegistry(registry_path)
        assert pruning_registry.prune_missing_files([tmp_path], [existing]) == 1
        assert pruning_registry.flush() is True

        unavailable_registry = OffsetRegistry(registry_path)
        unavailable_registry.set(str(deleted), 50, 0)
        assert unavailable_registry.flush() is True

        assert OffsetRegistry(registry_path).get(str(deleted)) == (0, 0)

    def test_prune_tombstone_compacts_without_replaying_unchanged_stale_entries(self, monkeypatch, tmp_path):
        from brainlayer import watcher as watcher_module

        now = [1_000.0]
        monkeypatch.setattr(watcher_module.time, "time", lambda: now[0])
        registry_path = tmp_path / "offsets.json"
        existing = tmp_path / "existing.jsonl"
        deleted = tmp_path / "deleted.jsonl"
        existing.write_text('{"id":"kept"}\n')

        initial = OffsetRegistry(registry_path)
        initial.set(str(existing), 100, 1)
        initial.set(str(deleted), 200, 2)
        assert initial.flush() is True
        stale_registry = OffsetRegistry(registry_path)

        pruning_registry = OffsetRegistry(registry_path)
        assert pruning_registry.prune_missing_files([tmp_path], [existing]) == 1
        assert pruning_registry.flush() is True

        now[0] += 24 * 60 * 60 + 1
        compacting_registry = OffsetRegistry(registry_path)
        compacting_registry.set(str(existing), 300, 1)
        assert compacting_registry.flush() is True
        assert watcher_module._OFFSET_TOMBSTONES_KEY not in json.loads(registry_path.read_text())

        stale_registry.set(str(existing), 400, 1)
        assert stale_registry.flush() is True
        reloaded = OffsetRegistry(registry_path)
        assert reloaded.get(str(existing)) == (400, 1)
        assert reloaded.get(str(deleted)) == (0, 0)

    def test_prune_missing_files_preserves_offsets_under_unavailable_root(self, tmp_path):
        unavailable_root = tmp_path / "unmounted"
        missing = unavailable_root / "session.jsonl"
        reg = OffsetRegistry(tmp_path / "offsets.json")
        reg.set(str(missing), 100, 1)

        assert reg.prune_missing_files([unavailable_root]) == 0
        assert reg.get(str(missing)) == (100, 1)

    def test_prune_missing_files_preserves_empty_mounted_root(self, tmp_path):
        empty_mountpoint = tmp_path / "mounted-sessions"
        empty_mountpoint.mkdir()
        missing = empty_mountpoint / "session.jsonl"
        reg = OffsetRegistry(tmp_path / "offsets.json")
        reg.set(str(missing), 100, 1)

        assert reg.prune_missing_files([empty_mountpoint]) == 0
        assert reg.get(str(missing)) == (100, 1)

    def test_prune_missing_files_does_not_let_parent_authorize_unavailable_nested_root(self, tmp_path):
        parent_root = tmp_path / "data"
        nested_root = parent_root / "transcripts"
        parent_root.mkdir()
        nested_root.mkdir()
        live_parent_file = parent_root / "live.jsonl"
        missing_nested_file = nested_root / "session.jsonl"
        live_parent_file.write_text('{"id":"live"}\n')
        reg = OffsetRegistry(tmp_path / "offsets.json")
        reg.set(str(missing_nested_file), 100, 1)

        assert reg.prune_missing_files([parent_root, nested_root], [live_parent_file]) == 0
        assert reg.get(str(missing_nested_file)) == (100, 1)

        live_nested_file = nested_root / "sibling.jsonl"
        live_nested_file.write_text('{"id":"mounted"}\n')
        assert (
            reg.prune_missing_files(
                [parent_root, nested_root],
                [live_parent_file, live_nested_file],
            )
            == 1
        )
        assert reg.get(str(missing_nested_file)) == (0, 0)

    def test_prune_missing_files_preserves_offsets_under_broken_symlink_subtree(self, tmp_path):
        root = tmp_path / "projects"
        root.mkdir()
        live_file = root / "live.jsonl"
        live_file.write_text('{"id":"live"}\n')
        mounted_volume = tmp_path / "mounted-volume"
        mounted_volume.mkdir()
        unavailable_project = root / "unavailable-project"
        unavailable_project.symlink_to(mounted_volume, target_is_directory=True)
        missing = unavailable_project / "session.jsonl"
        reg = OffsetRegistry(tmp_path / "offsets.json")
        reg.set(str(missing), 100, 1)
        mounted_volume.rmdir()

        assert reg.prune_missing_files([root], [live_file]) == 0
        assert reg.get(str(missing)) == (100, 1)

    def test_prune_missing_files_preserves_offsets_under_empty_nested_subtree(self, tmp_path):
        root = tmp_path / "projects"
        root.mkdir()
        live_file = root / "live.jsonl"
        live_file.write_text('{"id":"live"}\n')
        empty_mountpoint = root / "mounted-project"
        empty_mountpoint.mkdir()
        missing = empty_mountpoint / "session.jsonl"
        reg = OffsetRegistry(tmp_path / "offsets.json")
        reg.set(str(missing), 100, 1)

        assert reg.prune_missing_files([root], [live_file]) == 0
        assert reg.get(str(missing)) == (100, 1)
        assert reg.last_prune_complete is False

    def test_prune_missing_files_preserves_broken_symlink_file(self, tmp_path):
        root = tmp_path / "projects"
        root.mkdir()
        live_file = root / "live.jsonl"
        live_file.write_text('{"id":"live"}\n')
        target = tmp_path / "mounted-session.jsonl"
        target.write_text('{"id":"mounted"}\n')
        unavailable_file = root / "session.jsonl"
        unavailable_file.symlink_to(target)
        reg = OffsetRegistry(tmp_path / "offsets.json")
        reg.set(str(unavailable_file), 100, 1)
        target.unlink()

        assert reg.prune_missing_files([root], [live_file]) == 0
        assert reg.get(str(unavailable_file)) == (100, 1)
        assert reg.last_prune_complete is False

    def test_prune_missing_files_skips_stat_errors(self, monkeypatch, tmp_path):
        root = tmp_path / "sessions"
        root.mkdir()
        live_file = root / "live.jsonl"
        live_file.write_text('{"id":"live"}\n')
        inaccessible = root / "inaccessible.jsonl"
        inaccessible.write_text('{"id":"inaccessible"}\n')
        reg = OffsetRegistry(tmp_path / "offsets.json")
        reg.set(str(inaccessible), 100, 1)
        original_stat = Path.stat
        original_is_file = Path.is_file

        def fail_inaccessible(path, *args, **kwargs):
            if path.name == inaccessible.name:
                raise PermissionError("denied")
            return original_stat(path, *args, **kwargs)

        def python_314_is_file(path):
            if path.name == inaccessible.name:
                return False
            return original_is_file(path)

        def lstat_still_succeeds(path, *args, **kwargs):
            return os.lstat(path)

        monkeypatch.setattr(Path, "stat", fail_inaccessible)
        monkeypatch.setattr(Path, "is_file", python_314_is_file)
        monkeypatch.setattr(Path, "lstat", lstat_still_succeeds)

        assert reg.prune_missing_files([root], [live_file]) == 0
        assert reg.get(str(inaccessible)) == (100, 1)
        assert reg.last_prune_complete is False

    def test_malformed_tombstones_are_sanitized(self, tmp_path):
        from brainlayer import watcher as watcher_module

        registry_path = tmp_path / "offsets.json"
        registry_path.write_text(
            json.dumps(
                {
                    watcher_module._OFFSET_TOMBSTONES_KEY: {
                        "/valid.jsonl": 123.0,
                        "/string.jsonl": "invalid",
                        "/bool.jsonl": True,
                        "/nan.jsonl": float("nan"),
                    }
                }
            )
        )

        reg = OffsetRegistry(registry_path)

        assert reg._removed == {
            "/valid.jsonl": {
                "removed_at": 123.0,
                "generation": 0,
                "inode": 0,
            }
        }
        reg.set("/string.jsonl", 100, 1)
        assert reg.flush() is True

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

    def test_corrupt_line_stops_before_unparsed_bytes(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_text('{"good":"line"}\nnot json at all\n{"also":"good"}\n')

        tailer = JSONLTailer(str(f))
        lines = tailer.read_new_lines()
        first_line_end = len(b'{"good":"line"}\n')

        assert lines == [{"good": "line", "_line_end_offset": first_line_end}]
        assert tailer.offset == first_line_end
        assert tailer._buffer.startswith(b"not json at all\n")
        assert tailer.last_error is not None

    def test_read_new_lines_limits_bytes_per_call(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_text(json.dumps({"role": "user", "content": "x" * 256}) + "\n")
        tailer = JSONLTailer(str(f))

        assert tailer.read_new_lines(max_bytes=64) == []
        assert tailer.offset == 0
        assert len(tailer._buffer) == 64

    def test_read_new_lines_stops_reading_once_line_limit_is_buffered(self, tmp_path):
        f = tmp_path / "test.jsonl"
        record = json.dumps({"role": "user", "content": "x" * 2048}) + "\n"
        f.write_text(record * 200)
        tailer = JSONLTailer(str(f))

        lines = tailer.read_new_lines(max_lines=2, max_bytes=1024 * 1024)

        assert len(lines) == 2
        assert len(tailer._buffer) < 64 * 1024

    def test_read_new_lines_bounds_an_oversized_incomplete_record(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_bytes(b'{"role":"user","content":"' + b"x" * 20_000)
        tailer = JSONLTailer(str(f), max_record_bytes=4096)

        for _ in range(10):
            assert tailer.read_new_lines(max_bytes=1024) == []

        assert len(tailer._buffer) <= 4097
        assert type(tailer.last_error).__name__ == "OversizedJSONLRecordError"
        assert tailer.offset == 0

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

    def test_flush_callback_watermark_is_forwarded_to_batch_callback(self):
        confirmed = []
        indexer = BatchIndexer(
            on_flush=lambda items: {"/tmp/source.jsonl": items[-1]["_line_end_offset"]},
            batch_size=2,
            on_confirm_batch=lambda watermarks, _batch: confirmed.append(watermarks),
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

    def test_failure_isolation_retains_deferred_entry_without_confirming_its_gap(self):
        deferred = {
            "id": "deferred",
            "_source_file": "/tmp/codex.jsonl",
            "_line_end_offset": 555,
        }
        failing = {"id": "failing", "_source_file": "/tmp/other.jsonl", "_line_end_offset": 100}
        confirmed = []

        class DeferredWatermark(dict):
            def __init__(self, item):
                super().__init__({item["_source_file"]: item["_line_end_offset"]})
                self.deferred_entries = [item]
                self.inserted = 0

        def isolate_mixed_batch(items):
            if len(items) > 1 or items[0] is failing:
                raise RuntimeError("isolate this flush")
            return DeferredWatermark(items[0])

        indexer = BatchIndexer(
            on_flush=isolate_mixed_batch,
            batch_size=2,
            on_confirm_batch=lambda watermarks, _batch: confirmed.append(watermarks),
        )

        indexer.add([deferred, failing])

        assert indexer._buffer == [deferred, failing]
        assert confirmed == []
        assert indexer.total_failed_inputs == 1
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

    def test_first_poll_prunes_and_flushes_deleted_offset_entries(self, tmp_path):
        project = self._make_project_dir(tmp_path)
        existing = project / "existing.jsonl"
        deleted = project / "deleted.jsonl"
        existing.write_text('{"id":"existing"}\n')
        registry_path = tmp_path / "offsets.json"
        registry_path.write_text(
            json.dumps(
                {
                    str(existing): {"offset": existing.stat().st_size, "inode": existing.stat().st_ino},
                    str(deleted): {"offset": 100, "inode": 999},
                }
            )
        )

        watcher = JSONLWatcher(
            watch_dir=tmp_path / "projects",
            registry_path=registry_path,
            on_flush=lambda items: None,
            registry_flush_interval_s=3600,
        )

        watcher.poll_once()

        persisted = json.loads(registry_path.read_text())
        assert str(existing) in persisted
        assert str(deleted) not in persisted

    def test_poll_retries_pruning_after_unavailable_startup_root(self, tmp_path):
        root = tmp_path / "projects"
        root.mkdir()
        project = root / "project"
        project.mkdir()
        deleted = project / "deleted.jsonl"
        registry_path = tmp_path / "offsets.json"
        registry_path.write_text(
            json.dumps(
                {
                    str(deleted): {"offset": 100, "inode": 999},
                }
            )
        )
        watcher = JSONLWatcher(
            watch_dir=root,
            registry_path=registry_path,
            on_flush=lambda items: None,
            registry_flush_interval_s=3600,
        )

        watcher.poll_once()
        assert watcher._offset_prune_complete is False

        live = project / "live.jsonl"
        live.write_text('{"id":"live"}\n')
        watcher.poll_once()

        assert watcher._offset_prune_complete is True
        assert watcher.registry.get(str(deleted)) == (0, 0)

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

    def test_poll_once_calls_tick_callback_even_without_new_jsonl_lines(self, tmp_path):
        ticks: list[str] = []
        watcher = JSONLWatcher(
            watch_dir=tmp_path / "empty-projects",
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda _items: None,
            on_tick=lambda: ticks.append("tick"),
        )

        assert watcher.poll_once() == 0
        assert ticks == ["tick"]

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

    def test_same_inode_rewind_persists_lower_confirmed_offset(self, tmp_path):
        project = self._make_project_dir(tmp_path)
        transcript = project / "s1.jsonl"
        original = b'{"id":"first"}\n{"id":"second"}\n'
        rewound = b'{"id":"first"}\n'
        transcript.write_bytes(original)

        def confirm_all(items):
            return {item["_source_file"]: item["_line_end_offset"] for item in items}

        registry_path = tmp_path / "offsets.json"
        watcher = JSONLWatcher(
            watch_dir=tmp_path / "projects",
            registry_path=registry_path,
            on_flush=confirm_all,
            batch_size=1,
            registry_flush_interval_s=3600,
        )
        assert watcher.poll_once() == 2
        assert watcher.registry.flush() is True
        stale_registry = OffsetRegistry(registry_path)
        original_inode = transcript.stat().st_ino

        transcript.write_bytes(rewound)
        assert transcript.stat().st_ino == original_inode
        assert watcher.poll_once() == 1
        assert watcher.registry.flush() is True

        stale_registry.set(str(transcript), len(original), original_inode)
        assert stale_registry.flush() is True
        assert OffsetRegistry(registry_path).get(str(transcript)) == (len(rewound), original_inode)

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

    def test_default_roots_include_cursor_agent_transcripts(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)
        cursor_session = tmp_path / ".cursor" / "sessions" / "session.jsonl"
        cursor_agent_transcript = (
            tmp_path / ".cursor" / "projects" / "repo" / "agent-transcripts" / "agent-session" / "agent-session.jsonl"
        )
        unrelated_project_jsonl = tmp_path / ".cursor" / "projects" / "repo" / "state.jsonl"
        for path in (cursor_session, cursor_agent_transcript, unrelated_project_jsonl):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"type":"message","payload":{"role":"user","content":"cursor line"}}\n')

        watcher = JSONLWatcher(
            watch_roots=default_watch_roots(home=tmp_path),
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda x: None,
        )

        files = set(watcher._discover_jsonl_files())

        assert str(cursor_session) in files
        assert str(cursor_agent_transcript) in files
        assert str(unrelated_project_jsonl) not in files
        assert watcher.provider_for_file(str(cursor_session)) == "cursor"
        assert watcher.provider_for_file(str(cursor_agent_transcript)) == "cursor-agent-transcripts"

    def test_default_roots_include_codex_and_gemini_sessions(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)
        codex_session = tmp_path / ".codex" / "sessions" / "2026" / "07" / "worker.jsonl"
        gemini_session = tmp_path / ".gemini" / "sessions" / "worker.jsonl"
        for path in (codex_session, gemini_session):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"role":"user","content":"worker line"}\n')

        watcher = JSONLWatcher(
            watch_roots=default_watch_roots(home=tmp_path),
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda x: None,
        )

        files = set(watcher._discover_jsonl_files())

        assert str(codex_session) in files
        assert str(gemini_session) in files
        assert watcher.provider_for_file(str(codex_session)) == "codex"
        assert watcher.provider_for_file(str(gemini_session)) == "gemini"

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

    def test_poll_drains_buffered_lines_before_checkpointing_oversized_append(self, tmp_path, monkeypatch):
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        rollout = sessions / "rollout.jsonl"
        encoded_lines = [
            (json.dumps({"role": "user", "content": f"buffered line {idx} with enough content"}) + "\n").encode()
            for idx in range(3)
        ]
        rollout.write_bytes(b"".join(encoded_lines))
        monkeypatch.setenv("BRAINLAYER_WATCH_MAX_FILE_BYTES", "256")
        flushed = []

        def confirm_all(items):
            flushed.extend(items)
            return {item["_source_file"]: item["_line_end_offset"] for item in items}

        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=confirm_all,
            batch_size=1,
            max_lines_per_file=1,
        )

        assert watcher.poll_once() == 1
        tailer = watcher._tailers[str(rollout)]
        assert tailer.offset == len(encoded_lines[0])
        assert tailer._buffer == b"".join(encoded_lines[1:])

        with rollout.open("ab") as file_handle:
            file_handle.write(json.dumps({"role": "user", "content": "x" * 512}).encode() + b"\n")
        oversized_size = rollout.stat().st_size

        assert watcher.poll_once() == 1
        assert [item["message"]["content"][0]["text"] for item in flushed] == [
            "buffered line 0 with enough content",
            "buffered line 1 with enough content",
        ]
        assert tailer.offset == len(encoded_lines[0]) + len(encoded_lines[1])
        assert tailer._buffer == encoded_lines[2]
        assert watcher.registry.get(str(rollout)) == (tailer.offset, rollout.stat().st_ino)
        assert tailer.offset < oversized_size

    def test_poll_fully_indexes_file_larger_than_read_window(self, tmp_path, monkeypatch):
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        rollout = sessions / "rollout.jsonl"
        expected = [f"entry {index} " + "x" * 80 for index in range(4)]
        rollout.write_text("".join(json.dumps({"role": "user", "content": content}) + "\n" for content in expected))
        assert rollout.stat().st_size > 128
        monkeypatch.setenv("BRAINLAYER_WATCH_MAX_FILE_BYTES", "128")
        flushed = []

        def confirm_all(items):
            flushed.extend(items)
            return {item["_source_file"]: item["_line_end_offset"] for item in items}

        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=confirm_all,
            batch_size=1,
        )

        for _ in range(12):
            watcher.poll_once()
            if watcher.registry.get(str(rollout))[0] == rollout.stat().st_size:
                break

        assert [item["message"]["content"][0]["text"] for item in flushed] == expected
        assert watcher.registry.get(str(rollout)) == (rollout.stat().st_size, rollout.stat().st_ino)

    def test_poll_does_not_checkpoint_dropped_tail_past_unconfirmed_record(self, tmp_path):
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        rollout = sessions / "rollout.jsonl"
        rollout.write_text(
            json.dumps({"role": "user", "content": "indexable record"})
            + "\n"
            + json.dumps({"type": "response_item", "payload": {"type": "function_call"}})
            + "\n"
        )

        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda _items: None,
            batch_size=10,
            flush_interval_ms=360000,
        )

        assert watcher.poll_once() == 1
        assert watcher.indexer.has_buffered_source(str(rollout))
        assert watcher.registry.get(str(rollout)) == (0, 0)

    def test_poll_bounds_large_file_without_starving_healthy_file(
        self,
        tmp_path,
        monkeypatch,
    ):
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        oversized = sessions / "oversized.jsonl"
        healthy = sessions / "healthy.jsonl"
        oversized.write_text(json.dumps({"role": "user", "content": "x" * 256}) + "\n")
        healthy.write_text(json.dumps({"role": "user", "content": "healthy"}) + "\n")
        os.utime(oversized, (2000, 2000))
        os.utime(healthy, (1000, 1000))
        monkeypatch.setenv("BRAINLAYER_WATCH_MAX_FILE_BYTES", "128")

        flushed = []

        def confirm_all(items):
            flushed.extend(items)
            return {item["_source_file"]: item["_line_end_offset"] for item in items}

        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=confirm_all,
            batch_size=1,
        )

        assert watcher.poll_once() == 1
        assert [item["_source_file"] for item in flushed] == [str(healthy)]
        assert watcher.registry.get(str(oversized)) == (0, 0)
        assert watcher._tailers[str(oversized)].offset == 0
        assert len(watcher._tailers[str(oversized)]._buffer) == 128

    def test_poll_never_checkpoints_past_unparsed_window(self, tmp_path, monkeypatch):
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        oversized = sessions / "oversized.jsonl"
        oversized.write_text(json.dumps({"role": "user", "content": "x" * 256}) + "\n")
        registry_path = tmp_path / "offsets.json"
        monkeypatch.setenv("BRAINLAYER_WATCH_MAX_FILE_BYTES", "128")

        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=registry_path,
            on_flush=lambda _items: None,
            registry_flush_interval_s=3600,
        )

        assert watcher.poll_once() == 0
        assert watcher._tailers[str(oversized)].offset == 0
        assert len(watcher._tailers[str(oversized)]._buffer) == 128
        assert OffsetRegistry(registry_path).get(str(oversized)) == (0, 0)

    def test_poll_indexes_large_inode_replacement_from_start(self, tmp_path, monkeypatch):
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        rollout = sessions / "rollout.jsonl"
        rollout.write_text(json.dumps({"role": "user", "content": "x" * 512}) + "\n")
        original_size = rollout.stat().st_size
        original_inode = rollout.stat().st_ino
        monkeypatch.setenv("BRAINLAYER_WATCH_MAX_FILE_BYTES", "128")
        flushed = []

        def confirm_all(items):
            flushed.extend(items)
            return {item["_source_file"]: item["_line_end_offset"] for item in items}

        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=confirm_all,
            batch_size=1,
        )
        watcher.registry.set(str(rollout), original_size, original_inode)
        watcher._tailers[str(rollout)] = JSONLTailer(str(rollout), offset=original_size)

        replacement = sessions / "replacement.tmp"
        replacement.write_text(json.dumps({"role": "user", "content": "y" * 256}) + "\n")
        os.replace(replacement, rollout)
        assert rollout.stat().st_ino != original_inode

        for _ in range(8):
            watcher.poll_once()
            if flushed:
                break

        assert [item["message"]["content"][0]["text"] for item in flushed] == ["y" * 256]
        assert watcher.registry.get(str(rollout)) == (
            rollout.stat().st_size,
            rollout.stat().st_ino,
        )

    def test_poll_discards_unconfirmed_buffer_when_inode_is_replaced(self, tmp_path, monkeypatch):
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        rollout = sessions / "rollout.jsonl"
        rollout.write_bytes(b'{"role":"user","content":"' + b"x" * 256)
        monkeypatch.setenv("BRAINLAYER_WATCH_MAX_FILE_BYTES", "128")
        flushed = []

        def confirm_all(items):
            flushed.extend(items)
            return {item["_source_file"]: item["_line_end_offset"] for item in items}

        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=confirm_all,
            batch_size=1,
        )
        watcher.poll_once()
        original_inode = rollout.stat().st_ino
        assert watcher.registry.get(str(rollout)) == (0, 0)
        assert watcher._tailers[str(rollout)]._buffer

        replacement = sessions / "replacement.jsonl"
        replacement.write_text(json.dumps({"role": "user", "content": "replacement"}) + "\n")
        os.replace(replacement, rollout)
        assert rollout.stat().st_ino != original_inode

        watcher.poll_once()

        assert [item["message"]["content"][0]["text"] for item in flushed] == ["replacement"]
        assert watcher.registry.get(str(rollout)) == (rollout.stat().st_size, rollout.stat().st_ino)

    def test_old_inode_flush_watermark_cannot_advance_replacement_offset(self, tmp_path, monkeypatch):
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        rollout = sessions / "rollout.jsonl"
        old_records = [f"old-{index}-" + "x" * 64 for index in range(20)]
        rollout.write_text("".join(json.dumps({"role": "user", "content": content}) + "\n" for content in old_records))
        state = {"fail": True}

        def flush(items):
            if state["fail"]:
                raise RuntimeError("retain old-inode batch")
            watermarks = {}
            for item in items:
                source = item["_source_file"]
                watermarks[source] = max(watermarks.get(source, 0), item["_line_end_offset"])
            return watermarks

        def capture_alarm(code, message, context):
            raise BrainLayerAlarm(code, message, context)

        monkeypatch.setattr("brainlayer.watcher.raise_alarm", capture_alarm)
        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=flush,
            batch_size=20,
        )
        watcher.poll_once()
        assert watcher.registry.get(str(rollout)) == (0, 0)
        assert len(watcher.indexer._buffer) == 20

        replacement = sessions / "replacement.jsonl"
        replacement.write_text(
            "".join(json.dumps({"role": "user", "content": f"new-{index}"}) + "\n" for index in range(3))
        )
        os.replace(replacement, rollout)
        replacement_size = rollout.stat().st_size
        replacement_inode = rollout.stat().st_ino
        watcher.poll_once()

        state["fail"] = False
        watcher.indexer.flush()

        assert watcher.registry.get(str(rollout)) == (replacement_size, replacement_inode)

        with rollout.open("a") as file_handle:
            file_handle.write(json.dumps({"role": "user", "content": "new-append"}) + "\n")
        watcher.poll_once()
        watcher.indexer.flush()
        assert watcher.registry.get(str(rollout)) == (rollout.stat().st_size, replacement_inode)

    def test_replacement_between_read_and_confirmation_restarts_at_zero(self, tmp_path, monkeypatch):
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        rollout = sessions / "rollout.jsonl"
        valid_line = (json.dumps({"role": "user", "content": "old valid record"}) + "\n").encode()
        malformed_line = b'{"role":"user","content":}\n'
        rollout.write_bytes(valid_line + malformed_line)
        old_inode = rollout.stat().st_ino
        registry_path = tmp_path / "offsets.json"
        replacement_text = "replacement must be read from its first byte"
        replacement = sessions / "replacement.tmp"
        replacement.write_text(json.dumps({"role": "user", "content": replacement_text}) + "\n")
        replacement_inode = replacement.stat().st_ino
        flushed = []

        def replace_during_flush(items):
            flushed.extend(items)
            os.replace(replacement, rollout)
            return {item["_source_file"]: item["_line_end_offset"] for item in items}

        monkeypatch.setenv("BRAINLAYER_WATCHER_QUARANTINE_DIR", str(tmp_path / "quarantine"))
        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=registry_path,
            on_flush=replace_during_flush,
            batch_size=1,
            registry_flush_interval_s=3600,
        )

        assert watcher.poll_once() == 1
        assert rollout.stat().st_ino == replacement_inode
        assert watcher.registry.get(str(rollout)) == (len(valid_line + malformed_line), old_inode)
        assert watcher.registry.flush() is True

        replacement_items = []

        def confirm_replacement(items):
            replacement_items.extend(items)
            return {item["_source_file"]: item["_line_end_offset"] for item in items}

        fresh_watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=registry_path,
            on_flush=confirm_replacement,
            batch_size=1,
            registry_flush_interval_s=3600,
        )

        assert fresh_watcher._ensure_tailer(str(rollout)).offset == 0
        assert fresh_watcher.poll_once() == 1
        assert [item["message"]["content"][0]["text"] for item in replacement_items] == [replacement_text]

    def test_poll_indexes_large_same_inode_rewind_from_start(self, tmp_path, monkeypatch):
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        rollout = sessions / "rollout.jsonl"
        rollout.write_text(json.dumps({"role": "user", "content": "x" * 512}) + "\n")
        original_size = rollout.stat().st_size
        original_inode = rollout.stat().st_ino
        monkeypatch.setenv("BRAINLAYER_WATCH_MAX_FILE_BYTES", "128")
        flushed = []
        rewinds = []

        def confirm_all(items):
            flushed.extend(items)
            return {item["_source_file"]: item["_line_end_offset"] for item in items}

        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=confirm_all,
            on_rewind=lambda *args: rewinds.append(args),
            batch_size=1,
        )
        watcher.registry.set(str(rollout), original_size, original_inode)
        watcher._tailers[str(rollout)] = JSONLTailer(str(rollout), offset=original_size)

        with rollout.open("w") as file_handle:
            file_handle.write(json.dumps({"role": "user", "content": "y" * 256}) + "\n")
        assert rollout.stat().st_ino == original_inode

        for _ in range(8):
            watcher.poll_once()
            if flushed:
                break

        assert [item["message"]["content"][0]["text"] for item in flushed] == ["y" * 256]
        assert watcher.registry.get(str(rollout)) == (
            rollout.stat().st_size,
            rollout.stat().st_ino,
        )
        assert rewinds == [(str(rollout), "rollout", original_size, rollout.stat().st_size)]

    def test_poll_does_not_checkpoint_past_retained_unconfirmed_entries(self, tmp_path, monkeypatch):
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        rollout = sessions / "rollout.jsonl"
        rollout.write_text(json.dumps({"role": "user", "content": "pending"}) + "\n")
        monkeypatch.setenv("BRAINLAYER_WATCH_MAX_FILE_BYTES", "128")

        def fail_flush(_items):
            raise RuntimeError("write unavailable")

        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=fail_flush,
            batch_size=10,
            flush_interval_ms=360000,
        )

        assert watcher.poll_once() == 1
        assert watcher.registry.get(str(rollout)) == (0, 0)
        with rollout.open("a") as file_handle:
            file_handle.write(json.dumps({"role": "user", "content": "x" * 256}) + "\n")

        assert watcher.poll_once() == 0
        assert watcher.registry.get(str(rollout)) == (0, 0)
        assert len(watcher.indexer._buffer) == 1

    def test_poll_does_not_checkpoint_past_partially_confirmed_watermark(self, tmp_path, monkeypatch):
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        rollout = sessions / "rollout.jsonl"
        rollout.write_text(
            json.dumps({"role": "user", "content": "first"})
            + "\n"
            + json.dumps({"role": "user", "content": "second"})
            + "\n"
        )
        monkeypatch.setenv("BRAINLAYER_WATCH_MAX_FILE_BYTES", "128")

        def confirm_first_only(items):
            return {str(rollout): items[0]["_line_end_offset"]}

        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=confirm_first_only,
            batch_size=2,
            flush_interval_ms=360000,
        )

        assert watcher.poll_once() == 2
        confirmed_offset, confirmed_inode = watcher.registry.get(str(rollout))
        assert confirmed_offset < watcher._tailers[str(rollout)].offset
        assert watcher.indexer._buffer == []

        with rollout.open("a") as file_handle:
            file_handle.write(json.dumps({"role": "user", "content": "x" * 256}) + "\n")

        assert watcher.poll_once() == 0
        assert watcher.registry.get(str(rollout)) == (confirmed_offset, confirmed_inode)

    def test_file_processing_failure_raises_alarm_and_surfaces_in_health(self, tmp_path, monkeypatch):
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        rollout = sessions / "rollout.jsonl"
        rollout.write_text(json.dumps({"role": "user", "content": "must not be checkpointed"}) + "\n")
        health_path = tmp_path / "watcher-health.json"
        alarms = []

        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda _items: None,
            health_path=health_path,
        )

        def forced_failure(_filepath):
            raise OSError("forced read failure")

        def capture_alarm(code, message, context):
            alarms.append((code, message, context))
            raise BrainLayerAlarm(code, message, context)

        monkeypatch.setattr(watcher, "_ensure_tailer", forced_failure)
        monkeypatch.setattr("brainlayer.watcher.raise_alarm", capture_alarm)
        assert watcher.poll_once() == 0
        payload = json.loads(health_path.read_text())

        assert watcher.registry.get(str(rollout)) == (0, 0)
        assert alarms[0][0] == "watcher_file_ingestion_failed"
        assert alarms[0][2]["file_path"] == str(rollout)
        assert payload["alerting"] is True
        assert "file_ingestion_failure" in payload["alert_reasons"]
        assert payload["file_ingestion_failure_count"] == 1
        assert payload["file_ingestion_failures"][0]["file_path"] == str(rollout)

    def test_normalization_failure_retries_without_crossing_failed_record(self, tmp_path, monkeypatch):
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        rollout = sessions / "rollout.jsonl"
        expected = ["first", "second"]
        rollout.write_text("".join(json.dumps({"role": "user", "content": content}) + "\n" for content in expected))
        flushed = []

        def confirm_all(items):
            flushed.extend(items)
            return {item["_source_file"]: item["_line_end_offset"] for item in items}

        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=confirm_all,
            batch_size=1,
            max_lines_per_file=1,
        )
        original_normalize = watcher._normalize_lines
        attempts = 0

        def fail_once(filepath, lines):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("forced normalization failure")
            return original_normalize(filepath, lines)

        def capture_alarm(code, message, context):
            raise BrainLayerAlarm(code, message, context)

        monkeypatch.setattr(watcher, "_normalize_lines", fail_once)
        monkeypatch.setattr("brainlayer.watcher.raise_alarm", capture_alarm)

        assert watcher.poll_once() == 0
        assert watcher.registry.get(str(rollout)) == (0, 0)

        watcher.poll_once()
        watcher.poll_once()

        assert [item["message"]["content"][0]["text"] for item in flushed] == expected
        assert watcher.registry.get(str(rollout))[0] == rollout.stat().st_size

    def test_malformed_record_is_quarantined_and_later_records_continue(self, tmp_path, monkeypatch):
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        rollout = sessions / "rollout.jsonl"
        first = json.dumps({"role": "user", "content": "first"}) + "\n"
        malformed = b"not json at all\n"
        last = json.dumps({"role": "user", "content": "last"}) + "\n"
        rollout.write_bytes(first.encode() + malformed + last.encode())
        health_path = tmp_path / "watcher-health.json"
        quarantine_dir = tmp_path / "quarantine"
        monkeypatch.setenv("BRAINLAYER_WATCHER_QUARANTINE_DIR", str(quarantine_dir))
        alarms = []
        flushed = []

        def confirm_all(items):
            flushed.extend(items)
            return {item["_source_file"]: item["_line_end_offset"] for item in items}

        def capture_alarm(code, message, context):
            alarms.append((code, message, context))
            raise BrainLayerAlarm(code, message, context)

        monkeypatch.setattr("brainlayer.watcher.raise_alarm", capture_alarm)
        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=confirm_all,
            batch_size=1,
            health_path=health_path,
        )

        for _ in range(3):
            watcher.poll_once()

        payload = json.loads(health_path.read_text())
        quarantined = list(quarantine_dir.glob("watcher-parse-*.jsonl.bad"))
        assert [item["message"]["content"][0]["text"] for item in flushed] == ["first", "last"]
        assert watcher.registry.get(str(rollout))[0] == rollout.stat().st_size
        assert len(quarantined) == 1
        assert quarantined[0].read_bytes() == malformed
        assert len(alarms) == 1
        assert alarms[0][0] == "watcher_file_ingestion_failed"
        assert alarms[0][2]["disposition"] == "quarantined"
        assert payload["quarantined_record_count_total"] == 1
        assert payload["quarantined_records"][0]["file_path"] == str(rollout)
        assert "quarantined_record" in payload["alert_reasons"]

    def test_quarantined_offset_waits_for_prior_indexable_record_confirmation(self, tmp_path, monkeypatch):
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        rollout = sessions / "rollout.jsonl"
        first = json.dumps({"role": "user", "content": "first"}) + "\n"
        malformed = b"not json at all\n"
        rollout.write_bytes(first.encode() + malformed)
        monkeypatch.setenv("BRAINLAYER_WATCHER_QUARANTINE_DIR", str(tmp_path / "quarantine"))

        def confirm_all(items):
            return {item["_source_file"]: item["_line_end_offset"] for item in items}

        def capture_alarm(code, message, context):
            raise BrainLayerAlarm(code, message, context)

        monkeypatch.setattr("brainlayer.watcher.raise_alarm", capture_alarm)

        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=confirm_all,
            batch_size=10,
            flush_interval_ms=360_000,
        )

        watcher.poll_once()
        assert watcher.registry.get(str(rollout)) == (0, 0)

        watcher.indexer.flush()

        assert watcher.registry.get(str(rollout))[0] == rollout.stat().st_size

    def test_quarantine_write_failure_keeps_record_and_offset_unmodified(self, tmp_path, monkeypatch):
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        rollout = sessions / "rollout.jsonl"
        rollout.write_bytes(b"not json at all\n")
        invalid_quarantine_dir = tmp_path / "not-a-directory"
        invalid_quarantine_dir.write_text("occupied")
        monkeypatch.setenv("BRAINLAYER_WATCHER_QUARANTINE_DIR", str(invalid_quarantine_dir))
        alarms = []

        def capture_alarm(code, message, context):
            alarms.append((code, message, context))
            raise BrainLayerAlarm(code, message, context)

        monkeypatch.setattr("brainlayer.watcher.raise_alarm", capture_alarm)
        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda _items: None,
        )

        watcher.poll_once()

        tailer = watcher._tailers[str(rollout)]
        assert tailer.offset == 0
        assert tailer._buffer == b"not json at all\n"
        assert watcher.registry.get(str(rollout)) == (0, 0)
        assert len(alarms) == 1
        assert alarms[0][2]["error_type"] == "FileExistsError"

    def test_growing_blocked_record_emits_one_alarm(self, tmp_path, monkeypatch):
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        rollout = sessions / "rollout.jsonl"
        rollout.write_bytes(b'{"role":"user","content":"' + b"x" * 64)
        monkeypatch.setenv("BRAINLAYER_WATCH_MAX_FILE_BYTES", "16")
        monkeypatch.setenv("BRAINLAYER_WATCH_MAX_RECORD_BYTES", "32")
        alarms = []

        def capture_alarm(code, message, context):
            alarms.append((code, message, context))
            raise BrainLayerAlarm(code, message, context)

        monkeypatch.setattr("brainlayer.watcher.raise_alarm", capture_alarm)
        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda _items: None,
        )

        for _ in range(4):
            with rollout.open("ab") as file_handle:
                file_handle.write(b"x")
            watcher.poll_once()

        assert len(alarms) == 1
        assert watcher.registry.get(str(rollout)) == (0, 0)
        assert len(watcher._tailers[str(rollout)]._buffer) <= 33

    def test_health_caps_failure_details_and_reports_overflow(self, tmp_path):
        health_path = tmp_path / "watcher-health.json"
        watcher = JSONLWatcher(
            watch_roots=[],
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda _items: None,
            health_path=health_path,
        )
        watcher._file_ingestion_failures = {
            f"/tmp/failure-{index}.jsonl": {
                "file_path": f"/tmp/failure-{index}.jsonl",
                "error": "forced",
                "_fingerprint": ("OSError", "forced"),
            }
            for index in range(150)
        }

        watcher._write_health_snapshot([])

        payload = json.loads(health_path.read_text())
        assert payload["file_ingestion_failure_count"] == 150
        assert len(payload["file_ingestion_failures"]) == 100
        assert payload["file_ingestion_failures_overflow_count"] == 50

    def test_negative_watch_read_window_falls_back_to_default(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv("BRAINLAYER_WATCH_MAX_FILE_BYTES", "-1")

        watcher = JSONLWatcher(
            watch_roots=[],
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda _items: None,
        )

        assert watcher.max_read_bytes_per_file == 100 * 1024 * 1024
        assert any("BRAINLAYER_WATCH_MAX_FILE_BYTES='-1'" in record.getMessage() for record in caplog.records)

    def test_invalid_watch_read_window_falls_back_to_default(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv("BRAINLAYER_WATCH_MAX_FILE_BYTES", "invalid")

        watcher = JSONLWatcher(
            watch_roots=[],
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda _items: None,
        )

        assert watcher.max_read_bytes_per_file == 100 * 1024 * 1024
        assert any("BRAINLAYER_WATCH_MAX_FILE_BYTES='invalid'" in record.getMessage() for record in caplog.records)

    def test_poll_ingests_append_after_large_file_is_fully_consumed(self, tmp_path, monkeypatch):
        sessions = tmp_path / "codex" / "sessions"
        sessions.mkdir(parents=True)
        rollout = sessions / "rollout.jsonl"
        rollout.write_text(json.dumps({"role": "user", "content": "x" * 256}) + "\n")
        monkeypatch.setenv("BRAINLAYER_WATCH_MAX_FILE_BYTES", "128")
        flushed = []

        def confirm_all(items):
            flushed.extend(items)
            return {item["_source_file"]: item["_line_end_offset"] for item in items}

        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", sessions)],
            registry_path=tmp_path / "offsets.json",
            on_flush=confirm_all,
            batch_size=1,
        )

        for _ in range(8):
            watcher.poll_once()
            if watcher.registry.get(str(rollout))[0] == rollout.stat().st_size:
                break
        with rollout.open("a") as file_handle:
            file_handle.write(json.dumps({"role": "user", "content": "small append"}) + "\n")

        for _ in range(4):
            watcher.poll_once()
            if len(flushed) == 2:
                break
        assert [item["message"]["content"][0]["text"] for item in flushed] == ["x" * 256, "small append"]

    def test_zero_watch_read_window_falls_back_to_default(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv("BRAINLAYER_WATCH_MAX_FILE_BYTES", "0")

        watcher = JSONLWatcher(
            watch_roots=[],
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda _items: None,
        )

        assert watcher.max_read_bytes_per_file == 100 * 1024 * 1024
        assert any("BRAINLAYER_WATCH_MAX_FILE_BYTES='0'" in record.getMessage() for record in caplog.records)

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

    def test_health_snapshot_emits_alarm_without_stopping_on_zero_db_writes_while_active(self, tmp_path):
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
        assert watcher.poll_once() == 0

        payload = json.loads(health_path.read_text())
        assert payload["providers"] == ["codex"]
        assert payload["alerting"] is True
        assert "coverage_drop" in payload["alert_reasons"]
        assert payload["active_jsonl_entries_per_minute"] > 0
        assert payload["db_realtime_inserts_per_minute"] == 0

    def test_health_snapshot_holds_zero_write_alarm_across_quiet_burst_without_stopping(self, tmp_path):
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
        assert watcher.poll_once() == 0

        payload = json.loads(health_path.read_text())
        assert payload["alerting"] is True
        assert payload["watcher_chunks_output_per_minute"] > 0

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
        assert watcher.poll_once() == 1

        payload = json.loads(health_path.read_text())
        assert payload["alerting"] is True
        assert payload["db_realtime_inserts_per_minute"] == 0

    def test_health_snapshot_emits_alarm_when_db_probe_fails_while_active(self, tmp_path):
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
        assert watcher.poll_once() == 0

        payload = json.loads(health_path.read_text())
        assert payload["db_realtime_inserts_per_minute"] is None
        assert payload["db_probe_failed"] is True
        assert payload["alerting"] is True

    def test_health_snapshot_emits_alarm_when_flush_fails_while_active(self, tmp_path):
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
        assert watcher.poll_once() == 0

        payload = json.loads(health_path.read_text())
        assert payload["failed_flush_inputs_per_minute"] > 0
        assert payload["active_jsonl_entries_per_minute"] > 0
        assert payload["watcher_chunks_output_per_minute"] == 0
        assert payload["alerting"] is True

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
