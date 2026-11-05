"""#732: a vanished JSONL is not a parse failure, and a reappeared path must not be skipped.

Every test drives the production `JSONLWatcher.poll_once` path over synthetic files in tmp_path.
Orderings that a live machine produces by race (a file deleted between discovery and read, a
path deleted and recreated inside one poll window) are forced deterministically.
"""

from __future__ import annotations

import json
import os

from brainlayer.alarm import BrainLayerAlarm
from brainlayer.watcher import JSONLWatcher, WatchRoot


def _line(content: str) -> str:
    return json.dumps({"role": "user", "content": content}) + "\n"


def _texts(items: list[dict]) -> list[str]:
    return [item["message"]["content"][0]["text"] for item in items]


def _make_watcher(tmp_path, monkeypatch, *, on_rewind=None):
    sessions = tmp_path / "codex" / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    flushed: list[dict] = []
    alarms: list[tuple[str, str, dict]] = []

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
        on_rewind=on_rewind,
        batch_size=1,
        health_path=tmp_path / "watcher-health.json",
    )
    return watcher, sessions, flushed, alarms


def _health(tmp_path) -> dict:
    return json.loads((tmp_path / "watcher-health.json").read_text())


def _poll(watcher, times: int = 3) -> None:
    for _ in range(times):
        watcher.poll_once()


def test_file_deleted_between_discovery_and_read_is_not_a_parse_alarm(tmp_path, monkeypatch):
    watcher, sessions, flushed, alarms = _make_watcher(tmp_path, monkeypatch)
    rollout = sessions / "rollout.jsonl"
    rollout.write_text(_line("about to vanish"))
    discover = watcher._discover_jsonl_files

    def discover_then_delete():
        files = discover()
        rollout.unlink()  # the race: listed by discovery, gone before the read
        return files

    monkeypatch.setattr(watcher, "_discover_jsonl_files", discover_then_delete)
    watcher.poll_once()
    monkeypatch.setattr(watcher, "_discover_jsonl_files", discover)
    _poll(watcher)

    payload = _health(tmp_path)
    assert alarms == [], "a deleted file must not raise watcher_file_ingestion_failed ('could not be parsed')"
    assert payload["file_ingestion_failure_count"] == 0
    assert "file_ingestion_failure" not in payload["alert_reasons"]
    assert payload["vanished_source_count_total"] == 1

    rollout.write_text(_line("came back"))
    _poll(watcher)
    assert _texts(flushed) == ["came back"]
    assert alarms == []


def test_deleted_tracked_file_is_counted_once_across_polls(tmp_path, monkeypatch):
    watcher, sessions, flushed, alarms = _make_watcher(tmp_path, monkeypatch)
    rollout = sessions / "rollout.jsonl"
    rollout.write_text(_line("first"))
    _poll(watcher)
    rollout.unlink()
    _poll(watcher, times=5)

    assert _texts(flushed) == ["first"]
    assert alarms == []
    assert _health(tmp_path)["vanished_source_count_total"] == 1
    # The checkpoint is not retired by a live watcher; startup pruning owns that, with mount evidence.
    assert watcher.registry.get(str(rollout))[0] > 0


def test_same_inode_with_new_bytes_after_vanishing_is_reingested_not_skipped(tmp_path, monkeypatch):
    """The reused-inode case, made deterministic: the path vanishes, and what reappears under
    the SAME inode is a different, longer file. Resuming at the old offset drops its head."""
    watcher, sessions, flushed, alarms = _make_watcher(tmp_path, monkeypatch)
    rollout = sessions / "rollout.jsonl"
    rollout.write_text(_line("old session line one") + _line("old session line two"))
    _poll(watcher)
    assert _texts(flushed) == ["old session line one", "old session line two"]
    old_inode = rollout.stat().st_ino
    generation_before = watcher.registry.generation(str(rollout))

    parked = tmp_path / "parked.jsonl"
    os.rename(rollout, parked)
    _poll(watcher)
    replacement = _line("new head A") + _line("new head B") + _line("new head C") + _line("new tail")
    with open(parked, "r+b") as handle:  # same inode, new content, longer than the old offset
        handle.write(replacement.encode())
        handle.truncate()
    os.rename(parked, rollout)
    assert rollout.stat().st_ino == old_inode
    flushed.clear()
    _poll(watcher)

    assert _texts(flushed) == ["new head A", "new head B", "new head C", "new tail"]
    assert watcher.registry.get(str(rollout)) == (len(replacement), old_inode)
    # A new offset generation, as for an inode change: confirmations still in flight for the
    # old file can no longer advance the checkpoint of the new one.
    assert watcher.registry.generation(str(rollout)) > generation_before
    assert alarms == []


def test_recreated_inside_one_poll_window_with_reused_inode_is_reingested(tmp_path, monkeypatch):
    """No poll ever observes the gap, so only the bytes can tell the files apart."""
    watcher, sessions, flushed, _alarms = _make_watcher(tmp_path, monkeypatch)
    rollout = sessions / "rollout.jsonl"
    rollout.write_text(_line("before " + "x" * 40))
    _poll(watcher)
    old_inode = rollout.stat().st_ino
    replacement = _line("after " + "y" * 40) + _line("after two")
    with open(rollout, "r+b") as handle:  # stands in for unlink + create handed the freed inode
        handle.write(replacement.encode())
        handle.truncate()
    assert rollout.stat().st_ino == old_inode
    flushed.clear()
    _poll(watcher)

    assert _texts(flushed) == ["after " + "y" * 40, "after two"]
    assert watcher.registry.get(str(rollout))[0] == len(replacement)


def test_transient_rename_and_back_resumes_without_reingesting(tmp_path, monkeypatch):
    watcher, sessions, flushed, alarms = _make_watcher(tmp_path, monkeypatch)
    rollout = sessions / "rollout.jsonl"
    rollout.write_text(_line("one") + _line("two"))
    _poll(watcher)
    parked = tmp_path / "parked.jsonl"
    os.rename(rollout, parked)
    _poll(watcher)
    os.rename(parked, rollout)
    with rollout.open("a") as handle:
        handle.write(_line("three"))
    _poll(watcher)

    assert _texts(flushed) == ["one", "two", "three"]
    assert alarms == []


def test_new_inode_after_vanishing_is_ingested_from_the_start(tmp_path, monkeypatch):
    watcher, sessions, flushed, _alarms = _make_watcher(tmp_path, monkeypatch)
    rollout = sessions / "rollout.jsonl"
    rollout.write_text(_line("old one") + _line("old two"))
    _poll(watcher)
    keep = tmp_path / "keep.jsonl"
    os.link(rollout, keep)  # holds the old inode so the new file cannot reuse it
    rollout.unlink()
    _poll(watcher)
    rollout.write_text(_line("fresh one") + _line("fresh two") + _line("fresh three"))
    assert rollout.stat().st_ino != keep.stat().st_ino
    flushed.clear()
    _poll(watcher)

    assert _texts(flushed) == ["fresh one", "fresh two", "fresh three"]


def test_shorter_file_after_vanishing_still_takes_the_rewind_path(tmp_path, monkeypatch):
    rewinds: list[tuple[str, int, int]] = []
    watcher, sessions, flushed, _alarms = _make_watcher(
        tmp_path,
        monkeypatch,
        on_rewind=lambda path, _session, old, new: rewinds.append((path, old, new)),
    )
    rollout = sessions / "rollout.jsonl"
    original = _line("kept") + _line("reverted " + "z" * 30)
    rollout.write_text(original)
    _poll(watcher)
    parked = tmp_path / "parked.jsonl"
    os.rename(rollout, parked)
    _poll(watcher)
    with open(parked, "r+b") as handle:
        handle.truncate(len(_line("kept")))
    os.rename(parked, rollout)
    _poll(watcher)

    assert rewinds == [(str(rollout), len(original), len(_line("kept")))]


def test_file_missing_from_discovery_but_still_on_disk_is_not_declared_vanished(tmp_path, monkeypatch):
    """Only lstat ENOENT confirms a deletion; a discovery miss alone (a stat hiccup) does not."""
    watcher, sessions, _flushed, _alarms = _make_watcher(tmp_path, monkeypatch)
    rollout = sessions / "rollout.jsonl"
    rollout.write_text(_line("still here"))
    _poll(watcher)
    discover = watcher._discover_jsonl_files
    monkeypatch.setattr(watcher, "_discover_jsonl_files", lambda: [path for path in discover() if path != str(rollout)])
    _poll(watcher)

    assert _health(tmp_path)["vanished_source_count_total"] == 0
