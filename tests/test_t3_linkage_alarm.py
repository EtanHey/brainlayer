"""Pending provenance must stay fatal without emitting once per retained entry."""

import json
import subprocess
import sys

import pytest

from brainlayer import alarm, t3_provenance
from brainlayer.alarm import BrainLayerAlarm


@pytest.fixture
def pending(tmp_path, monkeypatch):
    source = tmp_path / "rollout.jsonl"
    session = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    source.write_text(json.dumps({"type": "session_meta", "payload": {"id": session, "originator": "T3 Code"}}) + "\n")
    monkeypatch.setattr(t3_provenance, "resolve_t3_sources", lambda *_: [(tmp_path / "state.sqlite", 2)])
    events = []
    monkeypatch.setattr(alarm, "emit_alarm", lambda event: events.append(event))
    clock = [10000.0]
    monkeypatch.setattr("time.time", lambda: clock[0])

    def check(linked=False):
        return t3_provenance.is_t3_app_initiated_codex_session(
            source, linked_session_ids={session} if linked else set(), projection_version=2
        )

    return source, events, clock, check


def test_pending_retries_raise_but_emit_once_and_persist_status(pending, monkeypatch):
    source, events, clock, check = pending
    for _ in range(20):
        with pytest.raises(BrainLayerAlarm, match="t3_runtime_linkage_pending"):
            check()
    assert len(events) == 1
    from brainlayer import t3_linkage_alarm

    records = list(t3_linkage_alarm.status_directory().glob("*.json"))
    assert len(records) == 1
    status = json.loads(records[0].read_text())
    assert status["state"] == "pending"
    assert status["path"] == str(source)
    assert status["code"] == "t3_runtime_linkage_pending"
    # A fresh interpreter, as after a supervisor restart, must honor the durable receipt.
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys, time
from pathlib import Path
from brainlayer import alarm, t3_linkage_alarm
time.time = lambda: 10000.0
alarm.emit_alarm = lambda event: print('EMITTED')
try:
    t3_linkage_alarm.observe(Path(sys.argv[1]), 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
                            [(Path(sys.argv[1]).parent / 'state.sqlite', 2)], pending=True)
except alarm.BrainLayerAlarm:
    print('PENDING')
""",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert child.stdout.strip() == "PENDING"
    t3_linkage_alarm._CACHE.clear()
    with pytest.raises(BrainLayerAlarm):
        check()
    assert len(events) == 1
    clock[0] += 3600
    with pytest.raises(BrainLayerAlarm):
        check()
    assert len(events) == 2


def test_linkage_resolution_and_recurrence_realert(pending):
    _, events, _, check = pending
    with pytest.raises(BrainLayerAlarm):
        check()
    assert check(linked=True)
    from brainlayer import t3_linkage_alarm

    record = next(t3_linkage_alarm.status_directory().glob("*.json"))
    assert json.loads(record.read_text())["state"] == "resolved"
    with pytest.raises(BrainLayerAlarm):
        check()
    assert len(events) == 2


def test_changed_identity_realerts_without_realerting_for_append(pending):
    source, events, _, check = pending
    with pytest.raises(BrainLayerAlarm):
        check()
    with source.open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(BrainLayerAlarm):
        check()
    assert len(events) == 1
    source.write_text(
        source.read_text().replace("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
    )
    with pytest.raises(BrainLayerAlarm):
        check()
    assert len(events) == 2


def test_status_write_failure_stays_loud_and_pending(pending, monkeypatch, tmp_path, caplog):
    from brainlayer import t3_linkage_alarm

    _, events, _, check = pending
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("block status writes")
    monkeypatch.setattr(t3_linkage_alarm, "status_directory", lambda: blocked)
    for _ in range(2):
        with pytest.raises(BrainLayerAlarm):
            check()
    assert len(events) == 2
    assert "Could not persist T3 linkage alarm status" in caplog.text


def test_corrupt_status_realerts_and_repairs(pending):
    from brainlayer import t3_linkage_alarm

    _, events, _, check = pending
    with pytest.raises(BrainLayerAlarm):
        check()
    record = next(t3_linkage_alarm.status_directory().glob("*.json"))
    record.write_text("{broken")
    t3_linkage_alarm._CACHE.clear()
    with pytest.raises(BrainLayerAlarm):
        check()
    assert len(events) == 2
    assert json.loads(record.read_text())["state"] == "pending"


def test_projection_selection_change_realerts(pending, monkeypatch, tmp_path):
    _, events, _, check = pending
    with pytest.raises(BrainLayerAlarm):
        check()
    monkeypatch.setattr(t3_provenance, "resolve_t3_sources", lambda *_: [(tmp_path / "other-state.sqlite", 2)])
    with pytest.raises(BrainLayerAlarm):
        check()
    assert len(events) == 2


def test_busy_status_lock_does_not_block_or_swallow_pending(pending, monkeypatch):
    from brainlayer import t3_linkage_alarm

    _, events, _, check = pending

    def busy(*_):
        raise BlockingIOError("status lock busy")

    monkeypatch.setattr(t3_linkage_alarm.fcntl, "flock", busy)
    with pytest.raises(BrainLayerAlarm):
        check()
    assert len(events) == 1


def test_other_process_resolution_invalidates_cached_pending(pending):
    from brainlayer import t3_linkage_alarm

    _, events, _, check = pending
    with pytest.raises(BrainLayerAlarm):
        check()
    record = next(t3_linkage_alarm.status_directory().glob("*.json"))
    status = json.loads(record.read_text())
    status["state"] = "resolved"
    record.write_text(json.dumps(status))
    with pytest.raises(BrainLayerAlarm):
        check()
    assert len(events) == 2
