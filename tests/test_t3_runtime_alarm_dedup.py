"""#1402: runtime T3 alarms stay fatal on every flush but emit CRITICAL only on change or hourly."""

import json
import os
import sqlite3
import subprocess
import sys

import pytest

from brainlayer import alarm, t3_linkage_alarm
from brainlayer.alarm import BrainLayerAlarm
from brainlayer.t3_provenance import t3_app_codex_session_ids

SESSION = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
SENTINEL = "RAW-CURSOR-SENTINEL-do-not-log"


def _legacy_db(path, *, columns=("thread_id", "provider_name", "resume_cursor_json"), rows=()):
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(f"CREATE TABLE provider_session_runtime ({', '.join(f'{c} TEXT' for c in columns)})")
        if rows:
            conn.executemany("INSERT INTO provider_session_runtime VALUES (?, ?, ?)", rows)
    return path


def _make(kind, path):
    if kind == "unavailable":
        path.unlink(missing_ok=True)
        return path
    if kind == "schema_drift":
        return _legacy_db(path, columns=("thread_id",))
    if kind == "linkage_invalid":
        return _legacy_db(path, rows=[("t", "codex", f"not json {SENTINEL}")])
    return _legacy_db(path, rows=[("t", "codex", json.dumps({"threadId": SESSION}))])


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(alarm, "emit_alarm", events.append)
    clock = [10000.0]
    monkeypatch.setattr("time.time", lambda: clock[0])
    state = tmp_path / "state.sqlite"

    def poll(times=1):
        codes = []
        for _ in range(times):
            with pytest.raises(BrainLayerAlarm) as raised:
                t3_app_codex_session_ids(state, projection_version=1)
            codes.append(raised.value.code)
        return codes

    return state, events, clock, poll


@pytest.mark.parametrize("kind", ["unavailable", "schema_drift", "linkage_invalid"])
def test_persistent_runtime_alarm_raises_every_poll_but_emits_hourly(runtime, kind):
    state, events, clock, poll = runtime
    _make(kind, state)
    assert poll(20) == [f"t3_runtime_{kind}"] * 20
    assert [e.code for e in events] == [f"t3_runtime_{kind}"]
    clock[0] += 3599
    poll()
    assert len(events) == 1
    clock[0] += 1
    poll()
    assert len(events) == 2


def test_changed_runtime_alarm_realerts(runtime):
    state, events, _, poll = runtime
    _make("schema_drift", state)
    poll(3)
    _make("linkage_invalid", state)
    poll(3)
    assert [e.code for e in events] == ["t3_runtime_schema_drift", "t3_runtime_linkage_invalid"]


def test_resolution_removes_status_and_recurrence_realerts(runtime):
    state, events, _, poll = runtime
    _make("schema_drift", state)
    poll(2)
    assert list(t3_linkage_alarm.status_directory().glob("*.json"))
    _make("healthy", state)
    assert t3_app_codex_session_ids(state, projection_version=1) == {SESSION}
    assert list(t3_linkage_alarm.status_directory().iterdir()) == []
    _make("schema_drift", state)
    poll(2)
    assert len(events) == 2


def test_status_receipt_survives_fresh_interpreter_with_real_emitter(runtime):
    state, events, _, _ = runtime
    _make("schema_drift", state)
    script = """
import sys
from brainlayer.alarm import BrainLayerAlarm
from brainlayer.t3_provenance import t3_app_codex_session_ids
for _ in range(5):
    try:
        t3_app_codex_session_ids(sys.argv[1], projection_version=1)
    except BrainLayerAlarm as exc:
        print(exc.code)
"""
    env = {**os.environ, "AXIOM_TOKEN": ""}
    runs = [
        subprocess.run([sys.executable, "-c", script, str(state)], capture_output=True, text=True, env=env, check=True)
        for _ in range(2)
    ]
    assert [r.stdout.split() for r in runs] == [["t3_runtime_schema_drift"] * 5] * 2
    first, second = (r.stderr.count("BRAINLAYER_ALARM t3_runtime_schema_drift") for r in runs)
    # One emission (stderr print + logging fallback) in the first process; the receipt silences the second.
    assert (first, second) == (2, 0)


def test_busy_status_lock_still_emits_without_error_traceback(runtime, monkeypatch, caplog):
    state, events, _, poll = runtime
    _make("schema_drift", state)

    def busy(*_):
        raise BlockingIOError("status lock busy")

    monkeypatch.setattr(t3_linkage_alarm.fcntl, "flock", busy)
    poll(2)
    assert len(events) == 2
    assert "Could not persist" not in caplog.text
    assert not [r for r in caplog.records if r.exc_info]


def test_status_and_alarm_never_carry_raw_runtime_rows(runtime):
    state, events, _, poll = runtime
    _make("linkage_invalid", state)
    poll()
    (record,) = t3_linkage_alarm.status_directory().glob("*.json")
    assert SENTINEL not in record.read_text()
    assert SENTINEL not in events[0].human_message()
