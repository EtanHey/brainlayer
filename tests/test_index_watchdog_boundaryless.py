"""The `brainlayer index` runtime cap must hold in a phase that reaches no boundary.

Observed on the M1 2026-09-05: `com.brainlayer.index` ran 12h13m against a 4h cap with no
alarm, ~100% CPU, WAL unchanged, no JSONL open, stdout silent for 12h.

Measured against real apsw (3.51.2.0), the plan's hypothesis is wrong: the progress handler
in `upsert_chunks` is not blind -- it ticks 7,287 times during a 0.68s FTS5 rebuild and ~48
times a second during a vec0 KNN, and its `id=` registration works. The real hole is that
the cap is enforced ONLY inside `upsert_chunks` and at the per-file boundaries in
`index_fast`; opening the writer store and loading the embedding model have no cap at all.

So the cap has to be owned beside the work, by a thread a blocked main thread cannot delay.
These tests pin the three things that thread must guarantee: it interrupts, it alarms with
the phase named, and it alarms *before* the stuck phase unwinds.

Fakes only -- no embedding model, no canonical DB.
"""

import threading
from time import monotonic as _mono
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from brainlayer.alarm import BrainLayerAlarm
from brainlayer.cli import app

# Real wall-clock budgets. The cap is what we assert on; the ceiling only keeps a
# regressed build from hanging the suite instead of failing it.
CAP_S = 1.0
CEILING_S = 20.0


def _prepare_index_source(tmp_path, monkeypatch):
    """One source file that yields exactly one chunk, with the real clock left alone."""
    source = tmp_path / "projects"
    project = source / "watchdog-boundaryless"
    project.mkdir(parents=True)
    (project / "session.jsonl").write_text("{}\n")

    monkeypatch.setattr("brainlayer.pipeline.extract.parse_jsonl", lambda _path: [{}])
    monkeypatch.setattr("brainlayer.pipeline.classify.classify_content", lambda entry: entry)
    monkeypatch.setattr("brainlayer.pipeline.chunk.chunk_content", lambda _entry: [object()])
    return source


class _StallingRuntimeStore:
    """A writer store whose write phase only unwinds when the connection is interrupted.

    This is the defect's shape, not a convenience: the stall deliberately ignores every
    opcode-counting hook and responds solely to `conn.interrupt()`.
    """

    def __init__(self, ceiling_s: float = CEILING_S) -> None:
        self.conn = SimpleNamespace(interrupt=self._interrupt)
        self.ceiling_s = ceiling_s
        self.interrupt_calls = 0
        self.stalled_s: float | None = None
        self.closed = False
        self._interrupted = threading.Event()

    def _interrupt(self) -> None:
        self.interrupt_calls += 1
        self._interrupted.set()

    def stall(self) -> int:
        started = _mono()
        interrupted = self._interrupted.wait(self.ceiling_s)
        self.stalled_s = _mono() - started
        if interrupted:
            # apsw surfaces sqlite3_interrupt as InterruptError out of sqlite3_step.
            raise RuntimeError("apsw.InterruptError: interrupted")
        return 5

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True
        return False


def _install(monkeypatch, store, alarms):
    monkeypatch.setenv("BRAINLAYER_INDEX_MAX_RUNTIME_S", str(CAP_S))
    monkeypatch.setattr("brainlayer.runtime_store.open_writer_store", lambda _path: store)
    monkeypatch.setattr("brainlayer.alarm.emit_alarm", lambda alarm: alarms.append(alarm) or True)

    def fake_index(_chunks, **_kwargs):
        return store.stall()

    monkeypatch.setattr("brainlayer.index_new.index_chunks_to_sqlite", fake_index)


def test_boundaryless_write_phase_is_interrupted_at_the_cap(tmp_path, monkeypatch):
    source = _prepare_index_source(tmp_path, monkeypatch)
    store = _StallingRuntimeStore()
    alarms: list[BrainLayerAlarm] = []
    _install(monkeypatch, store, alarms)

    started = _mono()
    result = CliRunner().invoke(app, ["index", str(source)])
    wall_s = _mono() - started

    # The lever: another thread pulled sqlite3_interrupt. Today this is 0.
    assert store.interrupt_calls >= 1, "the cap never interrupted the boundary-less phase"
    assert store.stalled_s is not None
    assert store.stalled_s < store.ceiling_s, "the phase ran to its own ceiling, not to the cap"
    # Stopped near the cap, not hours past it.
    assert wall_s < CAP_S + 10.0, f"index ran {wall_s:.1f}s against a {CAP_S}s cap"
    assert result.exit_code != 0
    assert store.closed is True


def test_boundaryless_stall_alarms_with_its_phase_name(tmp_path, monkeypatch):
    source = _prepare_index_source(tmp_path, monkeypatch)
    store = _StallingRuntimeStore()
    alarms: list[BrainLayerAlarm] = []
    _install(monkeypatch, store, alarms)

    result = CliRunner().invoke(app, ["index", str(source)])

    assert result.exit_code != 0
    codes = [alarm.code for alarm in alarms]
    assert codes.count("INDEX_RUNTIME_EXCEEDED") == 1, f"expected exactly one cap alarm, got {codes}"
    alarm = next(a for a in alarms if a.code == "INDEX_RUNTIME_EXCEEDED")
    # A 12h silent log is only impossible if the alarm says WHICH phase was silent.
    assert alarm.context.get("phase"), f"cap alarm carries no phase: {alarm.context}"
    assert alarm.context["max_runtime_s"] == pytest.approx(CAP_S)
    assert alarm.context["elapsed_s"] >= CAP_S


def test_cap_alarm_is_emitted_while_the_phase_is_still_stuck(tmp_path, monkeypatch):
    """ "Not silent past the cap" means the alarm lands before the stall unwinds."""
    source = _prepare_index_source(tmp_path, monkeypatch)
    store = _StallingRuntimeStore()
    alarms: list[BrainLayerAlarm] = []
    alarm_seen = threading.Event()
    _install(monkeypatch, store, alarms)

    def emit(alarm):
        alarms.append(alarm)
        if alarm.code == "INDEX_RUNTIME_EXCEEDED":
            alarm_seen.set()
        return True

    monkeypatch.setattr("brainlayer.alarm.emit_alarm", emit)

    def fake_index(_chunks, **_kwargs):
        value = store.stall()
        # By the time the stall unwinds, the alarm must already have been emitted.
        assert alarm_seen.is_set(), "the cap alarm waited for the stuck phase to finish"
        return value

    monkeypatch.setattr("brainlayer.index_new.index_chunks_to_sqlite", fake_index)

    result = CliRunner().invoke(app, ["index", str(source)])

    assert result.exit_code != 0
    assert alarm_seen.is_set()


def test_a_long_phase_heartbeats_instead_of_going_silent(tmp_path, monkeypatch):
    """The 12h-silent stdout log is the other half of the defect: report while stuck."""
    source = _prepare_index_source(tmp_path, monkeypatch)
    store = _StallingRuntimeStore(ceiling_s=2.0)
    alarms: list[BrainLayerAlarm] = []
    _install(monkeypatch, store, alarms)
    # Cap well past the stall so the run ends by completing, not by the cap: the heartbeat
    # has to work on its own, not as a side effect of the alarm.
    monkeypatch.setenv("BRAINLAYER_INDEX_MAX_RUNTIME_S", "600")
    monkeypatch.setenv("BRAINLAYER_INDEX_HEARTBEAT_S", "0.25")

    result = CliRunner().invoke(app, ["index", str(source)])

    assert result.exit_code == 0, result.output
    assert store.interrupt_calls == 0, "a run inside its cap must never be interrupted"
    heartbeats = [line for line in result.stderr.splitlines() if line.startswith("BRAINLAYER_INDEX_HEARTBEAT")]
    assert heartbeats, f"a {store.ceiling_s}s phase produced no heartbeat: {result.stderr!r}"
    assert "phase=embed_and_upsert:session.jsonl" in heartbeats[-1]
    assert "last_progress_age_s=" in heartbeats[-1]
    assert alarms == []
