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
import time
from pathlib import Path
from time import monotonic as _mono
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

# Imported at module scope BEFORE any patching, on purpose. `index_new` binds
# `open_writer_store` at ITS import time; if that import first happens inside a test that has
# already monkeypatched `brainlayer.runtime_store.open_writer_store`, index_new captures the
# fake permanently -- monkeypatch restores runtime_store, not index_new -- and every later
# test in the session gets the fake. That is what broke test_context_pipeline.
from brainlayer import index_new, index_watchdog
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

    def wait_for_interrupt(self) -> bool:
        """Block until the watchdog interrupts, WITHOUT raising -- for callers that want to
        raise a failure of their own once the cap has already passed."""
        return self._interrupted.wait(self.ceiling_s)

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


def _patch_open_writer_store(monkeypatch, fake) -> None:
    """Patch every binding of `open_writer_store`, and only via monkeypatch so all are undone."""
    monkeypatch.setattr("brainlayer.runtime_store.open_writer_store", fake)
    monkeypatch.setattr(index_new, "open_writer_store", fake)


def _cap_alarm(alarms: list[BrainLayerAlarm]) -> BrainLayerAlarm:
    """The single INDEX_RUNTIME_EXCEEDED alarm, or a readable failure instead of StopIteration."""
    matches = [alarm for alarm in alarms if alarm.code == "INDEX_RUNTIME_EXCEEDED"]
    assert len(matches) == 1, f"expected exactly one cap alarm, got {[a.code for a in alarms]}"
    return matches[0]


def _install(monkeypatch, store, alarms):
    monkeypatch.setenv("BRAINLAYER_INDEX_MAX_RUNTIME_S", str(CAP_S))

    def fake_open(_path, on_connection=None):
        # Faithful to the real path: the writer hands its connection over before the probe.
        if on_connection is not None:
            on_connection(store.conn)
        return store

    _patch_open_writer_store(monkeypatch, fake_open)
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
    alarm = _cap_alarm(alarms)
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


class _StallingOpenStore(_StallingRuntimeStore):
    """A writer store whose *open* stalls, after handing its connection over.

    This is the phase the M1 evidence points at -- schema validation inside
    `open_writer_store`, which has no cap of its own. `WriterRuntimeStore._init_runtime_db`
    creates the connection before it runs any probe SQL, so the `on_connection` hook can hand
    the interrupt lever over while the probe is still running.
    """

    def open(self, on_connection):
        if on_connection is not None:
            on_connection(self.conn)
        self.stall()
        return self


def test_a_stalling_open_store_is_alarmed_and_interrupted(tmp_path, monkeypatch):
    source = _prepare_index_source(tmp_path, monkeypatch)
    store = _StallingOpenStore()
    alarms: list[BrainLayerAlarm] = []
    monkeypatch.setenv("BRAINLAYER_INDEX_MAX_RUNTIME_S", str(CAP_S))
    monkeypatch.setattr("brainlayer.alarm.emit_alarm", lambda alarm: alarms.append(alarm) or True)
    _patch_open_writer_store(monkeypatch, lambda _path, on_connection=None: store.open(on_connection))
    monkeypatch.setattr("brainlayer.index_new.index_chunks_to_sqlite", lambda *_a, **_k: 0)

    started = _mono()
    result = CliRunner().invoke(app, ["index", str(source)])
    wall_s = _mono() - started

    assert store.interrupt_calls >= 1, "open_store was never interruptible"
    assert store.stalled_s is not None and store.stalled_s < store.ceiling_s
    assert wall_s < CAP_S + 10.0, f"open_store ran {wall_s:.1f}s against a {CAP_S}s cap"
    assert result.exit_code != 0
    alarm = _cap_alarm(alarms)
    assert alarm.context["phase"] == "open_store", alarm.context


def test_a_slow_discovery_shares_the_one_deadline(tmp_path, monkeypatch):
    """The watchdog must adopt the CLI's deadline, not start a fresh budget when it arms."""
    source = _prepare_index_source(tmp_path, monkeypatch)
    monkeypatch.setenv("BRAINLAYER_INDEX_MAX_RUNTIME_S", "7")

    clock = {"now": 1000.0}
    monkeypatch.setattr("brainlayer.cli.time.monotonic", lambda: clock["now"])

    real_rglob = Path.rglob

    def slow_rglob(self, pattern):
        clock["now"] += 5.0  # discovery burns 5 of the 7s budget
        return real_rglob(self, pattern)

    monkeypatch.setattr(Path, "rglob", slow_rglob)

    built: list[object] = []
    real_watchdog = index_watchdog.IndexWatchdog

    def capture(**kwargs):
        instance = real_watchdog(**kwargs)
        built.append(instance)
        return instance

    monkeypatch.setattr("brainlayer.index_watchdog.IndexWatchdog", capture)
    _patch_open_writer_store(monkeypatch, lambda _path, on_connection=None: _StallingRuntimeStore(ceiling_s=0.01))
    monkeypatch.setattr("brainlayer.index_new.index_chunks_to_sqlite", lambda *_a, **_k: 0)
    monkeypatch.setattr("brainlayer.alarm.emit_alarm", lambda _alarm: True)

    CliRunner().invoke(app, ["index", str(source)])

    assert len(built) == 1
    # start 1000.0 + cap 7.0 -- NOT the instant at which the watchdog armed.
    assert built[0].deadline_monotonic == 1007.0


def test_a_discovery_that_outruns_the_cap_is_stopped_and_named(tmp_path, monkeypatch):
    """A walk that burns the whole budget must stop the run, naming `discover`."""
    source = _prepare_index_source(tmp_path, monkeypatch)
    monkeypatch.setenv("BRAINLAYER_INDEX_MAX_RUNTIME_S", "7")
    alarms: list[BrainLayerAlarm] = []

    clock = {"now": 1000.0}
    monkeypatch.setattr("brainlayer.cli.time.monotonic", lambda: clock["now"])

    real_rglob = Path.rglob

    def stalled_rglob(self, pattern):
        clock["now"] += 9.0  # past the 7s cap
        return real_rglob(self, pattern)

    monkeypatch.setattr(Path, "rglob", stalled_rglob)
    _patch_open_writer_store(monkeypatch, lambda _path, on_connection=None: _StallingRuntimeStore(ceiling_s=0.01))
    monkeypatch.setattr("brainlayer.index_new.index_chunks_to_sqlite", lambda *_a, **_k: 0)
    monkeypatch.setattr("brainlayer.alarm.emit_alarm", lambda alarm: alarms.append(alarm) or True)

    result = CliRunner().invoke(app, ["index", str(source)])

    assert result.exit_code != 0
    alarm = _cap_alarm(alarms)
    assert alarm.context["phase"] == "discover", alarm.context


def test_a_stuck_discovery_alarms_and_heartbeats_while_still_walking(tmp_path, monkeypatch):
    """`rglob` is one opaque call -- a stuck walk must not be silent until it returns.

    Real time here, not a fake clock: the point is that the watchdog THREAD reports while the
    main thread is still blocked inside the directory walk.
    """
    source = _prepare_index_source(tmp_path, monkeypatch)
    monkeypatch.setenv("BRAINLAYER_INDEX_MAX_RUNTIME_S", str(CAP_S))
    monkeypatch.setenv("BRAINLAYER_INDEX_HEARTBEAT_S", "0.3")
    alarms: list[BrainLayerAlarm] = []
    monkeypatch.setattr("brainlayer.alarm.emit_alarm", lambda alarm: alarms.append(alarm) or True)

    real_rglob = Path.rglob
    seen: dict[str, object] = {}

    def stuck_rglob(self, pattern):
        found = list(real_rglob(self, pattern))

        def walk():
            # Blocks well past the cap, exactly like a walk on a wedged mount.
            time.sleep(CAP_S + 2.0)
            seen["alarms_during_walk"] = len(alarms)
            yield from found

        return walk()

    monkeypatch.setattr(Path, "rglob", stuck_rglob)
    _patch_open_writer_store(monkeypatch, lambda _path, on_connection=None: _StallingRuntimeStore(ceiling_s=0.01))
    monkeypatch.setattr("brainlayer.index_new.index_chunks_to_sqlite", lambda *_a, **_k: 0)

    result = CliRunner().invoke(app, ["index", str(source)])

    assert seen.get("alarms_during_walk"), "the run was silent while discovery was stuck"
    alarm = _cap_alarm(alarms)
    assert alarm.context["phase"] == "discover", alarm.context
    heartbeats = [ln for ln in result.stderr.splitlines() if ln.startswith("BRAINLAYER_INDEX_HEARTBEAT")]
    assert any("phase=discover" in ln for ln in heartbeats), f"no discover heartbeat: {heartbeats}"
    assert result.exit_code != 0


def test_an_unrelated_failure_after_the_cap_keeps_its_own_identity(tmp_path, monkeypatch):
    """A real failure landing after the deadline must not be relabelled as the cap.

    Cap exit stays fail-closed; diagnosis must not fail open -- the operator still needs the
    schema mismatch / I/O error / OOM that actually broke the run.
    """
    source = _prepare_index_source(tmp_path, monkeypatch)
    store = _StallingRuntimeStore()
    alarms: list[BrainLayerAlarm] = []
    _install(monkeypatch, store, alarms)

    def fail_after_the_cap(_chunks, **_kwargs):
        assert store.wait_for_interrupt(), "the watchdog never expired, so this proves nothing"
        raise RuntimeError("runtime database schema fingerprint mismatch")

    monkeypatch.setattr("brainlayer.index_new.index_chunks_to_sqlite", fail_after_the_cap)

    result = CliRunner().invoke(app, ["index", str(source)])

    assert result.exit_code != 0
    combined = result.stdout + result.stderr
    assert "runtime database schema fingerprint mismatch" in combined, (
        f"the original failure was rewritten as the cap: {combined!r}"
    )
    # The cap alarm still fires (it fired at the deadline, before this failure existed) AND
    # the original error keeps its type and message in what the operator actually sees.
    assert any(a.code == "INDEX_RUNTIME_EXCEEDED" for a in alarms), "the cap went unreported"
    assert "RuntimeError" in combined, combined
    assert "unwind_error=RuntimeError" in combined, combined


def test_a_failed_watchdog_alarm_still_leaves_a_record(tmp_path, monkeypatch):
    """`alarm_emitted` must mean the alarm landed, or the run exits 1 with no record."""
    source = _prepare_index_source(tmp_path, monkeypatch)
    store = _StallingRuntimeStore()
    recorded: list[BrainLayerAlarm] = []
    calls = {"n": 0}

    def flaky_emit(alarm):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("telemetry sink down")
        recorded.append(alarm)
        return True

    monkeypatch.setenv("BRAINLAYER_INDEX_MAX_RUNTIME_S", str(CAP_S))
    monkeypatch.setattr("brainlayer.alarm.emit_alarm", flaky_emit)
    _patch_open_writer_store(
        monkeypatch, lambda _path, on_connection=None: (on_connection and on_connection(store.conn), store)[1]
    )
    monkeypatch.setattr("brainlayer.index_new.index_chunks_to_sqlite", lambda *_a, **_k: store.stall())

    result = CliRunner().invoke(app, ["index", str(source)])

    assert result.exit_code != 0
    assert calls["n"] >= 2, "the CLI never retried after the watchdog's alarm failed"
    assert [a.code for a in recorded] == ["INDEX_RUNTIME_EXCEEDED"]
