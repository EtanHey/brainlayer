"""Unit coverage for `IndexWatchdog` on an injected clock -- no threads, no sleeping."""

from time import monotonic as _mono

import pytest

from brainlayer.index_watchdog import HEARTBEAT_PREFIX, IndexWatchdog, _heartbeat_interval_s_env
from brainlayer.vector_store import IndexDeadlineExceeded


class _Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _watchdog(clock, **kwargs):
    kwargs.setdefault("started_at", clock.now)
    kwargs.setdefault("max_runtime_s", 10.0)
    kwargs.setdefault("monotonic", clock)
    kwargs.setdefault("emit", lambda alarm: True)
    return IndexWatchdog(**kwargs)


def test_deadline_emits_one_alarm_naming_the_phase_and_pulls_interrupt():
    clock = _Clock()
    interrupts: list[int] = []
    alarms = []
    watchdog = _watchdog(
        clock,
        interrupt=lambda: interrupts.append(1),
        emit=lambda alarm: alarms.append(alarm) or True,
        counters=lambda: {"committed_chunks": 42},
    )
    watchdog.set_phase("embed_and_upsert:session.jsonl")

    clock.now += 9.0
    assert watchdog.expired is False

    clock.now += 2.0  # now past the 10s cap
    watchdog._on_deadline()

    assert watchdog.expired is True
    assert watchdog.alarm_emitted is True
    assert len(alarms) == 1
    assert alarms[0].code == "INDEX_RUNTIME_EXCEEDED"
    assert alarms[0].context["phase"] == "embed_and_upsert:session.jsonl"
    assert alarms[0].context["committed_chunks"] == 42
    assert alarms[0].context["stopped_by"] == "watchdog"
    assert alarms[0].context["elapsed_s"] == pytest.approx(11.0)
    assert interrupts == [1]

    # Repeated deadline ticks must not multiply the alarm.
    watchdog._on_deadline()
    watchdog._on_deadline()
    assert len(alarms) == 1


def test_interrupt_retries_are_bounded_so_the_rollback_survives():
    clock = _Clock()
    interrupts: list[int] = []
    watchdog = _watchdog(clock, interrupt=lambda: interrupts.append(1))

    clock.now += 20.0
    for _ in range(10):
        watchdog._on_deadline()

    assert len(interrupts) == 3, "interrupt retries must stay bounded"


def test_interrupt_attached_after_the_store_opens_is_still_used():
    clock = _Clock()
    interrupts: list[int] = []
    watchdog = _watchdog(clock)  # no connection yet: the store is not open

    watchdog.set_interrupt(lambda: interrupts.append(1))
    clock.now += 20.0
    watchdog._on_deadline()

    assert interrupts == [1]


def test_deadline_without_a_connection_still_alarms():
    clock = _Clock()
    alarms = []
    watchdog = _watchdog(clock, emit=lambda alarm: alarms.append(alarm) or True, phase="open_store")

    clock.now += 20.0
    watchdog._on_deadline()

    assert watchdog.expired is True
    assert len(alarms) == 1
    assert alarms[0].context["phase"] == "open_store"


def test_a_failing_counter_never_suppresses_the_alarm():
    clock = _Clock()
    alarms = []

    def boom():
        raise RuntimeError("counter blew up")

    watchdog = _watchdog(clock, emit=lambda alarm: alarms.append(alarm) or True, counters=boom)
    clock.now += 20.0
    watchdog._on_deadline()

    assert len(alarms) == 1
    assert "committed_chunks" not in alarms[0].context


def test_raise_if_expired_is_a_noop_before_the_cap_and_raises_after():
    clock = _Clock()
    watchdog = _watchdog(clock)

    watchdog.raise_if_expired()  # no cap reached yet

    clock.now += 20.0
    watchdog._on_deadline()
    with pytest.raises(IndexDeadlineExceeded) as excinfo:
        watchdog.raise_if_expired(processed_count=7)
    assert excinfo.value.processed_count == 7


def test_heartbeat_carries_the_phase_and_the_age_of_the_last_progress():
    clock = _Clock()
    lines: list[str] = []
    watchdog = _watchdog(clock, heartbeat_sink=lines.append)

    watchdog.set_phase("embed_and_upsert:session.jsonl")
    clock.now += 4.0
    watchdog.note_progress()
    clock.now += 6.0
    watchdog._heartbeat()

    assert len(lines) == 1
    line = lines[0]
    assert line.startswith(HEARTBEAT_PREFIX)
    assert "phase=embed_and_upsert:session.jsonl" in line
    assert "elapsed_s=10.0" in line
    assert "last_progress_age_s=6.0" in line
    assert "max_runtime_s=10.0" in line


def test_set_phase_resets_the_progress_age():
    clock = _Clock()
    watchdog = _watchdog(clock)

    clock.now += 30.0
    watchdog.set_phase("parse:next.jsonl")
    assert watchdog.context()["last_progress_age_s"] == pytest.approx(0.0)
    assert watchdog.phase == "parse:next.jsonl"


def test_a_failing_heartbeat_sink_never_kills_the_thread():
    clock = _Clock()

    def boom(_line):
        raise RuntimeError("sink blew up")

    watchdog = _watchdog(clock, heartbeat_sink=boom)
    watchdog._heartbeat()  # must not raise


@pytest.mark.parametrize("raw", ["0", "-5", "abc", "nan", "inf"])
def test_bad_heartbeat_interval_falls_back_to_the_default(monkeypatch, raw):
    monkeypatch.setenv("BRAINLAYER_INDEX_HEARTBEAT_S", raw)
    assert _heartbeat_interval_s_env() == 300.0


def test_heartbeat_interval_env_override(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_INDEX_HEARTBEAT_S", "45")
    assert _heartbeat_interval_s_env() == 45.0


def test_stop_joins_the_thread_and_is_idempotent():
    watchdog = IndexWatchdog(started_at=_mono(), max_runtime_s=3600.0, emit=lambda alarm: True, poll_interval_s=0.01)
    with watchdog as armed:
        assert armed._thread is not None
        assert armed._thread.is_alive()
    assert watchdog._thread is None
    watchdog.stop()  # second stop must not raise


def test_heartbeats_continue_after_the_deadline():
    """A wedge that cannot be aborted must keep reporting, not alarm once and go quiet.

    Going quiet after the alarm is the second half of the original failure: 12h of silent
    log. A model load or an open_store with no connection yet cannot be interrupted, so the
    heartbeat is the only signal left.
    """
    clock = _Clock()
    lines: list[str] = []
    alarms = []
    watchdog = _watchdog(
        clock,
        max_runtime_s=0.0,  # already expired
        emit=lambda alarm: alarms.append(alarm) or True,
        heartbeat_interval_s=0.02,
        poll_interval_s=0.01,
        heartbeat_sink=lines.append,
    )
    watchdog.set_phase("open_store")

    with watchdog:
        deadline = _mono() + 5.0
        while len(lines) < 3 and _mono() < deadline:
            clock.now += 0.05  # the watchdog reads the caller's clock

    assert watchdog.expired is True
    assert len(alarms) == 1, "the alarm must not repeat"
    assert len(lines) >= 3, f"heartbeats stopped after the deadline: {lines}"
    assert all("phase=open_store" in line for line in lines)


def test_the_cap_alarm_never_repeats_when_emit_reports_no_telemetry():
    """`emit_alarm` returns the Axiom result, not whether a record was written.

    Found live, not by a mock: a 6s-cap run against the real DB printed the same
    INDEX_RUNTIME_EXCEEDED 13 times, because a falsy return was read as "did not land" and
    every poll re-alarmed. stderr and logging are emit_alarm's guaranteed paths.
    """
    clock = _Clock()
    alarms = []
    watchdog = _watchdog(clock, emit=lambda alarm: alarms.append(alarm) and False)

    clock.now += 20.0
    for _ in range(8):
        watchdog._on_deadline()

    assert len(alarms) == 1, f"the cap alarm repeated {len(alarms)} times"
    assert watchdog.alarm_emitted is True, "a falsy telemetry result must still count as a record"


def test_a_raising_emit_leaves_no_record_but_still_does_not_repeat():
    clock = _Clock()
    calls = {"n": 0}

    def boom(_alarm):
        calls["n"] += 1
        raise RuntimeError("sink down")

    watchdog = _watchdog(clock, emit=boom)
    clock.now += 20.0
    for _ in range(8):
        watchdog._on_deadline()

    assert calls["n"] == 1, "a failing emit must not be retried every poll"
    assert watchdog.alarm_emitted is False, "the CLI still owes a fallback alarm"
    assert watchdog.expired is True
