"""Unit coverage for `IndexWatchdog` on an injected clock -- no threads, no sleeping."""

import pytest

from brainlayer.index_watchdog import HEARTBEAT_PREFIX, IndexWatchdog, _heartbeat_interval_s_env
from brainlayer.vector_store import IndexDeadlineExceeded


class _Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _watchdog(clock, **kwargs):
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
    watchdog = IndexWatchdog(max_runtime_s=3600.0, emit=lambda alarm: True, poll_interval_s=0.01)
    with watchdog as armed:
        assert armed._thread is not None
        assert armed._thread.is_alive()
    assert watchdog._thread is None
    watchdog.stop()  # second stop must not raise
