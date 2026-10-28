"""Deterministic host starvation; no sleeps or signals reach real processes."""

import signal
from types import SimpleNamespace

import pytest

from brainlayer import supervisor_guard as guard


@pytest.fixture
def simulate_guard(monkeypatch):
    def run(steps, *, diagnostics_fail=False, timeout=30, expected_reaped=None):
        clock, rows, kills, waits, reaped = [0.0], [], [], [], []
        steps = iter(steps)
        pending = [b""]

        def select(read, write, error, timeout):
            overshoot, data = next(steps)
            waits.append(timeout)
            clock[0] += timeout + overshoot
            data = data() if callable(data) else data
            pending[0] = data
            return ([0] if data is not None else []), [], []

        def emit(**row):
            rows.append(row)
            if diagnostics_fail:
                raise OSError("diagnostic sink unavailable")

        monkeypatch.setattr(guard.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(guard.select, "select", select)
        monkeypatch.setattr(guard.os, "getppid", lambda: 1234)
        monkeypatch.setattr(guard.os, "write", lambda *_: 1)
        monkeypatch.setattr(guard.os, "close", lambda *_: None)
        monkeypatch.setattr(guard.os, "read", lambda *_: pending[0])
        monkeypatch.setattr(guard.os, "kill", lambda *args: kills.append(args))
        monkeypatch.setattr(guard, "reap_groups", lambda groups, *args, **kwargs: reaped.append(groups.copy()))
        guard.run_guard(1234, timeout, 99, 0.1, parent_birth="birth", diagnostics=SimpleNamespace(emit=emit))
        assert list(steps) == [], "guard exited before the simulated incident ended"
        if expected_reaped is not None:
            assert reaped == [expected_reaped]
        return rows, kills, waits

    return run


@pytest.mark.parametrize("stall", [30, 60, 90])
def test_starved_guard_rearms_and_emits_stall(simulate_guard, stall):
    rows, kills, waits = simulate_guard([(stall, None), (0, b"")])
    assert not kills and waits == [30, 30]
    row = next(row for row in rows if row["event"] == "guard-stall-rearm")
    assert row["wait_overshoot_s"] == stall
    assert row["stall_rearms"] == 1 and row["grace_used_s"] == 30
    assert (row["parent_pid"], row["parent_start_time"]) == (1234, "birth")
    assert not any(row["event"] == "guard-timeout" for row in rows)


@pytest.mark.parametrize("overshoot", [0, 1, 5])
def test_responsive_guard_kills_silent_parent_at_normal_timeout(simulate_guard, overshoot):
    rows, kills, waits = simulate_guard([(overshoot, None)])
    assert kills == [(1234, signal.SIGKILL)] and waits == [30]
    assert rows[-1]["reason"] == "heartbeat-timeout"


def test_repeated_stalls_exhaust_three_rearm_cap(simulate_guard):
    rows, kills, waits = simulate_guard([(60, None)] * 4)
    assert kills == [(1234, signal.SIGKILL)] and waits == [30] * 4
    rearms = [row for row in rows if row["event"] == "guard-stall-rearm"]
    assert [row["stall_rearms"] for row in rearms] == [1, 2, 3]
    assert rows[-1]["reason"] == "heartbeat-timeout"


@pytest.mark.parametrize("timeout,grants", [(120, [120, 240, 300]), (900, [300])])
def test_cumulative_stall_grace_is_capped_at_300_seconds(simulate_guard, timeout, grants):
    rows, kills, waits = simulate_guard([(60, None)] * (len(grants) + 1), timeout=timeout)
    rearms = [row for row in rows if row["event"] == "guard-stall-rearm"]
    assert [row["grace_used_s"] for row in rearms] == grants
    assert waits == [timeout] + [grants[0]] + [b - a for a, b in zip(grants, grants[1:])]
    assert kills == [(1234, signal.SIGKILL)] and rows[-1]["reason"] == "heartbeat-timeout"


def test_real_heartbeat_resets_cumulative_stall_grace(simulate_guard):
    rows, kills, _ = simulate_guard([(60, None)] * 3 + [(0, b"H\n")] + [(60, None)] * 3 + [(0, b"")], timeout=120)
    assert not kills
    assert [row["grace_used_s"] for row in rows if row["event"] == "guard-stall-rearm"] == [120, 240, 300] * 2


def test_stall_coinciding_with_eof_reaps_without_rearm(simulate_guard):
    rows, kills, _ = simulate_guard([(0, b"+2468\n"), (60, b"")], expected_reaped={2468})
    assert not kills
    assert not any(row["event"] in ("guard-stall-rearm", "guard-timeout") for row in rows)


def test_real_heartbeat_after_stall_resets_cap(simulate_guard):
    rows, kills, _ = simulate_guard([(60, None)] * 3 + [(0, b"H\n")] + [(60, None)] * 3 + [(0, b"")])
    assert not kills
    assert [row["stall_rearms"] for row in rows if row["event"] == "guard-stall-rearm"] == [1, 2, 3] * 2


def test_registration_after_stall_does_not_reset_cap(simulate_guard):
    rows, kills, _ = simulate_guard([(60, b"+2468\n")] * 4 + [(0, None)])
    assert kills == [(1234, signal.SIGKILL)]
    assert len([row for row in rows if row["event"] == "guard-stall-rearm"]) == 3
    assert rows[-1]["registered_pgids"] == [2468]


def test_stall_grace_does_not_disable_silent_parent_kill(simulate_guard):
    rows, kills, waits = simulate_guard([(60, None), (0, None)])
    assert kills == [(1234, signal.SIGKILL)] and waits == [30, 30]
    assert rows[-1]["last_heartbeat_age_s"] == 120


def test_quiesced_guard_never_rearms_on_stall(simulate_guard):
    rows, kills, _ = simulate_guard([(0, b"Q\n"), (60, None)])
    assert kills == [(1234, signal.SIGKILL)]
    assert rows[-1]["reason"] == "shutdown-timeout"
    assert not any(row["event"] == "guard-stall-rearm" for row in rows)


def test_failed_stall_diagnostic_cannot_disarm_enforcement(simulate_guard):
    _, kills, _ = simulate_guard([(60, None)] * 4, diagnostics_fail=True)
    assert kills == [(1234, signal.SIGKILL)]


def test_operation_overrun_suppression_still_kills_with_responsive_guard(simulate_guard, tmp_path):
    from brainlayer.service_supervisor import Operation, Supervisor

    supervisor = Supervisor({}, tmp_path / "unused")
    operation = Operation("blocked-operation", -121, -1)
    supervisor.operations[object()] = operation
    rows, kills, _ = simulate_guard([(0, lambda: b"H\n" if supervisor._heartbeat_allowed() else None)])
    assert operation.reported
    assert kills == [(1234, signal.SIGKILL)] and rows[-1]["reason"] == "heartbeat-timeout"
