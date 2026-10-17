"""Pure readiness checks: fake time/query and private text artifacts only."""

import pytest

from tests.calendar_watch_readiness import wait_for_calendar_watch


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def status(pid):
    return {"components": {"watch": {"pid": pid, "paused": pid is None, "error": "fixture-state"}}}


def wait(query, path, clock):
    return wait_for_calendar_watch(query, path, monotonic=clock.monotonic, sleep=clock.sleep)


def test_transient_status_timeout_retries_inside_original_deadline(tmp_path):
    clock, calls = Clock(), []

    def query():
        calls.append(clock.now)
        clock.now += 0.6
        if len(calls) == 1:
            raise TimeoutError("transient status read")
        return status(123)

    wait(query, tmp_path, clock)
    assert calls == [0.0, 0.62]
    assert clock.now < 3


def test_persistent_timeout_keeps_deadline_last_cause_and_private_diagnostics(tmp_path):
    clock, calls, timeouts = Clock(), [], []
    (tmp_path / "calendar").mkdir()
    for name, content in {
        "calendar.stderr": "calendar fixture stderr",
        "calendar/runs.jsonl": '{"event":"finished"}',
        "errors": "supervisor fixture stderr",
        "step": "456",
        "guard": "789",
    }.items():
        (tmp_path / name).write_text(content)
    (tmp_path / "control.sock").touch()  # Presence marker; no socket or process is created.

    def query():
        assert clock.now < 3, "query started after readiness deadline"
        calls.append(clock.now)
        clock.now += 0.6
        error = TimeoutError(f"status timeout {len(calls)}")
        timeouts.append(error)
        raise error

    with pytest.raises(AssertionError) as failure:
        wait(query, tmp_path, clock)
    assert len(calls) == 5  # Fixed 0.6 s requests and 0.02 s polling, no reset or final query.
    assert 3 <= clock.now < 3.6  # Last in-flight request may finish after the deadline.
    assert failure.value.__cause__ is timeouts[-1]
    message = str(failure.value)
    for text in (
        "3s",
        "status timeout 5",
        "calendar fixture stderr",
        '"event":"finished"',
        "supervisor fixture stderr",
        "control.sock",
        "True",
        "step",
        "456",
        "guard",
        "789",
        "last_components=None",
    ):
        assert text in message


@pytest.mark.parametrize("pid", [None, 0])
def test_missing_live_watch_pid_never_passes_or_queries_after_deadline(tmp_path, pid):
    clock, calls = Clock(), []

    def query():
        assert clock.now < 3, "query started after readiness deadline"
        calls.append(clock.now)
        clock.now += 0.6
        return status(pid)

    with pytest.raises(AssertionError, match="fixture-state") as failure:
        wait(query, tmp_path, clock)
    assert len(calls) == 5
    assert failure.value.__cause__ is None
    assert "last_components=" in str(failure.value)


@pytest.mark.parametrize("error", [ValueError("malformed response"), ConnectionRefusedError("no listener")])
def test_unexpected_exception_propagates(tmp_path, error):
    clock = Clock()

    def query():
        raise error

    with pytest.raises(type(error)) as failure:
        wait(query, tmp_path, clock)
    assert failure.value is error
    assert clock.now == 0 and not clock.sleeps


@pytest.mark.parametrize("finish", [3.0, 3.1])
def test_late_status_cannot_grant_postdeadline_success(tmp_path, finish):
    clock, calls = Clock(), []

    def query():
        calls.append(clock.now)
        clock.now = finish
        return status(123)

    with pytest.raises(AssertionError, match="3s"):
        wait(query, tmp_path, clock)
    assert calls == [0.0]


def test_timeout_after_valid_state_retains_state_without_diagnostic_query(tmp_path):
    clock, calls = Clock(), []
    last_timeout = TimeoutError("last response stalled")

    def query():
        assert clock.now < 3
        calls.append(clock.now)
        clock.now += 0.6
        if len(calls) == 1:
            return status(None)
        raise last_timeout

    with pytest.raises(AssertionError, match="fixture-state") as failure:
        wait(query, tmp_path, clock)
    assert failure.value.__cause__ is last_timeout
    assert len(calls) == 5
