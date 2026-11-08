"""Pure readiness checks: fake time/query and private text artifacts only."""

import os
from pathlib import Path

import pytest

from tests.calendar_watch_readiness import _fixture_diagnostics, wait_for_calendar_watch

TRUNCATED = "…[truncated after 16384 characters]"


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
        "'control.sock exists': True",
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


def fail_with_timeouts(path, clock):
    timeouts = []

    def query():
        assert clock.now < 3, "query started after readiness deadline"
        clock.now += 0.6
        timeouts.append(TimeoutError(f"status timeout {len(timeouts) + 1}"))
        raise timeouts[-1]

    with pytest.raises(AssertionError, match="not ready within 3s") as failure:
        wait(query, path, clock)
    assert failure.value.__cause__ is timeouts[-1]
    return str(failure.value)


def test_false_pid_at_2_99_sleeps_only_the_remainder_and_sends_no_later_query(tmp_path):
    clock, calls = Clock(), []

    def query():
        assert clock.now < 3, "query started after readiness deadline"
        calls.append(clock.now)
        clock.now = 2.99
        return status(None)

    with pytest.raises(AssertionError, match="fixture-state") as failure:
        wait(query, tmp_path, clock)
    assert calls == [0.0]
    assert clock.sleeps == [pytest.approx(0.01)]
    assert clock.now == 3.0
    assert failure.value.__cause__ is None


def test_absent_endpoint_is_bound_to_control_sock_key(tmp_path):
    assert "'control.sock exists': False" in fail_with_timeouts(tmp_path, Clock())


def test_endpoint_stat_error_cannot_mask_readiness_failure_or_cause(tmp_path, monkeypatch):
    real_exists = Path.exists

    def exists(self, *args, **kwargs):
        if self.name == "control.sock":
            raise PermissionError("injected endpoint stat failure")
        return real_exists(self, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", exists)
    message = fail_with_timeouts(tmp_path, Clock())
    assert "'control.sock exists': \"unknown: PermissionError('injected endpoint stat failure')\"" in message


def test_diagnostics_skip_symlinks_and_nonregular_artifacts_and_report_read_errors(tmp_path, monkeypatch):
    outside, fixture = tmp_path / "outside", tmp_path / "fixture"
    outside.mkdir()
    fixture.mkdir()
    (outside / "runs.jsonl").write_text("linked-dir-content")
    (outside / "target").write_text("linked-file-content")
    (fixture / "calendar").symlink_to(outside, target_is_directory=True)
    (fixture / "calendar.stderr").symlink_to(outside / "target")
    (fixture / "step").mkdir()
    os.mkfifo(fixture / "drain")  # Opening a FIFO would block; it must never be read.
    (fixture / "errors").write_text("never-read-content")
    real_open = Path.open

    def open_(self, *args, **kwargs):
        if self.name == "errors":
            raise PermissionError("injected read failure")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_)
    message = fail_with_timeouts(fixture, Clock())
    artifacts = _fixture_diagnostics(fixture)
    assert artifacts["calendar.stderr"] == artifacts["calendar/runs.jsonl"] == "symlink (not read)"
    for name in ("step", "drain", "watch"):
        assert artifacts[name] == "missing or not a regular file"
    assert artifacts["errors"] == "unreadable: PermissionError('injected read failure')"
    assert "linked-" not in message and "never-read" not in message


@pytest.mark.parametrize("tail, suffix", [("", ""), ("TAIL", TRUNCATED)])
def test_diagnostic_text_limit_counts_characters_and_labels_truncation(tmp_path, tail, suffix):
    (tmp_path / "errors").write_text("é" * 16384 + tail, encoding="utf-8")  # 2 bytes per char.
    assert _fixture_diagnostics(tmp_path)["errors"] == "é" * 16384 + suffix
    assert (TRUNCATED in fail_with_timeouts(tmp_path, Clock())) is bool(suffix)
