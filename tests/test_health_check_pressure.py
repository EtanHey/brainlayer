import subprocess

import pytest

import brainlayer.health_check as health
from tests.test_stability_health_check import _clock_at, _make_db, _ps_output_stub


@pytest.mark.parametrize("live_socket", [True, False])
def test_canary_timeout_recycles_only_helper_when_daemon_answers(tmp_path, monkeypatch, live_socket):
    db = tmp_path / "brain.db"
    _make_db(db, total=3, vector_rows=3)
    commands = []
    child = "456 123 /python -m brainlayer.brainbar_hybrid_helper --socket-path /tmp/brainbar-hybrid-123.sock"
    monkeypatch.setattr(health, "_brainbar_answers_ping", lambda _: live_socket, raising=False)

    def runner(args):
        commands.append(args)
        output = ""
        if args[:2] == ["launchctl", "print"]:
            output = "pid = 123"
        elif args[:2] == ["ps", "-o"]:
            output = "S"
        elif args[0] == "ps":
            output = child
        return subprocess.CompletedProcess(args, 0, output, "")

    def timeout(*_):
        raise TimeoutError("socket read timed out")

    config = health.HealthCheckConfig(db_path=db, state_path=tmp_path / "state", heal=True)
    for minute in (25, 30):
        result = health.run_health_check(
            config,
            ps_output_fn=_ps_output_stub(""),
            socket_request_fn=timeout,
            command_runner=runner,
            now_fn=_clock_at(minute),
        )
    daemon_restarts = [a for a in result.actions if a == "kickstart:com.brainlayer.brainbar-daemon"]
    if live_socket:
        assert "recycle_helper:456" in result.actions
        assert ["kill", "-TERM", "456"] in commands
        assert not daemon_restarts
    else:
        assert daemon_restarts
        assert not any(a[0] == "kill" for a in commands)


@pytest.mark.parametrize(
    "rows",
    [
        "456 999 /python -m brainlayer.brainbar_hybrid_helper --socket-path /tmp/brainbar-hybrid-123.sock",
        "456 123 /python -m brainlayer.brainbar_hybrid_helper --socket-path /tmp/brainbar-hybrid-999.sock",
        "456 123 /python -m other --socket-path /tmp/brainbar-hybrid-123.sock",
        "",
    ],
)
def test_helper_identity_mismatch_never_signals_or_restarts(rows, monkeypatch):
    monkeypatch.setattr(health, "_launchd_process_state", lambda *_: (123, "S"))
    commands = []

    def runner(args):
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, rows, "")

    assert health._recycle_hybrid_helper("daemon", runner).startswith("recycle_helper_skipped:")
    assert all(args[0] == "ps" for args in commands)


@pytest.mark.parametrize(
    "response,expected",
    [
        ({"id": 2, "result": {}}, True),
        ({"id": 1, "result": {}}, False),
        ({"id": 2, "error": {}}, False),
        ([], False),
    ],
)
def test_ping_requires_matching_success_response(monkeypatch, tmp_path, response, expected):
    monkeypatch.setattr(health, "_brainbar_request", lambda *_: response)
    assert health._brainbar_answers_ping(tmp_path / "socket") is expected
