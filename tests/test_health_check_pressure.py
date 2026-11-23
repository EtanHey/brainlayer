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
    monkeypatch.setattr(health, "_brainbar_answers_ping", lambda *_: live_socket, raising=False)

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
    assert health._brainbar_answers_ping(tmp_path / "socket", 10) is expected


@pytest.mark.parametrize("change", ["child_changed", "daemon_changed", "child_not_unique"])
def test_revalidation_changes_never_signal(change, monkeypatch):
    child = "456 123 /python -m brainlayer.brainbar_hybrid_helper --socket-path /tmp/brainbar-hybrid-123.sock"
    states = iter([(123, "S"), (999, "S") if change == "daemon_changed" else (123, "S")])
    monkeypatch.setattr(health, "_launchd_process_state", lambda *_: next(states))
    commands = []

    def runner(args):
        commands.append(args)
        output = child
        if change == "child_not_unique":
            output += "\n" + child.replace("456", "457")
        if change == "child_changed" and args[1] == "-p":
            output = child.replace("456 123", "456 999")
        return subprocess.CompletedProcess(args, 0, output, "")

    assert health._recycle_hybrid_helper("daemon", runner) == f"recycle_helper_skipped:{change}"
    assert all(args[0] == "ps" for args in commands)


def test_slow_alive_ping_uses_canary_budget_and_only_recycles_helper(tmp_path, monkeypatch):
    import json
    import socket
    import threading
    import time
    from pathlib import Path

    db = tmp_path / "brain.db"
    _make_db(db, total=3, vector_rows=3)
    monkeypatch.chdir(tmp_path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind("ping.sock")
    server.listen(1)
    server.settimeout(5)
    elapsed = []
    errors = []

    def slow_daemon():
        try:
            with server.accept()[0] as conn:
                conn.settimeout(5)
                request = json.loads(conn.recv(4096))
                assert request["method"] == "ping"
                start = time.monotonic()
                threading.Event().wait(2.2)
                elapsed.append(time.monotonic() - start)
                conn.sendall(b'{"jsonrpc":"2.0","id":2,"result":{}}\n')
        except Exception as exc:
            errors.append(exc)
        finally:
            server.close()

    commands = []
    child = "456 123 /python -m brainlayer.brainbar_hybrid_helper --socket-path /tmp/brainbar-hybrid-123.sock"

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
        raise TimeoutError("canary timed out")

    config = health.HealthCheckConfig(
        db_path=db,
        state_path=tmp_path / "state",
        socket_path=Path("ping.sock"),
        heal=True,
        heal_min_consecutive_failures=1,
    )
    # Observe the actual socket budget; do not replace the ping with a constant.
    budgets = []
    request_fn = health._brainbar_request

    def observe(path, request, budget):
        budgets.append(budget)
        return request_fn(path, request, budget)

    monkeypatch.setattr(health, "_brainbar_request", observe)
    worker = threading.Thread(target=slow_daemon)
    worker.start()
    try:
        result = health.run_health_check(
            config,
            ps_output_fn=_ps_output_stub(""),
            socket_request_fn=timeout,
            command_runner=runner,
            now_fn=_clock_at(25),
        )
    finally:
        worker.join(6)
    assert not worker.is_alive() and not errors
    assert 2 < elapsed[0] < config.socket_timeout_seconds
    assert budgets == [config.socket_timeout_seconds]
    assert "recycle_helper:456" in result.actions
    assert "kickstart:com.brainlayer.brainbar-daemon" not in result.actions
