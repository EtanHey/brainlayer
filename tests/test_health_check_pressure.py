import subprocess

import pytest

import brainlayer.health_check as health
from tests.test_stability_health_check import _clock_at, _make_db, _ok_canary, _ps_output_stub


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


# --- Heal gate: never restart a BrainBar that answers ping, nor one that does not while the Mac is
# under memory pressure. Four kickstarts on 21-22 Nov restarted BrainBar inside overload episodes,
# dropped every in-flight request, forced a cold ~2.7 GB helper reload, and cured nothing.

DAEMON = "com.brainlayer.brainbar-daemon"
PRESSURE_KEY = "kern.memorystatus_vm_pressure_level"


def _retrieval_error(*_):
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [{"type": "text", "text": "Error: search failed"}], "isError": True},
    }


def _closed_without_response(*_):
    raise RuntimeError("BrainBar socket closed without response")


def _timed_out(*_):
    raise TimeoutError("timed out")


def _gate_runner(commands, *, pressure=None, state="S"):
    """Fake launchctl/ps/sysctl: the daemon is pid 123 in ``state``; ``pressure`` is the sysctl answer."""

    def runner(args):
        commands.append(args)
        if args[-1] == PRESSURE_KEY:
            if isinstance(pressure, BaseException):
                raise pressure
            if isinstance(pressure, subprocess.CompletedProcess):
                return pressure
            return subprocess.CompletedProcess(args, 0, f"{pressure}\n", "")
        output = ""
        if args[:2] == ["launchctl", "print"]:
            output = "pid = 123"
        elif args[:2] == ["ps", "-o"]:
            output = state
        return subprocess.CompletedProcess(args, 0, output, "")

    return runner


def _heal_once(tmp_path, monkeypatch, *, canary, ping_ok, runner):
    db = tmp_path / "brain.db"
    _make_db(db, total=3, vector_rows=3)
    pings = []
    monkeypatch.setattr(health, "_brainbar_answers_ping", lambda *args: pings.append(args) or ping_ok)
    config = health.HealthCheckConfig(
        db_path=db, state_path=tmp_path / "state", heal=True, heal_min_consecutive_failures=1
    )
    result = health.run_health_check(
        config,
        ps_output_fn=_ps_output_stub(""),
        socket_request_fn=canary,
        command_runner=runner,
        now_fn=_clock_at(25),
    )
    assert "brain_search_canary_failed" in [issue.code for issue in result.issues]
    return result, pings


def _kickstarts(commands):
    return [args for args in commands if args[:3] == ["launchctl", "kickstart", "-k"]]


def _daemon_kickstarts(commands):
    return [args for args in _kickstarts(commands) if args[-1].endswith(f"/{DAEMON}")]


def _sysctl_reads(commands):
    return [args for args in commands if args[-1] == PRESSURE_KEY]


@pytest.mark.parametrize("canary", [_retrieval_error, _closed_without_response], ids=["retrieval", "closed"])
def test_canary_failure_with_live_ping_skips_the_daemon_restart(tmp_path, monkeypatch, canary):
    commands = []
    result, pings = _heal_once(
        tmp_path, monkeypatch, canary=canary, ping_ok=True, runner=_gate_runner(commands, pressure=1)
    )
    assert not _daemon_kickstarts(commands)
    assert f"kickstart:{DAEMON}" not in result.actions
    assert f"heal_skip:{DAEMON}:brain_search_canary_failed:ping_ok" in result.actions
    assert len(pings) == 1
    assert not any(args[0] == "kill" for args in commands)


@pytest.mark.parametrize("level", [2, 4])
def test_dead_ping_under_memory_pressure_backs_off_instead_of_restarting(tmp_path, monkeypatch, level):
    commands = []
    result, _ = _heal_once(
        tmp_path, monkeypatch, canary=_timed_out, ping_ok=False, runner=_gate_runner(commands, pressure=level)
    )
    assert not _daemon_kickstarts(commands)
    assert f"kickstart:{DAEMON}" not in result.actions
    assert f"heal_backoff:{DAEMON}:brain_search_canary_failed:pressure={level}" in result.actions
    assert len(_sysctl_reads(commands)) == 1


@pytest.mark.parametrize("canary", [_timed_out, _retrieval_error], ids=["timeout", "retrieval"])
def test_dead_ping_without_pressure_still_restarts_the_daemon(tmp_path, monkeypatch, canary):
    commands = []
    result, _ = _heal_once(
        tmp_path, monkeypatch, canary=canary, ping_ok=False, runner=_gate_runner(commands, pressure=1)
    )
    assert f"kickstart:{DAEMON}" in result.actions
    assert len(_daemon_kickstarts(commands)) == 1
    assert not [action for action in result.actions if action.startswith(("heal_skip:", "heal_backoff:"))]


@pytest.mark.parametrize(
    "pressure,reason",
    [
        (OSError("sysctl missing"), "OSError"),
        (subprocess.CompletedProcess(["sysctl"], 1, "", "unknown oid"), "rc=1"),
        (subprocess.CompletedProcess(["sysctl"], 0, "warn\n", ""), "unparsed"),
    ],
    ids=["raises", "nonzero", "garbage"],
)
def test_unreadable_pressure_fails_open_to_todays_restart_and_says_so(tmp_path, monkeypatch, pressure, reason):
    commands = []
    result, _ = _heal_once(
        tmp_path, monkeypatch, canary=_timed_out, ping_ok=False, runner=_gate_runner(commands, pressure=pressure)
    )
    assert f"kickstart:{DAEMON}" in result.actions
    assert len(_daemon_kickstarts(commands)) == 1
    assert f"heal_pressure_unavailable:{DAEMON}:brain_search_canary_failed:{reason}" in result.actions


def test_uninterruptible_daemon_keeps_its_state_backoff_under_pressure(tmp_path, monkeypatch):
    commands = []
    result, _ = _heal_once(
        tmp_path, monkeypatch, canary=_timed_out, ping_ok=False, runner=_gate_runner(commands, pressure=2, state="U")
    )
    assert f"heal_backoff:{DAEMON}:brain_search_canary_failed:pid=123:state=U" in result.actions
    assert not [action for action in result.actions if "pressure=" in action]
    assert not _daemon_kickstarts(commands)


def test_other_labels_still_restart_under_memory_pressure(tmp_path):
    db = tmp_path / "brain.db"
    _make_db(db, total=4, vector_rows=3)
    commands = []
    result = health.run_health_check(
        health.HealthCheckConfig(db_path=db, state_path=tmp_path / "state", heal=True, heal_min_consecutive_failures=1),
        ps_output_fn=_ps_output_stub(
            "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py --interval 1 --backlog-batch 0\n"
        ),
        socket_request_fn=_ok_canary,
        command_runner=_gate_runner(commands, pressure=4),
        now_fn=_clock_at(25),
    )
    assert "hotlane_backlog_disabled" in [issue.code for issue in result.issues]
    assert [args[-1].rsplit("/", 1)[-1] for args in _kickstarts(commands)] == ["com.brainlayer.hotlane-brainbar"]
    assert not _sysctl_reads(commands)
