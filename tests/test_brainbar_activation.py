import json
import plistlib
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from brainlayer import brainbar_activation as activation
from brainlayer.cli import app


@pytest.fixture
def installed(monkeypatch, tmp_path):
    monkeypatch.setattr(activation.sys, "platform", "darwin")
    monkeypatch.delenv("BRAINLAYER_MCP_SOCKET", raising=False)
    monkeypatch.delenv("MCPLAYER_BRAINLAYER_FRONT", raising=False)
    agents = tmp_path / "LaunchAgents"
    agents.mkdir()
    app_path = tmp_path / "BrainBar.app" / "Contents" / "MacOS"
    app_path.mkdir(parents=True)
    for label, binary in activation.JOBS:
        executable = app_path / binary
        executable.touch(mode=0o700)
        (agents / f"{label}.plist").write_bytes(plistlib.dumps({"Label": label, "ProgramArguments": [str(executable)]}))
    calls = []

    def launchctl(args, **kwargs):
        calls.append(args[1:])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(activation.subprocess, "run", launchctl)
    monkeypatch.setattr(activation.Path, "is_socket", lambda _: True)
    monkeypatch.setattr(activation, "get_current_mcp_bridge_bin", lambda: "/fixture/bridge")
    verified = []
    monkeypatch.setattr(activation, "verify_mcp_transport", lambda **kw: verified.append(kw) or 5)
    return agents, calls, verified


def test_activate_replaces_both_jobs_once_then_verifies(installed):
    agents, calls, verified = installed
    report = activation.activate(agents, timeout_seconds=90)
    assert report["ok"] and report["tool_count"] == 5
    assert [call[0] for call in calls] == ["print", "bootout", "bootstrap"] * 2
    assert verified[0]["bridge_command"] == "/fixture/bridge"
    assert 0 < verified[0]["timeout_seconds"] <= 90


@pytest.mark.parametrize("failure", ["bootstrap", "verify", "socket"])
def test_failure_unloads_attempts_without_retry_or_config_rollback(installed, monkeypatch, failure):
    agents, calls, verified = installed
    if failure == "bootstrap":
        run = activation.subprocess.run

        def fail(args, **kwargs):
            result = run(args, **kwargs)
            return SimpleNamespace(returncode=7) if args[1] == "bootstrap" else result

        monkeypatch.setattr(activation.subprocess, "run", fail)
    elif failure == "verify":

        def fail_verify(**kwargs):
            raise RuntimeError("MCP initialize failed")

        monkeypatch.setattr(activation, "verify_mcp_transport", fail_verify)
    else:
        monkeypatch.setattr(activation.Path, "is_socket", lambda _: False)
        ticks = iter([0, 0, 0, 0, 0, 0, 0, 0, 100])
        monkeypatch.setattr(activation.time, "monotonic", lambda: next(ticks, 100))
        monkeypatch.setattr(activation.time, "sleep", lambda _: None)
    report = activation.activate(agents, timeout_seconds=90)
    assert not report["ok"] and report["reason"]
    assert not report["rollback_attempted"] and not report["service_ready"]
    assert sum(call[0] == "bootstrap" for call in calls) == (1 if failure == "bootstrap" else 2)
    assert calls[-1][0] == "bootout"
    if failure == "socket":
        assert not verified


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_deadline_does_not_launch(installed, timeout):
    agents, calls, _ = installed
    assert not activation.activate(agents, timeout_seconds=timeout)["ok"]
    assert not calls


def test_missing_plist_or_bridge_does_not_stop_old_jobs(installed, monkeypatch):
    agents, calls, _ = installed
    monkeypatch.setattr(activation, "get_current_mcp_bridge_bin", lambda: None)
    assert not activation.activate(agents)["ok"]
    assert not calls
    monkeypatch.setattr(activation, "get_current_mcp_bridge_bin", lambda: "/fixture/bridge")
    (agents / f"{activation.JOBS[1][0]}.plist").unlink()
    assert not activation.activate(agents)["ok"]
    assert not calls


def test_cli_emits_json_and_propagates_failure(monkeypatch):
    captured = []

    def fail(*args, **kwargs):
        captured.append(kwargs)
        return {"ok": False, "reason": "socket_timeout"}

    monkeypatch.setattr(activation, "activate", fail)
    result = CliRunner().invoke(app, ["brainbar", "activate", "--verify", "--timeout", "125"])
    assert result.exit_code == 1 and json.loads(result.stdout)["reason"] == "socket_timeout"
    assert captured == [{"timeout_seconds": 125.0}]


@pytest.mark.parametrize("payload", [[], {"Label": "wrong"}, {"ProgramArguments": 42}])
def test_malformed_plist_fails_before_any_launch(installed, payload):
    agents, calls, _ = installed
    (agents / f"{activation.JOBS[0][0]}.plist").write_bytes(plistlib.dumps(payload))
    assert activation.activate(agents)["stage"] == "preflight"
    assert not calls


def test_first_install_bootstraps_missing_labels(installed, monkeypatch):
    agents, calls, _ = installed
    run = activation.subprocess.run

    def unloaded(args, **kwargs):
        result = run(args, **kwargs)
        return SimpleNamespace(returncode=113) if args[1] == "print" else result

    monkeypatch.setattr(activation.subprocess, "run", unloaded)
    assert activation.activate(agents)["ok"]
    assert [call[0] for call in calls] == ["print", "bootstrap"] * 2


def test_failed_cleanup_is_reported_without_relaunch(installed, monkeypatch):
    agents, calls, _ = installed
    run = activation.subprocess.run

    def timeout(args, **kwargs):
        if args[1] == "bootstrap":
            run(args, **kwargs)
            raise activation.subprocess.TimeoutExpired(args, 10)
        result = run(args, **kwargs)
        return SimpleNamespace(returncode=1) if len(calls) > 3 else result

    monkeypatch.setattr(activation.subprocess, "run", timeout)
    report = activation.activate(agents)
    assert not report["ok"] and report["cleanup_errors"]
    assert [call[0] for call in calls] == ["print", "bootout", "bootstrap", "bootout"]


def test_socket_override_cannot_verify_a_different_daemon(installed, monkeypatch):
    agents, calls, _ = installed
    monkeypatch.setenv("BRAINLAYER_MCP_SOCKET", "/fixture/other.sock")
    assert not activation.activate(agents)["ok"]
    assert not calls
