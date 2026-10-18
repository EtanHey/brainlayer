import json
import plistlib
from types import SimpleNamespace

import pytest

from brainlayer import brainbar_auto_activation as auto
from brainlayer.brainbar_identity import JOBS


@pytest.fixture
def monitor(tmp_path, monkeypatch):
    agents = tmp_path / "agents"
    agents.mkdir()
    app = tmp_path / "BrainBar.app"
    (app / "Contents").mkdir(parents=True)
    info = app / "Contents/Info.plist"
    info.write_bytes(plistlib.dumps({"CFBundleShortVersionString": "1.5.99", "GitCommit": "abc"}))
    for label, binary in JOBS:
        (agents / f"{label}.plist").write_bytes(
            plistlib.dumps({"ProgramArguments": [str(app / "Contents/MacOS" / binary)]})
        )
    calls, alerts = [], []
    monkeypatch.setattr(auto, "brew_active", lambda: False)
    monkeypatch.setattr(auto, "loaded", lambda *a: SimpleNamespace(returncode=113))
    monkeypatch.setattr(auto, "report", lambda *args: alerts.append(args))

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=1, stdout='{"ok":false,"reason":"blocked"}', stderr="")

    monkeypatch.setattr(auto.subprocess, "run", run)
    return auto.AutoActivator(agents, tmp_path / "state.json"), info, calls, alerts


def test_failure_is_durable_once_per_identity_and_alerted(monitor):
    worker, info, calls, alerts = monitor
    worker.tick()
    auto.AutoActivator(worker.agents, worker.state).tick()
    assert len(calls) == 1 and calls[0][1]["timeout"] <= 110
    assert alerts[-1][0] == "brainbar-activation" and "blocked" in alerts[-1][1]
    assert json.loads(worker.state.read_text())["identity"]["git_commit"] == "abc"
    info.write_bytes(plistlib.dumps({"CFBundleShortVersionString": "1.5.99", "GitCommit": "def"}))
    worker.tick()
    assert len(calls) == 2


@pytest.mark.parametrize("blocked", ["brew", "loaded", "missing"])
def test_ineligible_state_never_attempts_or_consumes_identity(monitor, monkeypatch, blocked):
    worker, _, calls, _ = monitor
    if blocked == "brew":
        monkeypatch.setattr(auto, "brew_active", lambda: True)
    elif blocked == "loaded":
        monkeypatch.setattr(auto, "loaded", lambda *a: SimpleNamespace(returncode=0))
    else:
        (worker.agents / f"{JOBS[1][0]}.plist").unlink()
    worker.tick()
    assert not calls and not worker.state.exists()


def test_brew_started_during_preflight_defers_without_consuming_attempt(monitor, monkeypatch):
    worker, _, calls, _ = monitor
    states = iter([False, True])
    monkeypatch.setattr(auto, "brew_active", lambda: next(states))
    worker.tick()
    assert not calls and not worker.state.exists()


def test_success_clears_visible_alert(monitor, monkeypatch):
    worker, _, calls, alerts = monitor
    monkeypatch.setattr(
        auto.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout='{"ok":true}', stderr="")
    )
    worker.tick()
    assert alerts[-1] == ("brainbar-activation", None)


def test_process_scan_catches_brew_ruby_and_orphan_child(monkeypatch):
    for output in [
        "42 1 ruby /opt/homebrew/Library/Homebrew/brew.rb upgrade\n43 42 curl curl download",
        "43 1 ruby /opt/homebrew/Library/Homebrew/vendor/portable-ruby/bin/ruby",
    ]:
        monkeypatch.setattr(auto, "process_table", lambda: output)
        assert auto.brew_active()
    monkeypatch.setattr(auto, "process_table", lambda: "3 1 python python monitor.py")
    assert not auto.brew_active()


def test_supervisor_formula_wires_bounded_monitor(monkeypatch, tmp_path):
    from brainlayer import supervisor_jobs

    captured = {}
    manifest = tmp_path / "manifest"
    manifest.write_text(
        json.dumps(
            {
                "com.brainlayer.watch": {
                    "enabled": False,
                    "plist": {"Label": "com.brainlayer.watch", "ProgramArguments": ["unused"], "KeepAlive": True},
                }
            }
        )
    )
    manifest.chmod(0o600)
    monkeypatch.setattr(supervisor_jobs.sys, "platform", "darwin")

    class FakeSupervisor:
        def __init__(self, components, path):
            captured.update(components)

        async def run(self, **kwargs):
            pass

    monkeypatch.setattr(supervisor_jobs, "Supervisor", FakeSupervisor)
    supervisor_jobs.run_daemon(tmp_path / "manifest", tmp_path / "control.sock")
    child = captured["brainbar-activation"]
    assert child.interval <= 10 and child.run_timeout == 180
    assert "brainlayer.brainbar_auto_activation" in child.argv


def test_brew_deferral_is_not_a_failed_build_attempt(monitor, monkeypatch):
    worker, _, calls, alerts = monitor

    def deferred(*a, **k):
        return SimpleNamespace(returncode=1, stdout='{"ok":false,"stage":"deferred"}', stderr="")

    monkeypatch.setattr(auto.subprocess, "run", deferred)
    worker.tick()
    assert not worker.state.exists() and not alerts


def test_activation_timeout_is_visible_and_never_retried(monitor, monkeypatch):
    worker, _, calls, alerts = monitor

    def timeout(*args, **kwargs):
        calls.append(args)
        raise auto.subprocess.TimeoutExpired(args, 110)

    monkeypatch.setattr(auto.subprocess, "run", timeout)
    worker.tick()
    worker.tick()
    assert len(calls) == 1 and alerts[-1][1]


def test_monitor_inherits_only_alert_and_db_routing(monkeypatch, tmp_path):
    from brainlayer import supervisor_jobs
    from brainlayer.service_supervisor import Component

    captured = {}
    source = Component(
        ("unused",),
        env={
            "BRAINLAYER_DB": "/fixture/db",
            "BRAINLAYER_JOB_ALERT_PATH": "/fixture/alerts",
            "GOOGLE_API_KEY": "do-not-copy",
        },
    )
    manifest = tmp_path / "manifest"
    manifest.write_text(
        json.dumps(
            {
                "com.brainlayer.observability": {
                    "enabled": False,
                    "plist": {
                        "Label": "com.brainlayer.observability",
                        "ProgramArguments": ["unused"],
                        "KeepAlive": True,
                        "EnvironmentVariables": source.env,
                    },
                }
            }
        )
    )
    manifest.chmod(0o600)
    monkeypatch.setattr(supervisor_jobs.sys, "platform", "darwin")

    class FakeSupervisor:
        def __init__(self, components, path):
            captured.update(components)

        async def run(self, **kwargs):
            pass

    monkeypatch.setattr(supervisor_jobs, "Supervisor", FakeSupervisor)
    supervisor_jobs.run_daemon(tmp_path / "manifest", tmp_path / "control.sock")
    env = captured["brainbar-activation"].env
    assert env["BRAINLAYER_JOB_ALERT_PATH"] == "/fixture/alerts"
    assert env["BRAINLAYER_DB"] == "/fixture/db" and "GOOGLE_API_KEY" not in env


def test_interrupted_attempt_raises_attention_without_retry(monitor):
    worker, info, calls, alerts = monitor
    worker.state.write_text(
        json.dumps(
            {"identity": {"app": str(info.parents[1]), "version": "1.5.99", "git_commit": "abc"}, "stage": "attempted"}
        )
    )
    worker.tick()
    assert not calls and alerts[-1][0] == "brainbar-activation"
    assert "interrupted" in alerts[-1][1]
