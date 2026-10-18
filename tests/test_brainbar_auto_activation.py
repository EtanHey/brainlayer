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
    (app / "Contents/MacOS").mkdir(parents=True)
    info = app / "Contents/Info.plist"
    info.write_bytes(plistlib.dumps({"CFBundleShortVersionString": "1.5.99", "GitCommit": "abc"}))
    for label, binary in JOBS:
        (app / "Contents/MacOS" / binary).touch()
        (agents / f"{label}.plist").write_bytes(
            plistlib.dumps({"ProgramArguments": [str(app / "Contents/MacOS" / binary)]})
        )
    calls, alerts = [], []
    monkeypatch.setattr(auto, "brew_processes", lambda: [], raising=False)
    monkeypatch.setattr(auto, "running_bundle_matches", lambda *a: False, raising=False)
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
        monkeypatch.setattr(auto, "brew_processes", lambda: ["42 1 brew brew upgrade"])
    elif blocked == "loaded":
        monkeypatch.setattr(auto, "loaded", lambda *a: SimpleNamespace(returncode=0))
    else:
        (worker.agents / f"{JOBS[1][0]}.plist").unlink()
    worker.tick()
    assert not calls and not worker.state.exists()


def test_brew_started_during_preflight_defers_without_consuming_attempt(monitor, monkeypatch):
    worker, _, calls, _ = monitor
    states = iter([[], ["42 1 brew brew upgrade"]])
    monkeypatch.setattr(auto, "brew_processes", lambda: next(states))
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
            {"identity": {"app": str(info.parents[1]), **auto.app_identity(info.parents[1])}, "stage": "attempted"}
        )
    )
    worker.tick()
    assert not calls and alerts[-1][0] == "brainbar-activation"
    assert "interrupted" in alerts[-1][1]


@pytest.mark.parametrize("stage", ["ready", "failed"])
def test_same_build_reinstall_rearms_once(monitor, stage):
    worker, info, calls, _ = monitor
    app = info.parents[1]
    worker.state.write_text(json.dumps({"identity": {"app": str(app), **auto.app_identity(app)}, "stage": stage}))
    binary = app / "Contents/MacOS/BrainBar"
    replacement = binary.with_suffix(".new")
    replacement.touch()
    replacement.replace(binary)
    worker.tick()
    auto.AutoActivator(worker.agents, worker.state).tick()
    assert len(calls) == 1


def test_loaded_current_bundle_records_ready_and_respects_later_bootout(monitor, monkeypatch):
    worker, _, calls, alerts = monitor
    monkeypatch.setattr(auto, "loaded", lambda *a: SimpleNamespace(returncode=0))
    monkeypatch.setattr(auto, "running_bundle_matches", lambda *a: True)
    worker.tick()
    assert json.loads(worker.state.read_text())["stage"] == "ready"
    monkeypatch.setattr(auto, "running_bundle_matches", lambda *a: pytest.fail("ready identity should stay suppressed"))
    worker.tick()
    monkeypatch.setattr(auto, "loaded", lambda *a: SimpleNamespace(returncode=113))
    monkeypatch.setattr(auto, "running_bundle_matches", lambda *a: False)
    auto.AutoActivator(worker.agents, worker.state).tick()
    assert not calls


def test_loaded_stale_bundle_does_not_consume_identity(monitor, monkeypatch):
    worker, _, calls, _ = monitor
    monkeypatch.setattr(auto, "loaded", lambda *a: SimpleNamespace(returncode=0))
    worker.tick()
    assert not calls and not worker.state.exists()


def test_brew_deferral_alert_is_durable_names_all_matches_and_recovers(monitor, monkeypatch):
    worker, _, calls, alerts = monitor
    now = [1000.0]
    monkeypatch.setattr(auto.time, "time", lambda: now[0])
    processes = ["42 brew", "43 ruby"]
    monkeypatch.setattr(auto, "brew_processes", lambda: processes)
    # Brew can already have removed the cask plists; monitoring must still work.
    for path in worker.agents.glob("*.plist"):
        path.unlink()
    worker.tick()
    now[0] += 600
    auto.AutoActivator(worker.agents, worker.state).tick()
    assert not alerts
    now[0] += 1
    auto.AutoActivator(worker.agents, worker.state).tick()
    assert len(alerts) == 1 and all(row in alerts[0][1] for row in processes)
    now[0] += 100
    worker.tick()
    assert len(alerts) == 1 and not calls and not worker.state.exists()
    monkeypatch.setattr(auto, "brew_processes", lambda: [])
    worker.tick()
    assert alerts[-1] == (alerts[0][0], None)


@pytest.mark.parametrize("interruption", ["brew_ends", "label_loaded"])
def test_brew_deferral_requires_continuous_absence(monitor, monkeypatch, interruption):
    worker, _, calls, alerts = monitor
    now = [1000.0]
    monkeypatch.setattr(auto.time, "time", lambda: now[0])
    monkeypatch.setattr(auto, "brew_processes", lambda: ["42 1 brew brew upgrade"])
    worker.tick()
    now[0] += 599
    if interruption == "brew_ends":
        monkeypatch.setattr(auto, "brew_processes", lambda: [])
        monkeypatch.setattr(auto, "loaded", lambda *a: SimpleNamespace(returncode=0))
    else:
        monkeypatch.setattr(auto, "loaded", lambda *a: SimpleNamespace(returncode=0))
    worker.tick()
    monkeypatch.setattr(auto, "brew_processes", lambda: ["43 1 brew brew upgrade"])
    monkeypatch.setattr(auto, "loaded", lambda *a: SimpleNamespace(returncode=113))
    now[0] += 2
    worker.tick()
    assert not alerts and not calls


def test_process_scan_reports_every_matching_process(monkeypatch):
    rows = ["42 1 brew brew upgrade", "43 1 ruby /Homebrew/brew.rb reinstall"]
    monkeypatch.setattr(auto, "process_table", lambda: "\n".join(rows + ["44 1 python python worker.py"]))
    assert auto.brew_processes() == ["42 brew", "43 ruby"]


@pytest.mark.parametrize("token", ["ghp_" + "A" * 36, "opaque-synthetic-credential-123456789"])
def test_brew_deferral_discards_argv_before_every_alert_sink(tmp_path, monkeypatch, caplog, token):
    from brainlayer import job_alerts

    rows = [
        f"{pid} 1 /opt/homebrew/Library/Homebrew/shims/shared/curl "
        f'/opt/homebrew/Library/Homebrew/shims/shared/curl --header "Authorization: token {token}" '
        f"https://fixture.invalid/?access_token={token}"
        for pid in range(42, 49)
    ]
    monkeypatch.setattr(auto, "process_table", lambda: "\n".join(rows))
    monkeypatch.setattr(auto, "loaded", lambda *a: SimpleNamespace(returncode=113))
    now = [1000.0]
    monkeypatch.setattr(auto.time, "time", lambda: now[0])
    reasons, notifications = [], []
    alert_file = tmp_path / "job-alerts.json"

    def report(key, reason):
        reasons.append(reason)
        return job_alerts.report(key, reason, path=alert_file, notify=notifications.append)

    monkeypatch.setattr(auto, "report", report)
    worker = auto.AutoActivator(tmp_path / "agents", tmp_path / "state.json")
    worker.tick()
    now[0] += 601
    worker.tick()
    assert reasons and notifications and caplog.records and alert_file.is_file()
    for value in [
        *reasons,
        *notifications,
        alert_file.read_text(),
        caplog.text,
        repr([r.args for r in caplog.records]),
    ]:
        assert token not in value
        assert "Authorization" not in value and "fixture.invalid" not in value
        assert "/Homebrew/" not in value
    assert auto.brew_processes() == [f"{pid} curl" for pid in range(42, 49)]
    assert reasons == [
        "BrainBar activation deferred by Homebrew for >10 minutes: 42 curl; 43 curl; 44 curl; 45 curl; 46 curl; +2 more"
    ]


def test_process_scan_scrubs_and_bounds_executable_basename(monkeypatch):
    token = "ghp_" + "B" * 36
    monkeypatch.setattr(auto, "process_table", lambda: f"42 1 /private/{token} /Homebrew/child --secret private")
    assert auto.brew_processes() == ["42 [REDACTED:github]"]
    monkeypatch.setattr(auto, "process_table", lambda: f"42 1 /private/{'x' * 300} /Homebrew/child")
    assert auto.brew_processes() == ["42 " + "x" * 128]
