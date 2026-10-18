import json
import plistlib
from types import SimpleNamespace

import pytest

from brainlayer import brainbar_auto_activation as auto
from brainlayer.brainbar_identity import JOBS


@pytest.fixture
def monitor(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(tmp_path / "job-alerts.json"))
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


@pytest.mark.parametrize("error", [auto.subprocess.TimeoutExpired(["ps", "private-argv"], 5), OSError("private-argv")])
@pytest.mark.parametrize("codes", [(113, 113), (0, 113), (113, 1)])
def test_scan_failures_are_transient_durable_and_recover(monitor, monkeypatch, error, codes):
    from brainlayer import job_alerts

    worker, _, calls, alerts = monitor
    alert_file = worker.state.with_name("alerts.json")
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(alert_file))
    monkeypatch.setattr(
        auto,
        "report",
        lambda key, reason: (alerts.append((key, reason)), job_alerts.report(key, reason, notify=lambda _: None)),
    )
    worker.state.write_text('{"stage":"ready","identity":{}}')
    brew_state = worker.state.with_suffix(".brew.json")
    brew_state.write_text('{"since":1000,"alerted":true}')
    before = worker.state.read_bytes(), brew_state.read_bytes()

    def fail():
        raise error

    monkeypatch.setattr(auto, "brew_processes", fail)
    monkeypatch.setattr(
        auto, "loaded", lambda label: SimpleNamespace(returncode=codes[[x[0] for x in JOBS].index(label)])
    )
    for _ in range(11):
        auto.AutoActivator(worker.agents, worker.state).tick()
    assert not alerts and not calls
    assert (worker.state.read_bytes(), brew_state.read_bytes()) == before
    auto.AutoActivator(worker.agents, worker.state).tick()
    assert bool(alerts) == (codes == (113, 113))
    assert "private-argv" not in repr(alerts)
    if alerts:
        assert type(error).__name__ in alerts[-1][1]
    # Recovery must clear even if no plists remain, without erasing other failures.
    for path in worker.agents.glob("*.plist"):
        path.unlink()
    monkeypatch.setattr(auto, "brew_processes", lambda: [])
    worker.tick()
    assert "brainbar-activation" not in job_alerts.active_alerts()
    monkeypatch.setattr(auto, "brew_processes", fail)
    worker.tick()
    assert "brainbar-activation" not in job_alerts.active_alerts()


@pytest.mark.parametrize(
    "reason,cleared",
    [
        (
            "BrainBar activation monitor refused: Command '['/bin/ps', '-axo', 'pid=,ppid=,comm=,args=']' timed out after 5 seconds",
            True,
        ),
        ("BrainBar activation failed: blocked", False),
    ],
)
def test_successful_scan_clears_only_scan_alerts(monitor, monkeypatch, reason, cleared):
    from brainlayer import job_alerts

    worker, _, _, _ = monitor
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(worker.state.with_name("alerts.json")))
    job_alerts.report("brainbar-activation", reason, notify=lambda _: None)
    monkeypatch.setattr(auto, "report", lambda key, reason: job_alerts.report(key, reason, notify=lambda _: None))
    for path in worker.agents.glob("*.plist"):
        path.unlink()
    worker.tick()
    assert ("brainbar-activation" not in job_alerts.active_alerts()) == cleared


def test_second_scan_failure_never_launches_and_successful_scan_resets_count(monitor, monkeypatch):
    worker, _, calls, alerts = monitor
    scans = 0

    def scan():
        nonlocal scans
        scans += 1
        if scans % 2 == 0:
            raise OSError("private-argv")
        return []

    monkeypatch.setattr(auto, "brew_processes", scan)
    for _ in range(11):
        auto.AutoActivator(worker.agents, worker.state).tick()
    assert not calls and not alerts and not worker.state.exists()
    worker.tick()
    assert not calls and not worker.state.exists()
    assert not alerts
    assert json.loads(worker.state.with_suffix(".scan.json").read_text()) == {"failures": 1}


@pytest.mark.parametrize(
    "scan_reason",
    [
        "BrainBar activation process scan failed for >=12 ticks: TimeoutExpired",
        "BrainBar activation monitor refused: Command '['/bin/ps', '-axo']' timed out after 5 seconds",
        "BrainBar activation monitor refused: [Errno 11] Resource temporarily unavailable",
    ],
)
def test_scan_recovery_keeps_new_activation_failure_visible(monitor, monkeypatch, scan_reason):
    from brainlayer import job_alerts

    worker, _, calls, _ = monitor
    job_alerts.report("brainbar-activation", scan_reason, notify=lambda _: None)
    monkeypatch.setattr(auto, "report", lambda key, reason: job_alerts.report(key, reason, notify=lambda _: None))
    worker.tick()
    assert len(calls) == 1
    assert (
        job_alerts.active_alerts()["brainbar-activation"]
        == "BrainBar activation failed: blocked; run brainlayer brainbar activate"
    )
    worker.tick()
    assert len(calls) == 1
    assert "activation failed: blocked" in job_alerts.active_alerts()["brainbar-activation"]


@pytest.mark.parametrize(
    "error",
    [
        OSError("launchctl private-argv"),
        auto.subprocess.TimeoutExpired(["launchctl", "private-argv"], 5),
        RuntimeError("launchctl private-argv"),
    ],
)
def test_scan_failure_with_unknown_loaded_state_does_not_alert(monitor, monkeypatch, error):
    worker, _, calls, alerts = monitor
    worker.state.with_suffix(".scan.json").write_text('{"failures":11}')

    def fail_scan():
        raise OSError("ps private-argv")

    def fail_loaded(label):
        raise error

    monkeypatch.setattr(auto, "brew_processes", fail_scan)
    monkeypatch.setattr(auto, "loaded", fail_loaded)
    worker.tick()
    assert not alerts and not calls and not worker.state.exists()


def test_successful_scan_resets_counter_before_later_tick_error(monitor):
    worker, _, calls, alerts = monitor
    failures = worker.state.with_suffix(".scan.json")
    failures.write_text('{"failures":11}')
    (worker.agents / f"{JOBS[0][0]}.plist").write_text("invalid plist")
    with pytest.raises(plistlib.InvalidFileException):
        worker.tick()
    assert not failures.exists()
    assert not calls and not alerts


@pytest.mark.parametrize("legacy", [False, True])
def test_non_scan_errno_never_flaps_after_one_shot_migration(monitor, monkeypatch, legacy):
    from brainlayer import job_alerts

    worker, _, calls, alerts = monitor
    notifications = []
    if legacy:
        job_alerts.report(
            "brainbar-activation", "BrainBar activation monitor refused: [Errno 11] old scan", notify=lambda _: None
        )

    def report(key, reason):
        alerts.append((key, reason))
        return job_alerts.report(key, reason, notify=notifications.append)

    monkeypatch.setattr(auto, "report", report)

    def fail_identity(app):
        raise PermissionError(13, "fixture identity denied")

    def fail_scan():
        raise OSError("fixture scan unavailable")

    with monkeypatch.context() as scan_patch:
        scan_patch.setattr(auto, "brew_processes", fail_scan)
        worker.tick()
    assert not worker.state.with_suffix(".scan-migration.json").exists()
    monkeypatch.setattr(auto, "app_identity", fail_identity)
    for _ in range(3):
        with pytest.raises(PermissionError) as failure:
            auto.AutoActivator(worker.agents, worker.state).tick()
        report("brainbar-activation", f"BrainBar activation monitor refused: {failure.value}")
        assert "[Errno 13]" in job_alerts.active_alerts()["brainbar-activation"]
    assert sum(reason is None for _, reason in alerts) == int(legacy)
    assert sum(reason is not None for _, reason in alerts) == 3
    assert len(notifications) == 1 + int(legacy)
    assert worker.state.with_suffix(".scan-migration.json").is_file()
    assert not calls


def test_unwritable_migration_never_clears_or_flaps(monitor, monkeypatch):
    from brainlayer import job_alerts

    worker, _, calls, alerts = monitor
    reason = "BrainBar activation monitor refused: [Errno 11] old scan"
    job_alerts.report("brainbar-activation", reason, notify=lambda _: None)
    notifications = []

    def report(key, reason):
        alerts.append((key, reason))
        return job_alerts.report(key, reason, notify=notifications.append)

    def unwritable(path, payload):
        assert path == worker.state.with_suffix(".scan-migration.json")
        raise PermissionError(13, "fixture migration denied")

    monkeypatch.setattr(auto, "report", report)
    monkeypatch.setattr(auto, "_atomic_write_json", unwritable)
    for _ in range(3):
        with pytest.raises(PermissionError) as failure:
            auto.AutoActivator(worker.agents, worker.state).tick()
        report("brainbar-activation", f"BrainBar activation monitor refused: {failure.value}")
        assert job_alerts.active_alerts()["brainbar-activation"] == reason
    assert len(alerts) == 3 and all(reason is not None for _, reason in alerts)
    assert not notifications and not calls
    assert not worker.state.with_suffix(".scan-migration.json").exists()
