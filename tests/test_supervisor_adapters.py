"""Legacy health/watchdog policy with real supervisor control and private children."""

import json
import os
import subprocess
from datetime import UTC, datetime, timedelta

import pytest

import tests.test_service_supervisor as native_tests
from brainlayer import health_check
from brainlayer.service_supervisor import Supervisor
from brainlayer.supervisor_adapters import SupervisorCommands, adapt_commands
from tests.test_service_supervisor import native  # noqa: F401
from tests.test_stability_health_check import _make_db, _queue_backlog_config, _run_queue_backlog_health
from tests.test_throughput_watchdog import _config, _load_module, _progress


@pytest.fixture
def backend(request, monkeypatch):
    grace = getattr(request, "param", 1)
    path, start, request = request.getfixturevalue("native")
    code = native_tests.BOOT.replace("exit_timeout=1)", f"exit_timeout={grace})").replace(
        "'off':Component", "'hotlane':Component((sys.executable,'-c','import time; time.sleep(60)')), 'off':Component"
    )
    monkeypatch.setattr(native_tests, "BOOT", code)
    process, initial, _ = start()

    def ready():
        nonlocal initial
        assert process.poll() is None
        initial = request()["components"]
        return all(value["pid"] is not None for value in initial.values() if value["configured"])

    # The native file markers do not cover the hotlane added by this fixture.
    native_tests.eventually(ready)
    commands, disabled = [], {"output": "disabled services = {}", "code": 0}

    def fallback(args):
        commands.append(args)
        if args[:2] == ["launchctl", "print-disabled"]:
            return subprocess.CompletedProcess(args, disabled["code"], disabled["output"], "")
        if args[0] == "ps":
            return subprocess.run(args, capture_output=True, text=True)
        raise AssertionError(f"legacy command escaped adapter: {args}")

    client = SupervisorCommands(path / "control.sock", fallback, pause_path=path / "pause.sentinel")
    yield client, path, request, initial, commands, disabled
    process.terminate()
    assert process.wait(timeout=2) == 0


def test_health_heal_is_component_scoped_and_keeps_alerts(backend, tmp_path, monkeypatch):
    client, path, request, initial, commands, _ = backend
    monkeypatch.setenv("BRAINLAYER_SUPERVISOR_SOCKET", str(path / "control.sock"))
    config, state, _, _ = _queue_backlog_config(tmp_path, heal=True)
    config.job_opt_path = None
    config.health_check_label = config.observability_label = ""
    config.pause_sentinel_path = path / "pause.sentinel"
    config.job_alert_path = tmp_path / "alerts.json"
    config.job_alert_path.write_text(json.dumps({"wal-checkpoint": "busy"}))
    for tick in range(3):
        db = tmp_path / f"{tick}.db"
        _make_db(db, total=103, vector_rows=100)
        config.db_path = db
        result = _run_queue_backlog_health(
            config, client.fallback, now=datetime(2026, 10, 12, tzinfo=UTC) + timedelta(seconds=299 * tick)
        )
    current = request()["components"]
    assert current["hotlane"]["starts"] == 2
    assert current["watch"]["pid"] == initial["watch"]["pid"] and current["drain"]["pid"] == initial["drain"]["pid"]
    assert any(a.startswith("kickstart:com.brainlayer.hotlane-brainbar") for a in result.actions)
    assert json.loads(config.job_alert_path.read_text()) == {"wal-checkpoint": "busy"}
    assert not any(a[0] == "launchctl" and a[1] != "print-disabled" for a in commands)


@pytest.mark.parametrize("program", ["launchctl", "/bin/launchctl"])
@pytest.mark.parametrize("options", [[], ["-k"], ["-kp"]])
def test_managed_healer_cannot_kickstart_parent_supervisor(backend, program, options):
    client, _, request, initial, commands, _ = backend
    result = client([program, "kickstart", *options, f"gui/{os.getuid()}/com.brainlayer.daemon"])
    assert result.returncode != 0 and "parent supervisor" in result.stderr
    assert commands == []  # Never escape to the real launchctl runner.
    current = request()["components"]
    assert all(current[name]["pid"] == value["pid"] for name, value in initial.items())


def test_parent_restart_refusal_after_delayed_hotlane_start(request, monkeypatch):
    # Hold the added component until native.start() has captured its early snapshot.
    scheduled_start = """
import json
hotlane_release=asyncio.Event()
class ScheduledSupervisor(Supervisor):
 async def _start_pending(self, child):
  if child is self.components['hotlane']: await hotlane_release.wait()
  await super()._start_pending(child)
 async def _control(self, reader, writer):
  if hotlane_release.is_set():
   await self._wait_start(self.components['hotlane'])
  else:
   (p/'before-hotlane.json').write_text(json.dumps(self.status()))
  await super()._control(reader, writer)
  hotlane_release.set()
"""
    monkeypatch.setattr(
        native_tests, "BOOT", native_tests.BOOT.replace("s=Supervisor(", scheduled_start + "\ns=ScheduledSupervisor(")
    )
    backend = request.getfixturevalue("backend")
    client, path, status, initial, _, _ = backend
    before = json.loads((path / "before-hotlane.json").read_text())["components"]
    assert before["hotlane"]["pid"] is None and before["hotlane"]["starts"] == 0
    current = status()["components"]
    print(
        json.dumps(
            {
                phase: {name: {key: value[key] for key in ("pid", "starts")} for name, value in states.items()}
                for phase, states in (("before_hotlane", before), ("initial", initial), ("current", current))
            },
            sort_keys=True,
        )
    )
    assert current["hotlane"]["pid"] is not None and current["hotlane"]["starts"] == 1

    def forbidden_control(*args, **kwargs):
        pytest.fail("parent restart refusal must not contact supervisor control")

    monkeypatch.setattr(client, "request", forbidden_control)
    test_managed_healer_cannot_kickstart_parent_supervisor(backend, "/bin/launchctl", ["-k"])
    assert all(value["starts"] == 1 for value in initial.values() if value["configured"])


def test_parent_restart_refusal_survives_control_outage_and_keeps_unmanaged_jobs(tmp_path):
    commands = []

    def fallback(args):
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    client = SupervisorCommands(tmp_path / "missing.sock", fallback)
    client.control_error = "supervisor control unavailable"
    assert client(["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/com.brainlayer.daemon"]).returncode != 0
    assert commands == []
    app = ["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/com.brainlayer.brainbar"]
    assert client(app).returncode == 0
    assert commands == [app]


@pytest.mark.parametrize("gate", ["pause", "daemon-pause", "corrupt", "null", "operator", "unknown", "off"])
def test_stop_signals_block_heals_and_bootstrap(backend, gate):
    client, path, request, initial, commands, disabled = backend
    label = "com.brainlayer.watch"
    if gate in {"pause", "daemon-pause"}:
        held = "com.brainlayer.daemon" if gate == "daemon-pause" else label
        (path / "pause.sentinel").write_text(json.dumps({"labels": [held]}))
    elif gate == "corrupt":
        (path / "pause.sentinel").write_text("{")
    elif gate == "null":
        (path / "pause.sentinel").write_text("null")
    elif gate == "operator":
        disabled["output"] = f'"{label}" => disabled'
    elif gate == "unknown":
        disabled["code"] = 1
    else:
        client.aliases[label] = "off"
    for args in (
        ["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{label}"],
        ["launchctl", "bootstrap", f"gui/{os.getuid()}", f"/archived/{label}.plist"],
    ):
        assert client(args).returncode != 0
    assert request()["components"]["watch"]["pid"] == initial["watch"]["pid"]
    assert not any(a[0] == "launchctl" and a[1] != "print-disabled" for a in commands)


def test_expired_pause_and_loaded_metadata_without_revival(backend):
    client, path, request, initial, _, _ = backend
    label = "com.brainlayer.watch"
    (path / "pause.sentinel").write_text(json.dumps({"labels": [label], "expires_at": "2000-01-01T00:00:00Z"}))
    target = f"gui/{os.getuid()}/{label}"
    assert f"pid = {initial['watch']['pid']}" in client(["launchctl", "print", target]).stdout
    assert client(["launchctl", "kickstart", "-k", target]).returncode == 0
    assert request()["components"]["watch"]["pid"] != initial["watch"]["pid"]
    assert (
        client(["launchctl", "bootstrap", f"gui/{os.getuid()}", "/archived/com.brainlayer.enrichment.plist"]).returncode
        != 0
    )

    assert client(["launchctl", "bootout", target]).returncode == 0
    assert request()["components"]["watch"]["pid"] is None
    assert client(["launchctl", "bootstrap", f"gui/{os.getuid()}", f"/archived/{label}.plist"]).returncode == 0
    assert request()["components"]["watch"]["pid"] is not None
    assert client(["launchctl", "print", "gui/9999/" + label]).returncode != 0


def test_throughput_policy_uses_control_and_retains_recovery_log(backend, tmp_path, monkeypatch):
    client, path, request, initial, commands, _ = backend
    monkeypatch.setenv("BRAINLAYER_SUPERVISOR_SOCKET", str(path / "control.sock"))
    monkeypatch.setenv("BRAINLAYER_PAUSE_SENTINEL", str(client.pause_path))
    module = _load_module()
    config = _config(module, tmp_path, stall_threshold=1)
    from dataclasses import replace

    config = replace(config, watch_label="com.brainlayer.watch", log_path=tmp_path / "throughput.log")
    for now in (1000, 1060):
        result = module.run_once(
            config,
            now_epoch=now,
            progress_reader=lambda _: _progress(module, 40, 40),
            operational_progress_reader=lambda _: module.OperationalProgress(None, None, None),
            source_probe=lambda *_: module.SourceEvidence(1, 100, 1, 999),
            command_runner=client.fallback,
            alert_fn=module._best_effort_alert,
        )
    assert result.action == "respawn:com.brainlayer.watch"
    assert request()["components"]["watch"]["pid"] != initial["watch"]["pid"]
    assert config.log_path.exists()
    assert all(a[0] in {"ps", "launchctl"} for a in commands)


@pytest.mark.parametrize("elapsed,advanced", [(239, False), (240, True), (299, True)])
def test_progress_comparison_accepts_jittered_five_minute_ticks(elapsed, advanced):
    now = datetime(2026, 10, 12, tzinfo=UTC)
    state = dict(hotlane_missing_vectors=3, hotlane_sample_at=now.isoformat())
    assert health_check._hotlane_progress_sample(state, 4, now + timedelta(seconds=elapsed)) == (advanced, False)


def test_inactive_mode_and_shutdown_fits_default_launchd(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_SUPERVISOR_SOCKET", raising=False)

    def runner(args):
        return args

    assert adapt_commands(runner) is runner
    assert Supervisor({}, tmp_path / "unused").shutdown_timeout <= 15
    monkeypatch.setenv("BRAINLAYER_SUPERVISOR_SOCKET", str(tmp_path / "missing"))
    with pytest.raises(OSError):
        adapt_commands(runner)


@pytest.fixture
def reply_endpoint(request):
    import socket
    import threading

    path = request.getfixturevalue("native")[0] / "reply.sock"
    server = socket.socket(socket.AF_UNIX)
    server.bind(str(path))
    path.chmod(0o600)
    server.listen(1)
    threads = []

    def serve(payload, delay=0):
        def reply():
            import time

            with server.accept()[0] as connection:
                connection.recv(65536)
                time.sleep(delay)
                try:
                    connection.sendall(json.dumps(payload).encode() + b"\n")
                except BrokenPipeError:
                    pass

        thread = threading.Thread(target=reply)
        thread.start()
        threads.append(thread)
        return path

    yield serve
    server.close()
    for thread in threads:
        thread.join(timeout=3)
        assert not thread.is_alive()


@pytest.mark.parametrize("failure", ["missing", "slow", "malformed"])
def test_control_outage_preserves_health_result_badge_and_canary(tmp_path, monkeypatch, reply_endpoint, failure):
    path = tmp_path / "missing.sock"
    if failure != "missing":
        path = reply_endpoint({"components": []}, delay=2.1 if failure == "slow" else 0)
    monkeypatch.setenv("BRAINLAYER_SUPERVISOR_SOCKET", str(path))
    config, state, _, _ = _queue_backlog_config(tmp_path, heal=True)
    config.badge_state_path = tmp_path / "badge.json"
    commands = []

    def runner(args):
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    result = _run_queue_backlog_health(config, runner)
    assert result.canary_status == "results"
    assert config.badge_state_path.exists() and state.exists()
    assert any(i.code == "supervisor_control_unavailable" and i.severity == "critical" for i in result.issues)
    assert result.ok is False
    assert any("failed:" in action for action in result.actions)
    assert not any(a[0] == "launchctl" and a[1] in {"bootstrap", "kickstart", "bootout"} for a in commands)


def test_read_only_health_does_not_build_control_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_SUPERVISOR_SOCKET", str(tmp_path / "missing.sock"))
    config, _, _, _ = _queue_backlog_config(tmp_path, heal=False)
    result = _run_queue_backlog_health(config, lambda a: subprocess.CompletedProcess(a, 0, "", ""))
    assert result.canary_status == "results"
    assert not any(i.code == "supervisor_control_unavailable" for i in result.issues)


@pytest.mark.parametrize(
    "payload", [[], None, 4, {"components": []}, {"components": None}, {"components": {"watch": []}}]
)
def test_malformed_control_reply_returns_failed_command(reply_endpoint, payload):
    client = SupervisorCommands(reply_endpoint(payload), lambda a: None)
    result = client(["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/com.brainlayer.watch"])
    assert result.returncode == 127
    assert "supervisor control failed" in result.stderr


@pytest.mark.parametrize("payload", [{}, {"reason": "maintenance finished"}])
def test_label_free_pause_sentinel_does_not_hold_component(backend, payload):
    client, path, request, initial, _, _ = backend
    (path / "pause.sentinel").write_text(json.dumps(payload))
    assert client(["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/com.brainlayer.watch"]).returncode == 0
    assert request()["components"]["watch"]["pid"] != initial["watch"]["pid"]


def test_action_timeout_covers_advertised_exit_grace(backend, monkeypatch):
    client, _, request, _, _, _ = backend
    states = request()["components"]
    states["watch"]["exit_timeout"] = 60
    calls = []

    def control(action="status", component="*", *, timeout=None):
        calls.append((action, component, timeout))
        return states

    monkeypatch.setattr(client, "request", control)
    assert client(["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/com.brainlayer.watch"]).returncode == 0
    assert calls[-1][2] >= 65


def test_shared_socket_resolver(tmp_path, monkeypatch):
    from pathlib import Path

    from brainlayer.paths import supervisor_socket_path

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("BRAINLAYER_SUPERVISOR_SOCKET", raising=False)
    default = tmp_path / ".local/share/brainlayer/supervisor/control.sock"
    assert supervisor_socket_path() == Supervisor({}).path == default
    monkeypatch.setenv("BRAINLAYER_SUPERVISOR_SOCKET", str(tmp_path / "custom.sock"))
    assert supervisor_socket_path() == Supervisor({}).path == tmp_path / "custom.sock"


@pytest.mark.parametrize("backend", [21], indirect=True)
def test_slow_successful_restart_is_not_reported_as_failure(backend):
    import time

    client, _, request, initial, _, _ = backend
    started = time.monotonic()
    result = client(["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/com.brainlayer.watch"])
    assert result.returncode == 0, result.stderr
    assert time.monotonic() - started >= 20
    current = request()["components"]
    assert current["watch"]["pid"] != initial["watch"]["pid"]
    assert current["drain"]["pid"] == initial["drain"]["pid"]


@pytest.mark.parametrize("domain", [None, "gui/99999"])
def test_invalid_disabled_domain_does_not_escape_or_call_fallback(backend, domain):
    client, _, _, _, commands, _ = backend
    args = ["launchctl", "print-disabled"] + ([domain] if domain else [])
    assert client(args).returncode == 127
    assert not commands


def test_failed_drain_restart_is_reported_in_queue_summary(tmp_path):
    config, _, _, _ = _queue_backlog_config(tmp_path, heal=True)
    action = f"kickstart_failed:{config.drain_label}"
    summary = health_check._queue_heal_summary(
        config=config,
        result=health_check.HealthCheckResult(checked_at="", ok=False, actions=[action]),
        blocking_reason=None,
        previous_failures={},
        heal_failures={f"{config.drain_label}:queue_backed_up": 2},
        heal_tripped=set(),
        backlog_issue_active=True,
    )
    assert "heal=failing" in summary and action in summary and "heal_failures=2" in summary
