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
    path, start, request = request.getfixturevalue("native")
    code = native_tests.BOOT.replace(
        "'off':Component", "'hotlane':Component((sys.executable,'-c','import time; time.sleep(60)')), 'off':Component"
    )
    monkeypatch.setattr(native_tests, "BOOT", code)
    process, initial, _ = start()
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


@pytest.mark.parametrize("gate", ["pause", "corrupt", "null", "operator", "unknown", "off"])
def test_stop_signals_block_heals_and_bootstrap(backend, gate):
    client, path, request, initial, commands, disabled = backend
    label = "com.brainlayer.watch"
    if gate == "pause":
        (path / "pause.sentinel").write_text(json.dumps({"labels": [label]}))
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
