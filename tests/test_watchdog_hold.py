"""Owned watchdog holds: no real launchd, alerts or database state."""

import json
import os
import signal
import subprocess

import pytest

from brainlayer import job_alerts, maintenance


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    marker = tmp_path / "fleet-watchdog-hold.json"
    monkeypatch.setattr(maintenance, "PAUSE_SENTINEL_PATH", tmp_path / "pause.sentinel")
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(tmp_path / "alerts.json"))
    monkeypatch.setattr(job_alerts, "_notify", lambda _: None)
    monkeypatch.setattr(maintenance, "_pid_start_time", lambda pid: "current")
    monkeypatch.setattr(maintenance, "_pid_is_alive", lambda pid: pid == os.getpid())
    state = {"disabled": False, "loaded": True, "bootstraps": 0}

    def launchctl(args, **kwargs):
        if args[1] == "disable":
            assert json.loads(marker.read_text())["pid_start_time"] == "current"
            state["disabled"] = True
        if args[1] == "enable":
            state["disabled"] = False
        if args[1] == "bootstrap":
            state["loaded"] = True
            state["bootstraps"] += 1
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(maintenance, "run_command", launchctl)
    monkeypatch.setattr(maintenance, "is_launchd_label_disabled", lambda *a, **k: state["disabled"])
    monkeypatch.setattr(maintenance, "_service_is_loaded", lambda *a, **k: state["loaded"])
    monkeypatch.setattr(maintenance, "_bootout_service", lambda _: state.update(loaded=False) or True)
    monkeypatch.setattr(maintenance, "_resume_service", lambda *_: launchctl(["launchctl", "bootstrap"]))
    return marker, state, tmp_path / "alerts.json"


@pytest.mark.parametrize("owner", ["dead", "reused"])
def test_crashed_watchdog_hold_recovers_before_next_quiesce(fleet, monkeypatch, owner):
    marker, state, alerts = fleet
    maintenance._quiesce_services(("watch",), {})
    payload = json.loads(marker.read_text())
    payload.update(pid=12345, pid_start_time="old")
    marker.write_text(json.dumps(payload))
    monkeypatch.setattr(maintenance, "_pid_is_alive", lambda pid: owner == "reused" or pid == os.getpid())
    receipt = {}
    maintenance._quiesce_services(("watch",), receipt)
    assert receipt[maintenance._WATCHDOG_DISABLED_BEFORE] is False
    assert state["bootstraps"] == 1
    maintenance._resume_services(marker.parent, ("watch",), receipt)
    assert not state["disabled"] and state["loaded"] and state["bootstraps"] == 3
    assert not marker.exists()
    assert "fleet-watchdog-hold" in job_alerts.active_alerts(alerts)


def test_operator_disable_without_hold_is_preserved(fleet):
    marker, state, _ = fleet
    state["disabled"] = True
    receipt = {}
    maintenance._quiesce_services(("watch",), receipt)
    maintenance._resume_services(marker.parent, ("watch",), receipt)
    assert state["disabled"] and state["bootstraps"] == 1 and not marker.exists()


@pytest.mark.parametrize("step", ["enable", "bootstrap"])
def test_failed_watchdog_resume_retains_owned_marker(fleet, monkeypatch, step):
    marker, _, _ = fleet
    receipt = {}
    maintenance._quiesce_services(("watch",), receipt)

    def fail(*args, **kwargs):
        raise OSError("synthetic restore failure")

    monkeypatch.setattr(maintenance, "run_command" if step == "enable" else "_resume_service", fail)
    assert maintenance._resume_services(marker.parent, ("watch",), receipt)
    assert marker.exists()


def test_failed_injected_recovery_retains_marker(fleet):
    marker, _, _ = fleet
    marker.write_text(json.dumps({"pid": 12345, "pid_start_time": "old", "started_at": "fixture"}))
    with pytest.raises(maintenance.MaintenanceAbort):
        maintenance.recover_fleet_watchdog_hold(
            command_runner=lambda args: subprocess.CompletedProcess(args, 1, "", "")
        )
    assert marker.exists()


def test_live_hold_owner_refuses_without_service_changes(fleet):
    marker, state, _ = fleet
    marker.write_text(json.dumps({"pid": os.getpid(), "pid_start_time": "current", "started_at": "fixture"}))
    before = state.copy()
    with pytest.raises(maintenance.MaintenanceAbort) as error:
        maintenance._quiesce_services(("watch",), {})
    assert error.value.detail == "hold-active:com.etanhey.brainlayer-fleet-watchdog"
    assert state == before and marker.exists()


@pytest.mark.parametrize("entry", ["maintenance", "fts"])
def test_sigterm_unwinds_guarded_entry_and_restores_handler(fleet, monkeypatch, entry):
    marker, state, _ = fleet
    previous = signal.getsignal(signal.SIGTERM)

    def body(*args, **kwargs):
        receipt = {}
        try:
            maintenance._quiesce_services(("watch",), receipt)
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        finally:
            maintenance._resume_services(marker.parent, ("watch",), receipt)

    if entry == "maintenance":
        monkeypatch.setattr(maintenance, "run_maintenance", body)

        def call():
            maintenance.main(["--light"])

    else:
        monkeypatch.setattr(maintenance, "_run_coordinated_fts_repair_unlocked", body)

        def call():
            maintenance.run_coordinated_fts_repair(marker.parent / "fixture.db")

    with pytest.raises(SystemExit) as error:
        call()
    assert error.value.code == 143
    assert signal.getsignal(signal.SIGTERM) == previous
    assert not state["disabled"] and state["loaded"] and not marker.exists()
