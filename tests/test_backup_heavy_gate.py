"""Admission tests: real mkdir custody and real alerts, no DB or backup I/O."""

import signal
import tempfile
from pathlib import Path

import pytest

from brainlayer import backup_daily as backup
from brainlayer import backup_heavy_gate as gate
from brainlayer import job_alerts


@pytest.fixture
def admission(monkeypatch):
    root = Path.cwd() / ".build.noindex" / "pytest"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as directory:
        path = Path(directory)
        monkeypatch.setenv("BRAINLAYER_BACKUP_HEAVY_HOST", "m4")
        monkeypatch.setenv("BRAINLAYER_BACKUP_HEAVY_LOCK_PATH", str(path / "lock"))
        monkeypatch.setenv("BRAINLAYER_BACKUP_HEAVY_LOG_PATH", str(path / "heavy.log"))
        monkeypatch.setenv("BRAINLAYER_BACKUP_HEAVY_WINDOW_SECONDS", "20")
        monkeypatch.setenv("BRAINLAYER_BACKUP_HEAVY_RETRY_SECONDS", "10")
        monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(path / "alerts.json"))
        monkeypatch.delenv(backup.BACKUP_SUPERVISED_CHILD_ENV, raising=False)
        monkeypatch.delenv("BRAINLAYER_BACKUP_PHASE", raising=False)
        clock = [0.0]
        monkeypatch.setattr(gate.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(gate.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
        monkeypatch.setattr(gate, "_pressure_reason", lambda *args: None)
        monkeypatch.setattr(backup, "_configured_backup_timeout_seconds", lambda: 60)
        yield path, clock


@pytest.mark.parametrize("closed", ["lock", "pressure"])
def test_main_waits_then_runs_when_gate_opens(admission, monkeypatch, closed):
    path, clock = admission
    lock = path / "lock"
    if closed == "lock":
        lock.mkdir()
    else:
        monkeypatch.setattr(gate, "_pressure_reason", lambda *args: "swap too high" if clock[0] == 0 else None)
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds
        if closed == "lock":
            lock.rmdir()  # Only the simulated owning lane releases this lock.

    def run(seconds):
        assert sleeps == [10]
        assert lock.is_dir()
        return 0

    monkeypatch.setattr(gate.time, "sleep", sleep)
    monkeypatch.setattr(backup, "_supervise_backup_process", run)
    assert backup.main() == 0
    assert not lock.exists()
    assert "deferred:" in (path / "heavy.log").read_text()
    assert "released" in (path / "heavy.log").read_text()


def test_held_lock_expires_with_durable_alert_and_is_never_removed(admission, monkeypatch):
    path, clock = admission
    lock = path / "lock"
    lock.mkdir()
    monkeypatch.setattr(backup, "_supervise_backup_process", lambda *a, **kw: pytest.fail("backup started"))
    assert backup.main() == 75
    assert clock[0] == 20
    assert lock.is_dir()
    assert "by the heavy-work gate" in job_alerts.active_alerts()["backup-daily"]
    assert "took" not in (path / "heavy.log").read_text()


@pytest.mark.parametrize("failure", [RuntimeError("failed"), SystemExit(143), KeyboardInterrupt()])
def test_owned_lock_released_on_every_unwind(admission, monkeypatch, failure):
    path, _ = admission

    def run(*args):
        assert (path / "lock").is_dir()
        raise failure

    monkeypatch.setattr(backup, "_supervise_backup_process", run)
    with pytest.raises(type(failure)):
        backup.main()
    assert not (path / "lock").exists()
    assert (path / "heavy.log").read_text().count("released") == 1


def test_term_unwinds_owned_lock(admission):
    path, _ = admission
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit, match="143"):
        with gate.backup_heavy_gate():
            signal.raise_signal(signal.SIGTERM)
    assert not (path / "lock").exists()
    assert signal.getsignal(signal.SIGTERM) == previous


def test_snapshot_deadline_bounds_admission(admission, monkeypatch):
    path, clock = admission
    (path / "lock").mkdir()
    monkeypatch.setenv("BRAINLAYER_BACKUP_PHASE", "snapshot")
    monkeypatch.setenv(backup.BACKUP_CALENDAR_DEADLINE_ENV, "76")
    monkeypatch.setattr(backup, "default_snapshot_timeout_seconds", lambda: 60)
    monkeypatch.setattr(backup, "_supervise_backup_process", lambda *a, **kw: pytest.fail("backup started"))
    assert backup.main() == 75
    assert clock[0] == 15  # 76 - 60 receipt reserve - 1 second of work
    assert "backup-snapshot" in job_alerts.active_alerts()


@pytest.mark.parametrize(("swap", "load", "closed"), [(8.9, 11.9, False), (9, 1, True), (1, 12, True)])
def test_pressure_thresholds_are_strict(monkeypatch, swap, load, closed):
    monkeypatch.setattr(gate.sys, "platform", "darwin")
    monkeypatch.setattr(gate, "_sysctl", lambda key: f"total = 16000M used = {swap * 1024}M free = 1M")
    monkeypatch.setattr(gate.os, "getloadavg", lambda: (load, 0, 0))
    assert bool(gate._pressure_reason("m4", 9, 12, 44)) is closed


def test_probe_failure_is_closed(monkeypatch):
    monkeypatch.setattr(gate.sys, "platform", "darwin")
    monkeypatch.setattr(gate, "_sysctl", lambda key: "unknown")
    assert gate._pressure_reason("m4", 9, 12, 44) == "swap usage unavailable"


def test_closed_pressure_expires_releases_and_recovers(admission, monkeypatch):
    path, _ = admission
    monkeypatch.setattr(gate, "_pressure_reason", lambda *args: "swap too high")
    monkeypatch.setattr(backup, "_supervise_backup_process", lambda *args: pytest.fail("backup started"))
    assert backup.main() == 75
    assert not (path / "lock").exists()
    assert "backup-daily" in job_alerts.active_alerts()
    monkeypatch.setattr(gate, "_pressure_reason", lambda *args: None)
    monkeypatch.setattr(backup, "_supervise_backup_process", lambda *args: 0)
    assert backup.main() == 0
    assert not job_alerts.active_alerts()


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1"])
def test_invalid_config_fails_before_admission(admission, monkeypatch, value):
    path, _ = admission
    monkeypatch.setenv("BRAINLAYER_BACKUP_HEAVY_RETRY_SECONDS", value)
    with pytest.raises(ValueError):
        backup.main()
    assert not (path / "lock").exists()
    assert "backup-daily" in job_alerts.active_alerts()


@pytest.mark.parametrize(("brand", "host"), [("Apple M1 Pro", "m1"), ("Apple M4", "m4")])
def test_cpu_detection(monkeypatch, brand, host):
    monkeypatch.delenv("BRAINLAYER_BACKUP_HEAVY_HOST", raising=False)
    monkeypatch.setattr(gate.sys, "platform", "darwin")
    monkeypatch.setattr(gate, "_sysctl", lambda key: brand)
    assert gate._host() == host


def test_m1_disk_floor(monkeypatch):
    monkeypatch.setattr(gate.sys, "platform", "darwin")
    monkeypatch.setattr(gate, "_sysctl", lambda key: "used = 1M")
    monkeypatch.setattr(gate.os, "getloadavg", lambda: (1, 0, 0))
    from collections import namedtuple

    usage = namedtuple("Usage", "free")
    monkeypatch.setattr(gate.shutil, "disk_usage", lambda path: usage(43 * 1024**3))
    assert "free disk" in gate._pressure_reason("m1", 9, 12, 44)
