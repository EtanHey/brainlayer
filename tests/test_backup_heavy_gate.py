"""Admission tests: real mkdir custody and real alerts, no DB or backup I/O."""

import os
import select
import signal
import subprocess
import sys
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
        monkeypatch.setattr(backup, "_configured_backup_timeout_seconds", lambda: 3600)
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
        alert = job_alerts.active_alerts()["backup-daily"]
        assert "since " in alert
        assert ("swap too high" if closed == "pressure" else "lock held") in alert
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
    assert not job_alerts.active_alerts()
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
    assert "behind the heavy-work gate since " in job_alerts.active_alerts()["backup-daily"]
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
    monkeypatch.setenv(backup.BACKUP_CALENDAR_DEADLINE_ENV, "135")
    monkeypatch.setattr(backup, "default_snapshot_timeout_seconds", lambda: 60)
    monkeypatch.setattr(backup, "_supervise_backup_process", lambda *a, **kw: pytest.fail("backup started"))
    assert backup.main() == 75
    assert clock[0] == 15  # 135 - 60 receipt reserve - 60 seconds of useful work
    assert "backup-snapshot" in job_alerts.active_alerts()


@pytest.mark.parametrize(("swap", "load", "closed"), [(8.9, 11.9, False), (9, 1, True), (1, 12, True)])
def test_pressure_thresholds_are_strict(monkeypatch, swap, load, closed):
    monkeypatch.setattr(gate.sys, "platform", "darwin")
    monkeypatch.setattr(gate, "_sysctl", lambda key: f"total = 16000M used = {swap * 1024}M free = 1M")
    monkeypatch.setattr(gate.os, "getloadavg", lambda: (load, 0, 0))
    assert bool(gate._pressure_reason(9, 12)) is closed


def test_probe_failure_is_closed(monkeypatch):
    monkeypatch.setattr(gate.sys, "platform", "darwin")
    monkeypatch.setattr(gate, "_sysctl", lambda key: "unknown")
    assert gate._pressure_reason(9, 12) == "swap usage unavailable"


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


def test_m1_at_37_gib_still_runs_backup(admission, monkeypatch):
    import shutil

    path, _ = admission
    monkeypatch.setenv("BRAINLAYER_BACKUP_HEAVY_HOST", "m1")
    monkeypatch.setattr(gate.sys, "platform", "darwin")
    monkeypatch.setattr(gate, "_sysctl", lambda key: "used = 1M")
    monkeypatch.setattr(gate.os, "getloadavg", lambda: (1, 0, 0))
    from collections import namedtuple

    usage = namedtuple("Usage", "free")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage(37 * 1024**3))
    # Remove the fixture's pressure fake: exercise the production resource check.
    monkeypatch.setattr(gate, "_pressure_reason", REAL_PRESSURE_REASON)
    ran = []
    monkeypatch.setattr(backup, "_supervise_backup_process", lambda seconds: ran.append(seconds) or 0)
    assert backup.main() == 0
    assert ran == [3600]
    assert not (path / "lock").exists()


REAL_PRESSURE_REASON = gate._pressure_reason


def test_no_lock_churn_when_pressure_is_closed(admission, monkeypatch):
    path, _ = admission
    monkeypatch.setattr(gate, "_pressure_reason", lambda *args: "swap too high")
    assert backup.main() == 75
    log = (path / "heavy.log").read_text()
    assert "took" not in log and "released" not in log


def test_pressure_is_rechecked_under_lock(admission, monkeypatch):
    path, _ = admission
    probes = iter([None, "pressure changed", None, None])
    monkeypatch.setattr(gate, "_pressure_reason", lambda *args: next(probes))
    monkeypatch.setattr(backup, "_supervise_backup_process", lambda seconds: 0)
    assert backup.main() == 0
    log = (path / "heavy.log").read_text()
    assert "pressure changed" in log
    assert log.count("took") == log.count("released") == 1


def test_host_detection_failure_names_the_gate(admission, monkeypatch):
    def fail():
        raise OSError("sysctl unavailable")

    monkeypatch.setattr(gate, "_host", fail)
    assert backup.main() == 75
    assert "heavy-work gate host detection failed" in job_alerts.active_alerts()["backup-daily"]


def test_admission_wait_reduces_child_budget(admission, monkeypatch):
    _, clock = admission
    monkeypatch.setattr(gate, "_pressure_reason", lambda *args: "busy" if clock[0] == 0 else None)
    budgets = []
    monkeypatch.setattr(backup, "_supervise_backup_process", lambda seconds: budgets.append(seconds) or 0)
    assert backup.main() == 0
    assert budgets == [3590]


def test_sigterm_while_waiting_keeps_first_deferral_alert(admission, monkeypatch):
    monkeypatch.setattr(gate, "_pressure_reason", lambda *args: "swap too high")
    monkeypatch.setattr(gate.time, "sleep", lambda seconds: signal.raise_signal(signal.SIGTERM))
    with pytest.raises(SystemExit, match="143"):
        backup.main()
    assert "swap too high" in job_alerts.active_alerts()["backup-daily"]


def test_sigkill_mid_wait_leaves_durable_alert(admission):
    path, _ = admission
    read_fd, write_fd = os.pipe()
    script = """
import os, signal
from brainlayer import backup_daily as backup, backup_heavy_gate as gate
gate._pressure_reason = lambda *args: 'swap too high'
def wait(seconds):
    os.write(int(os.environ['BACKUP_TEST_READY_FD']), b'w')
    signal.pause()
gate.time.sleep = wait
backup._supervise_backup_process = lambda *args: (_ for _ in ()).throw(AssertionError('backup started'))
backup.main()
"""
    env = dict(os.environ, BACKUP_TEST_READY_FD=str(write_fd), PYTHONPATH=str(Path.cwd() / "src"))
    child = subprocess.Popen([sys.executable, "-c", script], env=env, pass_fds=(write_fd,), stdout=subprocess.DEVNULL)
    os.close(write_fd)
    try:
        assert select.select([read_fd], [], [], 10)[0], "child never reached the wait"
        assert os.read(read_fd, 1) == b"w"
        assert "since " in job_alerts.active_alerts(path / "alerts.json")["backup-daily"]
        child.kill()
        assert child.wait(timeout=5) == -signal.SIGKILL
        assert "swap too high" in job_alerts.active_alerts(path / "alerts.json")["backup-daily"]
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        os.close(read_fd)
