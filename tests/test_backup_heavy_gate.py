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

REAL_DISK_REASON = gate._disk_reason
REAL_EXPECTED_PHASE_SECONDS = gate.expected_phase_seconds


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
        monkeypatch.delenv("BRAINLAYER_BACKUP_UPLOAD_SNAPSHOT", raising=False)
        monkeypatch.setenv("BRAINLAYER_BACKUP_LOG_PATH", str(path / "journal.log"))
        clock = [0.0]
        monkeypatch.setattr(gate.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(gate.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
        monkeypatch.setattr(gate, "_pressure_reason", lambda *args: None)
        monkeypatch.setattr(gate, "_disk_reason", lambda host: None)
        monkeypatch.setattr(gate, "expected_phase_seconds", lambda: 60)
        monkeypatch.setattr(backup, "_configured_backup_timeout_seconds", lambda: 3600)
        yield path, clock


@pytest.fixture
def disk_admission(admission, monkeypatch):
    path, clock = admission
    free_gib = [58]
    monkeypatch.delenv("BRAINLAYER_BACKUP_HEAVY_HOST_FREE_GIB", raising=False)
    monkeypatch.setenv("BRAINLAYER_BACKUP_STAGING_DIR", str(path))
    monkeypatch.setattr(backup, "get_db_path", lambda: path / "synthetic.db")
    monkeypatch.setattr(backup, "_database_logical_size_bytes", lambda db: 16 * 1024**3)
    monkeypatch.setattr(backup, "_estimated_gzip_bytes", lambda staging, size: 8 * 1024**3)
    monkeypatch.setattr(backup, "_should_run_full_verify", lambda date: False)

    def space(volume, *, include_available):
        assert volume == path
        assert include_available is False  # Purgeable capacity cannot satisfy the reserve.
        free = int(free_gib[0] * 1024**3)
        return backup.disk_space.DiskSpace(200 * 1024**3, free, 200 * 1024**3 - free, "fake", "fake")

    monkeypatch.setattr(backup.disk_space, "measure_disk_space", space)
    monkeypatch.setattr(gate, "_disk_reason", REAL_DISK_REASON)
    return path, clock, free_gib


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
    monkeypatch.setattr(backup, "_configured_backup_timeout_seconds", lambda: 80)  # hard budget, 60s reserved
    monkeypatch.setattr(backup, "_supervise_backup_process", lambda *a, **kw: pytest.fail("backup started"))
    assert backup.main() == 75
    assert clock[0] == 20
    assert lock.is_dir()
    assert "remaining phase window exhausted" in job_alerts.active_alerts()["backup-daily"]
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


def test_pressure_window_ends_runs_under_lock_and_clears_alert_on_success(admission, monkeypatch):
    path, clock = admission
    monkeypatch.setattr(gate, "_pressure_reason", lambda *args: "swap too high")

    def run(seconds):
        assert clock[0] == 20
        assert (path / "lock").is_dir()
        assert "swap too high" in job_alerts.active_alerts()["backup-daily"]
        assert seconds == 3580
        return 0

    monkeypatch.setattr(backup, "_supervise_backup_process", run)
    assert backup.main() == 0
    assert not job_alerts.active_alerts()
    assert not (path / "lock").exists()
    assert "running past window" in (path / "heavy.log").read_text()
    assert "ignoring swap/load: swap too high" in (path / "heavy.log").read_text()


def test_past_window_still_waits_for_another_lanes_lock(admission, monkeypatch):
    path, clock = admission
    lock = path / "lock"
    lock.mkdir()
    monkeypatch.setattr(gate, "_pressure_reason", lambda *args: "load too high")

    def sleep(seconds):
        clock[0] += seconds
        if clock[0] == 30:
            lock.rmdir()  # The simulated owner releases, never the backup.

    def run(seconds):
        assert clock[0] == 30
        assert lock.is_dir()
        assert "backup-daily" in job_alerts.active_alerts()
        return 1

    monkeypatch.setattr(gate.time, "sleep", sleep)
    monkeypatch.setattr(backup, "_supervise_backup_process", run)
    assert backup.main() == 1
    assert not lock.exists()
    assert "backup-daily" in job_alerts.active_alerts()  # A failed forced run never clears it.


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


def test_m1_at_37_gib_still_runs_backup(disk_admission, monkeypatch):
    path, _, free_gib = disk_admission
    free_gib[0] = 37
    monkeypatch.setenv("BRAINLAYER_BACKUP_HEAVY_HOST", "m1")
    monkeypatch.setattr(gate.sys, "platform", "darwin")
    monkeypatch.setattr(gate, "_sysctl", lambda key: "used = 1M")
    monkeypatch.setattr(gate.os, "getloadavg", lambda: (1, 0, 0))
    monkeypatch.setattr(backup, "_should_run_full_verify", lambda date: True)  # peak 34 GiB, no extra M1 reserve
    # Remove the fixture's pressure fake: exercise the production resource check.
    monkeypatch.setattr(gate, "_pressure_reason", REAL_PRESSURE_REASON)
    ran = []
    monkeypatch.setattr(backup, "_supervise_backup_process", lambda seconds: ran.append(seconds) or 0)
    assert backup.main() == 0
    assert ran == [3600]
    assert not (path / "lock").exists()


REAL_PRESSURE_REASON = gate._pressure_reason


def test_m4_disk_floor_defers_and_alerts_even_past_pressure_window(disk_admission, monkeypatch):
    path, clock, _ = disk_admission
    monkeypatch.setattr(backup, "_configured_backup_timeout_seconds", lambda: 100)
    monkeypatch.setattr(backup, "_supervise_backup_process", lambda *args: pytest.fail("disk floor waived"))
    assert backup.main() == 75
    assert clock[0] == 40  # Includes retries beyond the 20-second pressure window.
    assert "needs 66.001 GiB, has 58.000 GiB" in job_alerts.active_alerts()["backup-daily"]
    assert not (path / "lock").exists()
    assert "running past window" not in (path / "heavy.log").read_text()


def test_full_verify_peak_streams_without_second_compressed_copy(disk_admission, monkeypatch):
    _, _, free_gib = disk_admission
    free_gib[0] = 70
    assert REAL_DISK_REASON("m4") is None  # ordinary: 16 raw + 8 compressed + 2 floor + 40 reserve
    monkeypatch.setattr(backup, "_should_run_full_verify", lambda date: True)
    assert REAL_DISK_REASON("m4") is None
    free_gib[0] = 66
    assert "needs 66.001 GiB, has 66.000 GiB" in REAL_DISK_REASON("m4")


def test_disk_floor_rechecks_under_lock(disk_admission, monkeypatch):
    path, clock, free_gib = disk_admission
    free_gib[0] = 100
    real_measure = backup.disk_space.measure_disk_space
    observations = []

    def measure(volume, **kwargs):
        observations.append((path / "lock").exists())
        free_gib[0] = 58 if len(observations) == 2 else 100
        return real_measure(volume, **kwargs)

    monkeypatch.setattr(backup.disk_space, "measure_disk_space", measure)
    monkeypatch.setattr(backup, "_supervise_backup_process", lambda seconds: 0)
    assert backup.main() == 0
    assert observations == [False, True, False, True]
    assert clock[0] == 10
    assert not job_alerts.active_alerts()


def test_disk_shortage_refreshes_existing_pressure_alert_without_clearing(disk_admission, monkeypatch):
    _, clock, free_gib = disk_admission
    free_gib[0] = 100
    monkeypatch.setattr(gate, "_pressure_reason", lambda *args: "swap too high")
    monkeypatch.setattr(backup, "_configured_backup_timeout_seconds", lambda: 100)

    def sleep(seconds):
        assert "backup-daily" in job_alerts.active_alerts()
        clock[0] += seconds
        free_gib[0] = 58

    monkeypatch.setattr(gate.time, "sleep", sleep)
    assert backup.main() == 75
    assert "needs 66.001 GiB, has 58.000 GiB" in job_alerts.active_alerts()["backup-daily"]


def test_host_floor_is_configurable_and_zero_is_valid(disk_admission, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_BACKUP_HEAVY_HOST_FREE_GIB", "0")
    assert REAL_DISK_REASON("m4") is None
    monkeypatch.setenv("BRAINLAYER_BACKUP_HEAVY_HOST_FREE_GIB", "60")
    assert "needs 86.001 GiB" in REAL_DISK_REASON("m4")


def test_disk_boundary_accepts_exact_peak_plus_reserve(disk_admission):
    _, _, free_gib = disk_admission
    free_gib[0] = 66 + 1 / 1024  # Includes the peak helper's 1 MiB copy allowance.
    assert REAL_DISK_REASON("m4") is None
    free_gib[0] -= 1 / 1024**3
    assert REAL_DISK_REASON("m4") is not None


def test_apfs_clone_budget_admits_m4_without_lowering_its_host_floor(disk_admission, monkeypatch):
    _, _, free_gib = disk_admission
    free_gib[0] = 60
    monkeypatch.setattr(backup.backup_clone, "supported", lambda *args: True)
    assert REAL_DISK_REASON("m4") is None
    free_gib[0] = 50
    assert "host floor" in REAL_DISK_REASON("m4")


def upload_receipt(path, *, streaming=False, name="2026-11-22.db.zst"):
    archive = path / name
    with archive.open("wb") as stream:
        stream.truncate(8 * 1024**3)  # Sparse metadata fixture; no payload is read or uploaded.
    row = {
        "snapshot": str(archive),
        "db": str(path / "synthetic.db"),
        "local_verified": True,
        "raw_sha256": "a" * 64 if streaming else None,
    }
    backup._append_json_log(backup._backup_log_path(None, db_path=path / "synthetic.db"), row)
    return archive


@pytest.mark.parametrize(("phase", "expected"), [("upload", 0), ("all", 75), ("snapshot", 75)])
def test_upload_at_60_gib_is_admitted_but_snapshot_is_not(disk_admission, monkeypatch, phase, expected):
    path, _, free_gib = disk_admission
    free_gib[0] = 60
    upload_receipt(path)
    monkeypatch.setenv("BRAINLAYER_BACKUP_PHASE", phase)
    monkeypatch.setattr(backup, "_configured_backup_timeout_seconds", lambda: 100)
    monkeypatch.setattr(backup, "_supervise_backup_process", lambda seconds: 0)
    assert backup.main() == expected


def test_upload_streaming_full_verification_needs_no_new_payload_files(disk_admission, monkeypatch):
    path, _, free_gib = disk_admission
    free_gib[0] = 43
    upload_receipt(path, streaming=True)
    monkeypatch.setenv("BRAINLAYER_BACKUP_PHASE", "upload")
    monkeypatch.setattr(backup, "_should_run_full_verify", lambda date: True)
    monkeypatch.setattr(backup, "_database_logical_size_bytes", lambda db: pytest.fail("upload opened DB"))
    monkeypatch.setattr(backup, "_estimated_gzip_bytes", lambda *args: pytest.fail("upload budgeted compression"))
    assert REAL_DISK_REASON("m4") is None


def test_upload_legacy_full_verification_reserves_only_new_restore_and_download(disk_admission, monkeypatch):
    path, _, free_gib = disk_admission
    free_gib[0] = 60
    upload_receipt(path)
    monkeypatch.setenv("BRAINLAYER_BACKUP_PHASE", "upload")
    monkeypatch.setattr(backup, "_should_run_full_verify", lambda date: True)
    assert "needs 66.001 GiB" in REAL_DISK_REASON("m4")  # 16 restore + 8 download + 2 + 40.


def test_upload_budget_covers_batch_candidates_but_honors_selected_target(disk_admission, monkeypatch):
    path, _, free_gib = disk_admission
    free_gib[0] = 60
    selected = upload_receipt(path, streaming=True)
    upload_receipt(path, name="2026-11-21.db.zst")
    monkeypatch.setenv("BRAINLAYER_BACKUP_PHASE", "upload")
    monkeypatch.setattr(backup, "_should_run_full_verify", lambda date: True)
    assert "needs 66.001 GiB" in REAL_DISK_REASON("m4")
    monkeypatch.setenv("BRAINLAYER_BACKUP_UPLOAD_SNAPSHOT", selected.name)
    assert REAL_DISK_REASON("m4") is None


def test_alert_tracks_disk_then_pressure_then_lock(disk_admission, monkeypatch):
    path, clock, free_gib = disk_admission
    lock = path / "lock"
    lock.mkdir()
    monkeypatch.setenv("BRAINLAYER_BACKUP_HEAVY_WINDOW_SECONDS", "100")
    monkeypatch.setattr(gate, "_pressure_reason", lambda *args: "swap too high" if clock[0] < 20 else None)

    def sleep(seconds):
        assert {0: "needs", 10: "swap too high", 20: "lock held"}[clock[0]] in job_alerts.active_alerts()[
            "backup-daily"
        ]
        clock[0] += seconds
        free_gib[0] = 100
        if clock[0] == 30:
            lock.rmdir()

    monkeypatch.setattr(gate.time, "sleep", sleep)
    monkeypatch.setattr(backup, "_supervise_backup_process", lambda seconds: 0)
    assert backup.main() == 0
    assert not job_alerts.active_alerts()


@pytest.mark.parametrize(("phase", "seconds"), [("snapshot", 2400), ("upload", 1800), ("all", 4200)])
def test_expected_duration_is_phase_specific(admission, monkeypatch, phase, seconds):
    monkeypatch.setenv("BRAINLAYER_BACKUP_PHASE", phase)
    monkeypatch.delenv(f"BRAINLAYER_BACKUP_HEAVY_EXPECTED_{phase.upper()}_SECONDS", raising=False)
    assert REAL_EXPECTED_PHASE_SECONDS() == seconds
    monkeypatch.setenv(f"BRAINLAYER_BACKUP_HEAVY_EXPECTED_{phase.upper()}_SECONDS", "123")
    assert REAL_EXPECTED_PHASE_SECONDS() == 123


def test_late_lock_release_cannot_start_a_phase_without_its_expected_budget(admission, monkeypatch):
    path, clock = admission
    lock = path / "lock"
    lock.mkdir()
    monkeypatch.setenv("BRAINLAYER_BACKUP_PHASE", "upload")
    monkeypatch.delenv("BRAINLAYER_BACKUP_HEAVY_EXPECTED_UPLOAD_SECONDS", raising=False)
    monkeypatch.setattr(gate, "expected_phase_seconds", REAL_EXPECTED_PHASE_SECONDS)
    monkeypatch.setattr(backup, "_configured_backup_timeout_seconds", lambda: 2000)

    def sleep(seconds):
        clock[0] += seconds
        if clock[0] >= 210:
            lock.rmdir()

    monkeypatch.setattr(gate.time, "sleep", sleep)
    monkeypatch.setattr(backup, "_supervise_backup_process", lambda seconds: 0)
    assert backup.main() == 75
    assert clock[0] == 200  # Reserve 1800 seconds for upload, never the old 60-second tail.
    assert "phase window exhausted" in job_alerts.active_alerts()["backup-daily"]
    assert lock.is_dir()


def test_initial_budget_shorter_than_expected_phase_defers_and_alerts(admission, monkeypatch):
    path, _ = admission
    monkeypatch.delenv("BRAINLAYER_BACKUP_HEAVY_EXPECTED_ALL_SECONDS", raising=False)
    monkeypatch.setattr(gate, "expected_phase_seconds", REAL_EXPECTED_PHASE_SECONDS)
    monkeypatch.setattr(backup, "_configured_backup_timeout_seconds", lambda: 1000)
    monkeypatch.setattr(backup, "_supervise_backup_process", lambda seconds: pytest.fail("short run started"))
    assert backup.main() == 75
    assert "needs 4200s expected work, has 1000s" in job_alerts.active_alerts()["backup-daily"]
    assert not (path / "lock").exists()


def test_unmeasurable_disk_is_not_waived_past_window(disk_admission, monkeypatch):
    monkeypatch.setattr(backup, "_configured_backup_timeout_seconds", lambda: 100)
    monkeypatch.setattr(backup, "_supervise_backup_process", lambda *args: pytest.fail("unknown disk admitted"))

    def fail(*args, **kwargs):
        raise OSError("synthetic disk probe failure")

    monkeypatch.setattr(backup.disk_space, "measure_disk_space", fail)
    assert backup.main() == 75
    assert "disk space measurement unavailable" in job_alerts.active_alerts()["backup-daily"]


@pytest.mark.parametrize("floor", ["-1", "nan", "inf"])
def test_invalid_host_floor_fails_closed(disk_admission, monkeypatch, floor):
    path, _, _ = disk_admission
    monkeypatch.setenv("BRAINLAYER_BACKUP_HEAVY_HOST_FREE_GIB", floor)
    with pytest.raises(ValueError):
        backup.main()
    assert not (path / "lock").exists()


def test_no_lock_churn_when_pressure_is_closed(admission, monkeypatch):
    path, _ = admission
    monkeypatch.setattr(gate, "_pressure_reason", lambda *args: "swap too high")
    monkeypatch.setattr(backup, "_configured_backup_timeout_seconds", lambda: 80)
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
gate._disk_reason = lambda host: None
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
