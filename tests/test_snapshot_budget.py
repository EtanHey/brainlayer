"""The weekly backup-snapshot budget is sized from the database it has to copy.

Evidence: MBP night of 2026-10-20 (orchestrator bl-lead report ``backup-rc124-measure.md``). The
calendar killed backup-snapshot at exactly 2700.3 s (rc=124) on a 20.75 GB raw snapshot:

    VACUUM INTO             03:45:11 -> 03:46:05      54 s
    full integrity_check    03:46:05 -> 03:55:03     538 s
    Python gzip -6 (live)   03:55:03 -> 04:30:10  >=2107 s  (killed at 9.68 GB; finished nights 8.9-9.8 GB)
    gunzip -t verify (live)  36 s standalone x ~4 live slowdown ~144 s
                                                  ----------
                                                  >=2843 s  = >=137 s/GB
"""

import datetime as dt

import pytest

from brainlayer import backup_daily as backup
from brainlayer import maintenance_calendar as calendar
from brainlayer import maintenance_control

GB = 1_000_000_000
MEASURED_20_75_GB_PHASES_S = 54 + 538 + 2107 + 144  # lower bound: gzip was still running at the kill
HEALTHY_WEEKLY_S = 815  # 10-17: maintenance-weekly ok, 04:21:19 -> 04:34:54


@pytest.fixture
def db_of_size(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_BACKUP_SNAPSHOT_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(tmp_path / "missing.env"))

    def make(size_bytes):
        db = tmp_path / "brainlayer.db"
        with db.open("wb") as handle:
            handle.truncate(size_bytes)  # sparse: no real bytes are written
        monkeypatch.setenv("BRAINLAYER_DB", str(db))
        return db

    return make


def snapshot_step(tmp_path):
    plan = calendar.calendar_steps("python", "cli", tmp_path, dt.date(2026, 10, 18))
    return next(s for s in plan.select() if s.name == "backup-snapshot")


def test_21_gb_budget_covers_the_measured_live_phases_with_margin(db_of_size, tmp_path):
    db_of_size(21 * GB)
    measured_s_per_gb = MEASURED_20_75_GB_PHASES_S / 20.75
    assert measured_s_per_gb < 140
    need_21_gb = measured_s_per_gb * 21
    budget = backup.default_snapshot_timeout_seconds()
    assert budget == 4200  # 21 GB x 200 s/GB
    assert budget >= 1.4 * need_21_gb  # >=40% headroom over a lower bound
    assert budget > 2700  # the budget that killed 10-20

    # 03:45:10 is when the 10-20 snapshot started (index ran its full 30 minutes).
    bounded = calendar._bounded_snapshot_step(snapshot_step(tmp_path), dt.datetime(2026, 10, 20, 3, 45, 10), 15)
    assert bounded.timeout == 4200


def test_budget_has_the_old_floor_and_a_ceiling_that_keeps_weekly(db_of_size):
    db_of_size(1_605_632_000)  # the M1 database, 1.6 GB
    assert backup.default_snapshot_timeout_seconds() == backup.SNAPSHOT_TIMEOUT_FLOOR_SECONDS == 2700
    db_of_size(40 * GB)
    assert backup.default_snapshot_timeout_seconds() == backup.SNAPSHOT_TIMEOUT_CEILING_SECONDS == 5400


def test_missing_database_falls_back_to_the_floor(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_DB", str(tmp_path / "absent.db"))
    assert backup.default_snapshot_timeout_seconds() == 2700


@pytest.mark.parametrize(
    ("db_gb", "weekly_hold_at_least"),
    [
        (21, 3290),  # snapshot ends 04:55:10; the hold is 06:00 - 04:55:10 - 600 s reserve
        (27, 2090),  # ceiling: snapshot ends 05:15:10
        (60, 2090),  # far past the ceiling: still 05:15:10
    ],
)
def test_weekly_keeps_its_window_when_the_snapshot_uses_its_whole_budget(
    db_of_size, tmp_path, db_gb, weekly_hold_at_least
):
    db_of_size(db_gb * GB)
    start = dt.datetime(2026, 10, 20, 3, 45, 10)
    bounded = calendar._bounded_snapshot_step(snapshot_step(tmp_path), start, 15)
    weekly_start = start + dt.timedelta(seconds=bounded.timeout + 15)  # plus group shutdown
    hold = maintenance_control.hold_budget(weekly_start)
    assert hold >= weekly_hold_at_least - 15
    assert hold >= 2 * HEALTHY_WEEKLY_S
    assert weekly_start.time() < dt.time(5, 20)


def test_06_00_clamp_still_wins_over_the_size_budget(db_of_size, tmp_path):
    db_of_size(21 * GB)
    bounded = calendar._bounded_snapshot_step(snapshot_step(tmp_path), dt.datetime(2026, 10, 20, 5, 0), 15)
    assert bounded.timeout == 3600 - 15 - 1


def test_internal_deadline_fires_before_the_calendar_kill_so_a_receipt_is_written(db_of_size, tmp_path):
    db_of_size(21 * GB)
    bounded = calendar._bounded_snapshot_step(snapshot_step(tmp_path), dt.datetime(2026, 10, 20, 3, 45, 10), 15)
    internal = int(bounded.env["BRAINLAYER_BACKUP_TIMEOUT_SECONDS"])
    assert int(bounded.env["BRAINLAYER_BACKUP_CALENDAR_SNAPSHOT_BUDGET_SECONDS"]) == internal
    # The supervisor waits `internal`, sends SIGALRM, waits up to 10 s, then writes its receipt.
    assert calendar.backup_deadline(bounded) - internal >= 10 + 30
    assert internal == bounded.timeout - backup.SNAPSHOT_RECEIPT_RESERVE_SECONDS


def test_entrypoint_size_default_never_exceeds_the_calendar_internal_budget(db_of_size, tmp_path, monkeypatch):
    db_of_size(21 * GB)
    bounded = calendar._bounded_snapshot_step(snapshot_step(tmp_path), dt.datetime(2026, 10, 20, 3, 45, 10), 15)
    for key in ("BRAINLAYER_BACKUP_TIMEOUT_SECONDS", "BRAINLAYER_BACKUP_CALENDAR_SNAPSHOT_BUDGET_SECONDS"):
        monkeypatch.setenv(key, bounded.env[key])
    monkeypatch.setenv("BRAINLAYER_BACKUP_PHASE", "snapshot")
    observed = []
    monkeypatch.setattr(backup, "_supervise_backup_process", lambda timeout: observed.append(timeout) or 0)
    monkeypatch.setattr("brainlayer.job_alerts.report", lambda *a, **kw: None)
    assert backup.main() == 0
    assert observed == [int(bounded.env["BRAINLAYER_BACKUP_TIMEOUT_SECONDS"])]
    assert observed[0] < bounded.timeout


def test_entrypoint_without_calendar_uses_the_size_default(db_of_size, monkeypatch):
    db_of_size(21 * GB)
    monkeypatch.delenv("BRAINLAYER_BACKUP_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("BRAINLAYER_BACKUP_CALENDAR_SNAPSHOT_BUDGET_SECONDS", raising=False)
    monkeypatch.setenv("BRAINLAYER_BACKUP_PHASE", "snapshot")
    observed = []
    monkeypatch.setattr(backup, "_supervise_backup_process", lambda timeout: observed.append(timeout) or 0)
    monkeypatch.setattr("brainlayer.job_alerts.report", lambda *a, **kw: None)
    assert backup.main() == 0
    assert observed == [4200]


@pytest.mark.parametrize("source", ["process", "env-file"])
def test_explicit_snapshot_timeout_override_still_wins(db_of_size, tmp_path, monkeypatch, source):
    db_of_size(21 * GB)
    if source == "process":
        monkeypatch.setenv("BRAINLAYER_BACKUP_SNAPSHOT_TIMEOUT_SECONDS", "1234")
    else:
        env_file = tmp_path / "brainlayer.env"
        env_file.write_text("BRAINLAYER_BACKUP_SNAPSHOT_TIMEOUT_SECONDS=1234\n")
        monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(env_file))
    bounded = calendar._bounded_snapshot_step(snapshot_step(tmp_path), dt.datetime(2026, 10, 20, 3, 45, 10), 15)
    assert bounded.timeout == 1234
    assert int(bounded.env["BRAINLAYER_BACKUP_TIMEOUT_SECONDS"]) < 1234
    if source == "process":
        monkeypatch.delenv("BRAINLAYER_BACKUP_TIMEOUT_SECONDS", raising=False)
        monkeypatch.delenv("BRAINLAYER_BACKUP_CALENDAR_SNAPSHOT_BUDGET_SECONDS", raising=False)
        monkeypatch.setenv("BRAINLAYER_BACKUP_PHASE", "snapshot")
        observed = []
        monkeypatch.setattr(backup, "_supervise_backup_process", lambda timeout: observed.append(timeout) or 0)
        monkeypatch.setattr("brainlayer.job_alerts.report", lambda *a, **kw: None)
        assert backup.main() == 0
        assert observed == [1234]


def test_tiny_window_reserves_proportionally_and_defers_when_nothing_fits(db_of_size, tmp_path):
    db_of_size(21 * GB)
    step = snapshot_step(tmp_path)
    bounded = calendar._bounded_snapshot_step(step, dt.datetime(2026, 10, 18, 5, 59), 2)
    assert bounded.timeout == 57
    assert 1 <= int(bounded.env["BRAINLAYER_BACKUP_TIMEOUT_SECONDS"]) < 57
    assert calendar._bounded_snapshot_step(step, dt.datetime(2026, 10, 18, 5, 59, 56), 2) is None
