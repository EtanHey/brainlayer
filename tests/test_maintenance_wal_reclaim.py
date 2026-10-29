"""Real WAL/VACUUM coverage with disposable databases and simulated service custody."""

import datetime as dt
import json
import sqlite3
import time
from types import SimpleNamespace

import pytest

from brainlayer import maintenance, maintenance_control, wal_checkpoint


@pytest.fixture
def weekly(tmp_path, monkeypatch):
    db = tmp_path / "brainlayer.db"
    resident = sqlite3.connect(db, isolation_level=None)
    resident.execute("PRAGMA journal_mode=WAL")
    resident.execute("PRAGMA wal_autocheckpoint=0")
    resident.execute("CREATE TABLE chunks(id INTEGER PRIMARY KEY, content BLOB)")
    resident.executemany("INSERT INTO chunks(content) VALUES (?)", [(b"x" * 4096,)] * 100)
    resident.execute("DELETE FROM chunks WHERE id > 50")
    events = []
    hold = SimpleNamespace(deadline=maintenance_control.hold_clock() + 3600)

    def close():
        events.append(("resume", wal_checkpoint.get_wal_size(str(db))))
        events.extend(maintenance.DEFAULT_SERVICES)

    hold.close = close

    def quiesce(_services, booted_out, *_args):
        events.append("hold")
        booted_out["_supervisor_hold"] = hold

    config = maintenance.MaintenanceConfig(
        db_path=db,
        queue_dir=tmp_path / "queue",
        quarantine_root=tmp_path / "quarantine",
        log_path=tmp_path / "maintenance.log",
        backup_staging_dir=tmp_path / "backup",
        now_fn=lambda: dt.datetime(2026, 10, 29, 4, 0),
    )
    receipt = {"snapshot": str(tmp_path / "snapshot.gz")}
    for name in ("_assert_running_merged_code", "_run_gates", "_emit_telemetry"):
        monkeypatch.setattr(maintenance, name, lambda *_a, **_kw: None)
    monkeypatch.setattr(maintenance, "_service_is_loaded", lambda *_a: True)
    monkeypatch.setattr(maintenance, "_quiesce_services", quiesce)
    monkeypatch.setattr(maintenance, "_weekly_backup", lambda _c: receipt)
    monkeypatch.setattr(maintenance, "_recent_weekly_verified_backup", lambda _c: receipt)
    monkeypatch.setattr(maintenance, "quarantine_stale_queue_files", lambda **_kw: maintenance.StaleQueueResult())
    monkeypatch.setattr(wal_checkpoint, "checkpoint_lock_path", lambda _db: tmp_path / "checkpoint.lock")
    try:
        yield config, resident, events, hold
    finally:
        resident.close()


def reclaim_row(config):
    rows = [json.loads(line) for line in config.log_path.read_text().splitlines()]
    row = next(row for row in rows if row.get("phase") in {"post-vacuum", "vacuum-failure"})
    assert row["mode"] == "TRUNCATE"
    assert row["duration_ms"] >= 0
    assert row["wal_before_bytes"] > 0
    return row


def test_real_vacuum_reclaims_wal_before_resident_services_resume(weekly):
    config, resident, events, _hold = weekly
    result = maintenance.run_maintenance("full", config=config)
    row = reclaim_row(config)
    assert row["phase"] == "post-vacuum"
    assert row["busy"] == 0 and row["deferred"] is False
    assert row["wal_after_bytes"] == 0
    assert result.checkpoint == (0, 0, 0)
    assert events == ["hold", ("resume", 0), *maintenance.DEFAULT_SERVICES]
    assert resident.execute("SELECT COUNT(*) FROM chunks").fetchone() == (50,)
    assert resident.execute("PRAGMA freelist_count").fetchone() == (0,)


def test_reader_transaction_defers_bounded_reclaim_then_release_allows_truncate(weekly):
    config, resident, events, _hold = weekly
    resident.execute("BEGIN")
    resident.execute("SELECT COUNT(*) FROM chunks").fetchone()
    started = time.monotonic()
    result = maintenance.run_maintenance("full", config=config)
    assert time.monotonic() - started < 2
    row = reclaim_row(config)
    assert row["busy"] == 1 and row["deferred"] is True
    assert row["wal_after_bytes"] > 0
    assert result.checkpoint[0] == 1
    assert events == ["hold", ("resume", row["wal_after_bytes"]), *maintenance.DEFAULT_SERVICES]
    resident.execute("ROLLBACK")
    assert wal_checkpoint.checkpoint(str(config.db_path), "TRUNCATE") == (0, 0, 0)
    assert wal_checkpoint.get_wal_size(str(config.db_path)) == 0


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_fault_after_vacuum_commit_keeps_original_failure_and_restores(weekly, monkeypatch, cleanup_fails):
    config, resident, events, _hold = weekly
    vacuum = maintenance._vacuum
    original = RuntimeError("fault after VACUUM commit")
    connect = sqlite3.connect
    closed = []

    class FaultConnection:
        def __init__(self, *args, **kwargs):
            self.conn = connect(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.conn, name)

        def execute(self, sql):
            self.conn.execute(sql)
            raise original

        def close(self):
            self.conn.close()
            closed.append(True)

    def fail_after_commit(*args, **kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(sqlite3, "connect", FaultConnection)
            vacuum(*args, **kwargs)

    monkeypatch.setattr(maintenance, "_vacuum", fail_after_commit)
    checkpoint = maintenance.checkpoint

    def check_cleanup(db, mode, **kwargs):
        if mode == "TRUNCATE":
            assert closed == [True], "cleanup ran before VACUUM connection closed"
            if cleanup_fails:
                raise sqlite3.OperationalError("cleanup fault")
        return checkpoint(db, mode, **kwargs)

    monkeypatch.setattr(maintenance, "checkpoint", check_cleanup)
    with pytest.raises(RuntimeError) as caught:
        maintenance.run_maintenance("full", config=config)
    assert caught.value is original
    row = reclaim_row(config)
    assert row["phase"] == "vacuum-failure" and row["deferred"] is cleanup_fails
    assert events[-3:] == list(maintenance.DEFAULT_SERVICES)
    assert resident.execute("SELECT COUNT(*) FROM chunks").fetchone() == (50,)
    assert resident.execute("PRAGMA freelist_count").fetchone() == (0,)
    if not cleanup_fails:
        assert events[1] == ("resume", 0)


def test_almost_expired_hold_skips_reclaim_without_extending_deadline(weekly, monkeypatch):
    config, _resident, events, hold = weekly
    vacuum = maintenance._vacuum
    deadline = hold.deadline

    def finish_at_cutoff(*args, **kwargs):
        result = vacuum(*args, **kwargs)
        monkeypatch.setattr(maintenance_control, "hold_clock", lambda: deadline - maintenance.HOLD_UNWIND_S + 1)
        return result

    monkeypatch.setattr(maintenance, "_vacuum", finish_at_cutoff)
    checkpoint = maintenance.checkpoint

    def no_truncate(db, mode, **kwargs):
        assert mode == "FULL", "reclaim started inside restore reserve"
        return checkpoint(db, mode, **kwargs)

    monkeypatch.setattr(maintenance, "checkpoint", no_truncate)
    maintenance.run_maintenance("full", config=config)
    row = reclaim_row(config)
    assert row["deferred"] is True and row["busy"] is None
    assert row["reason"] == "hold-unwind-reserve"
    assert hold.deadline == deadline
    assert events[-3:] == list(maintenance.DEFAULT_SERVICES)


def test_checkpoint_guard_contention_defers_without_waiting_under_hold(weekly, monkeypatch):
    config, _resident, events, _hold = weekly
    vacuum = maintenance._vacuum
    guard = wal_checkpoint.checkpoint_guard(config.db_path, blocking=False)

    def hold_guard_after_vacuum(*args, **kwargs):
        result = vacuum(*args, **kwargs)
        assert guard.__enter__()
        return result

    monkeypatch.setattr(maintenance, "_vacuum", hold_guard_after_vacuum)
    try:
        maintenance.run_maintenance("full", config=config)
        row = reclaim_row(config)
        assert row["reason"] == "checkpoint-guard-busy" and row["deferred"] is True
        assert row["duration_ms"] < 1000
        assert events[-3:] == list(maintenance.DEFAULT_SERVICES)
    finally:
        guard.__exit__(None, None, None)
