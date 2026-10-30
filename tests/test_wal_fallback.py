"""Independent reclamation on disposable WAL databases; no process control."""

import asyncio
import datetime as dt
import fcntl
import json
import os
import sqlite3
import time
from contextlib import asynccontextmanager

import pytest

from brainlayer import job_alerts
from brainlayer import maintenance_calendar as calendar
from brainlayer import wal_checkpoint as wal
from brainlayer.service_supervisor import Component, Supervisor
from brainlayer.supervisor_guard import component_gate
from tests.test_maintenance_calendar_completion import runner as calendar_runner  # noqa: F401
from tests.test_supervisor_jobs import short_path  # noqa: F401


@pytest.fixture
def database(tmp_path, monkeypatch):
    db = tmp_path / "db"
    control, staging = tmp_path / "supervisor", tmp_path / "backups"
    control.mkdir(mode=0o700)
    staging.mkdir()
    monkeypatch.setenv("BRAINLAYER_DB", str(db))
    monkeypatch.setenv("BRAINLAYER_SUPERVISOR_SOCKET", str(control / "control.sock"))
    monkeypatch.setenv("BRAINLAYER_BACKUP_STAGING_DIR", str(staging))
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(tmp_path / "alerts.json"))
    monkeypatch.setattr(wal, "checkpoint_lock_path", lambda _db: tmp_path / "checkpoint.lock")
    writer = sqlite3.connect(db, isolation_level=None)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("CREATE TABLE t(x)")
    writer.execute("INSERT INTO t VALUES (zeroblob(65536))")
    yield db, writer, control, staging
    writer.close()


def tick(**kwargs):
    return wal.periodic_checkpoint(high_water_bytes=4096, busy_timeout_seconds=0.02, **kwargs)


@pytest.mark.parametrize("phase", ["index", "maintenance-weekly"])
def test_calendar_dies_before_checkpoint_independent_tick_reclaims(database, request, monkeypatch, phase):
    runner = request.getfixturevalue("calendar_runner")
    scheduler_path = request.getfixturevalue("short_path") / "s"
    db, writer, _, _ = database
    executed = []

    @asynccontextmanager
    async def hold(_client, _guard, env, *_args):
        yield env

    async def execute(step, *_args, **_kwargs):
        executed.append(step.name)
        if phase == "maintenance-weekly":
            writer.execute("VACUUM")  # Committed, before fix 1's cleanup could execute.
        raise RuntimeError("calendar ended before reclamation")

    monkeypatch.setattr(calendar, "step_hold", hold)
    monkeypatch.setattr(calendar, "_execute_step", execute)
    plan = [calendar.Step(name, ("/unused",), 60) for name in (phase, "wal-checkpoint")]
    code = (
        "import pathlib; from brainlayer import wal_checkpoint as w; "
        "w.checkpoint_lock_path=lambda db:pathlib.Path(db).with_suffix('.checkpoint.lock'); "
        "w.periodic_checkpoint(high_water_bytes=4096, busy_timeout_seconds=0.02)"
    )
    child = Component(
        (os.sys.executable, "-c", code),
        env={key: value for key, value in os.environ.items() if key.startswith("BRAINLAYER_")}
        | {"PYTHONPATH": os.path.abspath("src")},
        interval=900,
        run_timeout=3,
        exit_timeout=0.1,
    )
    child.retry_at = time.monotonic() + 900

    async def exercise():
        scheduler = Supervisor({"wal-checkpoint-fallback": child}, scheduler_path)
        task = asyncio.create_task(scheduler.run())
        try:
            assert (
                await calendar.run_calendar(
                    plan,
                    runner.directory,
                    runner.logs,
                    now_fn=lambda: dt.datetime(2026, 10, 29, 3 if phase == "index" else 4, 30),
                )
                == 1
            )
            assert wal.get_wal_size(str(db)) > 4096 and not task.done()
            child.retry_at = 0  # The next independent supervisor tick is now eligible.
            async with asyncio.timeout(5):
                while child.last_exit_code is None:
                    await asyncio.sleep(0.02)
            assert child.last_exit_code == 0 and child.starts == 1
        finally:
            scheduler.stop.set()
            await task

    asyncio.run(exercise())
    result = json.loads(db.with_name("wal-fallback.json").read_text())
    assert result["outcome"] == "reclaimed" and result["attempts"] == 1
    assert result["wal_after_bytes"] == 0 and executed == [phase]
    assert writer.execute("SELECT length(x) FROM t").fetchone() == (65536,)


@pytest.mark.parametrize("gate", ["reader", "maintenance", "snapshot", "checkpoint"])
def test_bounded_defer_then_later_success(database, monkeypatch, gate):
    db, writer, control, staging = database
    reader = sqlite3.connect(db, isolation_level=None)
    if gate == "reader":
        reader.execute("BEGIN")
        reader.execute("SELECT * FROM t").fetchall()
        writer.execute("INSERT INTO t VALUES (1)")
    lock = None
    if gate in {"maintenance", "checkpoint"}:
        lock = component_gate(control, exclusive=True) if gate == "maintenance" else wal.checkpoint_guard(db)
        lock.__enter__()
    if gate == "snapshot":
        lock = (staging / ".backup.lock").open("a+b")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    monkeypatch.setattr(os, "kill", lambda *_a: pytest.fail("process control"))
    monkeypatch.setattr(os, "killpg", lambda *_a: pytest.fail("process control"))
    run = wal.subprocess.run

    def diagnostic_only(argv, **kwargs):
        assert argv[0] == "lsof", "only read-only holder diagnostics may spawn"
        return run(argv, **kwargs)

    monkeypatch.setattr(wal.subprocess, "run", diagnostic_only)
    try:
        started = time.monotonic()
        result = tick(now=1000)
        assert time.monotonic() - started < 1
        assert result["outcome"] == "deferred" and result["wal_after_bytes"] > 4096
        assert result["attempts"] == int(gate in {"reader", "checkpoint"})
        later = tick(now=2800)
        assert later["high_water_age_seconds"] == 1800
        assert later["holders"] is not None and "checkpointed_pages" in later
        assert "wal-checkpoint-fallback" in job_alerts.active_alerts()
    finally:
        reader.close()
        if gate == "snapshot":
            lock.close()
        elif lock:
            lock.__exit__(None, None, None)
    assert tick(now=3700)["outcome"] == "reclaimed"
    assert "wal-checkpoint-fallback" not in job_alerts.active_alerts()
    state = json.loads(db.with_name("wal-fallback.json").read_text())
    assert state["high_water_since"] is None and state["attempted_at"]


def test_low_water_does_not_open_database(database, monkeypatch):
    monkeypatch.setattr(wal, "checkpoint", lambda *_a, **_k: pytest.fail("checkpoint below threshold"))
    result = wal.periodic_checkpoint()
    assert result["outcome"] == "below-high-water" and result["attempts"] == 0


def test_default_high_water_makes_only_one_short_busy_attempt(database, monkeypatch):
    calls = []
    monkeypatch.setattr(wal, "get_wal_size", lambda _db: 256_000_000)
    monkeypatch.setattr(wal, "checkpoint", lambda *a, **k: calls.append((a, k)) or (1, 10, 5))
    result = wal.periodic_checkpoint()
    assert result["outcome"] == "deferred" and result["busy"] == 1
    assert result["log_pages"] == 10 and result["checkpointed_pages"] == 5
    assert len(calls) == 1 and calls[0][0][1] == "TRUNCATE"
    assert calls[0][1] == dict(guard_timeout_seconds=0, busy_timeout_seconds=0.1)
