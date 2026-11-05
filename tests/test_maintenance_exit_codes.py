"""#1040: deliberate deferrals exit 75; genuine failures exit a distinct, documented code.

Every test drives the production entry point (`maintenance.main`) through the real
`run_maintenance`, with only private boundaries injected: the launchctl/installer subprocess
runner, the burn drain batch, the latency clock, and a synthetic scratch database.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import sqlite3
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from brainlayer import job_alerts, maintenance
from brainlayer.drain import BurnDrainResult

DEFERRED = 75  # EX_TEMPFAIL
FAILED = 70  # EX_SOFTWARE
RUN_ID_ENV = "BRAINLAYER_MAINTENANCE_RUN_ID"
RUN_ID = "c" * 32


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """A private config, log and alert file; launchctl/installer calls are recorded, never run."""
    config = maintenance.MaintenanceConfig(
        db_path=tmp_path / "brainlayer.db",
        queue_dir=tmp_path / "queue",
        quarantine_root=tmp_path / "quarantine",
        log_path=tmp_path / "maintenance.log",
        repo_root=tmp_path,
        now_fn=lambda: dt.datetime(2026, 11, 5, 4, 5, tzinfo=dt.timezone.utc),
        quiet_window_start_hour=4,
        quiet_window_duration_minutes=120,
        idle_sample_seconds=0,
        recent_write_grace_seconds=0,
    )
    config.queue_dir.mkdir(parents=True)
    conn = sqlite3.connect(config.db_path)
    conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()
    monkeypatch.setenv("BRAINLAYER_MAINTENANCE_LOG_PATH", str(config.log_path))
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(tmp_path / "alerts.json"))
    monkeypatch.setenv(RUN_ID_ENV, RUN_ID)
    monkeypatch.delenv("BRAINLAYER_SUPERVISOR_SOCKET", raising=False)
    monkeypatch.setattr(maintenance, "PAUSE_SENTINEL_PATH", tmp_path / "pause.sentinel")
    monkeypatch.setattr(maintenance, "_launchd_dir_for_resume", lambda _root: tmp_path / "launchd")
    monkeypatch.setattr(maintenance, "_maintenance_keep_down_services", lambda: set())
    monkeypatch.setattr(maintenance, "_service_is_deliberately_paused", lambda _service: False)
    monkeypatch.setattr(maintenance, "collect_lsof_entries", lambda _paths: [])
    monkeypatch.setattr(maintenance, "_service_is_loaded", lambda _service, **_kw: False)
    calls: list[str] = []
    fail_resume: set[str] = set()

    def bootout(service):
        calls.append(f"bootout:{service}")
        return True

    def run_command(args, **kwargs):
        name = Path(args[0]).name
        if name == "install.sh":
            calls.append(f"resume:{args[1]}")
            if args[1] in fail_resume and kwargs.get("check"):
                raise subprocess.CalledProcessError(1, args, output="", stderr="ERROR: synthetic resume failure")
        elif args[:2] == ["launchctl", "bootstrap"]:
            calls.append("resume:fleet-watchdog")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(maintenance, "_bootout_service", bootout)
    monkeypatch.setattr(maintenance, "run_command", run_command)
    real = maintenance.run_maintenance
    monkeypatch.setattr(
        maintenance, "run_maintenance", lambda mode, *, dry_run: real(mode, config=config, dry_run=dry_run)
    )
    return SimpleNamespace(config=config, calls=calls, fail_resume=fail_resume, alerts=tmp_path / "alerts.json")


def _receipts(scratch, capsys):
    stdout = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    logged = json.loads(scratch.config.log_path.read_text().splitlines()[-1])
    return stdout, logged


def _assert_correlatable(record, *, outcome, exit_code):
    assert record["outcome"] == outcome
    assert record["exit_code"] == exit_code
    assert record["run_id"] == RUN_ID
    stamp = dt.datetime.fromisoformat(record["ts"])
    assert stamp.utcoffset() == dt.timedelta(0)


def test_light_resume_failure_is_a_failure_exit_not_tempfail(scratch, capsys):
    scratch.fail_resume.add("index")
    code = maintenance.main(["--light"])

    assert code == FAILED != DEFERRED
    # The original failure survives, and every booted-out service is still restored, in order.
    assert scratch.calls == [
        "bootout:fleet-watchdog",
        *(f"bootout:{s}" for s in maintenance.DEFAULT_SERVICES),
        *(f"resume:{s}" for s in maintenance.DEFAULT_SERVICES),
        "resume:fleet-watchdog",
    ]
    stdout, logged = _receipts(scratch, capsys)
    assert "failed to resume 1 launchd service: index" in stdout["reason"]
    assert "synthetic resume failure" in stdout["reason"]
    for record in (stdout, logged):
        assert record["status"] == "aborted"
        _assert_correlatable(record, outcome="failed", exit_code=FAILED)
    assert "maintenance-light" in job_alerts.active_alerts(scratch.alerts)


def test_deferral_that_cannot_restore_services_becomes_a_failure(scratch, capsys, monkeypatch):
    reason = "outside quiet window: now=05:59 start_hour=4 duration_minutes=120"

    def defer_mid_body(_db_path):
        raise maintenance.MaintenanceAbort(reason, code=DEFERRED)

    monkeypatch.setattr(maintenance, "_checkpoint_full", defer_mid_body)
    scratch.fail_resume.add("watch")

    assert maintenance.main(["--light"]) == FAILED
    stdout, _ = _receipts(scratch, capsys)
    assert stdout["reason"].startswith(reason + "; failed to resume 1 launchd service: watch")
    _assert_correlatable(stdout, outcome="failed", exit_code=FAILED)
    assert scratch.calls[-1] == "resume:fleet-watchdog"


def test_burn_drain_failure_is_a_failure_exit(scratch, capsys, monkeypatch):
    monkeypatch.setattr(maintenance, "burn_drain_once", lambda **_kw: BurnDrainResult(scanned_files=1, failed_files=1))

    assert maintenance.main(["--burn"]) == FAILED
    stdout, logged = _receipts(scratch, capsys)
    assert stdout["reason"] == "burn drain failed; queue files preserved"
    for record in (stdout, logged):
        _assert_correlatable(record, outcome="failed", exit_code=FAILED)
    assert scratch.calls[-1] == "resume:fleet-watchdog"


def test_pathological_search_latency_is_a_failure_exit(scratch, capsys, monkeypatch):
    ticks = iter(range(0, 10_000))
    # Each timed sample spans one deterministic "second": far past the 500 ms failure threshold.
    monkeypatch.setattr(maintenance.time, "perf_counter", lambda: float(next(ticks)))

    assert maintenance.main(["--light"]) == FAILED
    stdout, _ = _receipts(scratch, capsys)
    assert stdout["reason"].startswith("post-maintenance search latency pathological: 1000.0ms")
    _assert_correlatable(stdout, outcome="failed", exit_code=FAILED)


def test_quiet_window_gate_still_defers_with_tempfail(scratch, capsys):
    scratch.config.now_fn = lambda: dt.datetime(2026, 11, 5, 12, 0, tzinfo=dt.timezone.utc)

    assert maintenance.main(["--light"]) == DEFERRED
    stdout, logged = _receipts(scratch, capsys)
    for record in (stdout, logged):
        assert record["status"] == "deferred"
        _assert_correlatable(record, outcome="deferred", exit_code=DEFERRED)
    # Older BrainBar builds classify exit 75 by reason text; every deferral must stay on that list.
    assert maintenance._is_deliberate_deferral(stdout["reason"])
    assert scratch.calls == []
    assert job_alerts.active_alerts(scratch.alerts) == {}


def test_successful_run_record_carries_its_own_time_and_run_id(scratch, capsys):
    assert maintenance.main(["--light"]) == 0
    stdout, _ = _receipts(scratch, capsys)
    assert stdout["status"] == "ok"
    _assert_correlatable(stdout, outcome="ok", exit_code=0)


def test_manual_run_without_calendar_receipt_still_names_one_run(scratch, capsys, monkeypatch):
    monkeypatch.delenv(RUN_ID_ENV)
    scratch.config.now_fn = lambda: dt.datetime(2026, 11, 5, 12, 0, tzinfo=dt.timezone.utc)

    assert maintenance.main(["--light"]) == DEFERRED
    stdout, logged = _receipts(scratch, capsys)
    assert len(stdout["run_id"]) == 32 and stdout["run_id"] == logged["run_id"] != RUN_ID


def test_unreviewed_abort_defaults_to_failure_not_deferral():
    # An abort site that does not declare itself a deliberate deferral must never read as one.
    assert maintenance.MaintenanceAbort("anything").code == FAILED
    assert (getattr(maintenance, "EXIT_DEFERRED", None), getattr(maintenance, "EXIT_FAILED", None)) == (75, 70)


def test_calendar_run_start_receipt_id_reaches_child_and_terminal_record(tmp_path, monkeypatch):
    from brainlayer import maintenance_calendar as calendar
    from brainlayer import maintenance_control

    (tmp_path / "fixture.env").write_text("")
    monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(tmp_path / "fixture.env"))
    monkeypatch.setenv(RUN_ID_ENV, "f" * 32)  # Stale inherited id.
    monkeypatch.setattr(calendar, "report", lambda *_args: None)
    monkeypatch.setattr(maintenance_control, "hold_budget", lambda _now: 600.0)
    monkeypatch.setattr(
        calendar.maintenance_control,
        "control_client",
        lambda: SimpleNamespace(path=tmp_path / "unused.sock", request=lambda: {}, _disabled=lambda *_a, **_k: set()),
    )

    @asynccontextmanager
    async def no_supervisor_hold(_client, _guard, env, _step, _now=None):
        yield env  # The private child needs no real supervisor hold; the env is what is under test.

    monkeypatch.setattr(calendar, "step_hold", no_supervisor_hold)
    step = calendar.Step(
        "maintenance-nightly",
        (sys.executable, "-c", f"import os, sys; print(os.environ[{RUN_ID_ENV!r}]); sys.exit(70)"),
        5,
    )
    directory, logs = tmp_path / "calendar", tmp_path / "logs"
    now = lambda: dt.datetime(2026, 11, 5, 4, 30)  # noqa: E731
    for _ in range(2):
        asyncio.run(calendar.run_calendar([step], directory, logs, now_fn=now))
    rows = [json.loads(line) for line in (directory / "runs.jsonl").read_text().splitlines()]
    spawned = [r.get("maintenance_run_id") for r in rows if r["event"] == "spawn"]
    finished = [r for r in rows if r["event"] == "finished"]
    assert spawned == [r.get("maintenance_run_id") for r in finished]
    assert spawned == (logs / "maintenance-nightly.out.log").read_text().splitlines()
    assert len(set(spawned)) == 2 and "f" * 32 not in spawned
    # A genuine failure exit is recorded as failed, never as a deferral.
    assert [(r["status"], r["returncode"]) for r in finished] == [("failed", 70)] * 2
