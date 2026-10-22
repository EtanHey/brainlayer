"""An interrupted maintenance calendar must not cost another night of transcripts."""

import asyncio
import datetime as dt
import json
import sys

import pytest

from brainlayer import maintenance_calendar as calendar
from brainlayer.service_supervisor import Component
from brainlayer.supervisor_jobs import with_brainbar_monitor
from tests.test_maintenance_calendar_completion import runner  # noqa: F401


NOW = dt.datetime(2026, 10, 22, 23, 45, tzinfo=dt.UTC)


@pytest.mark.parametrize(
    "status,verified,uploaded,hours,expected",
    [
        ("uploaded", True, True, 40, ["jsonl-backup"]),
        ("uploaded", True, True, 2, []),
        ("no-op", True, False, 2, []),
        ("created", True, False, 2, ["jsonl-backup"]),
        ("failed", False, False, 2, ["jsonl-backup"]),
        ("uploaded", False, True, 2, ["jsonl-backup"]),
        ("uploaded", True, True, -2, ["jsonl-backup"]),
    ],
)
def test_recovery_checks_receipt_and_runs_only_transcripts(
    runner, monkeypatch, status, verified, uploaded, hours, expected
):
    receipt = runner.directory.parent / "transcript.log"
    receipt.write_text(json.dumps(dict(
        attempted_at=(NOW - dt.timedelta(hours=hours)).isoformat(),
        status=status, verified=verified, uploaded=uploaded,
    )) + "\n")
    monkeypatch.setenv("BRAINLAYER_JSONL_BACKUP_LOG_PATH", str(receipt))
    executed = []

    async def execute(step, env, **kwargs):
        executed.append(step.name)
        return 0

    monkeypatch.setattr(calendar, "_execute_step", execute)
    monkeypatch.setattr(calendar, "report", lambda *_a: None)
    plan = calendar.calendar_steps(sys.executable, "/unused", runner.logs, NOW.date())
    result = asyncio.run(calendar.run_calendar(
        plan, runner.directory, runner.logs,
        now_fn=lambda: NOW, transcript_recovery=True,
    ))
    assert result == 0
    assert executed == expected
    # Recovery must not supersede the nightly runner's scheduled-date coverage.
    assert not any(row["event"] == "runner" for row in runner.rows())


@pytest.mark.parametrize("contents", [None, "bad json\n", "{}\n"])
def test_missing_or_invalid_receipt_does_not_suppress_recovery(runner, monkeypatch, contents):
    receipt = runner.directory.parent / "transcript.log"
    if contents is not None:
        receipt.write_text(contents)
    monkeypatch.setenv("BRAINLAYER_JSONL_BACKUP_LOG_PATH", str(receipt))
    executed = []

    async def execute(step, env, **kwargs):
        executed.append(step.name)
        return 0

    monkeypatch.setattr(calendar, "_execute_step", execute)
    monkeypatch.setattr(calendar, "report", lambda *_a: None)
    plan = calendar.calendar_steps(sys.executable, "/unused", runner.logs, NOW.date())
    assert asyncio.run(calendar.run_calendar(
        plan, runner.directory, runner.logs, now_fn=lambda: NOW, transcript_recovery=True,
    )) == 0
    assert executed == ["jsonl-backup"]


def test_recovery_respects_existing_calendar_custody(runner, monkeypatch):
    async def forbidden(*_a, **_k):
        pytest.fail("recovery overlapped the active nightly calendar")

    monkeypatch.setattr(calendar, "_execute_step", forbidden)
    plan = calendar.calendar_steps(sys.executable, "/unused", runner.logs, NOW.date())
    with calendar.Custody(runner.directory):
        assert asyncio.run(calendar.run_calendar(
            plan, runner.directory, runner.logs, now_fn=lambda: NOW, transcript_recovery=True,
        )) == 0
    assert runner.rows()[-1]["event"] == "already-running"


def test_aborted_calendar_recovers_and_failed_backup_retries(runner, monkeypatch):
    runner.directory.mkdir()
    calendar.record(runner.directory, event="runner", scheduled_date="2026-10-22")
    calendar.record(runner.directory, event="finished", step="index", status="failed", returncode=124)
    # There is deliberately no nightly summary or transcript result after the index failure.
    receipt = runner.directory.parent / "transcript.log"
    monkeypatch.setenv("BRAINLAYER_JSONL_BACKUP_LOG_PATH", str(receipt))
    executed, notices = [], []

    async def execute(step, env, **kwargs):
        executed.append(step.name)
        ok = len(executed) > 1
        receipt.write_text(json.dumps(dict(
            attempted_at=NOW.isoformat(), status="uploaded" if ok else "failed",
            uploaded=ok, verified=ok,
        )) + "\n")
        return 0 if ok else 1

    monkeypatch.setattr(calendar, "_execute_step", execute)
    monkeypatch.setattr(calendar, "report", lambda key, reason: notices.append((key, reason)))
    plan = calendar.calendar_steps(sys.executable, "/unused", runner.logs, NOW.date())
    results = [asyncio.run(calendar.run_calendar(
        plan, runner.directory, runner.logs, now_fn=lambda: NOW, transcript_recovery=True,
    )) for _ in range(3)]
    assert results == [1, 0, 0]
    assert executed == ["jsonl-backup", "jsonl-backup"]
    assert notices[0][0] == "jsonl-backup" and notices[0][1] is not None
    assert notices[1] == ("jsonl-backup", None)
    assert [(row["step"], row["returncode"]) for row in runner.rows() if row["event"] == "finished"] == [
        ("index", 124), ("jsonl-backup", 1), ("jsonl-backup", 0),
    ]


def test_recovery_honors_transcript_disable(runner, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_JSONL_BACKUP_LOG_PATH", str(runner.directory.parent / "missing.log"))
    from types import SimpleNamespace

    monkeypatch.setattr(calendar.maintenance_control, "control_client", lambda: SimpleNamespace(
        path=runner.directory.parent / "unused.sock", request=lambda: {},
        _disabled=lambda *_a, **_k: {"com.brainlayer.jsonl-backup"},
    ))

    async def forbidden(*_a, **_k):
        pytest.fail("recovery revived a disabled transcript backup")

    monkeypatch.setattr(calendar, "_execute_step", forbidden)
    plan = calendar.calendar_steps(sys.executable, "/unused", runner.logs, NOW.date())
    assert asyncio.run(calendar.run_calendar(
        plan, runner.directory, runner.logs, now_fn=lambda: NOW, transcript_recovery=True,
    )) == 0
    row = next(row for row in runner.rows() if row["event"] == "finished")
    assert row["status"] == "deferred" and row["reason"] == "disabled"


def test_supervisor_admits_periodic_recovery_without_nightly_calendar(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    source = Component(("/unused",), env={"BRAINLAYER_ENV_FILE": str(tmp_path / "fixture.env")},
                       stderr=tmp_path / "logs/observability.err.log")
    components = with_brainbar_monitor({"observability": source}, tmp_path / "control.sock")
    child = components["transcript-backup-recovery"]
    assert child.interval == 3600
    assert child.run_timeout > 7200
    assert child.argv[:4] == (sys.executable, "-m", "brainlayer", "maintenance-calendar")
    assert "--recover-transcripts" in child.argv
    assert child.argv[child.argv.index("--state-dir") + 1] == str(tmp_path / "calendar")
    assert child.argv[child.argv.index("--logs-dir") + 1] == str(tmp_path / "logs")
    assert child.env["BRAINLAYER_ENV_FILE"] == str(tmp_path / "fixture.env")
