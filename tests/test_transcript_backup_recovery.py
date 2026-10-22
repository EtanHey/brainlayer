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
