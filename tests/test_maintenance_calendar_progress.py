"""The journal describes a running night before the terminal summary exists."""

import asyncio
import datetime as dt

from brainlayer import maintenance_calendar as calendar
from brainlayer import observability_backup
from tests.test_maintenance_calendar_completion import runner  # noqa: F401


def test_backup_wait_publishes_interim_summary_then_preserves_terminal_shape(runner, monkeypatch):  # noqa: F811
    names = ("jsonl-backup", "p0-counter", "wal-checkpoint", "backup-daily")
    executed = []

    async def execute(step, env, **kwargs):
        executed.append(step.name)
        if step.name == "backup-daily":
            # This runs before the backup returns: read the actual durable journal,
            # not a mocked record call or a reconstructed final result.
            rows = runner.rows()
            assert not any(row["event"] == "summary" for row in rows)
            progress = [row for row in rows if row["event"] == "progress"][-1]
            assert progress["status"] == "running"
            assert progress["complete"] is False
            assert progress["running_step"] == "backup-daily"
            assert progress["step_started_at"] >= progress["runner_started_at"]
            assert progress["scheduled_date"] == "2026-11-24"
            assert progress["kind"] == "nightly"
            assert progress["results"] == dict.fromkeys(names[:-1], "ok")
            parsed, status, skipped = observability_backup._read_calendar(runner.directory / "runs.jsonl")
            assert status == "read" and skipped == 0
            assert progress in parsed
            # Progress must not end the attempt in recovery history replay.
            assert calendar.failed_recovery_selection(runner.directory)["failures"] == [
                dict(step="backup-daily", reason="it did not finish in the scheduled night")
            ]
            await asyncio.sleep(0)  # The child is still running across an event-loop yield.
        return 0

    monkeypatch.setattr(calendar, "_execute_step", execute)
    # A small nightly plan isolates journal publication from maintenance holds.
    monkeypatch.setattr(calendar.CalendarPlan, "select", lambda self, _marker: self.candidates)
    now = dt.datetime(2026, 11, 24, 4, 45)
    assert (
        asyncio.run(
            calendar.run_calendar(
                calendar.CalendarPlan(now.date(), tuple(calendar.Step(name, ("/unused",), 60) for name in names)),
                runner.directory,
                runner.logs,
                now_fn=lambda: now,
            )
        )
        == 0
    )
    assert executed == list(names)
    final = runner.rows()[-1]
    assert set(final) == {
        "ts",
        "event",
        "kind",
        "complete",
        "scheduled_date",
        "runner_started_at",
        "failed",
        "deferred",
    }
    assert final.items() >= dict(event="summary", complete=True, failed=0, deferred=0).items()
    assert calendar.failed_recovery_selection(runner.directory)["failures"] == []
