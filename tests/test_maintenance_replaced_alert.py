"""A weekly replacement clears the nightly alert only after a successful child."""

import asyncio
import datetime as dt
import json
import sys
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from brainlayer import job_alerts
from brainlayer import maintenance_calendar as calendar
from tests import test_maintenance_calendar_completion as completion

runner = completion.runner
NOW = dt.datetime(2026, 11, 10, 4, 15)


@pytest.mark.parametrize("code", [0, 1, 75])
def test_real_weekly_replacement_updates_nightly_alert_and_deferrals(runner, monkeypatch, code):
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(runner.alerts))
    runner.directory.mkdir(mode=0o700)
    (runner.directory / "state.json").write_text(
        json.dumps(dict(deferrals={"maintenance-light": dict(date="2026-11-09", count=3)}))
    )
    job_alerts.report("maintenance-light", "old nightly failure", path=runner.alerts, notify=lambda *_: None)

    @asynccontextmanager
    async def no_hold(_client, _guard, env, _step, _now):
        yield env

    monkeypatch.setattr(calendar, "step_hold", no_hold)
    plan = calendar.calendar_steps(sys.executable, "/unused", runner.logs, NOW.date())
    plan = replace(
        plan,
        candidates=tuple(
            replace(
                step,
                argv=(sys.executable, "-c", f"raise SystemExit({code if step.name == 'maintenance-weekly' else 0})"),
            )
            for step in plan.candidates
        ),
    )
    assert asyncio.run(calendar.run_calendar(plan, runner.directory, runner.logs, now_fn=lambda: NOW)) == int(code == 1)
    weekly = next(
        row for row in runner.rows() if row.get("step") == "maintenance-weekly" and row["event"] == "finished"
    )
    assert weekly["replaces"] == "maintenance-nightly"
    assert ("maintenance-light" in job_alerts.active_alerts(runner.alerts)) == (code != 0)
    state = json.loads((runner.directory / "state.json").read_text())
    assert ("maintenance-light" in state["deferrals"]) == (code != 0)


@pytest.mark.parametrize("protected", [False, True])
def test_stopped_or_protected_replacement_preserves_nightly_episode(tmp_path, monkeypatch, protected):
    alerts = tmp_path / "alerts.json"
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(alerts))
    job_alerts.report("maintenance-light", "old failure", path=alerts, notify=lambda *_: None)
    state = dict(deferrals={"maintenance-light": dict(date="2026-11-09", count=3)})
    if protected:
        state["incomplete_run"] = dict(
            incomplete_steps=["maintenance-nightly"], step="maintenance-nightly", scheduled_date="2026-11-09"
        )
    calendar.finish(
        tmp_path,
        state,
        calendar.Step("maintenance-weekly", ("/unused",), 10),
        0,
        NOW.date(),
        2,
        stopped=not protected,
        replaces="maintenance-nightly",
        preserve_abandonment_alerts=protected,
    )
    assert "maintenance-light" in job_alerts.active_alerts(alerts)
    assert "maintenance-light" in state["deferrals"]
