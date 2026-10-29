"""Daytime retries use scheduled-night evidence and the existing recovery custody."""

import asyncio
import datetime as dt
import json
import plistlib
import sys
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from brainlayer import maintenance_calendar as calendar
from brainlayer.cli import app


def write_history(directory, rows):
    directory.mkdir(mode=0o700, exist_ok=True)
    (directory / "runs.jsonl").write_text(
        "".join(json.dumps(dict(ts=i + 1, **row)) + "\n" for i, row in enumerate(rows))
    )


def night(date="2026-10-28", steps=None):
    return dict(event="runner", kind="nightly", scheduled_date=date, steps=steps or list(calendar.NAMES))


def test_selection_only_latest_night_and_explicit_allowlist(tmp_path):
    write_history(
        tmp_path,
        [
            night("2026-10-27"),
            dict(event="finished", step="jsonl-backup", status="failed"),
            night(),
            dict(event="finished", step="jsonl-backup", status="ok", returncode=0),
            dict(event="runner", kind="manual", scheduled_date="2026-10-29", steps=["jsonl-backup"]),
            dict(event="finished", step="jsonl-backup", status="failed"),
        ],
    )
    selection = calendar.failed_recovery_selection(tmp_path)
    assert calendar.DAYTIME_RECOVERY_STEPS == frozenset({"jsonl-backup"})
    assert selection["scheduled_date"] == "2026-10-28"
    assert selection["steps"] == []
    assert {item["step"] for item in selection["deferred"]} == set(calendar.NAMES) - {"jsonl-backup"}
    assert all(item["reason"] for item in selection["deferred"])


@pytest.mark.parametrize("status", [None, "failed", "deferred", "stopped", "skipped"])
def test_failed_or_incomplete_is_retryable_and_successful_recovery_clears_it(tmp_path, status):
    rows = [night(steps=["jsonl-backup", "maintenance-weekly"])]
    if status:
        rows.append(dict(event="finished", step="jsonl-backup", status=status, reason="fixture reason"))
    write_history(tmp_path, rows)
    selection = calendar.failed_recovery_selection(tmp_path)
    assert selection["steps"] == ["jsonl-backup"]
    assert selection["failures"][0]["reason"] == (
        {
            "failed": "it stopped with an error",
            "deferred": "it was postponed",
            "stopped": "it stopped before finishing",
            "skipped": "it did not run",
        }.get(status, "it did not finish in the scheduled night")
    )
    rows += [
        dict(event="transcript-recovery", kind="recovery", scheduled_date="2026-10-28", steps=["jsonl-backup"]),
        dict(event="finished", step="jsonl-backup", status="ok", returncode=0),
    ]
    write_history(tmp_path, rows)
    assert calendar.failed_recovery_selection(tmp_path)["steps"] == []


@pytest.fixture
def retry(tmp_path, monkeypatch):
    directory = tmp_path / "calendar"
    write_history(directory, [night(steps=["jsonl-backup", "maintenance-weekly"])])
    monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(tmp_path / "fixture.env"))
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(tmp_path / "alerts.json"))
    disabled, executed = set(), []
    client = SimpleNamespace(path=tmp_path / "unused.sock", request=lambda: {}, _disabled=lambda *_a, **_k: disabled)
    monkeypatch.setattr(calendar.maintenance_control, "control_client", lambda: client)

    async def execute(step, env, **kwargs):
        assert "BRAINLAYER_MAINTENANCE_TOKEN" not in env
        executed.append(step.name)
        return 0

    monkeypatch.setattr(calendar, "_execute_step", execute)
    plan = calendar.calendar_steps(sys.executable, "/unused", tmp_path, dt.date(2026, 10, 29))
    result = {}

    def run(hour=12, status=False, minute=0):
        result.clear()
        return asyncio.run(
            calendar.run_calendar(
                context.plan,
                directory,
                tmp_path / "logs",
                failed_recovery=True,
                recovery_status=status,
                recovery_result=result,
                now_fn=lambda: dt.datetime(2026, 10, 29, hour, minute),
            )
        )

    context = SimpleNamespace(
        directory=directory, run=run, result=result, executed=executed, disabled=disabled, plan=plan
    )
    return context


def test_recovery_journals_original_night_and_never_runs_window_steps(retry):
    assert retry.run() == 0
    assert retry.executed == ["jsonl-backup"]
    assert retry.result["status"] == "accepted"
    rows = [json.loads(line) for line in (retry.directory / "runs.jsonl").read_text().splitlines()]
    assert any(
        r.get("event") == "transcript-recovery" and r["kind"] == "recovery" and r["scheduled_date"] == "2026-10-28"
        for r in rows
    )
    assert rows[-1]["kind"] == "recovery" and rows[-1]["scheduled_date"] == "2026-10-28"
    assert retry.run() == 0 and retry.result["status"] == "nothing-to-retry"
    assert retry.executed == ["jsonl-backup"]


@pytest.mark.parametrize("lock", ["", "recovery-admission", "full-calendar"])
def test_busy_custody_or_reserved_nightly_refuses_without_launching(retry, lock):
    with (
        calendar.Custody(retry.directory / lock),
        calendar.Custody(retry.directory / "recovery-admission") if lock == "full-calendar" else nullcontext(),
    ):
        assert retry.run() == 0
    assert retry.result["status"] == "refused" and retry.result["reason"]
    assert not retry.executed


@pytest.mark.parametrize("hour", [4, 5])
def test_quiet_window_is_reserved_even_when_nightly_not_started(retry, hour):
    assert retry.run(hour) == 0
    assert retry.result["status"] == "refused"
    assert "nightly maintenance" in retry.result["reason"]
    assert not retry.executed


def test_operator_disable_and_status_are_nonexecuting(retry):
    retry.disabled.add("com.brainlayer.jsonl-backup")
    assert retry.run() == 0 and retry.result["status"] == "refused"
    retry.disabled.clear()
    assert retry.run(status=True) == 0 and retry.result["status"] == "ready"
    assert not retry.executed


def test_cli_returns_json_and_preserves_legacy_option(monkeypatch, tmp_path):
    async def run(*args, **kwargs):
        kwargs["recovery_result"].update(status="nothing-to-retry", reason="No daytime-safe failures", steps=[])
        return 0

    monkeypatch.setattr(calendar, "run_calendar", run)
    result = CliRunner().invoke(
        app, ["maintenance-calendar", "--recover-failed", "--state-dir", str(tmp_path / "calendar")]
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["status"] == "nothing-to-retry"
    result = CliRunner().invoke(app, ["maintenance-calendar", "--recover-failed", "--recover-transcripts"])
    assert result.exit_code == 2


def test_no_history_and_malformed_history_never_start(retry):
    (retry.directory / "runs.jsonl").unlink()
    assert retry.run() == 0 and retry.result["status"] == "nothing-to-retry"
    (retry.directory / "runs.jsonl").write_text('{"event":')
    assert retry.run() == 1 and retry.result["status"] == "refused"
    assert not retry.executed


def test_busy_receipt_still_lists_failed_steps_for_display(retry):
    with calendar.Custody(retry.directory):
        retry.run(status=True)
    assert retry.result["status"] == "refused"
    assert {item["step"] for item in retry.result["failures"]} == {"jsonl-backup", "maintenance-weekly"}
    assert not retry.executed


def test_nonzero_exit_cannot_erase_a_failed_step(tmp_path):
    write_history(
        tmp_path,
        [night(steps=["jsonl-backup"]), dict(event="finished", step="jsonl-backup", status="ok", returncode=1)],
    )
    assert calendar.failed_recovery_selection(tmp_path)["steps"] == ["jsonl-backup"]


def test_cli_pre_admission_error_is_a_machine_readable_refusal(monkeypatch, tmp_path):
    async def fail(*args, **kwargs):
        raise PermissionError("Calendar directory is not owner-only")

    monkeypatch.setattr(calendar, "run_calendar", fail)
    result = CliRunner().invoke(
        app, ["maintenance-calendar", "--recover-failed", "--state-dir", str(tmp_path / "calendar")]
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout) == dict(
        status="refused",
        reason="Retry availability could not be checked. Check the maintenance log before trying again.",
        steps=[],
    )


def test_disabled_or_paused_gate_is_checked_again_before_spawn(retry, monkeypatch):
    client = calendar.maintenance_control.control_client()
    calls = []

    def disabled(*args, **kwargs):
        calls.append(kwargs)
        return set() if len(calls) == 1 else {"com.brainlayer.jsonl-backup"}

    monkeypatch.setattr(client, "_disabled", disabled)
    retry.run()
    assert retry.result["status"] == "failed"
    assert retry.result["results"] == {"jsonl-backup": "deferred"}
    assert all(call["include_paused"] is True for call in calls)
    assert not retry.executed


@pytest.mark.parametrize("lock", ["", "recovery-admission", "full-calendar"])
def test_status_busy_is_byte_identical_and_takes_no_custody(retry, monkeypatch, lock):
    with (
        calendar.Custody(retry.directory / lock),
        # A running full calendar also holds recovery-admission; the probe sees it there.
        calendar.Custody(retry.directory / "recovery-admission") if lock == "full-calendar" else nullcontext(),
    ):
        before = {p.relative_to(retry.directory): p.read_bytes() for p in retry.directory.rglob("*") if p.is_file()}
        monkeypatch.setattr(calendar, "record", lambda *_a, **_k: pytest.fail("probe wrote a journal row"))
        assert retry.run(status=True) == 0
        after = {p.relative_to(retry.directory): p.read_bytes() for p in retry.directory.rglob("*") if p.is_file()}
        assert before == after
        assert retry.result["status"] == "refused"
    assert not retry.executed


@pytest.mark.parametrize("failure", ["supervisor", "unreadable", "malformed"])
def test_status_errors_leave_history_bytes_identical(retry, monkeypatch, failure):
    if failure == "supervisor":

        def fail():
            raise ConnectionRefusedError("RAW SOCKET ERROR")

        monkeypatch.setattr(calendar.maintenance_control.control_client(), "request", fail)
    elif failure == "unreadable":

        def fail(*_a):
            raise PermissionError("RAW FILE ERROR")

        monkeypatch.setattr(calendar, "failed_recovery_selection", fail)
    else:
        with (retry.directory / "runs.jsonl").open("a") as out:
            out.write('{"event":')
    before = (retry.directory / "runs.jsonl").read_bytes()
    assert retry.run(status=True) == 1
    assert (retry.directory / "runs.jsonl").read_bytes() == before
    assert retry.result["status"] == "refused"
    assert "RAW" not in retry.result["reason"]
    assert "line 1" not in retry.result["reason"]
    assert not (retry.directory / "groups.json").exists()
    assert not (retry.directory / "logs").exists()


def test_status_probe_never_locks_full_calendar(retry, monkeypatch):
    # The nightly takes full-calendar outside its wait loop; a probe hold there skips the night.
    (retry.directory / "recovery-admission").mkdir(mode=0o700, exist_ok=True)
    with calendar.Custody(retry.directory / "full-calendar"):
        opened = []
        real_open = calendar.os.open
        monkeypatch.setattr(
            calendar.os, "open", lambda path, *a, **k: opened.append(str(path)) or real_open(path, *a, **k)
        )
        assert calendar._probe_calendar_custody(retry.directory) is True
    assert opened and not any("full-calendar" in path for path in opened)


def test_status_releases_probe_locks_before_supervisor_request(retry, monkeypatch):
    def request():
        with (
            calendar.Custody(retry.directory),
            calendar.Custody(retry.directory / "recovery-admission"),
            calendar.Custody(retry.directory / "full-calendar"),
        ):
            return {}

    monkeypatch.setattr(calendar.maintenance_control.control_client(), "request", request)
    assert retry.run(status=True) == 0
    assert retry.result["status"] == "ready"


@pytest.mark.parametrize("hour,minute", [(1, 30), (3, 0), (3, 59)])
def test_retry_reserves_lead_time_before_calendar_start(retry, hour, minute):
    assert retry.run(hour, minute=minute) == 0
    assert retry.result["status"] == "refused"
    assert not retry.executed


def test_retry_cutoff_comes_from_configured_start_and_runtime(retry):
    retry.plan = replace(retry.plan, nightly_start=dt.time(2, 45))
    retry.plan = replace(
        retry.plan,
        candidates=tuple(
            replace(s, timeout=3 * 3600) if s.name == "jsonl-backup" else s for s in retry.plan.candidates
        ),
    )
    assert retry.run(0, minute=30, status=True) == 0
    assert retry.result["status"] == "refused"
    start = dt.datetime.fromisoformat(retry.result["retry_pause_start"])
    assert start.date() == dt.date(2026, 10, 28)  # reservation crosses midnight
    assert retry.run(12, status=True) == 0
    assert retry.result["status"] == "ready"
    assert retry.run(12) == 0
    assert retry.executed == ["jsonl-backup"]


def test_calendar_start_reads_its_configured_plist(tmp_path):
    (tmp_path / "com.brainlayer.maintenance.plist").write_bytes(
        plistlib.dumps({"StartCalendarInterval": {"Hour": 2, "Minute": 45}})
    )
    plan = calendar.calendar_steps(sys.executable, "/unused", tmp_path, dt.date(2026, 10, 29))
    assert plan.nightly_start == dt.time(2, 45)


def test_torn_old_history_does_not_disable_clean_latest_night(retry):
    path = retry.directory / "runs.jsonl"
    path.write_text('{"event":"fin\n' + path.read_text())
    before = path.read_bytes()
    assert retry.run(status=True) == 0
    assert retry.result["status"] == "ready"
    assert path.read_bytes() == before
    assert retry.run() == 0
    assert retry.executed == ["jsonl-backup"]


def test_failure_reason_never_exposes_raw_ids_or_exit_codes(tmp_path):
    write_history(
        tmp_path,
        [
            night(steps=["jsonl-backup"]),
            dict(
                event="finished", step="jsonl-backup", status="failed", returncode=123, reason="unknown-internal-reason"
            ),
        ],
    )
    assert calendar.failed_recovery_selection(tmp_path)["failures"][0]["reason"] == "it stopped with an error"


def test_ready_status_creates_no_files_and_preserves_previous_boot_groups(retry, monkeypatch):
    path = retry.directory / "groups.json"
    path.write_text(json.dumps({"boot_time": "previous boot", "groups": {}}))
    before = {p.relative_to(retry.directory): p.read_bytes() for p in retry.directory.rglob("*") if p.is_file()}
    monkeypatch.setattr(calendar, "record", lambda *_a, **_k: pytest.fail("probe wrote a journal row"))
    monkeypatch.setattr(calendar, "launch_guard", lambda *_a, **_k: pytest.fail("probe launched a guard"))
    assert retry.run(status=True) == 0
    assert retry.result["status"] == "ready"
    after = {p.relative_to(retry.directory): p.read_bytes() for p in retry.directory.rglob("*") if p.is_file()}
    assert before == after


def test_status_without_history_creates_no_directory(retry):
    (retry.directory / "runs.jsonl").unlink()
    retry.directory.rmdir()
    assert retry.run(status=True) == 0
    assert retry.result["status"] == "nothing-to-retry"
    assert not retry.directory.exists()


def test_fractional_guard_timeout_emits_whole_second_pause_times():
    plan = calendar.CalendarPlan(dt.date(2026, 10, 29), (), dt.time(3, 15))
    _blocked, window, _reason = calendar._retry_window(plan, (), dt.datetime(2026, 10, 29, 12), 90.5, 10)
    for value in window.values():
        assert "." not in value
        dt.datetime.fromisoformat(value)


@pytest.mark.parametrize("template", [b"not a plist", plistlib.dumps({"Label": "x"})])
def test_unreadable_calendar_start_keeps_the_nightly_and_refuses_only_retry(tmp_path, template):
    (tmp_path / "com.brainlayer.maintenance.plist").write_bytes(template)
    plan = calendar.calendar_steps(sys.executable, "/unused", tmp_path, dt.date(2026, 10, 29))
    assert plan.candidates and plan.nightly_start is None
    with pytest.raises(ValueError, match="start is unavailable"):
        calendar._retry_window(plan, (), dt.datetime(2026, 10, 29, 12), 60, 10)
