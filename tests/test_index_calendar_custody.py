"""Synthetic budget, stalled watchdog, and existing calendar custody checks."""

import datetime as dt
import threading
from time import monotonic as real_monotonic
from types import SimpleNamespace

import apsw
import pytest
from typer.testing import CliRunner

import brainlayer.cli as cli
import brainlayer.index_new as adapter
import brainlayer.index_watchdog as watchdog_module
import brainlayer.maintenance_calendar as calendar
from brainlayer.vector_store import IndexDeadlineExceeded
from tests.test_cli_index_calendar import prepare
from tests.test_index_checkpoint_boundary import synthetic_vectors
from tests.test_index_completion import calendar_fixture as calendar_fixture


def test_stalled_calendar_alarms_while_writer_is_still_held(calendar_fixture, monkeypatch):
    source, db, _, _ = calendar_fixture
    entered, release, alarmed = threading.Event(), threading.Event(), threading.Event()
    observed, alarms = {}, []
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(db.parent / "pidfiles"))
    monkeypatch.setattr(cli, "_calendar_index_budget_s", lambda: 0.05)
    original = watchdog_module.IndexWatchdog

    def watched(**kw):
        kw.update(wall_monotonic=real_monotonic, poll_interval_s=0.01)
        if "yield_grace_s" in kw:
            kw["yield_grace_s"] = 0.05
        return original(**kw)

    def encode(chunks, **kw):
        entered.set()
        assert release.wait(3), "fixture observer failed to release encoder"
        return synthetic_vectors(chunks)

    def alarm(value):
        alarms.append(value)
        alarmed.set()

    def observe():
        try:
            observed["entered"] = entered.wait(2)
            observed["alarmed"] = alarmed.wait(0.7)
            observed["held"] = bool(list((db.parent / "pidfiles").glob("*.pid")))
        finally:
            release.set()

    monkeypatch.setattr(watchdog_module, "IndexWatchdog", watched)
    monkeypatch.setattr(adapter, "embed_chunks", encode)
    monkeypatch.setattr("brainlayer.alarm.emit_alarm", alarm)
    observer = threading.Thread(target=observe)
    observer.start()
    result = CliRunner().invoke(cli.app, ["index", str(source)])
    observer.join(3)
    assert not observer.is_alive()
    assert observed == dict(entered=True, alarmed=True, held=True), observed
    assert result.exit_code == 1 and len(alarms) == 1, result.output
    assert not list((db.parent / "pidfiles").glob("*.pid"))
    print("STALL alarm_before_encoder_release=true writer_held_at_alarm=true rc=1 pidfile_released=true")


@pytest.mark.parametrize(
    "now,expected", [(dt.datetime(2026, 10, 18, 3, 15), 1830), (dt.datetime(2026, 10, 18, 3, 50), 570)]
)
def test_outer_custody_matches_bounded_child_budget(tmp_path, monkeypatch, now, expected):
    env_file = tmp_path / "fixture.env"
    env_file.write_text("BRAINLAYER_INDEX_MAX_RUNTIME_S=14400\n")
    monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(env_file))
    step = calendar.Step("index", ("fixture-cli", "index"), 21600)
    # The real execution path consumes backup_deadline; this helper must route
    # index through the same existing custody rather than adding a signal site.
    bound = getattr(calendar, "_bounded_index_step", lambda s, now, shutdown: s)(step, now, 15)
    assert bound.timeout == expected
    assert float(bound.env["BRAINLAYER_INDEX_CALENDAR_BUDGET_S"]) == expected - 30
    assert bound.timeout + 15 < (now.replace(hour=4, minute=0) - now).total_seconds()


@pytest.mark.parametrize("hour,minute", [(3, 59), (4, 0), (5, 59)])
def test_no_index_child_admitted_without_maintenance_margin(tmp_path, monkeypatch, hour, minute):
    env_file = tmp_path / "fixture.env"
    env_file.write_text("")
    monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(env_file))
    now = dt.datetime(2026, 10, 18, hour, minute, 30)
    step = calendar.Step("index", ("fixture-cli",), 21600)
    bound = getattr(calendar, "_bounded_index_step", lambda s, now, shutdown: s)(step, now, 15)
    assert bound is None


def test_busy_budget_expiry_with_prior_progress_is_checkpointed(tmp_path, monkeypatch):
    source, clock, alarms = prepare(tmp_path, monkeypatch, chunks=66)
    calls = []

    def index(chunks, **kw):
        calls.append(kw["start_position"])
        if len(calls) == 2:
            clock[0] = 108.0
            raise apsw.BusyError("write busy deadline exceeded")
        return len(chunks)

    monkeypatch.setattr(adapter, "index_chunks_to_sqlite", index)
    result = CliRunner().invoke(cli.app, ["index", str(source)])
    assert result.exit_code == 0 and "INDEX_CHECKPOINTED" in result.output, result.output
    assert "resident_writer_busy" not in result.output and not alarms


def test_calendar_child_budget_is_an_upper_bound(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_INDEX_MAX_RUNTIME_S", "14400")
    monkeypatch.setenv("BRAINLAYER_INDEX_CALENDAR_BUDGET_S", "12")
    assert cli._calendar_index_budget_s(dt.datetime(2026, 10, 18, 3, 15)) == 12


def test_large_cleanup_allowance_still_leaves_quiet_window_margin(tmp_path, monkeypatch):
    path = tmp_path / "fixture.env"
    path.write_text("")
    monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(path))
    step = calendar.Step("index", ("fixture-cli",), 21600)
    bounded = calendar._bounded_index_step(step, dt.datetime(2026, 10, 18, 3, 50), 120)
    assert bounded.timeout + 120 == 585  # Fifteen seconds before 04:00, including cleanup.


def test_failed_escalation_still_fails_after_progress_and_retries_alarm(monkeypatch):
    clock, attempts = [100.0], []

    def emit(alarm):
        attempts.append(alarm)
        if len(attempts) == 1:
            raise OSError("synthetic alarm sink failure")

    monkeypatch.setattr("brainlayer.alarm.emit_alarm", emit)
    watchdog = watchdog_module.IndexWatchdog(
        started_at=100,
        max_runtime_s=1,
        monotonic=lambda: clock[0],
        wall_monotonic=lambda: clock[0],
        yield_grace_s=0.5,
        emit=lambda alarm: None,
    )
    clock[0] = 102
    watchdog._on_deadline()
    result = cli._exit_on_index_deadline(
        IndexDeadlineExceeded(0),
        watchdog=watchdog,
        max_runtime_s=1,
        start_monotonic=100,
        committed_chunks=64,
        files_completed=0,
        current_file=None,
        checkpoint=SimpleNamespace(path="fixture"),
    )
    assert result.exit_code == 1
    assert len(attempts) == 2  # Quiet cap record is not a substitute for the failed alarm.
