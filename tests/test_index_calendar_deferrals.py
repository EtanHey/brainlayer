"""Index rc75 streaks use the calendar's existing, deduplicated job report."""

import datetime as dt

import apsw
import pytest
from typer.testing import CliRunner

import brainlayer.cli as cli
import brainlayer.index_new as adapter
import brainlayer.maintenance_calendar as calendar
from tests.test_index_checkpoint_boundary import cap_after_commit, synthetic_vectors
from tests.test_index_completion import calendar_fixture as calendar_fixture


def test_index_consecutive_deferrals_report_at_existing_threshold(tmp_path, monkeypatch):
    reports = []
    monkeypatch.setattr(calendar, "report", lambda *args: reports.append(args))
    state = dict(deferrals={})
    step = calendar.Step("index", ("fixture",), 60)
    for day in (18, 19, 20):
        assert calendar.finish(tmp_path, state, step, 75, dt.date(2026, 10, day), 3) == "deferred"
    assert reports == [("index", "Deferred 3 nights")]
    assert state["deferrals"]["index"]["count"] == 3


@pytest.mark.parametrize("checkpointed", [False, True])
def test_real_success_or_checkpointed_index_resets_deferral_streak(calendar_fixture, monkeypatch, checkpointed):
    source, db, chunks, clock = calendar_fixture
    reports = []
    monkeypatch.setattr(calendar, "report", lambda *args: reports.append(args))
    monkeypatch.setattr(adapter, "embed_chunks", synthetic_vectors)
    if checkpointed:
        cap_after_commit(monkeypatch, clock)
    result = CliRunner().invoke(cli.app, ["index", str(source)])
    assert result.exit_code == 0, result.output
    assert ("INDEX_CHECKPOINTED" in result.output) == checkpointed
    state = dict(deferrals={"index": dict(date="2026-10-19", count=3)})
    calendar.finish(
        db.parent, state, calendar.Step("index", ("fixture",), 60), result.exit_code, dt.date(2026, 10, 20), 3
    )
    assert reports == [("index", None)]
    assert "index" not in state["deferrals"]


def test_busy_deferrals_do_not_also_raise_backlog_stall(calendar_fixture, monkeypatch):
    source, db, chunks, clock = calendar_fixture
    monkeypatch.setattr(adapter, "embed_chunks", synthetic_vectors)
    with monkeypatch.context() as cap:
        cap_after_commit(cap, clock)
        assert CliRunner().invoke(cli.app, ["index", str(source)]).exit_code == 0
    reports = []
    monkeypatch.setattr(calendar, "report", lambda *args: reports.append(args))

    def busy(*args, **kwargs):
        raise apsw.BusyError("synthetic resident writer busy")

    monkeypatch.setattr("brainlayer.runtime_store.open_writer_store", busy)
    state = dict(deferrals={})
    step = calendar.Step("index", ("fixture",), 60)
    for day in (18, 19, 20):
        clock[0] = 100.0
        result = CliRunner().invoke(cli.app, ["index", str(source)])
        assert result.exit_code == 75, result.output
        assert "INDEX_BACKLOG_STALLED" not in result.output
        calendar.finish(db.parent, state, step, result.exit_code, dt.date(2026, 10, day), 3)
    assert reports == [("index", "Deferred 3 nights")]
