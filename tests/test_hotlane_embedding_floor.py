"""Continuous load must not starve backlog work; explicit pauses remain absolute."""

import importlib
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("paused", [False, True])
def test_continuous_high_load_floor(tmp_path, monkeypatch, paused):
    h = importlib.import_module("scripts.hotlane_brainbar_daemon")
    monkeypatch.setenv("BRAINLAYER_HOTLANE_BACKLOG_MAX_LOAD", "8")
    monkeypatch.setenv("BRAINLAYER_HOTLANE_BACKLOG_FLOOR_SECONDS", "60")
    monkeypatch.setattr(h.os, "getloadavg", lambda: (25, 0, 0))
    if paused:
        (tmp_path / "hotlane.pause").touch()
    batches = []
    h.run(
        db_path=tmp_path / "fixture.db",
        interval=1,
        recent_limit=0,
        backlog_interval=10,
        backlog_batch=16,
        enrich_interval=0,
        enrich_limit=0,
        enrich_since_hours=0,
        model_factory=lambda: SimpleNamespace(embed_query=lambda _: []),
        vector_store_cls=lambda _: SimpleNamespace(close=lambda: None),
        cycle_fn=lambda **kw: batches.append(kw["backlog_batch"]) or h.CycleResult(),
        queue_depth_fn=lambda _: 0,
        high_priority_queue_depth_fn=lambda _: 0,
        time_fn=iter([0, 0, 59, 60, 61, 119, 120]).__next__,
        sleep_fn=lambda _: None,
        max_cycles=6,
    )
    assert [b for b in batches if b] == ([] if paused else [16, 16])


def test_default_load_threshold_and_override(monkeypatch):
    h = importlib.import_module("scripts.hotlane_brainbar_daemon")
    monkeypatch.setattr(h, "_performance_cores", lambda: 8)
    monkeypatch.delenv("BRAINLAYER_HOTLANE_BACKLOG_MAX_LOAD", raising=False)
    assert h._backlog_max_load() == 16
    monkeypatch.setenv("BRAINLAYER_HOTLANE_BACKLOG_MAX_LOAD", "9")
    assert h._backlog_max_load() == 9


def test_floor_deadline_reaches_health_note_and_clears_on_shrink(tmp_path):
    import json
    from datetime import datetime, timedelta, timezone

    from brainlayer import health_facts as hf
    from brainlayer import health_verdict as hv
    from brainlayer.hotlane_budget import HotlaneCpuSampler

    db = tmp_path / "fixture.db"
    sampler = HotlaneCpuSampler(db)
    sampler.set_backlog_floor(600)
    sampler.observe("high load")
    cpu = json.loads((tmp_path / "hotlane-cpu.json").read_text())
    now = datetime.now(timezone.utc)
    _, note = hf.embedding_activity(cpu, {"backlog_max_load": 16}, db, now)
    assert "next forced batch time unavailable" not in note
    assert "next forced batch" in note
    since, _ = hf.embedding_fact(9, {"count": 10, "since": (now - timedelta(hours=2)).isoformat()}, now, now)
    stalled = hv.Facts(now=now, embedding_stalled_since=now - timedelta(hours=2), embedding_activity=note)
    assert "paused by machine load" in hv.evaluate(stalled).check("embedding").note
    facts = hv.Facts(now=now, embedding_stalled_since=since, embedding_activity=note)
    check = hv.evaluate(facts).check("embedding")
    assert check.state == hv.GREEN
    assert "paused by machine load" not in check.note


@pytest.mark.parametrize("pause_during_burst", [False, True])
def test_sustained_arrivals_shrink_in_simulated_hour(tmp_path, monkeypatch, pause_during_burst):
    h = importlib.import_module("scripts.hotlane_brainbar_daemon")
    monkeypatch.setattr(h.os, "getloadavg", lambda: (25, 0, 0))
    monkeypatch.setenv("BRAINLAYER_HOTLANE_BACKLOG_MAX_LOAD", "8")
    clock, pending, arrived, drained = [0.0], [713.0], [0.0], [0]

    def advance(seconds):
        clock[0] += seconds
        incoming = seconds * 526 / 3600
        pending[0] += incoming
        arrived[0] += incoming

    stopped_at = []

    def cycle(**kw):
        assert not stopped_at, "batch ran after the deliberate pause"
        assert kw["recent_limit"] == 0
        n = min(kw["backlog_batch"], int(pending[0]))
        if n:
            advance(15)
            pending[0] -= n
            drained[0] += n
            if pause_during_burst and clock[0] >= 2100:
                (tmp_path / "hotlane.pause").touch()
                stopped_at.append(clock[0])
        return h.CycleResult(embedded=n, backlog_encoded=n)

    def sleep(_):
        advance(min(5, 3600 - clock[0]))
        if clock[0] >= 3600:
            h.STOP = True

    monkeypatch.setattr(h, "STOP", False)
    monkeypatch.setattr(h, "_backlog_pending_count", lambda _: int(pending[0]), raising=False)
    h.run(
        db_path=tmp_path / "fixture.db",
        interval=1,
        recent_limit=0,
        backlog_interval=10,
        backlog_batch=16,
        enrich_interval=0,
        enrich_limit=0,
        enrich_since_hours=0,
        model_factory=lambda: SimpleNamespace(embed_query=lambda _: []),
        vector_store_cls=lambda _: SimpleNamespace(close=lambda: None),
        cycle_fn=cycle,
        queue_depth_fn=lambda _: 1,
        high_priority_queue_depth_fn=lambda _: 1,
        time_fn=lambda: clock[0],
        sleep_fn=sleep,
    )
    if pause_during_burst:
        assert stopped_at
        return
    assert drained[0] > arrived[0]
    assert pending[0] < 713
    print(f"hour: arrivals={arrived[0]:.0f} drained={drained[0]} pending={pending[0]:.0f}")


def test_load_dip_resets_continuous_timer(tmp_path, monkeypatch):
    h = importlib.import_module("scripts.hotlane_brainbar_daemon")
    monkeypatch.setenv("BRAINLAYER_HOTLANE_BACKLOG_FLOOR_SECONDS", "60")
    monkeypatch.setenv("BRAINLAYER_HOTLANE_BACKLOG_MAX_LOAD", "8")
    loads = iter([25, 0, 25, 25, 25])
    monkeypatch.setattr(h.os, "getloadavg", lambda: (next(loads), 0, 0))
    batches = []
    h.run(
        db_path=tmp_path / "fixture.db",
        interval=1,
        recent_limit=0,
        backlog_interval=10,
        backlog_batch=16,
        enrich_interval=0,
        enrich_limit=0,
        enrich_since_hours=0,
        model_factory=lambda: SimpleNamespace(embed_query=lambda _: []),
        vector_store_cls=lambda _: SimpleNamespace(close=lambda: None),
        cycle_fn=lambda **kw: batches.append(kw["backlog_batch"]) or h.CycleResult(),
        queue_depth_fn=lambda _: 0,
        high_priority_queue_depth_fn=lambda _: 0,
        time_fn=iter([0, 0, 30, 40, 99, 100]).__next__,
        sleep_fn=lambda _: None,
        max_cycles=5,
    )
    assert batches == [0, 16, 0, 0, 16]


@pytest.mark.parametrize(
    "setting", ["BRAINLAYER_HOTLANE_BACKLOG_FLOOR_SECONDS", "BRAINLAYER_HOTLANE_BACKLOG_MAX_WAIT_SECONDS"]
)
def test_invalid_timer_names_actual_setting(tmp_path, monkeypatch, setting):
    h = importlib.import_module("scripts.hotlane_brainbar_daemon")
    monkeypatch.delenv("BRAINLAYER_HOTLANE_BACKLOG_FLOOR_SECONDS", raising=False)
    monkeypatch.setenv(setting, "-1")
    with pytest.raises(ValueError, match=setting):
        h._positive_seconds(setting, 300)
