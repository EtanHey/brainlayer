"""REV-S22-R1 repro: durable store-queue backpressure must still suppress the hot slice under high load."""

import importlib
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("yield_reason", ["high_load", "pause_marker"])
def test_high_load_does_not_lift_store_queue_backpressure(tmp_path, monkeypatch, yield_reason):
    sys.modules.pop("scripts.hotlane_brainbar_daemon", None)
    hotlane = importlib.import_module("scripts.hotlane_brainbar_daemon")
    monkeypatch.setenv("BRAINLAYER_HOTLANE_BACKLOG_MAX_LOAD", "2")
    monkeypatch.setattr(hotlane.os, "getloadavg", lambda: (3.0 if yield_reason == "high_load" else 0.0, 0, 0))
    if yield_reason == "pause_marker":
        (tmp_path / "hotlane.pause").touch()
    slices = []

    def spy(**kw):
        slices.append((kw["recent_limit"], kw["backlog_batch"]))
        return hotlane.CycleResult(embedded=0)

    monkeypatch.setattr(hotlane, "_run_split_cycle", spy)
    hotlane.run(
        db_path=tmp_path / "private.db",
        interval=0.25,
        recent_limit=5,
        backlog_interval=0,
        backlog_batch=16,
        enrich_interval=0,
        enrich_limit=0,
        enrich_since_hours=0,
        model_factory=lambda: SimpleNamespace(embed_query=lambda t: [0.25] * 1024),
        queue_depth_fn=lambda _: 3,
        high_priority_queue_depth_fn=lambda _: 3,  # durable store queue backed up
        time_fn=lambda: 100.0,
        sleep_fn=lambda _: None,
        max_cycles=3,
    )
    hot_slices = [s for s in slices if s[0] > 0]
    assert hot_slices == [], f"hot slice ran during store-queue backpressure: (recent_limit, backlog_batch)={slices}"
    assert slices == [], "a yielded backlog must leave both writer slices suppressed during queue drain"
