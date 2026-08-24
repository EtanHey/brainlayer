"""Incident 2026-08-24: BrainLayer daemons must go idle when there is no work.

hotlane pinned 120% CPU for 29 minutes because `run()` slept a FIXED `interval`
(1.0s) after every cycle, working or not -- so it re-scanned a 14.8 GB SQLite DB
once per second forever. These tests pin the contract: an idle cycle backs off,
and any real work snaps the cadence straight back to `interval`.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _load_hotlane_module():
    importlib.invalidate_caches()
    sys.modules.pop("scripts.hotlane_brainbar_daemon", None)
    return importlib.import_module("scripts.hotlane_brainbar_daemon")


class _FakeStore:
    def close(self):
        pass


class _FakeModel:
    def embed_query(self, _text):
        return [0.0]

    def embed_texts(self, texts):
        return [[0.0] * 1024 for _text in texts]


def _run(hotlane, *, cycle_fn, cycles, sleeps, interval=1.0, **extra):
    hotlane.run(
        db_path=Path("/tmp/unused.db"),
        interval=interval,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=0,
        enrich_interval=10.0,
        enrich_limit=0,
        enrich_since_hours=8760,
        vector_store_cls=lambda _path: _FakeStore(),
        model_factory=_FakeModel,
        cycle_fn=cycle_fn,
        queue_depth_fn=lambda _queue_dir: 0,
        high_priority_queue_depth_fn=lambda _queue_dir: 0,
        time_fn=iter([float(n) for n in range(cycles + 2)]).__next__,
        sleep_fn=sleeps.append,
        max_cycles=cycles,
        **extra,
    )


def test_idle_cycles_back_off_instead_of_rescanning_every_second():
    """No work for 5 cycles => the sleep must grow, not stay pinned at interval."""
    hotlane = _load_hotlane_module()
    sleeps: list[float] = []

    _run(hotlane, cycle_fn=lambda **_k: hotlane.CycleResult(), cycles=5, sleeps=sleeps)

    assert sleeps[0] == 1.0, "first idle cycle keeps the configured interval"
    assert sleeps[-1] > sleeps[0], f"idle cadence never backed off: {sleeps}"
    assert sleeps == sorted(sleeps), f"backoff must be monotonic: {sleeps}"


def test_idle_backoff_is_capped():
    """Backoff must not grow without bound -- a hot lane still has to be hot."""
    hotlane = _load_hotlane_module()
    sleeps: list[float] = []

    _run(hotlane, cycle_fn=lambda **_k: hotlane.CycleResult(), cycles=40, sleeps=sleeps)

    assert max(sleeps) <= hotlane.MAX_IDLE_INTERVAL_SECONDS, (
        f"backoff exceeded its cap: max={max(sleeps)} cap={hotlane.MAX_IDLE_INTERVAL_SECONDS}"
    )


def test_work_snaps_cadence_back_to_interval():
    """The moment a cycle embeds anything, latency must return to `interval`."""
    hotlane = _load_hotlane_module()
    sleeps: list[float] = []
    calls = {"n": 0}

    def cycle_fn(**_kwargs):
        calls["n"] += 1
        # idle for the first 4 cycles, then real work on the 5th
        if calls["n"] == 5:
            return hotlane.CycleResult(embedded=1)
        return hotlane.CycleResult()

    _run(hotlane, cycle_fn=cycle_fn, cycles=6, sleeps=sleeps)

    assert sleeps[3] > 1.0, f"should have backed off while idle: {sleeps}"
    assert sleeps[4] == 1.0, f"work must snap cadence back to interval: {sleeps}"


def test_enrichment_attempt_also_counts_as_work():
    """Enrichment is work too -- it must reset the backoff, not just embedding."""
    hotlane = _load_hotlane_module()
    sleeps: list[float] = []
    calls = {"n": 0}

    def cycle_fn(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 5:
            return hotlane.CycleResult(enrich_attempted=1)
        return hotlane.CycleResult()

    _run(hotlane, cycle_fn=cycle_fn, cycles=6, sleeps=sleeps)

    assert sleeps[4] == 1.0, f"enrichment work must reset backoff: {sleeps}"
