"""The TLA+ ingest-coverage model, tick by tick, against the real watcher and verdict.

formal/ingest-coverage/replay.py drives a real JSONLWatcher over a scratch home's real watch
roots, the real flush callback into a scratch DB, and the real watcher_health_fact + evaluate, with
JSONL in each wired agent's own format. Each tick must match the committed TLC trace: the
ingest state, what the monitor saw, and whether the newest record reached the DB.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "formal" / "ingest-coverage" / "replay.py"
_SPEC = importlib.util.spec_from_file_location("ingest_coverage_replay", _PATH)
replay = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(replay)


@pytest.fixture(autouse=True)
def _scratch_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BRAINLAYER_DB", str(tmp_path / "db" / "brainlayer.db"))
    monkeypatch.setenv("BRAINLAYER_T3_STATE_DB", str(tmp_path / "no-t3-state.db"))
    monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)
    monkeypatch.delenv("AXIOM_TOKEN", raising=False)


@pytest.mark.parametrize("provider", replay.AGENTS)
@pytest.mark.parametrize("scenario", sorted(replay.SCENARIOS))
def test_real_status_path_matches_the_model_every_tick(scenario, provider, tmp_path):
    trace = replay.load_trace(scenario)
    real = replay.run_real(provider, trace, tmp_path / "work")
    assert replay.mismatches(trace, real) == []


def test_every_scenario_has_a_committed_trace_for_the_current_script():
    for name, scenario in replay.SCENARIOS.items():
        trace = replay.load_trace(name)
        assert trace["events"] == scenario["events"], f"{name}: run replay.py --generate"
        assert trace["clock"] == scenario["clock"]
        assert trace["model"] == replay.MODEL


OPEN = ("drift", "rewrite_lossy")  # no watcher signal sees these (README "Open findings")


def test_every_alert_or_heartbeat_shape_leaves_green_within_two_ticks_and_stays_until_progress():
    """IngestFailureLeavesGreen on the traces the real code matched: stall > 2 ticks => amber or red."""
    for shape in replay.SHAPES:
        if shape in OPEN:
            continue
        ticks = replay.load_trace(shape)["ticks"]
        assert all(t["v"] in ("amber", "red") for t in ticks if t["stall"] > 2), shape


@pytest.mark.parametrize("shape", OPEN)
def test_drift_and_an_undetected_rewrite_are_still_open(shape):
    """Records no parser takes, or an edit the #732 fingerprint cannot see: green for the whole fault."""
    events, ticks = replay.SCENARIOS[shape]["events"], replay.load_trace(shape)["ticks"]
    stalled = [t for e, t in zip(events, ticks) if e["f"] == shape]
    assert stalled and all(t["backlog"] and t["v"] == "green" for t in stalled)


def test_the_replay_catches_a_wrong_model_tick(tmp_path):
    trace = replay.load_trace("sleep")
    trace["ticks"][5] = {**trace["ticks"][5], "v": "amber"}
    assert replay.mismatches(trace, replay.run_real("codex", trace, tmp_path / "work"))


def test_the_replay_catches_a_verdict_that_ignores_heartbeat_age(tmp_path, monkeypatch):
    import brainlayer.health_verdict as health_verdict

    monkeypatch.setattr(health_verdict, "_age_state", lambda age, amber, red: health_verdict.GREEN)
    trace = replay.load_trace("dead")
    assert replay.mismatches(trace, replay.run_real("claude", trace, tmp_path / "work"))


def test_the_replay_catches_a_watcher_that_stops_detecting_rewrites(tmp_path, monkeypatch):
    import brainlayer.watcher as watcher

    monkeypatch.setattr(watcher.JSONLTailer, "_verify_anchor", lambda self, handle: None)
    trace = replay.load_trace("rewrite")
    assert replay.mismatches(trace, replay.run_real("claude", trace, tmp_path / "work"))


def test_a_lossy_rewrite_lands_the_newest_record_and_loses_the_edited_one(tmp_path):
    """Why the replay checks every record: a newest-only check reports no backlog here (#1780 review r1)."""
    trace = replay.load_trace("rewrite_lossy")
    replay.run_real("claude", trace, tmp_path / "work")
    db = tmp_path / "work" / "db" / "brainlayer.db"
    ticks = [k for k, e in enumerate(trace["events"], 1) if e["f"] == "rewrite_lossy"]
    assert ticks
    for k in ticks:
        assert replay._db_has(db, replay._text("claude", k)[0]), k  # the appended turn is ingested
        assert not replay._db_has(db, replay._text("claude", 0, f"L{k:03d}")[0]), k  # the in-place edit never is


def test_a_replay_run_leaves_no_file_descriptor_open(tmp_path):
    """95 runs in one process exhausted select()'s 1024 fds in CI when each run left its stores open."""
    import gc
    import os

    def open_fds():
        gc.collect()
        return len(os.listdir("/dev/fd"))

    replay.run_real("gemini", replay.load_trace("coverage"), tmp_path / "warm")  # imports, caches, logging
    before = open_fds()
    for i in range(3):  # coverage opens both flush callbacks (the probed DB and "elsewhere")
        replay.run_real("gemini", replay.load_trace("coverage"), tmp_path / f"w{i}")
    assert open_fds() <= before
