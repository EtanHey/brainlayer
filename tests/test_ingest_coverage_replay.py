"""The TLA+ ingest-coverage model, tick by tick, against the real watcher and verdict.

formal/ingest-coverage/replay.py drives a real JSONLWatcher over a scratch home's real watch
roots, the real flush callback into a scratch DB, and the real watcher_fact + evaluate, with
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


def test_the_findings_are_real_today():
    """A lag, coverage, DB or missing-root stall keeps ingest green for the whole fault (README "Finding")."""
    for shape in ("lag", "coverage", "dbfail", "rootgone"):
        ticks = replay.load_trace(shape)["ticks"]
        stalled = [t for e, t in zip(replay.SCENARIOS[shape]["events"], ticks) if e["f"] == shape]
        assert stalled and all(t["backlog"] and t["v"] == "green" for t in stalled), shape


def test_the_replay_catches_a_wrong_model_tick(tmp_path):
    trace = replay.load_trace("sleep")
    trace["ticks"][5] = {**trace["ticks"][5], "v": "amber"}
    assert replay.mismatches(trace, replay.run_real("codex", trace, tmp_path / "work"))


def test_the_replay_catches_a_verdict_that_ignores_heartbeat_age(tmp_path, monkeypatch):
    import brainlayer.health_verdict as health_verdict

    monkeypatch.setattr(health_verdict, "_age_state", lambda age, amber, red: health_verdict.GREEN)
    trace = replay.load_trace("dead")
    assert replay.mismatches(trace, replay.run_real("claude", trace, tmp_path / "work"))
