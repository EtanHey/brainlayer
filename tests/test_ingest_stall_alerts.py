"""The ingest check reads the watcher's own stall signals, not only its heartbeat (#1780's finding).

Before this, a watcher that kept polling while nothing reached the DB (an unreadable transcript,
failing writes, writes landing elsewhere, a vanished root) wrote a fresh heartbeat every poll and
the check stayed green forever. formal/ingest-coverage models the rule; these pin it on the real
collectors, policy and watcher.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

import brainlayer.watcher as watcher_module
from brainlayer.health_verdict import Facts, evaluate
from brainlayer.watcher import JSONLWatcher
from tests.test_health_facts import NOW, _config, _verdict_ticks

STALLS = {
    "lag (unreadable transcript)": {"alert_reasons": ["file_ingestion_failure"]},
    "coverage or failing DB writes": {"alert_reasons": ["coverage_drop"]},
    "offset lag": {"alert_reasons": ["offset_lag"]},
    "DB probe failing for 5 min": {"db_probe_failed": True, "db_probe_failed_for_s": 300.0},
    "watched root gone": {"roots_missing": ["codex"]},
}


def _health(path: Path, at, **extra) -> None:
    payload = {"updated_at": at.isoformat(), "poll_count": 1, "alerting": False, "alert_reasons": []}
    path.write_text(json.dumps({**payload, **extra}))


def _ingest_states(tmp_path, stalled_ticks, extra, after=3):
    config = _config(tmp_path)
    ticks = []
    for i in range(stalled_ticks + after):
        at = NOW + timedelta(minutes=5 * i)
        signal = extra if i < stalled_ticks else {}
        ticks.append((at, lambda at=at, signal=signal: _health(config.watcher_health_path, at, **signal)))
    return [verdict.check("ingest").state for _, verdict in _verdict_ticks(config, ticks)]


@pytest.mark.parametrize("extra", STALLS.values(), ids=STALLS.keys())
def test_a_stall_signal_under_a_fresh_heartbeat_leaves_green_and_recovers_only_on_good_ticks(tmp_path, extra):
    states = _ingest_states(tmp_path, stalled_ticks=8, extra=extra)
    # amber at once (the watcher already debounced it), red after 6 alert ticks, then 2 good ticks
    assert states == ["amber"] * 6 + ["red"] * 2 + ["red", "green", "green"]


def test_one_failed_db_probe_is_a_momentary_lock_not_a_stall(tmp_path):
    states = _ingest_states(tmp_path, 3, {"db_probe_failed": True, "db_probe_failed_for_s": 30.0})
    assert set(states) == {"green"}


def test_quarantined_records_alone_never_move_ingest(tmp_path):
    states = _ingest_states(tmp_path, 3, {"alerting": True, "alert_reasons": ["quarantined_record"]})
    assert set(states) == {"green"}


def test_coverage_drop_while_maintenance_pauses_the_drain_is_not_a_stall():
    facts = Facts(now=NOW, watcher_updated_at=NOW, watcher_stall_codes=("coverage_drop",), stores_paused=True)
    assert evaluate(facts).check("ingest").state == "green"
    lag = Facts(now=NOW, watcher_updated_at=NOW, watcher_stall_codes=("file_ingestion_failure",), stores_paused=True)
    assert evaluate(lag).check("ingest").state == "amber"


class _Clock:
    def __init__(self, real):
        self._real, self.mono = real, 1000.0

    def monotonic(self):
        return self.mono

    def __getattr__(self, name):
        return getattr(self._real, name)


def _root_watcher(tmp_path, monkeypatch):
    clock = _Clock(watcher_module.time)
    monkeypatch.setattr(watcher_module, "time", clock)
    projects = tmp_path / "home" / ".claude" / "projects"
    health = tmp_path / "watcher-health.json"
    roots = watcher_module.default_watch_roots(tmp_path / "home")
    watcher = JSONLWatcher(
        watch_roots=roots, registry_path=tmp_path / "offsets.json", on_flush=lambda items: {}, health_path=health
    )
    return watcher, clock, projects, health


def test_the_watcher_publishes_a_vanished_root_after_its_debounce(tmp_path, monkeypatch):
    watcher, clock, projects, health = _root_watcher(tmp_path, monkeypatch)
    projects.mkdir(parents=True)

    def poll(advance=0.0):
        clock.mono += advance
        watcher.poll_once()
        return json.loads(health.read_text())["roots_missing"]

    assert poll() == []  # codex, cursor and gemini never existed: never "missing"
    projects.rename(projects.with_name("projects-moved"))
    assert poll(30) == []  # one poll is not an episode
    assert poll(299) == []
    assert poll(1) == ["claude"]  # 300 s after the first poll that saw it gone
    projects.with_name("projects-moved").rename(projects)
    assert poll(30) == []


def test_the_watcher_publishes_how_long_the_db_probe_has_failed(tmp_path, monkeypatch):
    watcher, clock, projects, health = _root_watcher(tmp_path, monkeypatch)
    watcher.db_path = tmp_path / "brainlayer.db"
    probe = iter([None, None, 3])
    monkeypatch.setattr(watcher, "_db_realtime_inserts_since_window_start", lambda: next(probe))
    seen = []
    for _ in range(3):
        watcher.poll_once()
        payload = json.loads(health.read_text())
        seen.append((payload["db_probe_failed"], payload["db_probe_failed_for_s"]))
        clock.mono += 300
    assert seen == [(True, 0.0), (True, 300.0), (False, 0.0)]
