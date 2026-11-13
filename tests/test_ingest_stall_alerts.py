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
from brainlayer.health_facts import watcher_health_fact
from brainlayer.health_verdict import UNOBSERVED, Facts, evaluate
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
    payload = {
        "updated_at": at.isoformat(),
        "poll_count": 1,
        "alerting": False,
        "alert_reasons": [],
        "db_probe_failed": False,
        "db_probe_failed_for_s": 0.0,
        "roots_missing": [],
    }
    path.write_text(json.dumps({**payload, **extra}))


def _states(config, payloads):
    """One health tick per payload, 5 min apart; each payload is written with that tick's updated_at."""
    ticks = []
    for i, payload in enumerate(payloads):
        at = NOW + timedelta(minutes=5 * i)
        write = lambda at=at, payload=payload: config.watcher_health_path.write_text(  # noqa: E731
            json.dumps({**payload, "updated_at": at.isoformat()})
        )
        ticks.append((at, write))
    return [verdict.check("ingest").state for _, verdict in _verdict_ticks(config, ticks)]


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
    # amber at once (the watcher already debounced it), red on the 6th alert check, then 2 good ticks
    assert states == ["amber"] * 5 + ["red"] * 3 + ["red", "green", "green"]


FULL = {
    "poll_count": 1,
    "alert_reasons": [],
    "db_probe_failed": False,
    "db_probe_failed_for_s": 0.0,
    "roots_missing": [],
}
BROKEN = {
    "truncated to its heartbeat": {"poll_count": 1},
    "probe failed, no duration": {**FULL, "db_probe_failed": True, "db_probe_failed_for_s": None},
    "duration as text": {**FULL, "db_probe_failed": True, "db_probe_failed_for_s": "600"},
    "duration as a bool": {**FULL, "db_probe_failed_for_s": True},
    "reasons as a string": {**FULL, "alert_reasons": "coverage_drop"},
    "roots missing is null": {**FULL, "roots_missing": None},
    "probe flag missing": {k: v for k, v in FULL.items() if k != "db_probe_failed"},
    "negative duration": {**FULL, "db_probe_failed": True, "db_probe_failed_for_s": -1},
    "duration while the probe is fine": {**FULL, "db_probe_failed_for_s": 30.0},
    "negative poll_count": {**FULL, "poll_count": -1},
    "poll_count as text": {**FULL, "poll_count": "3"},
}


@pytest.mark.parametrize("broken", BROKEN.values(), ids=BROKEN.keys())
def test_an_incomplete_payload_is_unobserved_and_never_clears_a_db_failure(tmp_path, broken):
    config = _config(tmp_path)
    config.watcher_health_path.write_text(json.dumps({"updated_at": NOW.isoformat(), **broken}))
    read = watcher_health_fact(config.watcher_health_path)
    heartbeat, codes = read[0], read[1]
    assert heartbeat == NOW and codes is UNOBSERVED
    failing = {**FULL, "alert_reasons": ["coverage_drop"]}
    states = _states(config, [FULL, failing, failing] + [broken] * 6)
    assert states[:3] == ["green", "amber", "amber"]
    assert "green" not in states[3:]  # fresh heartbeats, but no stall reading: never a good check


def test_a_payload_without_stall_fields_turns_amber_like_any_unknown(tmp_path):
    states = _states(_config(tmp_path), [FULL] + [{"poll_count": 1}] * 4)
    assert states == ["green", "green", "green", "amber", "amber"]


def test_a_re_read_snapshot_is_not_a_new_alert_check_even_across_sleep():
    """Codex r1 #4: an alert published before sleep, read again before the first poll after wake."""
    codes = ("file_ingestion_failure",)
    first = evaluate(
        Facts(
            now=NOW,
            watcher_updated_at=NOW,
            watcher_stall_codes=codes,
            awake_clock_s=0.0,
            boot_session="b",
        )
    )
    state = first.state
    assert state["checks"]["ingest"]["alert_ticks"] == 1
    for k in range(1, 11):  # lid closed for 2 h, then the same frozen file re-read ten times
        at = NOW + timedelta(hours=2, minutes=5 * k)
        verdict = evaluate(
            Facts(
                now=at,
                watcher_updated_at=NOW,
                watcher_stall_codes=codes,
                awake_clock_s=1.0,
                boot_session="b",
            ),
            state,
        )
        state = verdict.state
        assert state["checks"]["ingest"]["alert_ticks"] == 1, k
    newer = evaluate(
        Facts(
            now=at,
            watcher_updated_at=at,
            watcher_stall_codes=codes,
            awake_clock_s=2.0,
            boot_session="b",
        ),
        state,
    )
    assert newer.state["checks"]["ingest"]["alert_ticks"] == 2


def _snapshots(config, snapshots):
    """One health tick per (updated_at, poll_count, alert reasons), ticking at that updated_at; None re-reads."""
    ticks, last = [], NOW
    for snap in snapshots:
        if snap is None:
            last += timedelta(minutes=5)
            ticks.append((last, None))  # nothing new written: the same file is read again
            continue
        at, polls, reasons = snap
        last = at
        write = lambda at=at, polls=polls, reasons=reasons: _health(  # noqa: E731
            config.watcher_health_path, at, poll_count=polls, alert_reasons=reasons
        )
        ticks.append((at, write))
    runs = _verdict_ticks(config, ticks)
    return [v.check("ingest").state for _, v in runs], [v.state["checks"]["ingest"].get("alert_ticks") for _, v in runs]


def test_a_backwards_clock_correction_during_an_alert_still_escalates_on_the_6th_new_snapshot(tmp_path):
    """Codex r2 #3: a snapshot is new when updated_at OR poll_count changed, not only when it increased."""
    back = NOW - timedelta(hours=1)
    times = [NOW, NOW + timedelta(minutes=5)] + [back + timedelta(minutes=5 * i) for i in range(4)]
    states, _ = _snapshots(_config(tmp_path), [(at, n, ["coverage_drop"]) for n, at in enumerate(times, 1)])
    assert states == ["amber"] * 5 + ["red"]


def test_a_new_poll_count_under_the_same_updated_at_is_a_new_snapshot(tmp_path):
    _, ticks = _snapshots(_config(tmp_path), [(NOW, 1, ["coverage_drop"]), (NOW, 2, ["coverage_drop"])])
    assert ticks == [1, 2]


def test_a_frozen_file_re_read_keeps_its_count(tmp_path):
    _, ticks = _snapshots(_config(tmp_path), [(NOW, 1, ["coverage_drop"])] + [None] * 4)
    assert ticks == [1] * 5


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

    def time(self):
        return self._real.time() + self.wall_shift

    wall_shift = 0.0

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


def test_a_root_that_vanished_while_the_watcher_was_stopped_is_reported_after_restart(tmp_path, monkeypatch):
    """Codex r1 #1: offsets.json is the evidence that the root existed; a restart keeps it."""
    watcher, clock, projects, health = _root_watcher(tmp_path, monkeypatch)
    (projects / "-proj").mkdir(parents=True)
    (projects / "-proj" / "s.jsonl").write_text('{"type":"user","message":{"role":"user","content":"hi"}}\n')
    watcher.poll_once()
    transcript = projects / "-proj" / "s.jsonl"
    watcher.registry.set(str(transcript), transcript.stat().st_size, transcript.stat().st_ino)  # a confirmed read
    watcher.registry.flush()
    projects.rename(projects.with_name("projects-moved"))  # while no watcher runs
    restarted, clock, _, health = _root_watcher(tmp_path, monkeypatch)
    restarted.poll_once()
    assert json.loads(health.read_text())["roots_missing"] == []  # one poll is not an episode
    clock.mono += 300
    restarted.poll_once()
    assert json.loads(health.read_text())["roots_missing"] == ["claude"]


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


FOURTEEN_DAYS = 14 * 24 * 3600


def _offset(watcher, projects, age_s):
    """A transcript under the root whose offset was last confirmed ``age_s`` ago."""
    transcript = projects / "-proj" / "s.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text('{"type":"user","message":{"role":"user","content":"hi"}}\n')
    watcher.registry.set(str(transcript), transcript.stat().st_size, transcript.stat().st_ino)
    watcher.registry._data[str(transcript)]["mtime"] -= age_s
    watcher.registry.flush()


def test_a_root_deleted_on_purpose_after_14_quiet_days_never_alarms(tmp_path, monkeypatch):
    """Lead decision (F1): offsets only make a root expected if confirmed within the last 14 days."""
    watcher, clock, projects, health = _root_watcher(tmp_path, monkeypatch)
    _offset(watcher, projects, age_s=FOURTEEN_DAYS + 3600)
    projects.rename(projects.with_name("projects-moved"))
    restarted, clock, _, health = _root_watcher(tmp_path, monkeypatch)
    for _ in range(3):
        restarted.poll_once()
        assert json.loads(health.read_text())["roots_missing"] == []
        clock.mono += 300


def test_a_root_that_vanished_while_active_alarms_until_14_days_after_its_last_activity(tmp_path, monkeypatch):
    watcher, clock, projects, health = _root_watcher(tmp_path, monkeypatch)
    _offset(watcher, projects, age_s=13 * 24 * 3600)  # a quiet but recent provider
    projects.rename(projects.with_name("projects-moved"))
    restarted, clock, _, health = _root_watcher(tmp_path, monkeypatch)
    restarted.poll_once()
    clock.mono += 300
    restarted.poll_once()
    assert json.loads(health.read_text())["roots_missing"] == ["claude"]
    clock.wall_shift = 24 * 3600 + 60  # its last activity is now older than 14 days
    clock.mono += 30
    restarted.poll_once()
    assert json.loads(health.read_text())["roots_missing"] == []
