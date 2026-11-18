"""Catch-up on a large transcript within each poll, with every other file keeping its turn.

2026-11-18, M4: a 146 MB resumed transcript drained at max_lines_per_file (100) lines per 30 s
poll, about 7 hours, and the ingest check sat red on offset_lag the whole time while the lag
visibly shrank. These tests pin the three halves of the fix:

- catch-up rounds: after the normal pass, files still behind keep reading within a time budget;
- fairness: every small live file still advances on every poll;
- the verdict: a shrinking lag is ``catch_up.state = catching_up`` (amber, with an estimate),
  never ``offset_lag``; a lag that stops shrinking is ``offset_lag`` at once and red as before.

Every test except the budget/guard ones fails on main 4923d5a8.
"""

from __future__ import annotations

import json
import time
from datetime import timedelta

import pytest

from brainlayer.health_facts import watcher_health_fact
from brainlayer.health_verdict import UNOBSERVED
from brainlayer.watcher import CoverageWatchdog, JSONLWatcher, WatchRoot
from tests.test_health_facts import NOW, _config, _verdict_ticks

PAD = "x" * 160


def _line(tag: str, idx: int) -> str:
    return json.dumps({"role": "user", "content": f"{tag} line {idx} {PAD}"}) + "\n"


def _write_lines(path, tag: str, start: int, count: int) -> None:
    with path.open("a") as handle:
        handle.writelines(_line(tag, idx) for idx in range(start, start + count))


def _sessions(tmp_path):
    sessions = tmp_path / "codex" / "sessions"
    sessions.mkdir(parents=True)
    return sessions


def _watcher(tmp_path, sessions, flushed, **kwargs):
    return JSONLWatcher(
        watch_roots=[WatchRoot("codex", sessions)],
        registry_path=tmp_path / "offsets.json",
        on_flush=lambda items: flushed.extend(items),
        batch_size=10,
        health_path=tmp_path / "watcher-health.json",
        **kwargs,
    )


def _texts(flushed, tag: str) -> list[str]:
    return [
        item["message"]["content"][0]["text"].split(" ")[2]
        for item in flushed
        if item["message"]["content"][0]["text"].startswith(f"{tag} ")
    ]


# ── Catch-up rounds and fairness ─────────────────────────────────────────────


def test_a_large_finished_file_catches_up_in_one_poll_while_small_live_files_keep_their_turn(tmp_path):
    sessions = _sessions(tmp_path)
    big = sessions / "big.jsonl"
    _write_lines(big, "big", 0, 5000)
    small = [sessions / f"small-{i}.jsonl" for i in range(3)]
    for path in small:
        _write_lines(path, path.stem, 0, 1)
    flushed: list[dict] = []
    watcher = _watcher(tmp_path, sessions, flushed)  # production default: max_lines_per_file=100

    polls = 0
    while watcher._tailers.get(str(big)) is None or watcher._tailers[str(big)].offset < big.stat().st_size:
        assert polls < 3, "5,000 lines at 100 per poll would need 50 polls; catch-up must finish in 1-2"
        before = {path.stem: len(_texts(flushed, path.stem)) for path in small}
        watcher.poll_once()
        polls += 1
        for path in small:  # fairness: every small file advanced on this very poll
            assert len(_texts(flushed, path.stem)) > before[path.stem]
            _write_lines(path, path.stem, polls, 1)  # it is live: one more line before the next poll

    assert _texts(flushed, "big") == [str(i) for i in range(5000)]  # every line, in order, once
    health = json.loads((tmp_path / "watcher-health.json").read_text())
    assert health["catch_up"]["last_poll_rounds"] >= 1


def test_a_still_growing_large_file_catches_up_to_its_discovered_size_and_never_starves_the_rest(tmp_path):
    sessions = _sessions(tmp_path)
    big = sessions / "big.jsonl"
    _write_lines(big, "big", 0, 2000)
    small = sessions / "small.jsonl"
    _write_lines(small, "small", 0, 1)
    flushed: list[dict] = []
    grown = [2000]

    def grow_while_read(items):
        # The large session is still being written while the watcher reads it: every flush of
        # its lines appends ten more, so it never reaches EOF within a poll.
        flushed.extend(items)
        if any(item["_source_file"] == str(big) for item in items):
            _write_lines(big, "big", grown[0], 10)
            grown[0] += 10

    watcher = JSONLWatcher(
        watch_roots=[WatchRoot("codex", sessions)],
        registry_path=tmp_path / "offsets.json",
        on_flush=grow_while_read,
        batch_size=10,
        health_path=tmp_path / "watcher-health.json",
    )
    for poll in range(1, 4):
        discovered = big.stat().st_size
        started = time.monotonic()
        watcher.poll_once()
        assert time.monotonic() - started < 15  # the default 10 s budget bounds it; never chasing EOF
        # caught up to at least the size discovery saw, far past one 100-line slice
        assert watcher._tailers[str(big)].offset >= discovered
        assert len(_texts(flushed, "small")) == poll  # the small file had its turn on every poll
        _write_lines(small, "small", poll, 1)


def test_the_budget_bounds_catch_up_and_zero_restores_one_slice_per_poll(tmp_path, monkeypatch):
    sessions = _sessions(tmp_path)
    big = sessions / "big.jsonl"
    _write_lines(big, "big", 0, 3000)
    flushed: list[dict] = []

    def slow(items):
        time.sleep(0.02)
        flushed.extend(items)

    watcher = JSONLWatcher(
        watch_roots=[WatchRoot("codex", sessions)],
        registry_path=tmp_path / "offsets.json",
        on_flush=slow,
        batch_size=10,
        catch_up_budget_s=0.3,
    )
    started = time.monotonic()
    watcher.poll_once()
    elapsed = time.monotonic() - started
    assert elapsed < 0.3 + 1.5  # the round in flight finishes; no new round starts past the budget
    assert 100 < len(flushed) < 3000  # more than one slice, not the whole file

    monkeypatch.setenv("BRAINLAYER_WATCH_CATCHUP_BUDGET_S", "0")
    off = _watcher(tmp_path / "off", _sessions(tmp_path / "off"), [])
    assert off.catch_up_budget_s == 0


@pytest.mark.parametrize("raw", ["nan", "inf", "-1", "soon"])
def test_a_bad_budget_value_falls_back_to_the_default(tmp_path, monkeypatch, raw):
    monkeypatch.setenv("BRAINLAYER_WATCH_CATCHUP_BUDGET_S", raw)
    watcher = _watcher(tmp_path, _sessions(tmp_path), [])
    assert watcher.catch_up_budget_s == 10.0


def test_a_file_that_fails_every_read_takes_no_catch_up_rounds(tmp_path):
    sessions = _sessions(tmp_path)
    big = sessions / "big.jsonl"
    _write_lines(big, "big", 0, 500)
    calls = []

    def always_fails(items):
        calls.append(len(items))
        raise RuntimeError("writer unavailable")

    watcher = JSONLWatcher(
        watch_roots=[WatchRoot("codex", sessions)],
        registry_path=tmp_path / "offsets.json",
        on_flush=always_fails,
        batch_size=10,
        flush_interval_ms=360000,
        catch_up_budget_s=5,
    )
    started = time.monotonic()
    watcher.poll_once()
    assert time.monotonic() - started < 3  # did not spin the 5 s budget on a failing file
    # nothing was confirmed, so nothing was checkpointed, catch-up or not
    assert watcher.registry.get(str(big))[0] == 0


# ── What the watcher publishes ───────────────────────────────────────────────


def _lagging_watcher(tmp_path, monkeypatch, clock):
    """One slice per poll (budget 0, via the env so main ignores it), 10 lines, tiny threshold."""
    monkeypatch.setenv("BRAINLAYER_WATCH_CATCHUP_BUDGET_S", "0")
    sessions = _sessions(tmp_path)
    big = sessions / "big.jsonl"
    _write_lines(big, "big", 0, 400)
    watcher = _watcher(
        tmp_path,
        sessions,
        [],
        max_lines_per_file=10,
        coverage_watchdog=CoverageWatchdog(lag_threshold_bytes=1000, alert_after_s=60, now_fn=lambda: clock[0]),
    )
    return watcher, big


def _poll(watcher, clock, tmp_path):
    watcher.poll_once()
    clock[0] += 30
    return json.loads((tmp_path / "watcher-health.json").read_text())


def test_a_shrinking_lag_is_catching_up_with_an_estimate_not_offset_lag(tmp_path, monkeypatch):
    clock = [0.0]
    watcher, big = _lagging_watcher(tmp_path, monkeypatch, clock)
    health = [_poll(watcher, clock, tmp_path) for _ in range(5)]
    assert [h["catch_up"]["state"] for h in health] == ["behind", "behind", "catching_up", "catching_up", "catching_up"]
    assert all("offset_lag" not in h["alert_reasons"] for h in health)
    last = health[-1]["catch_up"]
    assert last["bytes_behind"] == health[-1]["max_offset_lag_bytes"] > 0
    assert last["files_behind"] == 1 and last["lagging_files"] == 1
    # about 10 lines (~1.9 KB) per 30 s poll: the estimate is bytes behind over that rate
    assert last["drain_bytes_per_s"] > 0
    assert last["eta_s"] == pytest.approx(last["bytes_behind"] / last["drain_bytes_per_s"], rel=0.01)


def test_a_lag_that_stops_shrinking_is_offset_lag_at_once(tmp_path, monkeypatch):
    clock = [0.0]
    watcher, big = _lagging_watcher(tmp_path, monkeypatch, clock)
    for _ in range(4):
        assert "offset_lag" not in _poll(watcher, clock, tmp_path)["alert_reasons"]
    states = []
    for poll in range(4):  # the session now writes 30 lines per poll; the watcher reads 10
        _write_lines(big, "big", 1000 + 30 * poll, 30)
        health = _poll(watcher, clock, tmp_path)
        states.append((health["catch_up"]["state"], "offset_lag" in health["alert_reasons"]))
    # within the 60 s progress span the growth shows; from then on it is a stall, as today
    assert states[-2:] == [("stalled", True), ("stalled", True)]


def test_one_stalled_file_keeps_offset_lag_while_a_bigger_file_drains_beside_it():
    clock = [0.0]
    watchdog = CoverageWatchdog(lag_threshold_bytes=100, alert_after_s=60, now_fn=lambda: clock[0])
    result = None
    for poll in range(5):
        lags = {"draining": 100_000 - 10_000 * poll, "stuck": 5_000}
        result = watchdog.evaluate(
            active_entries_per_minute=0,
            realtime_inserts_per_minute=0,
            max_offset_lag_bytes=max(lags.values()),
            lag_by_file=lags,
        )
        clock[0] += 30
    assert result["alert_reasons"] == ["offset_lag"]
    assert result["catch_up"]["state"] == "stalled"


# ── The verdict ──────────────────────────────────────────────────────────────


def _payload(**extra):
    return {
        "poll_count": 1,
        "alerting": False,
        "alert_reasons": [],
        "db_probe_failed": False,
        "db_probe_failed_for_s": 0.0,
        "roots_missing": [],
        **extra,
    }


CATCHING_UP = _payload(catch_up={"state": "catching_up", "bytes_behind": 107_000_000, "eta_s": 720.0})
STALLED = _payload(
    alerting=True, alert_reasons=["offset_lag"], catch_up={"state": "stalled", "bytes_behind": 107_000_000}
)


def _verdicts(config, payloads):
    ticks = []
    for i, payload in enumerate(payloads):
        at = NOW + timedelta(minutes=5 * i)
        write = lambda at=at, payload=payload: config.watcher_health_path.write_text(  # noqa: E731
            json.dumps({**payload, "poll_count": i + 1, "updated_at": at.isoformat()})
        )
        ticks.append((at, write))
    return [verdict.check("ingest") for _, verdict in _verdict_ticks(config, ticks)]


def test_a_shrinking_lag_is_amber_with_its_estimate_for_as_long_as_it_shrinks(tmp_path):
    checks = _verdicts(_config(tmp_path), [CATCHING_UP] * 10)
    assert [c.state for c in checks] == ["amber"] * 10  # never red: it is progress, not a stall
    assert "watcher catching up: 107.0 MB behind, about 12 min left" in checks[-1].note


def test_a_lag_that_stalls_after_catching_up_goes_red_after_its_threshold_as_today(tmp_path):
    checks = _verdicts(_config(tmp_path), [CATCHING_UP] * 4 + [STALLED] * 6)
    # catching up never counted: the stall counts from 1 and is red on its 6th check
    assert [c.state for c in checks] == ["amber"] * 9 + ["red"]


def test_a_malformed_catch_up_block_is_unreadable_not_a_clean_reading(tmp_path):
    config = _config(tmp_path)
    for bad in ("catching_up", {"state": 1}, ["catching_up"]):
        config.watcher_health_path.write_text(json.dumps({**_payload(catch_up=bad), "updated_at": NOW.isoformat()}))
        assert watcher_health_fact(config.watcher_health_path)[1] is UNOBSERVED
    config.watcher_health_path.write_text(json.dumps({**_payload(), "updated_at": NOW.isoformat()}))
    assert watcher_health_fact(config.watcher_health_path)[1] == ()  # an older watcher: no block, no code
