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
from brainlayer.watcher import CoverageWatchdog, JSONLWatcher, OffsetRegistry, WatchRoot
from tests.test_codex_zstd_stream import compress as _compress
from tests.test_codex_zstd_stream import rollout
from tests.test_health_facts import NOW, _config, _verdict_ticks

compress = _compress  # the zstd fixture, re-exported for the archive test

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


def _confirming(flushed):
    """A flush that succeeds and confirms what it wrote, like the real queue writer."""

    def flush(items):
        flushed.extend(items)
        confirmed: dict[str, int] = {}
        for item in items:
            source = item["_source_file"]
            confirmed[source] = max(confirmed.get(source, 0), item["_line_end_offset"])
        return confirmed

    return flush


def _watcher(tmp_path, sessions, flushed, **kwargs):
    return JSONLWatcher(
        watch_roots=[WatchRoot("codex", sessions)],
        registry_path=tmp_path / "offsets.json",
        on_flush=_confirming(flushed),
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


def test_a_failing_write_path_takes_no_catch_up_rounds_and_never_reads_as_catching_up(tmp_path):
    # Round 1 (a F3 = b F3): the old name claimed a failing READ; this is every FLUSH failing
    # (queue or DB write path down). Reading ahead into it only grows retained memory.
    sessions = _sessions(tmp_path)
    big = sessions / "big.jsonl"
    _write_lines(big, "big", 0, 3000)
    clock = [0.0]

    def always_fails(items):
        raise RuntimeError("writer unavailable")

    watcher = JSONLWatcher(
        watch_roots=[WatchRoot("codex", sessions)],
        registry_path=tmp_path / "offsets.json",
        on_flush=always_fails,
        batch_size=10,
        flush_interval_ms=360000,
        catch_up_budget_s=5,
        health_path=tmp_path / "watcher-health.json",
        coverage_watchdog=CoverageWatchdog(lag_threshold_bytes=1000, alert_after_s=60, now_fn=lambda: clock[0]),
    )
    states = []
    for _ in range(4):
        watcher.poll_once()
        clock[0] += 30
        health = json.loads((tmp_path / "watcher-health.json").read_text())
        assert watcher._catch_up_rounds == 0
        assert watcher.indexer.retained_failed_input_count() <= 4 * 100  # one slice a poll, as on main
        # bytes_behind is CONFIRMED progress: nothing was confirmed, so the whole file is behind
        assert health["catch_up"]["bytes_behind"] == big.stat().st_size
        states.append(health["catch_up"]["state"])
    assert "catching_up" not in states
    assert watcher.registry.get(str(big))[0] == 0  # nothing confirmed, nothing checkpointed


def test_an_archive_whose_decoder_reads_ahead_with_nothing_confirmed_is_a_stall(tmp_path, monkeypatch, compress):
    # CodeRabbit Major on caaa5a3e: a .zst file's lag mixed its PHYSICAL size with the DECODED read
    # offset, so decoder progress alone made the lag fall and read as catching_up while every
    # write failed. An archive has no confirmed decoded checkpoint to compare: never catching up.
    import os

    monkeypatch.setenv("BRAINLAYER_CODEX_ZST_ADMISSION", "bounded")
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))
    records = b"".join(
        (
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": os.urandom(300).hex()}],
                    },
                }
            )
            + "\n"
        ).encode()
        for _ in range(1000)
    )
    archive = compress("sessions/history.jsonl.zst", rollout() + records)
    assert archive.stat().st_size > 200_000  # random text: the archive stays large
    clock = [0.0]

    def always_fails(_items):
        raise RuntimeError("writer unavailable")

    watcher = JSONLWatcher(
        watch_roots=[WatchRoot("codex", tmp_path / "sessions")],
        registry_path=tmp_path / "offsets.json",
        on_flush=always_fails,
        batch_size=10,
        flush_interval_ms=360000,
        max_lines_per_file=10,
        health_path=tmp_path / "watcher-health.json",
        coverage_watchdog=CoverageWatchdog(lag_threshold_bytes=1000, alert_after_s=60, now_fn=lambda: clock[0]),
    )
    offsets, states = [], []
    for _ in range(8):
        watcher.poll_once()
        clock[0] += 30
        health = json.loads((tmp_path / "watcher-health.json").read_text())
        offsets.append(watcher._tailers[str(archive.with_suffix(""))].offset)
        states.append((health["catch_up"]["state"], "offset_lag" in health["alert_reasons"]))
    assert offsets == sorted(offsets) and offsets[-1] > offsets[0]  # the decoder did read ahead
    assert watcher.registry.get(str(archive.with_suffix("")))[0] == 0  # nothing was confirmed
    assert states[2:] == [("stalled", True)] * 6  # a stall from the alert window on, as on main


def test_a_lag_that_falls_only_because_the_file_shrank_is_not_catching_up():
    # CodeRabbit (review body) on caaa5a3e: a truncation lowers size - confirmed offset with
    # nothing confirmed. Progress needs the confirmed offset to advance in the same generation.
    clock = [0.0]
    watchdog = CoverageWatchdog(lag_threshold_bytes=1_048_576, now_fn=lambda: clock[0])
    states = []
    for poll in range(20):
        lag = 100_000_000 - 1_000_000 * poll  # the file shrinks 1 MB a poll
        result = watchdog.evaluate(
            active_entries_per_minute=0,
            realtime_inserts_per_minute=0,
            max_offset_lag_bytes=lag,
            lag_by_file={"f": lag},
            progress_by_file={"f": (0, poll // 5)},  # confirmed stays 0; each rewrite is a new generation
        )
        states.append(result["catch_up"]["state"])
        clock[0] += 30
    assert "catching_up" not in states
    assert states[-1] == "stalled"


def _flush_test_watcher(tmp_path, monkeypatch, on_flush, batch_size):
    monkeypatch.setenv("BRAINLAYER_WATCH_MAX_FILE_BYTES", "128")
    sessions = _sessions(tmp_path)
    rollout = sessions / "rollout.jsonl"
    watcher = JSONLWatcher(
        watch_roots=[WatchRoot("codex", sessions)],
        registry_path=tmp_path / "offsets.json",
        on_flush=on_flush,
        batch_size=batch_size,
        flush_interval_ms=360000,
        catch_up_budget_s=5,  # catch-up ON: the pinned tests' safety subject, unpinned (b F5)
    )
    return watcher, rollout


def test_with_catch_up_on_retained_unconfirmed_entries_are_never_checkpointed(tmp_path, monkeypatch):
    def fail_flush(_items):
        raise RuntimeError("write unavailable")

    watcher, rollout = _flush_test_watcher(tmp_path, monkeypatch, fail_flush, 10)
    rollout.write_text(json.dumps({"role": "user", "content": "pending"}) + "\n")
    watcher.poll_once()
    with rollout.open("a") as handle:
        handle.write(json.dumps({"role": "user", "content": "x" * 256}) + "\n")
    for _ in range(3):
        watcher.poll_once()
        assert watcher.registry.get(str(rollout)) == (0, 0)


def test_with_catch_up_on_the_checkpoint_never_passes_a_partial_watermark(tmp_path, monkeypatch):
    first_end = len(json.dumps({"role": "user", "content": "first"}) + "\n")
    watcher, rollout = _flush_test_watcher(
        tmp_path, monkeypatch, lambda items: {items[0]["_source_file"]: items[0]["_line_end_offset"]}, 2
    )
    rollout.write_text(
        json.dumps({"role": "user", "content": "first"})
        + "\n"
        + json.dumps({"role": "user", "content": "second"})
        + "\n"
    )
    watcher.poll_once()
    confirmed = watcher.registry.get(str(rollout))
    assert confirmed[0] == first_end
    with rollout.open("a") as handle:
        handle.write(json.dumps({"role": "user", "content": "x" * 256}) + "\n")
    for _ in range(3):
        watcher.poll_once()
        assert watcher.registry.get(str(rollout)) == confirmed


def test_with_catch_up_on_an_unparsed_window_is_never_checkpointed(tmp_path, monkeypatch):
    watcher, rollout = _flush_test_watcher(tmp_path, monkeypatch, lambda _items: None, 10)
    rollout.write_text(json.dumps({"role": "user", "content": "x" * 256}) + "\n")
    for _ in range(3):
        watcher.poll_once()
        watcher.registry.flush()
        assert OffsetRegistry(tmp_path / "offsets.json").get(str(rollout)) == (0, 0)


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
    # the first growing poll is above the window's low: a stall at once, and every poll after it
    assert states == [("stalled", True)] * 4


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
        write = lambda at=at, payload=payload, poll_count=i + 1: config.watcher_health_path.write_text(  # noqa: E731
            json.dumps({**payload, "poll_count": poll_count, "updated_at": at.isoformat()})
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
    for bad in (
        "catching_up",
        {"state": 1},
        ["catching_up"],
        None,  # CodeRabbit: an explicit null is not an older watcher's omission
        {"state": "catching_up"},  # CodeRabbit: progress needs a measured lag and ETA
        {"state": "catching_up", "bytes_behind": 107_000_000, "eta_s": None},
        {"state": "catching_up", "bytes_behind": -1, "eta_s": 720.0},
    ):
        config.watcher_health_path.write_text(json.dumps({**_payload(catch_up=bad), "updated_at": NOW.isoformat()}))
        assert watcher_health_fact(config.watcher_health_path)[1] is UNOBSERVED, bad
    config.watcher_health_path.write_text(json.dumps({**_payload(), "updated_at": NOW.isoformat()}))
    assert watcher_health_fact(config.watcher_health_path)[1] == ()  # an older watcher: no block, no code


@pytest.mark.parametrize("budget", [float("inf"), float("nan")])
def test_an_explicit_nonfinite_budget_falls_back_to_the_default(tmp_path, budget):
    watcher = _watcher(tmp_path, _sessions(tmp_path), [], catch_up_budget_s=budget)
    assert watcher.catch_up_budget_s == 10.0


def test_stop_during_catch_up_ends_the_rounds_after_the_slice_in_flight(tmp_path):
    sessions = _sessions(tmp_path)
    big = sessions / "big.jsonl"
    _write_lines(big, "big", 0, 3000)
    flushed: list[dict] = []
    confirm = _confirming(flushed)
    holder = {}

    def stop_after_first_round_slice(items):
        result = confirm(items)
        if holder["w"]._catch_up_rounds >= 1:  # stop() arrives while a catch-up round runs
            holder["w"].stop()
        return result

    watcher = JSONLWatcher(
        watch_roots=[WatchRoot("codex", sessions)],
        registry_path=tmp_path / "offsets.json",
        on_flush=stop_after_first_round_slice,
        batch_size=10,
        catch_up_budget_s=10,
    )
    holder["w"] = watcher
    watcher.poll_once()
    assert watcher._catch_up_rounds == 1
    assert len(_texts(flushed, "big")) == 200  # the normal slice plus the one round slice in flight


def test_a_denylisted_source_drops_its_growth_backfill_marker(tmp_path, monkeypatch):
    sessions = _sessions(tmp_path)
    big = sessions / "big.jsonl"
    _write_lines(big, "big", 0, 50)
    watcher = _watcher(tmp_path, sessions, [])
    watcher.indexer.on_flush = lambda _items: None  # unconfirmed: the marker survives the read
    watcher.poll_once()
    assert str(big) in watcher._growth_backfill
    monkeypatch.setenv("BRAINLAYER_INGEST_DENYLIST", str(big))
    watcher.poll_once()
    assert str(big) not in watcher._growth_backfill  # retired with its registry entry


# ── Round 1: amber only for genuine progress ─────────────────────────────────


def _watchdog_states(lags: list[int], **kwargs) -> list[str]:
    """One file's lag per 30 s poll through the real CoverageWatchdog; the catch_up state each poll."""
    clock = [0.0]
    watchdog = CoverageWatchdog(lag_threshold_bytes=1_048_576, now_fn=lambda: clock[0], **kwargs)
    states = []
    for lag in lags:
        result = watchdog.evaluate(
            active_entries_per_minute=0,
            realtime_inserts_per_minute=0,
            max_offset_lag_bytes=lag,
            lag_by_file={"f": lag},
        )
        assert ("offset_lag" in result["alert_reasons"]) == (result["catch_up"]["state"] == "stalled")
        states.append(result["catch_up"]["state"])
        clock[0] += 30
    return states


def test_a_lag_that_grows_overall_in_a_sawtooth_is_a_stall_never_catching_up():
    # Reviewer a's sawtooth: 5 MB behind, 200 KB read a poll, a 900 KB burst every 3rd poll;
    # over 6 h (720 polls) the lag grows 5.0 -> 76 MB. 0730dd20 called 237 of those polls
    # catching_up (a 60 s look-back); net progress over the whole window never happens.
    lag, lags = 5_000_000, []
    for poll in range(720):
        lag -= 200_000
        if poll % 3 == 2:
            lag += 900_000
        lags.append(lag)
    assert lags[-1] > 70_000_000
    states = _watchdog_states(lags)
    assert states[10:].count("catching_up") == 0
    assert states[-1] == "stalled"


def test_a_reader_re_reading_from_zero_every_few_polls_is_a_stall():
    # Reviewer a's re-read loop: the lag drops for 5 polls, then jumps back (offset reset to 0).
    lags = [20_000_000 - 1_000_000 * (poll % 6) for poll in range(120)]
    assert _watchdog_states(lags)[12:].count("catching_up") == 0


@pytest.mark.parametrize("step", [1, 1024], ids=["1 byte a poll (95-year ETA)", "1 KB a poll (37-day ETA)"])
def test_a_lag_that_barely_shrinks_is_a_stall_not_catching_up(step):
    lags = [107_000_000 - step * poll for poll in range(400)]
    assert _watchdog_states(lags)[10:].count("catching_up") == 0


def test_a_genuine_drain_under_the_eta_ceiling_is_catching_up():
    # 107 MB at about 330 KB a poll is about 2.7 h: under the 6 h ceiling.
    lags = [107_000_000 - 330_000 * poll for poll in range(40)]
    states = _watchdog_states(lags)
    assert states[10:] == ["catching_up"] * 30


S = STALLED
C = CATCHING_UP


def _first_red(checks) -> int | None:
    return next((i for i, c in enumerate(checks) if c.state == "red"), None)


@pytest.mark.parametrize(
    "pattern",
    [[S, S, C], [S, S, S, S, S, C]],
    ids=["stalled-stalled-catching_up", "5x stalled then catching_up"],
)
def test_a_catching_up_snapshot_holds_the_count_so_a_flickering_stall_still_goes_red_on_time(tmp_path, pattern):
    # Reviewer b's patterns: on 0730dd20 each catching_up reset the count, so neither ever went red.
    payloads = (pattern * 10)[:24]
    checks = _verdicts(_config(tmp_path), payloads)
    sixth_stall = [i for i, p in enumerate(payloads) if p is S][5]
    assert _first_red(checks) == sixth_stall  # red on the 6th stall snapshot, as main counts them
    later_stalls = [i for i, p in enumerate(payloads) if p is S][6:]
    assert all(checks[i].state == "red" for i in later_stalls)  # every later stall snapshot is red


def test_a_red_check_steps_down_to_amber_during_a_genuine_catch_up_and_back_to_red_when_it_stops(tmp_path):
    # Round 1 item 4 (a F4): red from an earlier stall, then a measured catch-up is amber with its
    # ETA (0730dd20 held red for the whole catch-up); a stall after it is red again at once.
    checks = _verdicts(_config(tmp_path), [S] * 6 + [C] * 3 + [S])
    assert [c.state for c in checks] == ["amber"] * 5 + ["red"] + ["amber"] * 3 + ["red"]
    assert "about 12 min left" in checks[6].note
