"""A T3 session that T3 never links is held for a bounded window, never counted as a stall.

2026-11-19, M1 on 1.6.6: eight T3-origin Codex sessions of 12 Nov were never linked by T3 (it kept
only the next Codex session of each thread). The bridge held their entries unconfirmed forever
(``t3_runtime_linkage_pending``), and #1827 measures lag from the confirmed offset, so each whole
file counted as behind. No catch-up round could drain it (the reader was already at EOF), and the
ingest check went red with nothing Etan could do.

These tests pin the fix, through the real watcher, the real flush callback and the real verdict:

- inside the hold window a held file is published as ``catch_up.held_for_linkage``, not as lag:
  no ``stalled``, no ``offset_lag``, never red (fails on main 6f7f7ae4: stalled, then red);
- after the window it is indexed with the ``t3-unlinked`` provenance and its offset is confirmed
  (fails on main: held forever, offset 0);
- a link that arrives inside the window keeps today's behaviour (``t3-app-session``);
- a held file still unconfirmed past the window is ordinary lag again, and red as before.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import timedelta
from pathlib import Path

import pytest

from brainlayer.watcher import CoverageWatchdog, JSONLWatcher, WatchRoot
from brainlayer.watcher_bridge import FlushWatermarks, create_flush_callback
from tests.test_health_facts import NOW, _config, _verdict_ticks

DAY_S = 24 * 3600


def _uuid7(started_at: float, tail: str = "7abc-8def-0123456789ab") -> str:
    """A Codex-style UUIDv7 session id whose embedded time is ``started_at``."""
    millis = f"{int(started_at * 1000):012x}"
    return f"{millis[:8]}-{millis[8:12]}-{tail}"


def _projection(tmp_path: Path, linked: list[str]) -> Path:
    """A T3 V2 provider projection linking only ``linked`` Codex session ids."""
    path = tmp_path / "statev2.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS orchestration_v2_projection_provider_threads "
            "(provider_thread_id TEXT, provider TEXT, payload_json TEXT, driver TEXT)"
        )
        conn.execute("DELETE FROM orchestration_v2_projection_provider_threads")
        conn.executemany(
            "INSERT INTO orchestration_v2_projection_provider_threads VALUES (?, ?, ?, ?)",
            [
                (
                    f"pt-{i}",
                    "codex",
                    json.dumps({"driver": "codex", "nativeThreadRef": {"driver": "codex", "nativeId": sid}}),
                    "codex",
                )
                for i, sid in enumerate(linked)
            ],
        )
    return path


def _rollout(tmp_path: Path, session_id: str, lines: int = 20) -> Path:
    """A T3-origin Codex rollout: session_meta header, then ``lines`` user messages."""
    source = (
        tmp_path / "home" / ".codex" / "sessions" / "2026" / "11" / f"rollout-2026-11-12T13-10-04-{session_id}.jsonl"
    )
    source.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "type": "session_meta",
            "payload": {"id": session_id, "originator": "T3 Code", "source": "vscode", "cwd": "/fixture/repo"},
        }
    ] + [
        {
            "type": "response_item",
            "timestamp": "2026-11-12T11:10:05Z",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": f"held linkage message {i} " + "x" * 160}],
            },
        }
        for i in range(lines)
    ]
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return source


def _env(monkeypatch, tmp_path: Path, state: Path) -> Path:
    queue = tmp_path / "queue"
    monkeypatch.setenv("BRAINLAYER_T3_STATE_DB", str(state))
    monkeypatch.setenv("BRAINLAYER_T3_PROJECTION_VERSION", "2")
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue))
    return queue


def _watcher(tmp_path: Path, source: Path, health: Path, clock: list[float], on_flush=None, **kwargs) -> JSONLWatcher:
    return JSONLWatcher(
        **kwargs,
        watch_roots=[WatchRoot("codex", source.parents[2])],
        registry_path=tmp_path / "offsets.json",
        on_flush=on_flush or create_flush_callback(arbitrated=True),
        batch_size=10,
        flush_interval_ms=0,
        health_path=health,
        coverage_watchdog=CoverageWatchdog(lag_threshold_bytes=1000, alert_after_s=60, now_fn=lambda: clock[0]),
    )


def _events(queue: Path, source: Path) -> list[dict]:
    return [
        event
        for path in sorted(queue.glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if (event := json.loads(line))["source_file"] == str(source)
    ]


def _ingest_states(tmp_path: Path, watcher: JSONLWatcher, clock: list[float], ticks: int) -> list[str]:
    """Poll the real watcher once per 5-minute health tick and run the real verdict on its health file."""
    config = _config(tmp_path)

    def poll(at, index):
        def run():
            clock[0] += 300
            watcher.poll_once()
            payload = json.loads(watcher.health_path.read_text())
            payload.update(updated_at=at.isoformat(), poll_count=index + 1)
            config.watcher_health_path.write_text(json.dumps(payload))

        return run

    plan = [(NOW + timedelta(minutes=5 * i), poll(NOW + timedelta(minutes=5 * i), i)) for i in range(ticks)]
    return [verdict.check("ingest").state for _, verdict in _verdict_ticks(config, plan)]


def test_a_held_never_linked_session_inside_the_window_is_held_for_linkage_not_a_stall(tmp_path, monkeypatch):
    session = _uuid7(time.time() - 60)  # started a minute ago: well inside the 24 h window
    source = _rollout(tmp_path, session)
    _env(monkeypatch, tmp_path, _projection(tmp_path, linked=[]))
    clock = [0.0]
    watcher = _watcher(tmp_path, source, tmp_path / "watcher-health.json", clock)

    states = _ingest_states(tmp_path, watcher, clock, ticks=8)

    health = json.loads(watcher.health_path.read_text())
    assert watcher.registry.get(str(source))[0] == 0  # still held: nothing confirmed yet
    assert health["catch_up"]["state"] != "stalled"
    assert "offset_lag" not in health["alert_reasons"]
    assert health["catch_up"]["files_behind"] == 0
    assert health["catch_up"]["bytes_behind"] == 0
    held = health["catch_up"]["held_for_linkage"]
    assert (held["files"], held["bytes"]) == (1, source.stat().st_size)
    assert 0 <= held["oldest_age_s"] < DAY_S and held["window_s"] == DAY_S
    assert "red" not in states and states[-1] == "green"


def test_after_the_window_the_session_is_indexed_as_t3_unlinked_and_its_offset_confirmed(tmp_path, monkeypatch):
    session = _uuid7(time.time() - 7 * DAY_S)  # the M1's sessions: a week old, never linked
    source = _rollout(tmp_path, session)
    queue = _env(monkeypatch, tmp_path, _projection(tmp_path, linked=[]))
    clock = [0.0]
    watcher = _watcher(tmp_path, source, tmp_path / "watcher-health.json", clock)

    watcher.poll_once()

    events = _events(queue, source)
    assert len(events) == 20
    assert {(event["provenance_class"], event["source_class"]) for event in events} == {("t3-unlinked", "cli-agent")}
    assert watcher.registry.get(str(source))[0] == source.stat().st_size
    assert watcher.indexer._buffer == []
    health = json.loads(watcher.health_path.read_text())
    assert health["catch_up"]["held_for_linkage"]["files"] == 0
    assert health["catch_up"]["bytes_behind"] == 0


def test_a_link_that_arrives_inside_the_window_keeps_todays_behaviour(tmp_path, monkeypatch):
    session = _uuid7(time.time() - 30)
    source = _rollout(tmp_path, session)
    queue = _env(monkeypatch, tmp_path, _projection(tmp_path, linked=[]))
    clock = [0.0]
    watcher = _watcher(tmp_path, source, tmp_path / "watcher-health.json", clock)

    watcher.poll_once()
    assert _events(queue, source) == []
    assert watcher.registry.get(str(source))[0] == 0

    _projection(tmp_path, linked=[session])  # T3 links the session a few seconds later
    watcher.poll_once()

    events = _events(queue, source)
    assert {event["provenance_class"] for event in events} == {"t3-app-session"}
    assert len(events) == 20
    assert watcher.registry.get(str(source))[0] == source.stat().st_size


def test_a_held_session_still_unconfirmed_past_the_window_is_lag_and_reaches_red(tmp_path, monkeypatch):
    """A hold that outlives its window is a real fault: it must not hide behind held_for_linkage."""
    session = _uuid7(time.time() - 7 * DAY_S)
    source = _rollout(tmp_path, session)
    _env(monkeypatch, tmp_path, _projection(tmp_path, linked=[]))
    expired = time.time() - 7 * DAY_S

    def stuck(items):  # a release that never happens: every entry deferred, tagged as an old hold
        for item in items:
            item["_t3_linkage_hold_started_at"] = expired
        return FlushWatermarks({}, deferred_entries=items)

    clock = [0.0]
    watcher = _watcher(tmp_path, source, tmp_path / "watcher-health.json", clock, on_flush=stuck)

    states = _ingest_states(tmp_path, watcher, clock, ticks=8)

    health = json.loads(watcher.health_path.read_text())
    assert "offset_lag" in health["alert_reasons"]
    assert health["catch_up"]["state"] == "stalled"
    assert health["catch_up"]["bytes_behind"] == source.stat().st_size
    assert "red" in states


def test_a_partly_read_held_file_still_counts_its_unread_bytes_as_lag(tmp_path, monkeypatch):
    """Only bytes already read and held for linkage leave the lag; unread bytes are ordinary lag."""
    session = _uuid7(time.time() - 60)
    source = _rollout(tmp_path, session, lines=400)
    _env(monkeypatch, tmp_path, _projection(tmp_path, linked=[]))
    clock = [0.0]
    # No catch-up rounds: one 100-line slice per poll.
    watcher = _watcher(tmp_path, source, tmp_path / "watcher-health.json", clock, catch_up_budget_s=0)

    watcher.poll_once()

    read = watcher._tailers[str(source)].offset
    size = source.stat().st_size
    assert 0 < read < size
    health = json.loads(watcher.health_path.read_text())
    assert health["catch_up"]["held_for_linkage"]["bytes"] == read
    assert health["catch_up"]["bytes_behind"] == size - read


def test_the_hold_clock_is_the_sessions_own_start_and_the_window_is_tunable(tmp_path, monkeypatch):
    from brainlayer.t3_provenance import (
        DEFAULT_T3_LINKAGE_HOLD_S,
        codex_session_started_at,
        t3_linkage_hold_started_at,
        t3_linkage_hold_window_s,
    )

    started = time.time() - 3 * DAY_S
    assert codex_session_started_at(_uuid7(started)) == pytest.approx(started, abs=0.001)
    assert codex_session_started_at("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa") is None  # not a UUIDv7
    source = _rollout(tmp_path, _uuid7(started), lines=1)
    assert t3_linkage_hold_started_at(source) == pytest.approx(started, abs=0.001)

    # A copied or touched file keeps its session's start: the clock is in the data, not the inode.
    os.utime(source, None)
    assert t3_linkage_hold_started_at(source) == pytest.approx(started, abs=0.001)

    # Without a UUIDv7 the file's mtime is the (later, so longer-holding) fallback.
    plain = _rollout(tmp_path / "plain", "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", lines=1)
    assert t3_linkage_hold_started_at(plain) == pytest.approx(plain.stat().st_mtime, abs=0.001)

    assert DEFAULT_T3_LINKAGE_HOLD_S == DAY_S
    monkeypatch.delenv("BRAINLAYER_T3_LINKAGE_HOLD_S", raising=False)
    assert t3_linkage_hold_window_s() == DAY_S
    monkeypatch.setenv("BRAINLAYER_T3_LINKAGE_HOLD_S", "3600")
    assert t3_linkage_hold_window_s() == 3600
    for bad in ("", "nan", "inf", "-1", "soon"):
        monkeypatch.setenv("BRAINLAYER_T3_LINKAGE_HOLD_S", bad)
        assert t3_linkage_hold_window_s() == DAY_S, bad


def _append_message(source: Path, text: str) -> None:
    record = {
        "type": "response_item",
        "timestamp": "2026-11-19T18:00:00Z",
        "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
    }
    with source.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _rows(db: Path, source: Path) -> dict[str, str]:
    # The same SQLite library as the bridge's writer: a second library (sqlite3) on the same WAL
    # file in one process breaks the other's POSIX locks.
    import apsw

    conn = apsw.Connection(str(db))
    try:
        return dict(conn.execute("SELECT id, provenance_class FROM chunks WHERE source_file = ?", (str(source),)))
    finally:
        conn.close()


def test_a_late_link_after_release_relabels_in_place_with_no_duplicate_and_no_lost_row(tmp_path, monkeypatch):
    """T3 links a session AFTER it was indexed as t3-unlinked (never seen on either Mac, but possible).

    What happens: new entries of the session are t3-app-session at once; a re-read from 0 (a
    restart that lost its offsets) adds no row, because chunk ids are content hashes of the source;
    the rows already indexed are relabelled t3-app-session in place by the reviewed, reversible
    retag (scripts/retag_t3_app_provenance.py --apply, the lead runs it), never re-inserted.
    """
    from scripts.retag_t3_app_provenance import retag_t3_app_chunks, rollback_t3_app_chunks

    session = _uuid7(time.time() - 7 * DAY_S)
    source = _rollout(tmp_path, session)
    state = _projection(tmp_path, linked=[])
    _env(monkeypatch, tmp_path, state)
    db = tmp_path / "brainlayer.db"
    clock = [0.0]
    flush = create_flush_callback(db_path=db, arbitrated=False)
    watcher = _watcher(tmp_path, source, tmp_path / "watcher-health.json", clock, on_flush=flush)

    watcher.poll_once()
    released = _rows(db, source)
    assert len(released) == 20 and set(released.values()) == {"t3-unlinked"}

    _projection(tmp_path, linked=[session])  # T3 links it late
    _append_message(
        source,
        "Once the thread finally showed up in T3, Etan asked for the release notes to name the "
        "bounded hold, the evidence behind its window, and the relabel step the lead runs.",
    )
    watcher.poll_once()
    linked = _rows(db, source)
    assert len(linked) == 21 and set(released) < set(linked)
    assert {linked[chunk_id] for chunk_id in released} == {"t3-unlinked"}
    assert [linked[chunk_id] for chunk_id in set(linked) - set(released)] == ["t3-app-session"]

    # A restart that lost its offsets re-reads the whole file: no duplicate, nothing lost.
    rereader = JSONLWatcher(
        watch_roots=[WatchRoot("codex", source.parents[2])],
        registry_path=tmp_path / "offsets-after-restart.json",
        on_flush=flush,
        batch_size=10,
        flush_interval_ms=0,
    )
    rereader.poll_once()
    assert _rows(db, source) == linked

    # The reviewed relabel: in place, same ids, reversible. It runs as its own process in
    # production; here the watcher's writer closes first (it is sqlite3, the writer apsw).
    flush.close()
    report = retag_t3_app_chunks(db_path=db, state_db=state, apply=True, rollback_artifact=tmp_path / "rollback.jsonl")
    assert report["retagged_chunks"] == 20
    relabelled = _rows(db, source)
    assert set(relabelled) == set(linked) and set(relabelled.values()) == {"t3-app-session"}
    assert rollback_t3_app_chunks(db_path=db, rollback_artifact=tmp_path / "rollback.jsonl")["restored_chunks"] == 20
    assert _rows(db, source) == linked


def test_the_hold_never_confirms_a_byte_the_reader_has_not_consumed(tmp_path, monkeypatch):
    """Neither the hold nor its release may move a confirmed offset past what the reader consumed.

    Inside the window nothing is confirmed while the reader advances; after it, each poll confirms
    exactly the line ends it read, slice by slice, never the file's size ahead of the reader.
    """
    held_source = _rollout(tmp_path / "held", _uuid7(time.time() - 60), lines=400)
    released_source = _rollout(tmp_path / "released", _uuid7(time.time() - 7 * DAY_S), lines=400)
    _env(monkeypatch, tmp_path, _projection(tmp_path, linked=[]))

    for source, releases in ((held_source, False), (released_source, True)):
        clock = [0.0]
        watcher = _watcher(
            source.parents[5], source, source.parents[5] / "watcher-health.json", clock, catch_up_budget_s=0
        )
        size = source.stat().st_size
        data = source.read_bytes()
        previous = 0
        for _ in range(8):
            watcher.poll_once()
            read = watcher._tailers[str(source)].offset
            confirmed = watcher.registry.get(str(source))[0]
            assert confirmed <= read, (source.name, confirmed, read)
            if releases:
                assert confirmed == read  # every consumed line, and only those
                assert confirmed == 0 or data[confirmed - 1 : confirmed] == b"\n"
                assert confirmed >= previous
                previous = confirmed
            else:
                assert confirmed == 0  # held: nothing confirmed while the reader advances
            if read == size:
                break
        assert watcher._tailers[str(source)].offset == size


def _append_no_entry_tail(source: Path, records: int) -> None:
    """Records no parser turns into an entry: token counts and tool output, as a Codex session ends."""
    with source.open("a", encoding="utf-8") as handle:
        for i in range(records):
            handle.write(
                json.dumps(
                    {"type": "event_msg", "payload": {"type": "token_count", "info": {"total": i}, "pad": "z" * 200}}
                )
                + "\n"
            )


def _confirming(items):
    confirmed: dict[str, int] = {}
    for item in items:
        confirmed[item["_source_file"]] = max(confirmed.get(item["_source_file"], 0), item["_line_end_offset"])
    return confirmed


def test_a_finished_files_no_entry_tail_is_confirmed_once_its_last_entries_flush(tmp_path, monkeypatch):
    """Live check, 19 Nov: 9 of 15 released sessions confirmed short of EOF (e.g. 179,949 of
    3,025,612 B), because the tail after the last message holds no entry and was skipped while that
    message was still buffered; a reader at EOF never read again. Fails on main 6f7f7ae4.
    """
    source = tmp_path / "home" / ".codex" / "sessions" / "2026" / "11" / "rollout-2026-11-19T10-00-00-plain.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "plain", "cwd": "/fixture/repo"}})
        + "\n"
        + "".join(
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": f"plain message {i} " + "q" * 80}],
                    },
                }
            )
            + "\n"
            for i in range(9)  # under batch_size 10: still buffered when the tail is read
        ),
        encoding="utf-8",
    )
    _append_no_entry_tail(source, 40)
    clock = [0.0]
    watcher = _watcher(tmp_path, source, tmp_path / "watcher-health.json", clock, on_flush=_confirming)

    for _ in range(3):
        watcher.poll_once()

    assert watcher._tailers[str(source)].offset == source.stat().st_size
    assert watcher.registry.get(str(source))[0] == source.stat().st_size
    health = json.loads(watcher.health_path.read_text())
    assert health["catch_up"]["bytes_behind"] == 0


def test_a_released_sessions_no_entry_tail_is_confirmed_to_eof(tmp_path, monkeypatch):
    session = _uuid7(time.time() - 7 * DAY_S)
    source = _rollout(tmp_path, session, lines=13)
    _append_no_entry_tail(source, 40)
    _env(monkeypatch, tmp_path, _projection(tmp_path, linked=[]))
    clock = [0.0]
    watcher = _watcher(tmp_path, source, tmp_path / "watcher-health.json", clock)

    for _ in range(3):
        watcher.poll_once()

    assert watcher.registry.get(str(source))[0] == source.stat().st_size
    health = json.loads(watcher.health_path.read_text())
    assert (health["catch_up"]["bytes_behind"], health["catch_up"]["files_behind"]) == (0, 0)


def test_a_held_sessions_no_entry_tail_is_held_for_linkage_then_confirmed_on_release(tmp_path, monkeypatch):
    """Inside the window the whole read range, tail included, is held_for_linkage (never lag);
    once the window ends the tail is confirmed with the entries, never left behind."""
    session = _uuid7(time.time() - 2 * 3600)
    source = _rollout(tmp_path, session, lines=13)
    _append_no_entry_tail(source, 40)
    _env(monkeypatch, tmp_path, _projection(tmp_path, linked=[]))
    clock = [0.0]
    watcher = _watcher(tmp_path, source, tmp_path / "watcher-health.json", clock)

    watcher.poll_once()
    health = json.loads(watcher.health_path.read_text())
    assert health["catch_up"]["held_for_linkage"]["bytes"] == source.stat().st_size
    assert health["catch_up"]["bytes_behind"] == 0
    assert watcher.registry.get(str(source))[0] == 0

    monkeypatch.setenv("BRAINLAYER_T3_LINKAGE_HOLD_S", "3600")  # the window ends
    for _ in range(2):
        watcher.poll_once()
    assert watcher.registry.get(str(source))[0] == source.stat().st_size
    health = json.loads(watcher.health_path.read_text())
    assert health["catch_up"]["bytes_behind"] == 0 and health["catch_up"]["held_for_linkage"]["files"] == 0


def test_a_shorter_window_releases_a_younger_session(tmp_path, monkeypatch):
    session = _uuid7(time.time() - 2 * 3600)
    source = _rollout(tmp_path, session)
    queue = _env(monkeypatch, tmp_path, _projection(tmp_path, linked=[]))
    monkeypatch.setenv("BRAINLAYER_T3_LINKAGE_HOLD_S", "3600")
    clock = [0.0]
    watcher = _watcher(tmp_path, source, tmp_path / "watcher-health.json", clock)

    watcher.poll_once()

    assert {event["provenance_class"] for event in _events(queue, source)} == {"t3-unlinked"}
    assert watcher.registry.get(str(source))[0] == source.stat().st_size
