"""A forked Claude Code chat starts at its fork point (Etan, 2026-11-18).

Claude Code writes a fork (/side, a resumed fork) as the parent's entries copied under the
fork's sessionId, each stamped ``forkedFrom={sessionId: <parent>, messageUuid}``, followed by
the fork's own entries without it. The watcher skips the copies (the offset still advances)
instead of feeding them to content dedupe only when the parent transcript sits beside the fork,
is tracked, and the WATCHER has already written rows for it. A Stop-hook row
(source='realtime', no source_end_offset) never counts. Otherwise the copies ingest as before,
so the only copy is never lost (red-team 1828-a F1).
"""

import json
import sqlite3
import uuid
from pathlib import Path

import pytest

import brainlayer.watcher as watcher_module
from brainlayer.watcher import JSONLTailer, JSONLWatcher, WatchRoot

PARENT = "79b59b17-5523-4b46-b5a1-9b6bbb588201"
FORK = "e1336d1e-b487-4391-a564-01b04634833f"
FORK_OF_FORK = "0f0f0f0f-1111-4222-8333-444455556666"
MISSING = "deadbeef-0000-4000-8000-000000000000"


def _entry(session_id: str, index: int, *, forked_from: str | None = None, kind: str = "own") -> dict:
    entry = {
        "type": "user" if index % 2 == 0 else "assistant",
        "uuid": str(uuid.uuid4()),
        "sessionId": session_id,
        "timestamp": f"2026-11-18T06:{index // 60 % 60:02d}:{index % 60:02d}.000Z",
        "message": {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"{kind} message number {index} with enough words to be a substantive entry",
        },
    }
    if forked_from is not None:
        entry["forkedFrom"] = {"sessionId": forked_from, "messageUuid": str(uuid.uuid4())}
    return entry


def _write_fork(path: Path, session_id: str, parent: str, copies: int, own: int) -> list[dict]:
    rows = [_entry(session_id, i, forked_from=parent, kind="copy") for i in range(copies)]
    rows += [_entry(session_id, copies + i, kind="own") for i in range(own)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


def _write_parent(projects: Path, session_id: str, entries: int = 4) -> Path:
    """The parent transcript beside the fork, as Claude Code leaves it."""
    path = projects / f"{session_id}.jsonl"
    path.write_text("".join(json.dumps(_entry(session_id, i, kind="parent")) + "\n" for i in range(entries)))
    return path


# A row the watcher's drain writes, and the row the Stop hook writes for every session.
WATCHER_ROW = ("realtime_watcher", 120)
HOOK_ROW = ("realtime", None)


def _db(tmp_path: Path, held: tuple[str, ...], *, row: tuple[str, int | None] = WATCHER_ROW) -> Path:
    db_path = tmp_path / "brainlayer.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE chunks (id TEXT PRIMARY KEY, conversation_id TEXT, source TEXT, created_at TEXT,"
        " source_end_offset INTEGER)"
    )
    conn.execute("CREATE INDEX idx_chunks_conversation ON chunks(conversation_id)")
    source, end_offset = row
    for index, session_id in enumerate(held):
        conn.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, '2026-11-18', ?)", (f"rt-{index}", session_id, source, end_offset)
        )
    conn.commit()
    conn.close()
    return db_path


def _from(sink: "_Sink", path: Path) -> list[str]:
    """Uuids the sink received from one file, in order (the parent file is read too)."""
    return [entry["uuid"] for entry in sink.entries if entry["_source_file"] == str(path)]


class _Sink:
    """on_flush that records entries and confirms their offsets, like the real bridge."""

    def __init__(self):
        self.entries: list[dict] = []

    def __call__(self, items: list[dict]) -> dict[str, int]:
        self.entries.extend(items)
        confirmed: dict[str, int] = {}
        for item in items:
            confirmed[item["_source_file"]] = max(confirmed.get(item["_source_file"], 0), item["_line_end_offset"])
        return confirmed


def _watcher(tmp_path: Path, projects: Path, sink: _Sink, db_path: Path | None) -> JSONLWatcher:
    return JSONLWatcher(
        # The Claude root is ~/.claude/projects; transcripts sit one folder below it.
        watch_roots=[WatchRoot("claude", projects.parent)],
        registry_path=tmp_path / "offsets.json",
        on_flush=sink,
        batch_size=1,
        health_path=tmp_path / "watcher-health.json",
        db_path=db_path,
    )


def _drain(watcher: JSONLWatcher, path: Path, max_polls: int = 50) -> int:
    size = path.stat().st_size
    for poll in range(1, max_polls + 1):
        watcher.poll_once()
        watcher.indexer.flush()
        tailer = watcher._tailers.get(str(path))
        if tailer is not None and tailer.offset >= size and not tailer._buffer:
            return poll
    return max_polls + 1


@pytest.fixture
def projects(tmp_path):
    folder = tmp_path / "projects" / "-Users-x-Gits-narrationlayer"
    folder.mkdir(parents=True)
    return folder


def test_fork_of_a_held_parent_ingests_only_its_own_entries_and_reaches_the_end_fast(tmp_path, projects):
    parent_path = _write_parent(projects, PARENT)
    fork_path = projects / f"{FORK}.jsonl"
    rows = _write_fork(fork_path, FORK, PARENT, copies=1_000, own=7)
    sink = _Sink()
    watcher = _watcher(tmp_path, projects, sink, _db(tmp_path, held=(PARENT,)))

    polls = _drain(watcher, fork_path)

    assert _from(sink, fork_path) == [row["uuid"] for row in rows if "forkedFrom" not in row]
    assert not any("forkedFrom" in entry for entry in sink.entries)
    # 1,000 copies at the 100-line cap would take 11 polls; skipped copies do not count toward it.
    assert polls == 1
    watcher.registry.flush()
    assert watcher.registry.get(str(fork_path))[0] == fork_path.stat().st_size
    health = json.loads((tmp_path / "watcher-health.json").read_text())
    assert health["fork_copies_skipped_total"] == 1_000
    assert health["fork_copy_files"] == [
        {
            "path": str(fork_path),
            "parent_session_id": PARENT,
            "copies_skipped": 1_000,
            "offset": fork_path.stat().st_size,
            "size": fork_path.stat().st_size,
            "parent_path": str(parent_path),
            "parent_confirmed_offset": parent_path.stat().st_size,
            "parent_size": parent_path.stat().st_size,
        }
    ]


def test_fork_copies_alone_confirm_their_offset_without_flushing_anything(tmp_path, projects):
    _write_parent(projects, PARENT)
    fork_path = projects / f"{FORK}.jsonl"
    _write_fork(fork_path, FORK, PARENT, copies=300, own=0)
    sink = _Sink()
    watcher = _watcher(tmp_path, projects, sink, _db(tmp_path, held=(PARENT,)))

    assert _drain(watcher, fork_path) == 1
    watcher.registry.flush()

    assert _from(sink, fork_path) == []
    assert watcher.registry.get(str(fork_path))[0] == fork_path.stat().st_size


# ── red-team 1828-a F1: "held" must mean the watcher indexed the parent, and will finish it ──


def test_parent_held_only_by_stop_hook_rows_ingests_every_copy(tmp_path, projects):
    # The Stop hook writes a source='realtime' row for every session before the watcher reads
    # a line of it; 38 real conversations are held only by such rows (17 with no transcript).
    _write_parent(projects, PARENT)
    fork_path = projects / f"{FORK}.jsonl"
    rows = _write_fork(fork_path, FORK, PARENT, copies=50, own=1)
    sink = _Sink()
    watcher = _watcher(tmp_path, projects, sink, _db(tmp_path, held=(PARENT,), row=HOOK_ROW))

    _drain(watcher, fork_path)

    assert _from(sink, fork_path) == [row["uuid"] for row in rows]
    assert json.loads((tmp_path / "watcher-health.json").read_text())["fork_copies_skipped_total"] == 0


def test_parent_transcript_gone_ingests_every_copy_even_with_watcher_rows(tmp_path, projects):
    # Claude Code's age cleanup and the 14-day local retention remove the OLDER parent first;
    # the fork is then the only full copy, whatever the DB once held.
    assert not (projects / f"{PARENT}.jsonl").exists()
    fork_path = projects / f"{FORK}.jsonl"
    rows = _write_fork(fork_path, FORK, PARENT, copies=300, own=2)
    sink = _Sink()
    watcher = _watcher(tmp_path, projects, sink, _db(tmp_path, held=(PARENT,)))

    _drain(watcher, fork_path)

    assert _from(sink, fork_path) == [row["uuid"] for row in rows]


def test_parent_only_held_by_one_hook_row_with_its_transcript_gone_ingests_every_copy(tmp_path, projects):
    # Reviewer a's first driver, inverted: on cd7c7052 this flushed 2 of 302 entries.
    fork_path = projects / f"{FORK}.jsonl"
    rows = _write_fork(fork_path, FORK, PARENT, copies=300, own=2)
    sink = _Sink()
    watcher = _watcher(tmp_path, projects, sink, _db(tmp_path, held=(PARENT,), row=HOOK_ROW))

    _drain(watcher, fork_path)

    assert _from(sink, fork_path) == [row["uuid"] for row in rows]


def test_red_team_b_d1_one_realtime_hook_row_and_no_parent_file_loses_no_copy(tmp_path, projects):
    # Reviewer b's D1, inverted: the Stop hook path (drain.py:1183, hooks/brainbar-stop-index.py,
    # src/brainlayer/hooks/indexer.py:249) writes source='realtime' rows with conversation_id =
    # the session id, whether or not the watcher ever read that JSONL. On cd7c7052 one such row
    # made the parent "held" and all 50 copies were consumed with nothing stored.
    fork_path = projects / f"{FORK}.jsonl"
    history = [
        _entry(FORK, i, forked_from=PARENT, kind=f"parent-only decision {i}: we chose option {i} because")
        for i in range(50)
    ]
    own = [_entry(FORK, 100 + i) for i in range(2)]
    fork_path.write_text("".join(json.dumps(row) + "\n" for row in history + own))
    assert not (projects / f"{PARENT}.jsonl").exists()
    sink = _Sink()
    watcher = _watcher(tmp_path, projects, sink, _db(tmp_path, held=(PARENT,), row=HOOK_ROW))

    _drain(watcher, fork_path)
    watcher.registry.flush()

    flushed = set(_from(sink, fork_path))
    assert [row for row in history if row["uuid"] not in flushed] == []
    assert len(flushed) == 52
    assert watcher.registry.get(str(fork_path))[0] == fork_path.stat().st_size


def test_parent_transcript_in_another_folder_or_denylisted_ingests_every_copy(tmp_path, projects, monkeypatch):
    # Only <fork dir>/<parent>.jsonl counts: a parent elsewhere is not one this rule can vouch for.
    elsewhere = projects.parent / "-Users-x-Gits-other"
    elsewhere.mkdir()
    _write_parent(elsewhere, PARENT)
    fork_path = projects / f"{FORK}.jsonl"
    rows = _write_fork(fork_path, FORK, PARENT, copies=40, own=1)
    sink = _Sink()
    watcher = _watcher(tmp_path, projects, sink, _db(tmp_path, held=(PARENT,)))
    _drain(watcher, fork_path)
    assert _from(sink, fork_path) == [row["uuid"] for row in rows]

    # Beside the fork but excluded from the index: the watcher will never read it.
    second = tmp_path / "second"
    second_projects = second / "projects" / "p"
    second_projects.mkdir(parents=True)
    parent_path = _write_parent(second_projects, PARENT)
    second_fork = second_projects / f"{FORK}.jsonl"
    second_rows = _write_fork(second_fork, FORK, PARENT, copies=40, own=1)
    second_sink = _Sink()
    second_watcher = _watcher(second, second_projects, second_sink, _db(second, held=(PARENT,)))
    original = second_watcher._denylisted
    monkeypatch.setattr(second_watcher, "_denylisted", lambda path: path == str(parent_path) or original(path))
    _drain(second_watcher, second_fork)
    assert _from(second_sink, second_fork) == [row["uuid"] for row in second_rows]


def _interleaved(path: Path, layout: list[tuple[str, int]]) -> list[dict]:
    rows, index = [], 0
    for kind, count in layout:
        for _ in range(count):
            rows.append(_entry(FORK, index, forked_from=PARENT if kind == "copy" else None, kind=kind))
            index += 1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


def test_interleaved_copies_and_own_entries_emit_every_own_entry_once_in_order(tmp_path, projects, monkeypatch):
    # Ported from red-team 1828-a's drivers, now with the parent transcript beside the fork.
    monkeypatch.setattr(watcher_module, "_MAX_SKIPPED_RECORDS_PER_READ", 37)
    _write_parent(projects, PARENT)
    fork_path = projects / f"{FORK}.jsonl"
    rows = _interleaved(fork_path, [("own", 3), ("copy", 250), ("own", 120), ("copy", 90), ("own", 1), ("copy", 400)])
    sink = _Sink()
    watcher = _watcher(tmp_path, projects, sink, _db(tmp_path, held=(PARENT,)))

    for _ in range(40):
        watcher.poll_once()
        watcher.indexer.flush()

    assert _from(sink, fork_path) == [row["uuid"] for row in rows if "forkedFrom" not in row]
    watcher.registry.flush()
    assert watcher.registry.get(str(fork_path))[0] == fork_path.stat().st_size


def test_restart_in_the_middle_of_skipped_copies_loses_and_duplicates_nothing(tmp_path, projects, monkeypatch):
    monkeypatch.setattr(watcher_module, "_MAX_SKIPPED_RECORDS_PER_READ", 50)
    _write_parent(projects, PARENT)
    fork_path = projects / f"{FORK}.jsonl"
    rows = _interleaved(fork_path, [("own", 2), ("copy", 500), ("own", 4)])
    db_path = _db(tmp_path, held=(PARENT,))
    sink = _Sink()
    first = _watcher(tmp_path, projects, sink, db_path)
    first.poll_once()
    first.indexer.flush()
    first.registry.flush()
    assert 0 < first.registry.get(str(fork_path))[0] < fork_path.stat().st_size

    second = _watcher(tmp_path, projects, sink, db_path)  # a restart resumes from offsets.json
    for _ in range(20):
        second.poll_once()
        second.indexer.flush()

    assert _from(sink, fork_path) == [row["uuid"] for row in rows if "forkedFrom" not in row]


def test_health_shows_how_far_the_watcher_has_read_the_parent_a_skip_relies_on(tmp_path, projects):
    # red-team 1828-a F2: a skip made while the parent is still being read is visible.
    parent_path = projects / f"{PARENT}.jsonl"
    parent_path.write_text("".join(json.dumps(_entry(PARENT, i, kind="parent")) + "\n" for i in range(250)))
    fork_path = projects / f"{FORK}.jsonl"
    _write_fork(fork_path, FORK, PARENT, copies=60, own=1)
    watcher = _watcher(tmp_path, projects, _Sink(), _db(tmp_path, held=(PARENT,)))

    watcher.poll_once()
    watcher.indexer.flush()
    watcher.poll_once()  # the health snapshot of this poll sees the first poll's confirmation

    row = json.loads((tmp_path / "watcher-health.json").read_text())["fork_copy_files"][0]
    assert row["parent_path"] == str(parent_path)
    assert row["parent_size"] == parent_path.stat().st_size
    assert 0 < row["parent_confirmed_offset"] < row["parent_size"]  # 100 of 250 lines per poll


def test_fork_of_a_parent_not_held_ingests_every_copy_as_before(tmp_path, projects):
    fork_path = projects / f"{FORK}.jsonl"
    rows = _write_fork(fork_path, FORK, PARENT, copies=250, own=5)
    sink = _Sink()
    watcher = _watcher(tmp_path, projects, sink, _db(tmp_path, held=()))

    _drain(watcher, fork_path)

    assert [entry["uuid"] for entry in sink.entries] == [row["uuid"] for row in rows]
    health = json.loads((tmp_path / "watcher-health.json").read_text())
    assert health["fork_copies_skipped_total"] == 0
    assert health["fork_copy_files"] == []


def test_fork_naming_a_missing_or_malformed_parent_is_ingested(tmp_path, projects):
    missing_path = projects / f"{FORK}.jsonl"
    missing_rows = _write_fork(missing_path, FORK, MISSING, copies=20, own=2)
    malformed_path = projects / f"{FORK_OF_FORK}.jsonl"
    malformed_rows = _write_fork(malformed_path, FORK_OF_FORK, "not-a-session-id", copies=20, own=2)
    sink = _Sink()
    # The DB holds unrelated sessions only, including one whose id is a prefix-free stranger.
    watcher = _watcher(tmp_path, projects, sink, _db(tmp_path, held=(PARENT,)))

    _drain(watcher, missing_path)
    _drain(watcher, malformed_path)

    seen = {entry["uuid"] for entry in sink.entries}
    assert {row["uuid"] for row in missing_rows} <= seen
    assert {row["uuid"] for row in malformed_rows} <= seen
    assert json.loads((tmp_path / "watcher-health.json").read_text())["fork_copies_skipped_total"] == 0


def test_fork_of_a_fork_skips_only_when_its_immediate_parent_is_held(tmp_path, projects):
    # Claude Code overwrites forkedFrom on a re-fork, so it names the IMMEDIATE parent.
    _write_fork(projects / f"{FORK}.jsonl", FORK, PARENT, copies=5, own=2)  # the immediate parent
    held_path = projects / f"{FORK_OF_FORK}.jsonl"
    held_rows = _write_fork(held_path, FORK_OF_FORK, FORK, copies=150, own=3)
    sink = _Sink()
    watcher = _watcher(tmp_path, projects, sink, _db(tmp_path, held=(FORK,)))
    _drain(watcher, held_path)
    assert _from(sink, held_path) == [row["uuid"] for row in held_rows[150:]]

    # Grandparent held, immediate parent not: never skip on a guess about the chain.
    other_tmp = tmp_path / "second"
    other_projects = other_tmp / "projects" / "p"
    other_projects.mkdir(parents=True)
    _write_parent(other_projects, PARENT)
    _write_fork(other_projects / f"{FORK}.jsonl", FORK, PARENT, copies=5, own=2)
    chain_path = other_projects / f"{FORK_OF_FORK}.jsonl"
    chain_rows = _write_fork(chain_path, FORK_OF_FORK, FORK, copies=150, own=3)
    chain_sink = _Sink()
    chain_watcher = _watcher(other_tmp, other_projects, chain_sink, _db(other_tmp, held=(PARENT,)))
    _drain(chain_watcher, chain_path)
    assert _from(chain_sink, chain_path) == [row["uuid"] for row in chain_rows]


def test_without_a_db_path_fork_copies_are_ingested(tmp_path, projects):
    fork_path = projects / f"{FORK}.jsonl"
    rows = _write_fork(fork_path, FORK, PARENT, copies=30, own=2)
    sink = _Sink()
    watcher = _watcher(tmp_path, projects, sink, db_path=None)

    _drain(watcher, fork_path)

    assert [entry["uuid"] for entry in sink.entries] == [row["uuid"] for row in rows]


def test_unreadable_db_fails_closed_and_ingests_the_copies(tmp_path, projects):
    fork_path = projects / f"{FORK}.jsonl"
    rows = _write_fork(fork_path, FORK, PARENT, copies=30, own=2)
    not_a_db = tmp_path / "brainlayer.db"
    not_a_db.write_bytes(b"this is not a sqlite database at all, just bytes" * 100)
    sink = _Sink()
    watcher = _watcher(tmp_path, projects, sink, db_path=not_a_db)

    _drain(watcher, fork_path)

    assert [entry["uuid"] for entry in sink.entries] == [row["uuid"] for row in rows]


def test_a_record_naming_its_own_session_as_parent_is_not_skipped(tmp_path, projects):
    fork_path = projects / f"{FORK}.jsonl"
    rows = _write_fork(fork_path, FORK, FORK, copies=10, own=1)
    sink = _Sink()
    watcher = _watcher(tmp_path, projects, sink, _db(tmp_path, held=(FORK,)))

    _drain(watcher, fork_path)

    assert [entry["uuid"] for entry in sink.entries] == [row["uuid"] for row in rows]


def test_skipped_copies_are_neutral_growth_not_unrecognised_lines(tmp_path, projects):
    _write_parent(projects, PARENT)
    fork_path = projects / f"{FORK}.jsonl"
    _write_fork(fork_path, FORK, PARENT, copies=400, own=0)
    watcher = _watcher(tmp_path, projects, _Sink(), _db(tmp_path, held=(PARENT,)))

    watcher.poll_once()

    tailer = watcher._tailers[str(fork_path)]
    assert tailer.lines_read_total == 400
    assert tailer.neutral_lines_total == 400
    # The parent's 4 lines are first-ingest backlog, which growth excludes; the copies are neutral.
    read, recognised = watcher._growth_poll.get("claude", (0, 0))
    assert (read, recognised) == (0, 0)


class TestTailerSkipRecord:
    def _file(self, tmp_path: Path, copies: int, own: int, pad: int = 0) -> Path:
        path = tmp_path / "t.jsonl"
        rows = [{"kind": "copy", "i": i, "pad": "x" * pad} for i in range(copies)]
        rows += [{"kind": "own", "i": i} for i in range(own)]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def test_skipped_records_do_not_count_toward_max_lines(self, tmp_path):
        path = self._file(tmp_path, copies=500, own=8)
        tailer = JSONLTailer(str(path))
        tailer.skip_record = lambda entry: entry.get("kind") == "copy"

        lines = tailer.read_new_lines(max_lines=5)

        assert [line["i"] for line in lines] == [0, 1, 2, 3, 4]
        assert tailer.skipped_records_total == 500
        assert all(line["kind"] == "own" for line in lines)
        assert lines[-1]["_line_end_offset"] == tailer.offset

    def test_the_per_read_skip_budget_bounds_one_read(self, tmp_path, monkeypatch):
        monkeypatch.setattr(watcher_module, "_MAX_SKIPPED_RECORDS_PER_READ", 120)
        # ~1 KiB records: one 64 KiB read chunk holds ~60 of them.
        path = self._file(tmp_path, copies=1_000, own=1, pad=1_000)
        tailer = JSONLTailer(str(path))
        tailer.skip_record = lambda entry: entry.get("kind") == "copy"

        assert tailer.read_new_lines(max_lines=50) == []
        # The budget is checked between passes; one pass parses at most what it buffered
        # (max_lines records, rounded up to whole 64 KiB chunks), so it overshoots by less than that.
        assert 120 <= tailer.skipped_records_total < 120 + 2 * 64
        assert tailer.offset < path.stat().st_size

    def test_the_byte_window_still_bounds_a_skipping_read(self, tmp_path):
        path = self._file(tmp_path, copies=1_000, own=1)
        tailer = JSONLTailer(str(path))
        tailer.skip_record = lambda entry: entry.get("kind") == "copy"

        tailer.read_new_lines(max_lines=10, max_bytes=4_096)

        assert tailer.offset + len(tailer._buffer) <= 4_096

    def test_without_a_predicate_a_read_is_one_pass_as_before(self, tmp_path):
        path = self._file(tmp_path, copies=500, own=8)
        tailer = JSONLTailer(str(path))

        lines = tailer.read_new_lines(max_lines=5)

        assert [line["kind"] for line in lines] == ["copy"] * 5
        assert tailer.skipped_records_total == 0

    def test_a_corrupt_record_after_skipped_copies_stops_at_it(self, tmp_path):
        path = self._file(tmp_path, copies=200, own=0)
        good_end = path.stat().st_size
        with path.open("a") as handle:
            handle.write("{not json\n")
            handle.write(json.dumps({"kind": "own", "i": 0}) + "\n")
        tailer = JSONLTailer(str(path))
        tailer.skip_record = lambda entry: entry.get("kind") == "copy"

        assert tailer.read_new_lines(max_lines=5) == []
        assert tailer.offset == good_end
        assert tailer.last_error is not None
