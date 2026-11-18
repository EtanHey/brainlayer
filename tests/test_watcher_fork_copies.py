"""A forked Claude Code chat starts at its fork point (Etan, 2026-11-18).

Claude Code writes a fork (/side, a resumed fork) as the parent's entries copied under the
fork's sessionId, each stamped ``forkedFrom={sessionId: <parent>, messageUuid}``, followed by
the fork's own entries without it. When BrainLayer already holds the parent, the watcher skips
the copies (the offset still advances) instead of feeding them to content dedupe; when it does
not, the copies ingest as before so the only copy is never lost.
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


def _db(tmp_path: Path, held: tuple[str, ...]) -> Path:
    db_path = tmp_path / "brainlayer.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, conversation_id TEXT, source TEXT, created_at TEXT)")
    conn.execute("CREATE INDEX idx_chunks_conversation ON chunks(conversation_id)")
    for index, session_id in enumerate(held):
        conn.execute("INSERT INTO chunks VALUES (?, ?, 'claude_code', '2026-11-18')", (f"rt-{index}", session_id))
    conn.commit()
    conn.close()
    return db_path


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
    fork_path = projects / f"{FORK}.jsonl"
    rows = _write_fork(fork_path, FORK, PARENT, copies=1_000, own=7)
    sink = _Sink()
    watcher = _watcher(tmp_path, projects, sink, _db(tmp_path, held=(PARENT,)))

    polls = _drain(watcher, fork_path)

    own_uuids = {row["uuid"] for row in rows if "forkedFrom" not in row}
    assert {entry["uuid"] for entry in sink.entries} == own_uuids
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
        }
    ]


def test_fork_copies_alone_confirm_their_offset_without_flushing_anything(tmp_path, projects):
    fork_path = projects / f"{FORK}.jsonl"
    _write_fork(fork_path, FORK, PARENT, copies=300, own=0)
    sink = _Sink()
    watcher = _watcher(tmp_path, projects, sink, _db(tmp_path, held=(PARENT,)))

    assert _drain(watcher, fork_path) == 1
    watcher.registry.flush()

    assert sink.entries == []
    assert watcher.registry.get(str(fork_path))[0] == fork_path.stat().st_size


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
    held_path = projects / f"{FORK_OF_FORK}.jsonl"
    held_rows = _write_fork(held_path, FORK_OF_FORK, FORK, copies=150, own=3)
    sink = _Sink()
    watcher = _watcher(tmp_path, projects, sink, _db(tmp_path, held=(FORK,)))
    _drain(watcher, held_path)
    assert {entry["uuid"] for entry in sink.entries} == {row["uuid"] for row in held_rows[150:]}

    # Grandparent held, immediate parent not: never skip on a guess about the chain.
    other_tmp = tmp_path / "second"
    other_projects = other_tmp / "projects" / "p"
    other_projects.mkdir(parents=True)
    chain_path = other_projects / f"{FORK_OF_FORK}.jsonl"
    chain_rows = _write_fork(chain_path, FORK_OF_FORK, FORK, copies=150, own=3)
    chain_sink = _Sink()
    chain_watcher = _watcher(other_tmp, other_projects, chain_sink, _db(other_tmp, held=(PARENT,)))
    _drain(chain_watcher, chain_path)
    assert [entry["uuid"] for entry in chain_sink.entries] == [row["uuid"] for row in chain_rows]


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
    fork_path = projects / f"{FORK}.jsonl"
    _write_fork(fork_path, FORK, PARENT, copies=400, own=0)
    watcher = _watcher(tmp_path, projects, _Sink(), _db(tmp_path, held=(PARENT,)))

    watcher.poll_once()

    tailer = watcher._tailers[str(fork_path)]
    assert tailer.lines_read_total == 400
    assert tailer.neutral_lines_total == 400
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
