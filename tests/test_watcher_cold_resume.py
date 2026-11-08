"""#1656 DB-copy rehearsal: a resumed byte offset must never skip records it has not seen.

The rehearsal (docs.local receipt DBCOPY-1656) resumed a watcher from offsets copied out of a
prior run. After an unchanged first poll, the source was rewritten under the SAME inode with a
longer file whose records are the same length as the old ones and share a long suffix. The
64-byte tail anchor matched, the old offset was trusted, and new head messages 3/4 were never
stored (4 rows instead of 6). These tests drive production `JSONLWatcher.poll_once` into the
real watcher flush callback on a scratch DB, so a skipped record is a missing row.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from contextlib import closing

from brainlayer.watcher import JSONLWatcher, WatchRoot
from brainlayer.watcher_bridge import create_flush_callback

META = {
    "type": "session_meta",
    "payload": {"id": "cold-resume-1656", "cwd": "/fixture/cold-resume-1656", "model_provider": "fixture"},
}


def _text(n: int) -> str:
    # Equal length for every n, and a shared suffix longer than the tail anchor. The distinct
    # body keeps simhash near-duplicate merging (3-bit threshold) from collapsing two messages,
    # so a missing row can only mean the watcher skipped a record.
    digest = "".join(hashlib.sha256(f"{n}:{i}".encode()).hexdigest() for i in range(6))
    body = " ".join(digest[i : i + 8] for i in range(0, len(digest), 8))
    return (
        f"COLDRESUME1656 message {n}: {body}. "
        + "The same padding closes every record so the bytes before any record boundary match. " * 4
    ).rstrip()


def _encode(records: list[dict]) -> bytes:
    return b"".join((json.dumps(record) + "\n").encode() for record in records)


def _source(numbers: tuple[int, ...]) -> bytes:
    return _encode([META] + [{"role": "user", "content": _text(n)} for n in numbers])


class _Rig:
    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.sessions = tmp_path / "home" / ".codex" / "sessions"
        self.sessions.mkdir(parents=True)
        self.source = self.sessions / "rollout-cold-resume-1656.jsonl"
        self.db_path = tmp_path / "scratch.db"
        self.flush = create_flush_callback(db_path=self.db_path, arbitrated=False)

    def watcher(self, registry_name: str) -> JSONLWatcher:
        return JSONLWatcher(
            watch_roots=[WatchRoot("codex", self.sessions)],
            registry_path=self.tmp_path / registry_name,
            on_flush=self.flush,
            batch_size=1,
            flush_interval_ms=0,
            registry_flush_interval_s=0,
            health_path=self.tmp_path / f"{registry_name}.health.json",
        )

    @staticmethod
    def drain(watcher: JSONLWatcher, polls: int = 2) -> None:
        for _ in range(polls):
            watcher.poll_once()
            watcher.indexer.flush()
            watcher.registry.flush()

    def copy_registry(self, src: str, dst: str) -> None:
        shutil.copyfile(self.tmp_path / src, self.tmp_path / dst)

    def rewrite_same_inode(self, data: bytes) -> None:
        inode = self.source.stat().st_ino
        with open(self.source, "r+b") as handle:
            handle.write(data)
            handle.truncate()
        assert self.source.stat().st_ino == inode

    def contents(self) -> list[str]:
        # Read-only and closed. `with sqlite3.connect(...)` only commits; the read-write handle
        # it leaves is closed later by GC. stdlib sqlite3 and APSW link separate SQLite copies
        # whose POSIX locks cannot see each other, so that close looks like the last connection,
        # checkpoints and unlinks the -wal the flush callback's APSW writer still has open, and
        # every later row lands in an unlinked file: the [1, 2] the probe receipt saw.
        with closing(sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)) as conn:
            rows = conn.execute("SELECT content FROM chunks WHERE source_file = ?", (str(self.source),)).fetchall()
        return [row[0] for row in rows]

    def messages(self) -> list[int]:
        return sorted(int(content.split("message ")[1].split(":")[0]) for content in self.contents())


def _first_run(rig: _Rig) -> bytes:
    old = _source((1, 2))
    rig.source.write_bytes(old)
    rig.drain(rig.watcher("offsets.json"))
    assert rig.messages() == [1, 2]
    rig.copy_registry("offsets.json", "copied.json")
    return old


def test_fixture_defeats_a_tail_only_anchor():
    """Guard the fixture itself: the bytes before the old offset are identical in both files."""
    old, new = _source((1, 2)), _source((3, 4, 5, 6))
    assert new[len(old) - 256 : len(old)] == old[-256:]


def test_cold_resume_unchanged_first_poll_then_longer_same_inode_rewrite_stores_every_row(tmp_path):
    rig = _Rig(tmp_path)
    old = _first_run(rig)
    resumed = rig.watcher("copied.json")
    assert resumed.registry.get(str(rig.source))[0] == len(old)
    rig.drain(resumed)  # the unchanged first poll
    assert rig.messages() == [1, 2]

    rig.rewrite_same_inode(_source((3, 4, 5, 6)))
    rig.drain(resumed)

    assert rig.messages() == [1, 2, 3, 4, 5, 6]
    assert resumed.registry.get(str(rig.source))[0] == len(_source((3, 4, 5, 6)))

    # A repeated run from the same copied offsets re-reads nothing it already stored.
    rig.copy_registry("offsets.json", "copied-again.json")
    rig.drain(rig.watcher("copied-again.json"))
    contents = rig.contents()
    assert len(contents) == len(set(contents)) == 6


def test_cold_resume_rewrite_before_any_poll_is_reingested_from_the_start(tmp_path):
    """No live anchor exists at all: only the digest persisted with the offset can tell."""
    rig = _Rig(tmp_path)
    _first_run(rig)
    rig.rewrite_same_inode(_source((3, 4, 5, 6)))
    resumed = rig.watcher("copied.json")
    rig.drain(resumed)

    assert rig.messages() == [1, 2, 3, 4, 5, 6]
    rig.drain(resumed)
    contents = rig.contents()
    assert len(contents) == len(set(contents)) == 6


def test_live_tailer_detects_a_boundary_aligned_same_inode_rewrite(tmp_path):
    rig = _Rig(tmp_path)
    rig.source.write_bytes(_source((1, 2)))
    watcher = rig.watcher("offsets.json")
    rig.drain(watcher)
    rig.rewrite_same_inode(_source((3, 4, 5, 6)))
    rig.drain(watcher)

    assert rig.messages() == [1, 2, 3, 4, 5, 6]


def test_cold_resume_append_while_down_resumes_at_the_verified_offset(tmp_path):
    """An append while down re-hashes the prefix once, then resumes at the proven offset."""
    rig = _Rig(tmp_path)
    old = _first_run(rig)
    entry = json.loads((tmp_path / "copied.json").read_text())[str(rig.source)]
    assert entry["prefix_sha256"] == hashlib.sha256(old).hexdigest()
    assert entry["prefix_stat"][0] == len(old)
    with rig.source.open("ab") as handle:
        handle.write(_encode([{"role": "user", "content": _text(7)}]))
    resumed = rig.watcher("copied.json")
    tailer = resumed._ensure_tailer(str(rig.source))
    assert tailer.offset == len(old)
    lines = tailer.read_new_lines()

    assert [line["content"] for line in lines] == [_text(7)]
    assert not tailer.replaced


def _strip_fingerprints(path) -> None:
    """Cut every entry back to the pre-#1656 shape: offset, inode, confirm mtime, generation."""
    data = json.loads(path.read_text())
    for value in data.values():
        if isinstance(value, dict):
            for key in ("fingerprint", "prefix_sha256", "prefix_stat"):
                value.pop(key, None)
    path.write_text(json.dumps(data))


def test_legacy_checkpoint_unmodified_since_confirm_resumes_without_reingest(tmp_path):
    rig = _Rig(tmp_path)
    old = _first_run(rig)
    _strip_fingerprints(tmp_path / "copied.json")
    entry_mtime = json.loads((tmp_path / "copied.json").read_text())[str(rig.source)]["mtime"]
    os.utime(rig.source, (entry_mtime - 5, entry_mtime - 5))
    tailer = rig.watcher("copied.json")._ensure_tailer(str(rig.source))

    assert tailer.read_new_lines() == []
    assert tailer.offset == len(old)
    assert not tailer.replaced


def test_legacy_checkpoint_modified_since_confirm_is_reread_from_the_start(tmp_path):
    rig = _Rig(tmp_path)
    _first_run(rig)
    _strip_fingerprints(tmp_path / "copied.json")
    rig.rewrite_same_inode(_source((3, 4, 5, 6)))
    entry_mtime = json.loads((tmp_path / "copied.json").read_text())[str(rig.source)]["mtime"]
    os.utime(rig.source, (entry_mtime + 5, entry_mtime + 5))
    rig.drain(rig.watcher("copied.json"))

    assert rig.messages() == [1, 2, 3, 4, 5, 6]


# ── #1656 round 2 (receipt DBCOPY-1656b): two more skip paths, ported from the probe harness ──


def test_middle_only_rewrite_after_resume_stores_every_row(tmp_path, monkeypatch):
    """The first 4096 bytes and the 256 before the offset survive; only the middle changes."""
    meta = json.loads(json.dumps(META))
    meta["payload"]["padding"] = "x" * 5000
    monkeypatch.setitem(globals(), "META", meta)
    rig = _Rig(tmp_path)
    old = _first_run(rig)
    resumed = rig.watcher("copied.json")
    rig.drain(resumed, 3)
    new = _source((3, 4, 5, 6))
    assert old[:4096] == new[:4096]
    assert old[-256:] == new[len(old) - 256 : len(old)]
    assert old != new[: len(old)]

    rig.rewrite_same_inode(new)
    rig.drain(resumed, 4)

    assert rig.messages() == [1, 2, 3, 4, 5, 6]


def _write_legacy_checkpoint_racing_a_rewrite(rig: _Rig, registry_name: str) -> dict:
    """Emit a pre-fingerprint checkpoint whose source was rewritten between read and ack.

    The rewrite runs inside the real flush callback, after the rows are stored and before the
    watcher confirms the offset, so no timestamp is forged: source mtime <= confirm time. The
    entry is then cut to exactly the fields the pre-#1656 watcher (8fc4f292) persisted.
    """
    rig.source.write_bytes(_source((1, 2)))
    changed = []

    def rewrite_before_confirm(items):
        result = rig.flush(items)
        if not changed:
            rig.rewrite_same_inode(_source((3, 4, 5, 6)))
            changed.append(True)
        return result

    prior = JSONLWatcher(
        watch_roots=[WatchRoot("codex", rig.sessions)],
        registry_path=rig.tmp_path / registry_name,
        on_flush=rewrite_before_confirm,
        batch_size=1,
        flush_interval_ms=0,
        registry_flush_interval_s=0,
        health_path=rig.tmp_path / "legacy-health.json",
    )
    prior.poll_once()
    prior.indexer.flush()
    prior.registry.flush()
    assert changed and rig.messages() == [1, 2]
    path = rig.tmp_path / registry_name
    data = json.loads(path.read_text())
    entry = {key: data[str(rig.source)][key] for key in ("offset", "inode", "mtime", "generation")}
    data[str(rig.source)] = entry
    path.write_text(json.dumps(data))
    assert entry["offset"] == len(_source((1, 2)))
    assert rig.source.stat().st_mtime <= entry["mtime"]
    return entry


def test_legacy_checkpoint_rewritten_between_read_and_ack_stores_every_row(tmp_path):
    rig = _Rig(tmp_path)
    _write_legacy_checkpoint_racing_a_rewrite(rig, "offsets.json")

    rig.drain(rig.watcher("offsets.json"), 4)

    assert rig.messages() == [1, 2, 3, 4, 5, 6]


def _rows(rig: _Rig) -> dict[str, tuple[str, int]]:
    with closing(sqlite3.connect(f"file:{rig.db_path}?mode=ro", uri=True)) as conn:
        query = "SELECT id, content, seen_count FROM chunks WHERE source_file = ?"
        return {row[0]: (row[1], row[2]) for row in conn.execute(query, (str(rig.source),))}


def test_replay_from_zero_after_a_digest_mismatch_adds_no_rows(tmp_path):
    """Restarting at byte 0 is safe: chunk ids are content hashes, so nothing is duplicated.

    It is not side-effect free: each replayed chunk's seen_count is bumped once
    (merge_existing_chunk_seen). Pinned here so that cost stays visible.
    """
    rig = _Rig(tmp_path)
    old = _first_run(rig)
    # Same length, same inode, same parsed records: only the digest can tell the bytes changed.
    respaced = old.replace(b'"user", "content"', b'"user" ,"content"')
    assert respaced != old and len(respaced) == len(old)
    rig.rewrite_same_inode(respaced)
    before = _rows(rig)
    rig.drain(rig.watcher("copied.json"))

    after = _rows(rig)
    assert len(before) == 2
    assert {chunk_id: content for chunk_id, (content, _) in after.items()} == {
        chunk_id: content for chunk_id, (content, _) in before.items()
    }
    assert {chunk_id: seen for chunk_id, (_, seen) in after.items()} == {
        chunk_id: seen + 1 for chunk_id, (_, seen) in before.items()
    }


def _count_proofs(monkeypatch) -> list[int]:
    from brainlayer.watcher import JSONLTailer

    calls: list[int] = []
    original = JSONLTailer._prove_prefix

    def counting(self, *args, **kwargs):
        calls.append(self.offset)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(JSONLTailer, "_prove_prefix", counting)
    return calls


def test_steady_state_never_rehashes_while_appending_and_once_when_idle(tmp_path, monkeypatch):
    calls = _count_proofs(monkeypatch)
    rig = _Rig(tmp_path)
    rig.source.write_bytes(_source((1,)))
    watcher = rig.watcher("offsets.json")
    rig.drain(watcher, 1)
    for n in (2, 3, 4):  # an append lands before every poll: never idle, never re-hashed
        with rig.source.open("ab") as handle:
            handle.write(_encode([{"role": "user", "content": _text(n)}]))
        rig.drain(watcher, 1)
    assert calls == []

    rig.drain(watcher, 3)  # first idle poll proves the prefix once; later ones skip the file
    assert calls == [len(_source((1, 2, 3, 4)))]
    assert rig.messages() == [1, 2, 3, 4]

    entry = json.loads((tmp_path / "offsets.json").read_text())[str(rig.source)]
    assert entry["prefix_sha256"] == hashlib.sha256(_source((1, 2, 3, 4))).hexdigest()
    assert entry["prefix_stat"][0] == len(_source((1, 2, 3, 4)))


def test_cold_resume_of_an_unchanged_file_reads_nothing(tmp_path, monkeypatch):
    rig = _Rig(tmp_path)
    _first_run(rig)
    calls = _count_proofs(monkeypatch)
    rig.drain(rig.watcher("copied.json"))

    assert calls == []
    assert rig.messages() == [1, 2]


def test_legacy_checkpoint_is_hashed_once_and_then_resumes_for_free(tmp_path, monkeypatch):
    rig = _Rig(tmp_path)
    old = _first_run(rig)
    _strip_fingerprints(tmp_path / "copied.json")
    calls = _count_proofs(monkeypatch)
    rig.drain(rig.watcher("copied.json"))
    assert calls == [len(old)]
    entry = json.loads((tmp_path / "copied.json").read_text())[str(rig.source)]
    assert entry["prefix_sha256"] == hashlib.sha256(old).hexdigest()

    rig.drain(rig.watcher("copied.json"))
    assert calls == [len(old)]
    assert rig.messages() == [1, 2]


def test_rig_reader_never_unlinks_the_writers_wal(tmp_path):
    """Pin the cause of the probe receipt's [1, 2]: a closed read-write stdlib reader.

    stdlib sqlite3 and APSW are separate SQLite copies in one process. A read-only reader may
    close freely; the rig used to leave a read-write one for GC, which deletes the live WAL.
    """
    import gc

    rig = _Rig(tmp_path)
    rig.source.write_bytes(_source((1, 2)))
    watcher = rig.watcher("offsets.json")
    rig.drain(watcher)
    assert rig.messages() == [1, 2]
    gc.collect()
    assert os.path.exists(f"{rig.db_path}-wal")
    with rig.source.open("ab") as handle:
        handle.write(_encode([{"role": "user", "content": _text(3)}]))
    rig.drain(watcher)

    assert rig.messages() == [1, 2, 3]
