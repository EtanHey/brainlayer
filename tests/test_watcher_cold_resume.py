"""#1656 DB-copy rehearsal: a resumed byte offset must never skip records it has not seen.

The rehearsal (docs.local receipt DBCOPY-1656) resumed a watcher from offsets copied out of a
prior run. After an unchanged first poll, the source was rewritten under the SAME inode with a
longer file whose records are the same length as the old ones and share a long suffix. The
64-byte tail anchor matched, the old offset was trusted, and new head messages 3/4 were never
stored (4 rows instead of 6). These tests drive production `JSONLWatcher.poll_once` into the
real watcher flush callback on a scratch DB, so a skipped record is a missing row.
"""

from __future__ import annotations

import gc
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
        with closing(sqlite3.connect(self.db_path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
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
    """No live anchor exists at all: only the fingerprint persisted with the offset can tell."""
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
    """The fingerprint must not turn every restart into a full re-read of active files."""
    rig = _Rig(tmp_path)
    old = _first_run(rig)
    entry = json.loads((tmp_path / "copied.json").read_text())[str(rig.source)]
    assert isinstance(entry.get("fingerprint"), str)
    with rig.source.open("ab") as handle:
        handle.write(_encode([{"role": "user", "content": _text(7)}]))
    resumed = rig.watcher("copied.json")
    tailer = resumed._ensure_tailer(str(rig.source))
    assert tailer.offset == len(old)
    lines = tailer.read_new_lines()

    assert [line["content"] for line in lines] == [_text(7)]
    assert not tailer.replaced


def _strip_fingerprints(path) -> None:
    data = json.loads(path.read_text())
    for value in data.values():
        if isinstance(value, dict):
            value.pop("fingerprint", None)
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


def test_scratch_readers_keep_the_writers_wal(tmp_path):
    rig = _Rig(tmp_path)
    _first_run(rig)
    gc.collect()
    assert rig.db_path.with_name(rig.db_path.name + "-wal").exists()
    rig.rewrite_same_inode(_source((3, 4, 5, 6)))
    rig.drain(rig.watcher("copied.json"))
    assert rig.messages() == [1, 2, 3, 4, 5, 6]
