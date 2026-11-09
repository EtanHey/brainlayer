"""Legacy archive replay is bounded and cannot turn old bytes into new sightings."""

import sqlite3
from contextlib import closing

from brainlayer.drain import drain_once
from brainlayer.vector_store import VectorStore
from brainlayer.watcher import WatchRoot
from brainlayer.watcher_bridge import create_flush_callback
from tests.test_codex_zst_watcher import compress as _compress
from tests.test_codex_zst_watcher import rollout, watcher

compress = _compress


def test_legacy_archive_replays_prefix_without_changing_rows(compress, tmp_path, monkeypatch):
    db, queue = tmp_path / "fixture.db", tmp_path / "queue"
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue))
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    VectorStore(db).close()
    old = rollout("Previously indexed original prefix prompt")
    tail = rollout("Fresh unique suffix after the old checkpoint").splitlines(keepends=True)[1]
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    plain = sessions / "resume.jsonl"
    plain.write_bytes(old)
    w, _ = watcher(tmp_path, monkeypatch)
    w.watch_roots = [WatchRoot("codex", sessions)]
    w.indexer.on_flush = create_flush_callback(db_path=db, arbitrated=True)
    w.poll_once()
    args = dict(db_path=db, queue_dir=queue, log_path=tmp_path / "drain.log", pause_sentinel_path=tmp_path / "absent")
    drain_once(**args)
    with closing(sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        before = dict(conn.execute("SELECT id, seen_count FROM chunks"))
    assert len(before) == 1
    path = compress("sessions/resume.jsonl.zst", old + tail)
    plain.unlink()
    stat = path.stat()
    w.registry.set(str(plain), len(old), 0, archive=dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, ino=stat.st_ino))
    w.registry.flush()
    w.stop()
    for _ in range(2):  # a restart during the bounded replay must retain the replay semantics
        peer, _ = watcher(tmp_path, monkeypatch, budget=17)
        peer.watch_roots = [WatchRoot("codex", sessions)]
        peer.indexer.on_flush = create_flush_callback(db_path=db, arbitrated=True)
        for _ in range(4):
            peer.poll_once()
        peer.registry.flush()
        peer.stop()
    peer, _ = watcher(tmp_path, monkeypatch, budget=17)
    peer.watch_roots = [WatchRoot("codex", sessions)]
    peer.indexer.on_flush = create_flush_callback(db_path=db, arbitrated=True)
    for _ in range(200):
        peer.poll_once()
    assert len(list(queue.glob("*.jsonl"))) == 1  # only the new suffix is delivered
    drain_once(**args)
    with closing(sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        after = dict(conn.execute("SELECT id, seen_count FROM chunks"))
    assert len(after) == 2 and {k: after[k] for k in before} == before
    assert peer.registry.get(str(plain))[0] == len(old + tail)
    assert peer.archive_status()["incomplete"] == 0
    assert peer.registry.archive(str(plain))["complete_offset"] == len(old + tail)
    peer.stop()


def test_reconciliation_survives_both_representation_changes(compress, tmp_path, monkeypatch):
    old = rollout("Already acknowledged prefix content")
    data = old + rollout("New suffix content following the replay").splitlines(keepends=True)[1]
    archive = compress("transition.jsonl.zst", data)
    path = archive.with_suffix("")
    w, entries = watcher(tmp_path, monkeypatch, max_lines=1)
    st = archive.stat()
    w.registry.set(str(path), len(old), 0, archive=dict(size=st.st_size, mtime_ns=st.st_mtime_ns, ino=st.st_ino))
    w.poll_once()  # legacy archive replay consumes only the header
    assert w.registry.get(str(path))[0] == len(old)
    path.write_bytes(data)
    w.poll_once()  # materialized plain replay also starts with the header
    w.poll_once()
    assert entries and entries[0].get("_archive_reconciliation") is True
    path.unlink()
    for _ in range(4):
        w.poll_once()
    assert entries[-1]["_line_end_offset"] == len(data)
    assert not entries[-1].get("_archive_reconciliation")
    w.stop()
