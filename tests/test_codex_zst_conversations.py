import json

"""Native metadata must not replace the file's conversation identity."""

import sqlite3

from brainlayer.watcher_bridge import create_flush_callback
from tests.test_codex_zst_watcher import compress as _compress
from tests.test_codex_zst_watcher import watcher

compress = _compress
from tests.test_codex_zstd_stream import rollout


def archive_entries(compress, tmp_path, monkeypatch):
    path = compress(
        "parent.jsonl.zst", rollout("Archive conversation identity should match the historical plain path.")
    )
    instance, entries = watcher(tmp_path, monkeypatch)
    instance.poll_once()
    instance.stop()
    return path, entries


def test_plain_archive_conversation_parity(compress, tmp_path, monkeypatch):
    path, entries = archive_entries(compress, tmp_path, monkeypatch)
    logical = path.with_suffix("")
    logical.write_bytes(rollout())
    plain = dict(entries[0])
    for key in ("sessionId", "_compressed_archive", "codex_native_session_id"):
        plain.pop(key, None)
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "plain-q"))
    create_flush_callback(tmp_path / "x.db", arbitrated=True)([plain])
    rows = [json.loads(p.read_text()) for p in (tmp_path / "plain-q").glob("*.jsonl")]
    logical.unlink()
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "archive-q"))
    create_flush_callback(tmp_path / "x.db", arbitrated=True)(entries)
    archives = [json.loads(p.read_text()) for p in (tmp_path / "archive-q").glob("*.jsonl")]
    assert rows[0]["conversation_id"] == archives[0]["conversation_id"] == logical.stem
    assert rows[0]["chunk_id"] == archives[0]["chunk_id"]
    assert archives[0]["metadata"]["codex_native_session_id"] == "native-id"


def test_rewind_reaches_archive_ingested_rows(compress, tmp_path, monkeypatch):
    from brainlayer.drain import drain_once
    from brainlayer.queue_io import enqueue_rewind_archive_batch
    from brainlayer.vector_store import VectorStore

    path, entries = archive_entries(compress, tmp_path, monkeypatch)
    db = tmp_path / "x.db"
    VectorStore(db).close()
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    create_flush_callback(db, arbitrated=True)(entries)
    args = dict(
        db_path=db,
        queue_dir=tmp_path / "queue",
        log_path=tmp_path / "drain.log",
        pause_sentinel_path=tmp_path / "absent",
    )
    assert drain_once(**args) == 1
    logical = path.with_suffix("")
    enqueue_rewind_archive_batch(
        [dict(filepath=str(logical), session_id=logical.stem, old_offset=entries[0]["_line_end_offset"], new_offset=0)]
    )
    drain_once(**args)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT archived_at FROM chunks").fetchone()[0] is not None


def test_fork_pair_keeps_two_conversations(compress, tmp_path, monkeypatch):
    compress("parent.jsonl.zst", rollout("Unique parent content for its separate conversation."))
    compress("fork.jsonl.zst", rollout("Unique fork content sharing a native header identity."))
    instance, entries = watcher(tmp_path, monkeypatch)
    instance.poll_once()
    instance.stop()
    create_flush_callback(tmp_path / "x.db", arbitrated=True)(entries)
    rows = [json.loads(p.read_text()) for p in (tmp_path / "queue").glob("*.jsonl")]
    assert {r["conversation_id"] for r in rows} == {"parent", "fork"}
