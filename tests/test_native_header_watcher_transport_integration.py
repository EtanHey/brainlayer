from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from brainlayer.drain import drain_once
from brainlayer.queue_io import enqueue_watcher_native_session_headers
from brainlayer.raw_source_archive import create_raw_source_archive
from brainlayer.vector_store import VectorStore
from brainlayer.watcher import JSONLWatcher, WatchRoot
from brainlayer.watcher_bridge import FlushWatermarks, create_flush_callback

SESSION_ID = "01a08d1e-42f6-7093-83a4-1c1b160f4911"
SESSION_ID_B = "01a08d1e-4373-7631-b908-1a465cb013d3"


def _header(session_id: str = SESSION_ID) -> bytes:
    return (json.dumps({"type": "session_meta", "payload": {"id": session_id}}, separators=(",", ":")) + "\n").encode()


def _assistant(text: str) -> bytes:
    return (
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                },
            },
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _watcher(
    sessions: Path,
    registry: Path,
    archive,
    callback,
    *,
    extra_roots: list[WatchRoot] | None = None,
) -> JSONLWatcher:
    return JSONLWatcher(
        watch_roots=[WatchRoot("codex", sessions), *(extra_roots or [])],
        registry_path=registry,
        on_flush=callback,
        batch_size=100,
        flush_interval_ms=60_000,
        registry_flush_interval_s=0,
        raw_archive=archive,
    )


def test_watcher_durable_header_handoff_then_drain_catalogs_without_early_catalog_claim(
    tmp_path: Path, monkeypatch
) -> None:
    db = tmp_path / "brainlayer.sqlite3"
    queue = tmp_path / "queue"
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue))
    monkeypatch.setenv("BRAINLAYER_T3_STATE_DB", str(tmp_path / "missing-t3.sqlite3"))
    VectorStore(db).close()
    archive = create_raw_source_archive(db, {})
    sessions = tmp_path / "codex" / "sessions"
    sessions.mkdir(parents=True)
    source = sessions / "rollout.jsonl"
    raw = _header()
    source.write_bytes(raw)
    current = _watcher(sessions, tmp_path / "offsets.json", archive, create_flush_callback(db, arbitrated=True))

    assert current.poll_once() == 1
    current.indexer.flush()
    assert current.registry.flush() is True
    assert current.registry.get(str(source))[0] == len(raw)
    assert current.registry.native_header_pending(str(source)) is None
    assert current.registry.native_header_root_receipt(str(source)) is not None
    queued = list(queue.glob("*.jsonl"))
    assert len(queued) == 1
    assert json.loads(queued[0].read_text())["kind"] == "watcher_native_session_header"
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='native_session_header_occurrences'"
        ).fetchone() == (0,)

    def no_embedding(_text: str) -> list[float]:
        raise AssertionError("header drain must not request embeddings")

    assert drain_once(db_path=db, queue_dir=queue, log_path=tmp_path / "drain.log", embed_fn=no_embedding) == 1
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT disposition,session_id FROM native_session_header_occurrences").fetchone() == (
            "canonical",
            SESSION_ID,
        )


def test_failed_header_handoff_holds_same_source_across_isolation_and_restart(tmp_path: Path, monkeypatch) -> None:
    from brainlayer import watcher_bridge

    db = tmp_path / "brainlayer.sqlite3"
    queue = tmp_path / "queue"
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue))
    monkeypatch.setenv("BRAINLAYER_T3_STATE_DB", str(tmp_path / "missing-t3.sqlite3"))
    archive = create_raw_source_archive(db, {})
    codex_dir = tmp_path / "codex" / "sessions"
    claude_root = tmp_path / "claude" / "projects"
    claude_project = claude_root / "project"
    codex_dir.mkdir(parents=True)
    claude_project.mkdir(parents=True)
    codex = codex_dir / "rollout.jsonl"
    claude = claude_project / "session.jsonl"
    codex.write_bytes(
        _header() + _assistant("Codex assistant content remains held until its header handoff is durable.")
    )
    claude.write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": "Unrelated Claude content advances despite the failed Codex header handoff."},
            }
        )
        + "\n"
    )

    def fail_header_handoff(*_args, **_kwargs):
        raise OSError("injected durable header handoff failure")

    monkeypatch.setattr(watcher_bridge, "enqueue_watcher_native_session_headers", fail_header_handoff, raising=False)
    registry = tmp_path / "offsets.json"
    first = _watcher(
        codex_dir,
        registry,
        archive,
        create_flush_callback(db, arbitrated=True),
        extra_roots=[WatchRoot("claude", claude_root)],
    )
    assert first.poll_once() == 3
    first.indexer.flush()
    assert first.registry.flush() is True
    assert first.registry.get(str(codex))[0] == 0
    assert first.registry.native_header_pending(str(codex)) is not None
    assert first.registry.get(str(claude))[0] == claude.stat().st_size
    assert len(first.indexer._buffer) == 2
    assert all("watcher_native_session_header" not in path.read_text() for path in queue.glob("*.jsonl"))

    monkeypatch.setattr(
        watcher_bridge, "enqueue_watcher_native_session_headers", enqueue_watcher_native_session_headers
    )
    restarted = _watcher(codex_dir, registry, archive, create_flush_callback(db, arbitrated=True))
    assert restarted.poll_once() == 2
    restarted.indexer.flush()
    assert restarted.registry.flush() is True
    assert restarted.registry.get(str(codex))[0] == codex.stat().st_size
    assert restarted.registry.native_header_pending(str(codex)) is None
    assert not restarted.indexer._buffer
    assert any("watcher_native_session_header" in path.read_text() for path in queue.glob("*.jsonl"))


def test_multiple_headers_flow_through_watcher_queue_and_catalog_as_canonical_repeat_conflict(
    tmp_path: Path, monkeypatch
) -> None:
    db = tmp_path / "brainlayer.sqlite3"
    queue = tmp_path / "queue"
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue))
    monkeypatch.setenv("BRAINLAYER_T3_STATE_DB", str(tmp_path / "missing-t3.sqlite3"))
    VectorStore(db).close()
    archive = create_raw_source_archive(db, {})
    sessions = tmp_path / "codex" / "sessions"
    sessions.mkdir(parents=True)
    source = sessions / "rollout.jsonl"
    source.write_bytes(_header() + _header() + _header(SESSION_ID_B))
    current = _watcher(sessions, tmp_path / "offsets.json", archive, create_flush_callback(db, arbitrated=True))

    assert current.poll_once() == 3
    current.indexer.flush()
    assert current.registry.get(str(source))[0] == source.stat().st_size
    assert current.registry.native_header_pending(str(source)) is None
    assert drain_once(db_path=db, queue_dir=queue, log_path=tmp_path / "drain.log") == 3
    with sqlite3.connect(db) as conn:
        dispositions = [
            row[0]
            for row in conn.execute(
                "SELECT disposition FROM native_session_header_occurrences ORDER BY record_start_offset"
            )
        ]
    assert dispositions == ["canonical", "repeat", "conflict"]


def test_mismatched_handoff_token_cannot_clear_pending_or_advance_offset(tmp_path: Path) -> None:
    sessions = tmp_path / "codex" / "sessions"
    sessions.mkdir(parents=True)
    source = sessions / "rollout.jsonl"
    source.write_bytes(_header())
    archive = create_raw_source_archive(tmp_path / "brainlayer.sqlite3", {})

    def callback(entries: list[dict]) -> FlushWatermarks:
        entry = entries[0]
        token = {
            "provider": entry["_provider"],
            "source_file": entry["_source_file"],
            "watcher_generation": entry["_source_generation"] + 1,
            "root_receipt": entry["_native_header_root_receipt"],
            "occurrence_receipt": entry["_raw_archive_receipt"],
            "record_start_offset": entry["_line_start_offset"],
            "record_end_offset": entry["_line_end_offset"],
            "raw_record_hash": entry["_raw_record_hash"],
        }
        return FlushWatermarks(
            {str(source): entry["_line_end_offset"]},
            native_header_handoffs=[token],
        )

    callback.bind_raw_archive = lambda _archive: None
    callback.native_header_transport_ready = lambda: True
    current = _watcher(sessions, tmp_path / "offsets.json", archive, callback)

    assert current.poll_once() == 1
    current.indexer.flush()
    assert current.registry.get(str(source))[0] == 0
    assert current.registry.native_header_pending(str(source)) is not None
    assert len(current.indexer._buffer) == 1
