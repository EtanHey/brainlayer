from __future__ import annotations

import json
from pathlib import Path

import apsw
import pytest

from brainlayer.drain import drain_once
from brainlayer.native_source_bindings import lookup_native_source_binding
from brainlayer.raw_source_archive import create_raw_source_archive
from brainlayer.vector_store import VectorStore
from brainlayer.watcher import JSONLWatcher, WatchRoot
from brainlayer.watcher_bridge import create_flush_callback

SESSION_A = "01a08d1e-42f6-7093-83a4-1c1b160f4911"
SESSION_B = "01a08d1e-4373-7631-b908-1a465cb013d3"


def _header(session_id: str, nickname: str) -> bytes:
    return (
        json.dumps(
            {"type": "session_meta", "payload": {"id": session_id, "agent_nickname": nickname}},
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


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


def _watcher(source_dir: Path, registry: Path, archive, callback) -> JSONLWatcher:
    return JSONLWatcher(
        watch_roots=[WatchRoot("codex", source_dir)],
        registry_path=registry,
        on_flush=callback,
        batch_size=100,
        flush_interval_ms=60_000,
        registry_flush_interval_s=0,
        raw_archive=archive,
    )


def _rows(db: Path) -> list[tuple[str, str, int]]:
    conn = apsw.Connection(str(db))
    try:
        return conn.execute(
            "SELECT record_key,native_generation_key,native_watcher_generation "
            "FROM chunk_source_records WHERE native_record_role='assistant' ORDER BY record_start_offset"
        ).fetchall()
    finally:
        conn.close()


def _binding(db: Path, key: str):
    conn = apsw.Connection(str(db))
    try:
        return lookup_native_source_binding(conn, record_key=key)
    finally:
        conn.close()


@pytest.fixture
def flow(tmp_path: Path, monkeypatch):
    db = tmp_path / "brainlayer.sqlite3"
    queue = tmp_path / "queue"
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue))
    monkeypatch.setenv("BRAINLAYER_T3_STATE_DB", str(tmp_path / "missing-t3.sqlite3"))
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    VectorStore(db).close()
    archive = create_raw_source_archive(db, {})
    source_dir = tmp_path / "codex" / "sessions"
    source_dir.mkdir(parents=True)
    return db, queue, archive, source_dir, source_dir / "rollout.jsonl"


def _queued_by_kind(queue: Path) -> dict[str, Path]:
    result = {}
    for path in queue.glob("*.jsonl"):
        event = json.loads(path.read_text().splitlines()[0])
        result[event["kind"]] = path
    return result


def test_arbitrated_watcher_chunk_can_arrive_before_header_and_lookup_waits_for_catalog(flow) -> None:
    db, queue, archive, source_dir, source = flow
    source.write_bytes(
        _header(SESSION_A, "Ada")
        + _assistant("This generated assistant answer is deliberately long enough for watcher indexing.")
    )
    watcher = _watcher(
        source_dir, source.parent.parent / "offsets.json", archive, create_flush_callback(db, arbitrated=True)
    )

    assert watcher.poll_once() == 2
    watcher.indexer.flush()
    queued = _queued_by_kind(queue)
    assert set(queued) == {"watcher_native_session_header", "watcher_chunk"}
    queued["watcher_chunk"].rename(queue / "000-watcher-chunk.jsonl")
    queued["watcher_native_session_header"].rename(queue / "999-native-header.jsonl")

    assert (
        drain_once(db_path=db, queue_dir=queue, batch_size=1, log_path=db.with_suffix(".log"), raw_archive=archive) == 1
    )
    key, generation, watcher_generation = _rows(db)[0]
    assert generation
    assert watcher_generation == 0
    assert _binding(db, key) is None

    assert (
        drain_once(db_path=db, queue_dir=queue, batch_size=1, log_path=db.with_suffix(".log"), raw_archive=archive) == 1
    )
    binding = _binding(db, key)
    assert binding["session_id"] == SESSION_A
    assert binding["header_context"]["/payload/agent_nickname"] == "Ada"


def test_direct_watcher_path_binds_physical_source_before_header_drain(flow) -> None:
    db, queue, archive, source_dir, source = flow
    source.write_bytes(
        _header(SESSION_A, "Direct")
        + _assistant("This direct-path assistant answer is deliberately long enough for watcher indexing.")
    )
    watcher = _watcher(
        source_dir, source.parent.parent / "offsets.json", archive, create_flush_callback(db, arbitrated=False)
    )

    assert watcher.poll_once() == 2
    watcher.indexer.flush()
    key, generation, _ = _rows(db)[0]
    assert generation
    assert _binding(db, key) is None
    assert {json.loads(path.read_text())["kind"] for path in queue.glob("*.jsonl")} == {"watcher_native_session_header"}

    assert drain_once(db_path=db, queue_dir=queue, log_path=db.with_suffix(".log"), raw_archive=archive) == 1
    assert _binding(db, key)["session_id"] == SESSION_A


def test_rewrite_creates_cross_generation_binding_and_late_conflict_taints_only_new_generation(flow) -> None:
    db, queue, archive, source_dir, source = flow
    source.write_bytes(
        _header(SESSION_A, "First")
        + _assistant(
            "The first generated assistant answer is deliberately much longer than the replacement so the watcher observes an explicit truncation and starts a new generation. "
            * 3
        )
    )
    registry = source.parent.parent / "offsets.json"
    first = _watcher(source_dir, registry, archive, create_flush_callback(db, arbitrated=True))
    assert first.poll_once() == 2
    first.indexer.flush()
    assert first.registry.flush() is True
    while list(queue.glob("*.jsonl")):
        drain_once(db_path=db, queue_dir=queue, log_path=db.with_suffix(".log"), raw_archive=archive)

    source.write_bytes(
        _header(SESSION_B, "Second")
        + _assistant(
            "The replacement assistant answer is distinct and deliberately long enough to pass every ordinary watcher content threshold."
        )
    )
    second = _watcher(source_dir, registry, archive, create_flush_callback(db, arbitrated=True))
    assert second.poll_once() == 2
    second.indexer.flush()
    while list(queue.glob("*.jsonl")):
        drain_once(db_path=db, queue_dir=queue, log_path=db.with_suffix(".log"), raw_archive=archive)

    rows = _rows(db)
    assert len(rows) == 2
    assert rows[0][1] != rows[1][1]
    assert len({row[2] for row in rows}) == 2
    assert {_binding(db, row[0])["session_id"] for row in rows} == {SESSION_A, SESSION_B}

    source.write_bytes(source.read_bytes() + _header(SESSION_A, "Conflict"))
    assert second.poll_once() == 1
    second.indexer.flush()
    while list(queue.glob("*.jsonl")):
        drain_once(db_path=db, queue_dir=queue, log_path=db.with_suffix(".log"), raw_archive=archive)
    bound = {_binding(db, row[0])["session_id"] if _binding(db, row[0]) else None for row in rows}
    assert bound == {SESSION_A, None}


def test_archive_failure_retains_queue_and_restart_drain_replays_binding(flow) -> None:
    db, queue, archive, source_dir, source = flow
    source.write_bytes(
        _header(SESSION_A, "Restart")
        + _assistant("This generated assistant answer survives a failed archive read and later drain restart.")
    )
    watcher = _watcher(
        source_dir, source.parent.parent / "offsets.json", archive, create_flush_callback(db, arbitrated=True)
    )
    assert watcher.poll_once() == 2
    watcher.indexer.flush()

    chunk_event_path = _queued_by_kind(queue)["watcher_chunk"]
    event = json.loads(chunk_event_path.read_text())
    receipt = event["native_source_binding"]["occurrence_receipt"]
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    chunk_sha = next((item["sha256"] for item in manifest.get("segments", []) if item.get("sha256")), None)
    assert chunk_sha
    object_path = archive.objects / chunk_sha
    saved = object_path.read_bytes()
    object_path.unlink()

    assert drain_once(db_path=db, queue_dir=queue, log_path=db.with_suffix(".log"), raw_archive=archive) < 2
    assert chunk_event_path.exists()
    assert not _rows(db)

    object_path.parent.mkdir(parents=True, exist_ok=True)
    object_path.write_bytes(saved)
    while list(queue.glob("*.jsonl")):
        drain_once(db_path=db, queue_dir=queue, log_path=db.with_suffix(".log"), raw_archive=archive)
    key = _rows(db)[0][0]
    assert _binding(db, key)["session_id"] == SESSION_A


def test_restarted_callback_resolves_exact_reserved_root_without_event_db_authority(flow) -> None:
    db, queue, archive, source_dir, source = flow
    source.write_bytes(_header(SESSION_A, "PersistedRoot"))
    registry = source.parent.parent / "offsets.json"
    first = _watcher(source_dir, registry, archive, create_flush_callback(db, arbitrated=True))
    assert first.poll_once() == 1
    first.indexer.flush()
    assert first.registry.flush() is True
    while list(queue.glob("*.jsonl")):
        drain_once(db_path=db, queue_dir=queue, log_path=db.with_suffix(".log"), raw_archive=archive)

    source.write_bytes(
        source.read_bytes()
        + _assistant("The post-restart assistant answer is long enough to exercise persisted root resolution.")
    )
    restarted = _watcher(source_dir, registry, archive, create_flush_callback(db, arbitrated=True))
    assert restarted.poll_once() == 1
    restarted.indexer.flush()
    event = json.loads(next(queue.glob("*.jsonl")).read_text())
    assert event["kind"] == "watcher_chunk"
    assert event["native_source_binding"]["root_receipt"] is None

    assert drain_once(db_path=db, queue_dir=queue, log_path=db.with_suffix(".log"), raw_archive=archive) == 1
    key = _rows(db)[0][0]
    assert _binding(db, key)["session_id"] == SESSION_A
