from __future__ import annotations

import json
import sqlite3

from brainlayer.drain import drain_once
from brainlayer.queue_io import enqueue_watcher_native_records
from brainlayer.vector_store import VectorStore
from brainlayer.watcher import BatchIndexer, JSONLWatcher
from brainlayer.watcher_bridge import create_flush_callback

_LONG_TEXT = (
    "This assistant response contains enough grounded detail to pass the normal "
    "watcher filters and create a searchable chunk for the invented fixture."
)


def _entry(
    source_file: str,
    start: int,
    end: int,
    digest_char: str,
    entry_type: str = "system",
    provider: str = "claude",
    session_id: str | None = "session-native",
    uuid: str = "native-control",
) -> dict:
    entry = {
        "type": entry_type,
        "_provider": provider,
        "_source_file": source_file,
        "_line_start_offset": start,
        "_line_end_offset": end,
        "_raw_record_hash": digest_char * 64,
        "uuid": uuid,
        "parentUuid": None,
        "compactMetadata": {"preservedSegment": {"marker": uuid}},
    }
    if session_id is not None:
        entry["sessionId"] = session_id
    if entry_type == "assistant":
        entry["message"] = {
            "role": "assistant",
            "content": [{"type": "text", "text": _LONG_TEXT}],
        }
    return entry


def _tool_use_entry(source_file: str, start: int, end: int, digest_char: str, target: str) -> dict:
    entry = _entry(source_file, start, end, digest_char, entry_type="assistant", uuid="native-tool")
    entry["message"] = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "tool-call-1",
                "name": "mcp__codex_app__send_message_to_thread",
                "input": {"threadId": target},
            }
        ],
    }
    return entry


def _target_rows(db_path) -> list[tuple]:
    with sqlite3.connect(db_path) as conn:
        try:
            return conn.execute(
                "SELECT target_extraction_version, declared_targets_json, target_diagnostics_json "
                "FROM native_record_links ORDER BY record_start_offset"
            ).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" not in str(exc):
                raise
            return []


def _drain(db_path, queue_dir, log_path) -> int:
    return drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=20, log_path=log_path)


def _native_rows(db_path) -> list[tuple]:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT provider, session_id, source_file, record_start_offset, record_end_offset, "
            "raw_record_hash, native_uuid, parent_uuid_state, declared_headers_json, "
            "preserved_segment_json FROM native_record_links ORDER BY record_start_offset"
        ).fetchall()


def test_filter_only_claude_control_is_cataloged_before_ack_without_classifier(tmp_path, monkeypatch):
    from brainlayer import watcher_bridge

    db_path = tmp_path / "direct.db"
    VectorStore(db_path).close()
    source_file = str(tmp_path / "session.jsonl")
    entry = _entry(source_file, 0, 90, "a")

    monkeypatch.setattr(
        watcher_bridge,
        "classify_content",
        lambda _entry: (_ for _ in ()).throw(AssertionError("unexpected classifier call")),
    )
    result = create_flush_callback(db_path, arbitrated=False)([entry])

    assert result[source_file] == 90
    rows = _native_rows(db_path)
    assert len(rows) == 1
    headers = json.loads(rows[0][8])
    assert headers == {"/parentUuid": None, "/type": "system", "/uuid": "native-control"}
    assert json.loads(rows[0][9]) == {"marker": "native-control"}
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone() == (0,)
        physical_index = conn.execute("PRAGMA index_info(idx_native_record_links_physical)").fetchall()
    assert [column[2] for column in physical_index] == [
        "provider",
        "session_id",
        "source_file",
        "record_start_offset",
        "record_end_offset",
        "raw_record_hash",
    ]


def test_mixed_chunk_and_control_batch_catalogs_each_physical_coordinate_once(tmp_path):
    db_path = tmp_path / "mixed.db"
    VectorStore(db_path).close()
    source_file = str(tmp_path / "session.jsonl")
    control = _entry(source_file, 0, 90, "a")
    assistant = _entry(source_file, 90, 290, "b", entry_type="assistant", uuid="native-assistant")

    result = create_flush_callback(db_path, arbitrated=False)([control, assistant])

    assert result[source_file] == 290
    assert [(row[3], row[4], row[5]) for row in _native_rows(db_path)] == [
        (0, 90, "a" * 64),
        (90, 290, "b" * 64),
    ]
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM native_record_links").fetchone() == (2,)
        assert conn.execute(
            "SELECT session_id, record_start_offset, record_end_offset, raw_record_hash FROM chunk_source_records"
        ).fetchone() == ("session-native", 90, 290, "b" * 64)


def test_direct_and_queued_native_catalog_content_match(tmp_path, monkeypatch):
    direct_db = tmp_path / "direct.db"
    queued_db = tmp_path / "queued.db"
    queue_dir = tmp_path / "queue"
    VectorStore(direct_db).close()
    VectorStore(queued_db).close()
    source_file = str(tmp_path / "session.jsonl")
    entries = [
        _entry(source_file, 0, 90, "a"),
        _entry(
            source_file=source_file,
            start=90,
            end=290,
            digest_char="b",
            entry_type="assistant",
            uuid="native-assistant",
        ),
    ]

    create_flush_callback(direct_db, arbitrated=False)(entries)
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue_dir))
    create_flush_callback(queued_db, arbitrated=True)(entries)
    native_batches = list(queue_dir.glob("watcher-native-record-*.jsonl"))
    assert len(native_batches) == 1
    assert len(native_batches[0].read_text(encoding="utf-8").splitlines()) == 2

    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    assert _drain(queued_db, queue_dir, tmp_path / "drain.log") == 3
    assert _native_rows(queued_db) == _native_rows(direct_db)


def test_tool_use_targets_match_direct_and_queued_and_excluded_reader(tmp_path, monkeypatch):
    direct_db = tmp_path / "direct-target.db"
    queued_db = tmp_path / "queued-target.db"
    queue_dir = tmp_path / "target-queue"
    VectorStore(direct_db).close()
    VectorStore(queued_db).close()
    source_file = str(tmp_path / "session.jsonl")
    target = "thread-" + "x" * 80
    entry = _tool_use_entry(source_file, 0, 240, "c", target)

    create_flush_callback(direct_db, arbitrated=False)([entry])
    create_flush_callback(direct_db, arbitrated=False)([entry])
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue_dir))
    create_flush_callback(queued_db, arbitrated=True)([entry])
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    assert _drain(queued_db, queue_dir, tmp_path / "target-drain.log") == 1

    assert _target_rows(direct_db) == _target_rows(queued_db)
    assert len(_target_rows(direct_db)) == 1
    version, targets_json, diagnostics_json = _target_rows(direct_db)[0]
    assert version == 1
    targets = json.loads(targets_json)
    assert targets[0]["target"] == target
    assert targets[0]["declared_unverified"] is True
    assert targets[0]["decoded_pointer"] == "/message/content/0/input/threadId"
    assert json.loads(diagnostics_json) == []
    with sqlite3.connect(direct_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone() == (0,)

    excluded = dict(entry)
    excluded["attributionAgent"] = "brain-worker"
    excluded["uuid"] = "excluded-tool"
    excluded_db = tmp_path / "excluded-target.db"
    VectorStore(excluded_db).close()
    create_flush_callback(excluded_db, arbitrated=False)([excluded])
    assert _target_rows(excluded_db) == []

def test_native_enqueue_error_cannot_confirm_offset(tmp_path, monkeypatch):
    from brainlayer import watcher_bridge

    db_path = tmp_path / "queued.db"
    VectorStore(db_path).close()
    entry = _entry(str(tmp_path / "session.jsonl"), 0, 90, "a")
    confirmations = []

    monkeypatch.setattr(
        watcher_bridge,
        "enqueue_watcher_native_records",
        lambda _records: (_ for _ in ()).throw(OSError("invented queue failure")),
    )
    indexer = BatchIndexer(
        create_flush_callback(db_path, arbitrated=True),
        batch_size=1,
        on_confirm_batch=lambda watermarks, _batch: confirmations.append(watermarks),
    )
    indexer.add([entry])

    assert confirmations == []
    assert indexer.retained_failed_input_count() == 1
    assert indexer.total_failed_inputs == 1


def test_native_drain_rolls_back_whole_batch_and_replays_idempotently(tmp_path, monkeypatch):
    from brainlayer import drain

    db_path = tmp_path / "drain.db"
    queue_dir = tmp_path / "queue"
    VectorStore(db_path).close()
    records = [
        {
            "provider": "claude",
            "session_id": "session-native",
            "source_file": "/fixture/session.jsonl",
            "record_start_offset": index * 100,
            "record_end_offset": index * 100 + 90,
            "raw_record_hash": digest_char * 64,
            "declared_headers": {"/uuid": f"native-{index}"},
        }
        for index, digest_char in enumerate(("a", "b"))
    ]
    queue_path = enqueue_watcher_native_records(records, queue_dir=queue_dir)
    original_catalog = drain.catalog_native_record
    calls = 0

    def fail_second(conn, **payload):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("invented apply failure")
        return original_catalog(conn, **payload)

    monkeypatch.setattr(drain, "catalog_native_record", fail_second)
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    assert _drain(db_path, queue_dir, tmp_path / "failed.log") == 0
    assert queue_path.exists()
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name = 'native_record_links'").fetchone() == (0,)

    monkeypatch.setattr(drain, "catalog_native_record", original_catalog)
    assert _drain(db_path, queue_dir, tmp_path / "replay.log") == 2
    assert not queue_path.exists()
    assert len(_native_rows(db_path)) == 2


def test_codex_is_unsupported_but_ordinary_ingest_continues(tmp_path):
    db_path = tmp_path / "codex.db"
    VectorStore(db_path).close()
    codex = _entry(
        str(tmp_path / ".codex" / "sessions" / "rollout.jsonl"),
        0,
        200,
        "c",
        entry_type="assistant",
        provider="codex",
        uuid="codex-shaped-id",
    )

    create_flush_callback(db_path, arbitrated=False)([codex])

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name = 'native_record_links'").fetchone() == (0,)
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone() == (1,)


def test_actual_watcher_denylist_blocks_memory_reader_before_native_catalog(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)
    db_path = tmp_path / "memory-reader.db"
    VectorStore(db_path).close()
    watch_dir = tmp_path / ".claude" / "projects"
    for role in ("brain-worker", "session-miner", "weave"):
        source = watch_dir / "project" / "session" / "subagents" / role / "agent.jsonl"
        source.parent.mkdir(parents=True)
        raw_entry = _entry(
            str(source),
            0,
            200,
            "f",
            entry_type="assistant",
            session_id=f"{role}-session",
            uuid=f"{role}-native",
        )
        source.write_text(
            json.dumps({key: value for key, value in raw_entry.items() if not key.startswith("_")}) + "\n",
            encoding="utf-8",
        )
        watcher = JSONLWatcher(
            watch_dir=watch_dir,
            registry_path=tmp_path / "offsets.json",
            on_flush=create_flush_callback(db_path, arbitrated=False),
            batch_size=1,
        )
        watcher.poll_once()

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name = 'native_record_links'").fetchone() == (0,)
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone() == (0,)

    ordinary = watch_dir / "project" / "session" / "subagents" / "workflow-subagent" / "agent.jsonl"
    ordinary.parent.mkdir(parents=True)
    raw_entry = _entry(
        str(ordinary),
        0,
        200,
        "e",
        entry_type="assistant",
        session_id="ordinary-session",
        uuid="ordinary-native",
    )
    raw_entry["attributionAgent"] = "workflow-subagent"
    ordinary.write_text(
        json.dumps({key: value for key, value in raw_entry.items() if not key.startswith("_")}) + "\n",
        encoding="utf-8",
    )
    watcher.poll_once()

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT native_uuid FROM native_record_links").fetchall() == [("ordinary-native",)]
