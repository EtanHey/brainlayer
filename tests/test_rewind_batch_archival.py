from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import apsw

from brainlayer.cli import _RewindArchiveBatcher
from brainlayer.drain import drain_once
from brainlayer.queue_io import enqueue_rewind_archive_batch, enqueue_watcher_chunk
from brainlayer.vector_store import VectorStore


def test_rewind_batcher_queues_full_offset_intents_without_opening_the_db(tmp_path: Path) -> None:
    queued: list[list[dict[str, object]]] = []
    batcher = _RewindArchiveBatcher(
        batch_size=2,
        flush_interval_ms=60_000,
        enqueue_batch=lambda events: queued.append(events) or tmp_path / "queued.jsonl",
        wall_clock=lambda: 123.5,
    )

    batcher.add("/tmp/session.jsonl", "session", 900, 400)
    assert batcher.maybe_flush("interval") == 0
    batcher.add("/tmp/other.jsonl", "other", 700, 300)

    assert batcher.maybe_flush("interval") == 2
    assert batcher.pending_count == 0
    assert not (tmp_path / "must-not-be-opened.db").exists()
    assert queued == [
        [
            {
                "filepath": "/tmp/session.jsonl",
                "session_id": "session",
                "old_offset": 900,
                "new_offset": 400,
                "rewind_detected_at": 123.5,
            },
            {
                "filepath": "/tmp/other.jsonl",
                "session_id": "other",
                "old_offset": 700,
                "new_offset": 300,
                "rewind_detected_at": 123.5,
            },
        ]
    ]


def test_rewind_batcher_queue_failure_preserves_pending_for_tick_retry(tmp_path: Path) -> None:
    attempts = 0

    def flaky_enqueue(events: list[dict[str, object]]) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("queue temporarily unavailable")
        assert len(events) == 1
        return tmp_path / "queued.jsonl"

    batcher = _RewindArchiveBatcher(
        batch_size=1,
        flush_interval_ms=1,
        enqueue_batch=flaky_enqueue,
    )
    batcher.add("/tmp/session.jsonl", "session", 900, 400)

    try:
        batcher.maybe_flush("threshold")
    except OSError as exc:
        assert str(exc) == "queue temporarily unavailable"
    else:
        raise AssertionError("first queue attempt should fail")

    assert batcher.pending_count == 1
    time.sleep(0.002)
    assert batcher.maybe_flush("tick") == 1
    assert batcher.pending_count == 0
    assert attempts == 2


def test_enqueue_rewind_archive_batch_is_durable_and_preserves_bounds(tmp_path: Path) -> None:
    path = enqueue_rewind_archive_batch(
        [
            {
                "filepath": "/tmp/session.jsonl",
                "session_id": "session",
                "old_offset": 900,
                "new_offset": 400,
            }
        ],
        queue_dir=tmp_path,
        detected_at=123.5,
    )

    event = json.loads(path.read_text(encoding="utf-8"))
    assert event["kind"] == "rewind_archive"
    assert event["source_file"] == "/tmp/session.jsonl"
    assert event["conversation_id"] == "session"
    assert event["old_offset"] == 900
    assert event["new_offset"] == 400
    assert event["rewind_detected_at"] == 123.5


def _prepare_db(path: Path) -> None:
    VectorStore(path).close()


def _insert_rewind_row(
    conn: sqlite3.Connection,
    *,
    chunk_id: str,
    source_file: str = "/tmp/session.jsonl",
    conversation_id: str = "session",
    source: str = "realtime_watcher",
    source_end_offset: int | None,
    source_last_queued_at: float | None = 50.0,
    archived_at: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO chunks (
            id, content, metadata, source_file, source, conversation_id,
            value_type, content_type, source_end_offset, source_last_queued_at,
            archived_at, archived, status
        ) VALUES (?, ?, '{}', ?, ?, ?, 'KNOWLEDGE', 'assistant_text', ?, ?, ?, ?, ?)
        """,
        (
            chunk_id,
            f"content {chunk_id}",
            source_file,
            source,
            conversation_id,
            source_end_offset,
            source_last_queued_at,
            archived_at,
            int(archived_at is not None),
            "archived" if archived_at else "active",
        ),
    )


def test_drain_archives_only_reverted_offset_window_and_fails_closed_for_legacy_rows(
    tmp_path: Path, monkeypatch
) -> None:
    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    log_path = tmp_path / "drain.log"
    _prepare_db(db_path)
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    with sqlite3.connect(db_path) as conn:
        _insert_rewind_row(conn, chunk_id="before", source_end_offset=100)
        _insert_rewind_row(conn, chunk_id="reverted-a", source_end_offset=101)
        _insert_rewind_row(conn, chunk_id="reverted-b", source_end_offset=200)
        _insert_rewind_row(conn, chunk_id="after-old", source_end_offset=201)
        _insert_rewind_row(conn, chunk_id="legacy-unknown", source_end_offset=None)
        _insert_rewind_row(conn, chunk_id="new-branch", source_end_offset=150, source_last_queued_at=125.0)
        _insert_rewind_row(conn, chunk_id="other-file", source_file="/tmp/other.jsonl", source_end_offset=150)
        _insert_rewind_row(conn, chunk_id="other-session", conversation_id="other", source_end_offset=150)
        _insert_rewind_row(conn, chunk_id="manual", source="manual", source_end_offset=150)
        _insert_rewind_row(
            conn,
            chunk_id="already-archived",
            source_end_offset=150,
            archived_at="2026-01-01T00:00:00Z",
        )
        conn.commit()

    enqueue_rewind_archive_batch(
        [
            {
                "filepath": "/tmp/session.jsonl",
                "session_id": "session",
                "old_offset": 200,
                "new_offset": 100,
            }
        ],
        queue_dir=queue_dir,
        detected_at=100.0,
    )

    assert drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=1, log_path=log_path) == 1
    assert not list(queue_dir.glob("*.jsonl"))
    with sqlite3.connect(db_path) as conn:
        rows = dict(conn.execute("SELECT id, archived_at IS NOT NULL FROM chunks"))
        lifecycle = conn.execute("SELECT value_type, archived, status FROM chunks WHERE id = 'reverted-a'").fetchone()

    assert {chunk_id for chunk_id, archived in rows.items() if archived} == {
        "reverted-a",
        "reverted-b",
        "already-archived",
    }
    assert lifecycle == ("ARCHIVED", 1, "archived")


def test_watcher_offset_is_persisted_and_same_source_replay_reactivates_archived_chunk(
    tmp_path: Path, monkeypatch
) -> None:
    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    _prepare_db(db_path)
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    content = "same watcher content reappeared on the surviving branch"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO chunks (
                id, content, metadata, source_file, source, conversation_id,
                value_type, content_type, source_end_offset, source_last_queued_at,
                archived_at, archived, status
            ) VALUES (
                'rt-replay', ?, '{}', '/tmp/session.jsonl', 'realtime_watcher', 'session',
                'ARCHIVED', 'assistant_text', 180, 50.0, '2026-01-01T00:00:00Z', 1, 'archived'
            )
            """,
            (content,),
        )
        conn.commit()

    enqueue_watcher_chunk(
        chunk_id="rt-replay",
        content=content,
        metadata={},
        source_file="/tmp/session.jsonl",
        source_end_offset=80,
        project="brainlayer",
        content_type="assistant_text",
        value_type="high",
        created_at="2026-07-18T12:00:00Z",
        conversation_id="session",
        queue_dir=queue_dir,
    )

    assert drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=1, log_path=tmp_path / "drain.log") == 1
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT source_end_offset, archived_at, archived, status, value_type,
                   source_last_queued_at IS NOT NULL
            FROM chunks WHERE id = 'rt-replay'
            """
        ).fetchone()

    assert row == (80, None, 0, "active", "high", 1)


def test_locked_rewind_archive_rolls_back_and_queue_file_retries(tmp_path: Path, monkeypatch) -> None:
    from brainlayer import drain

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    log_path = tmp_path / "drain.log"
    _prepare_db(db_path)
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    monkeypatch.setattr("brainlayer.drain._sleep", lambda _seconds: None)
    with sqlite3.connect(db_path) as conn:
        _insert_rewind_row(conn, chunk_id="reverted", source_end_offset=150)
        conn.commit()
    queued = enqueue_rewind_archive_batch(
        [
            {
                "filepath": "/tmp/session.jsonl",
                "session_id": "session",
                "old_offset": 200,
                "new_offset": 100,
            }
        ],
        queue_dir=queue_dir,
        detected_at=100.0,
    )

    original_apply = drain._apply_rewind_archive
    busy_attempts = 0

    def busy_archive(conn, event):
        nonlocal busy_attempts
        busy_attempts += 1
        raise apsw.BusyError("database is locked")

    monkeypatch.setattr(drain, "_apply_rewind_archive", busy_archive)
    assert drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=1, log_path=log_path) == 0
    assert busy_attempts == 5
    assert queued.exists()

    monkeypatch.setattr(drain, "_apply_rewind_archive", original_apply)
    assert drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=1, log_path=log_path) == 1
    assert not queued.exists()
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT archived_at IS NOT NULL FROM chunks WHERE id = 'reverted'").fetchone() == (1,)
