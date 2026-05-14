import multiprocessing as mp
import sqlite3
import time
from pathlib import Path


def _producer(queue_dir: str, worker_id: int, count: int) -> None:
    from brainlayer.queue_io import enqueue_store

    for index in range(count):
        enqueue_store(
            content=f"arbitration worker={worker_id} item={index}",
            memory_type="note",
            project="arbitration-test",
            tags=["arbitration", f"worker-{worker_id}"],
            importance=7,
            queue_dir=Path(queue_dir),
            source="test",
        )


def _create_minimal_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                source_file TEXT NOT NULL,
                project TEXT,
                content_type TEXT,
                value_type TEXT,
                char_count INTEGER,
                source TEXT,
                created_at TEXT,
                enriched_at TEXT,
                summary TEXT,
                tags TEXT,
                importance REAL,
                superseded_by TEXT
            );
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id UNINDEXED
            );
            CREATE TRIGGER chunks_fts_insert AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(content, summary, tags, chunk_id)
                VALUES (new.content, new.summary, new.tags, new.id);
            END;
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_drain_daemon_serializes_three_concurrent_producers(tmp_path):
    from brainlayer.drain import drain_once

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    log_path = tmp_path / "drain.log"
    _create_minimal_db(db_path)

    workers = [mp.Process(target=_producer, args=(str(queue_dir), worker, 1000)) for worker in range(3)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=20)
        assert worker.exitcode == 0

    deadline = time.monotonic() + 5
    total_drained = 0
    while time.monotonic() < deadline:
        total_drained += drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=250, log_path=log_path)
        with sqlite3.connect(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM chunks WHERE project = 'arbitration-test'").fetchone()[0]
            fts_count = conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'arbitration'").fetchone()[0]
        if count == 3000 and fts_count == 3000:
            break
        time.sleep(0.05)

    assert total_drained == 3000
    assert count == 3000
    assert fts_count == 3000
    assert not list(queue_dir.glob("*.jsonl"))
    assert "database is locked" not in log_path.read_text(encoding="utf-8").lower()


def test_queue_sanitizes_source_and_drain_preserves_supersedes(tmp_path):
    from brainlayer.drain import drain_once
    from brainlayer.queue_io import enqueue_store

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    _create_minimal_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO chunks (id, content, metadata, source_file)
            VALUES ('old-id', 'old content', '{}', 'seed')
            """
        )
        conn.commit()

    queued_path = enqueue_store(
        content="replacement content",
        project="arbitration-test",
        source="../unsafe/source",
        supersedes="old-id",
        queue_dir=queue_dir,
    )

    assert queued_path.parent == queue_dir
    assert ".." not in queued_path.name
    assert "/" not in queued_path.name
    assert drain_once(db_path=db_path, queue_dir=queue_dir) == 1

    with sqlite3.connect(db_path) as conn:
        replacement_id = conn.execute(
            "SELECT id FROM chunks WHERE content = 'replacement content'"
        ).fetchone()[0]
        superseded_by = conn.execute("SELECT superseded_by FROM chunks WHERE id = 'old-id'").fetchone()[0]

    assert superseded_by == replacement_id
