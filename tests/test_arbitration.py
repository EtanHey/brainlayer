import asyncio
import hashlib
import importlib
import json
import multiprocessing as mp
import re
import sqlite3
import sys
import threading
import time
from pathlib import Path

import apsw
import sqlite_vec


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
                content_hash TEXT,
                superseded_by TEXT
            );
            CREATE TABLE kg_entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL
            );
            CREATE TABLE kg_entity_chunks (
                entity_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                relevance REAL DEFAULT 1.0,
                context TEXT,
                PRIMARY KEY (entity_id, chunk_id)
            );
            CREATE TABLE chunk_vectors (
                chunk_id TEXT PRIMARY KEY,
                embedding BLOB
            );
            CREATE TABLE chunk_vectors_binary (
                chunk_id TEXT PRIMARY KEY,
                embedding BLOB
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


def _create_preview_trigger_db(path: Path) -> None:
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
                summary TEXT,
                tags TEXT,
                importance REAL,
                preview_text TEXT,
                seen_count INTEGER DEFAULT 1,
                last_seen_at TEXT,
                dedupe_hash TEXT,
                simhash TEXT,
                simhash_band_0 TEXT,
                simhash_band_1 TEXT,
                simhash_band_2 TEXT,
                simhash_band_3 TEXT,
                archived INTEGER DEFAULT 0,
                archived_at TEXT,
                superseded_by TEXT
            );
            CREATE TABLE preview_trigger_hits (chunk_id TEXT NOT NULL);
            CREATE TRIGGER chunks_preview_text_insert
            AFTER INSERT ON chunks
            WHEN new.preview_text IS NULL OR trim(new.preview_text) = ''
            BEGIN
                INSERT INTO preview_trigger_hits(chunk_id) VALUES (new.id);
                UPDATE chunks
                SET preview_text = trim(substr(replace(replace(replace(content, char(10), ' '), char(13), ' '), char(9), ' '), 1, 220))
                WHERE rowid = new.rowid;
            END;
            """
        )
        conn.commit()
    finally:
        conn.close()


def _connect_apsw(path: Path) -> apsw.Connection:
    conn = apsw.Connection(str(path))
    conn.enableloadextension(True)
    conn.loadextension(sqlite_vec.loadable_path())
    conn.enableloadextension(False)
    return conn


def _create_vec_db(path: Path) -> None:
    conn = _connect_apsw(path)
    try:
        conn.execute("""
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
                importance REAL
            )
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE chunk_vectors USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding FLOAT[1024]
            )
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE chunk_vectors_binary USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding BIT[1024]
            )
        """)
    finally:
        conn.close()


def _load_hotlane_module():
    importlib.invalidate_caches()
    sys.modules.pop("scripts.hotlane_brainbar_daemon", None)
    return importlib.import_module("scripts.hotlane_brainbar_daemon")


class _FastEmbeddingModel:
    def embed_query(self, _text: str) -> list[float]:
        return [0.03125] * 1024

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.03125] * 1024 for _text in texts]


def test_drain_default_queue_dir_expands_env_tilde(monkeypatch):
    from brainlayer.drain import _default_queue_dir

    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", "~/brainlayer-arbitration-test")

    queue_dir = _default_queue_dir()

    assert "~" not in str(queue_dir)
    assert queue_dir == Path.home() / "brainlayer-arbitration-test"


def test_drain_prioritizes_writes_before_enrichment(tmp_path, monkeypatch):
    from brainlayer.drain import drain_once

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    _create_minimal_db(db_path)
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")

    (queue_dir / "enrichment-000.jsonl").write_text(
        json.dumps({"kind": "enrichment_update", "chunk_id": "missing", "enrichment": {"summary": "later"}}) + "\n",
        encoding="utf-8",
    )
    (queue_dir / "watcher-999.jsonl").write_text(
        json.dumps(
            {
                "kind": "watcher_chunk",
                "chunk_id": "watcher-priority",
                "content": "Queued watcher writes must drain before enrichment backlog.",
                "created_at": "2026-05-27T04:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=1) == 1

    assert not (queue_dir / "watcher-999.jsonl").exists()
    assert (queue_dir / "enrichment-000.jsonl").exists()
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT id FROM chunks WHERE id = 'watcher-priority'").fetchone() == ("watcher-priority",)


def test_drain_yields_enrichment_when_interactive_store_is_queued(tmp_path, monkeypatch):
    """Interactive stores are higher priority than enrichment metadata writes."""
    from brainlayer.drain import drain_once
    from brainlayer.queue_io import enqueue_enrichment_updates, enqueue_store

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    _create_minimal_db(db_path)
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")

    with _connect_apsw(db_path) as conn:
        conn.execute(
            """
            INSERT INTO chunks (id, content, metadata, source_file, content_hash)
            VALUES ('enrich-low-priority', 'metadata can wait', '{}', 'fixture', 'hash-low')
            """
        )

    store_path = enqueue_store(
        content="interactive store must drain before enrichment",
        memory_type="note",
        project="arbitration-test",
        source="mcp",
        queue_dir=queue_dir,
    )
    enrichment_path = enqueue_enrichment_updates(
        [
            {
                "chunk_id": "enrich-low-priority",
                "content_hash": "hash-low",
                "enrichment": {"summary": "low priority summary"},
            }
        ],
        queue_dir=queue_dir,
    )

    assert drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=10) == 1

    assert not store_path.exists()
    assert enrichment_path.exists()
    with _connect_apsw(db_path) as conn:
        stored = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE content = 'interactive store must drain before enrichment'"
        ).fetchone()[0]
        summary = conn.execute("SELECT summary FROM chunks WHERE id = 'enrich-low-priority'").fetchone()[0]
    assert stored == 1
    assert summary is None


def test_drain_skips_stale_enrichment_for_rewritten_chunk(tmp_path, monkeypatch):
    from brainlayer.drain import drain_once

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    _create_minimal_db(db_path)
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    old_hash = hashlib.sha256("old content".encode("utf-8")).hexdigest()
    new_hash = hashlib.sha256("new content".encode("utf-8")).hexdigest()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO chunks(id, content, metadata, source_file, summary, content_hash)
            VALUES ('rewritten', 'new content', '{}', 'watcher', 'new summary', ?)
            """,
            (new_hash,),
        )
    (queue_dir / "enrichment-000.jsonl").write_text(
        json.dumps(
            {
                "kind": "enrichment_update",
                "chunk_id": "rewritten",
                "content_hash": old_hash,
                "enrichment": {"summary": "old stale summary", "tags": ["old"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=1) == 1

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT summary, tags, content_hash FROM chunks WHERE id = 'rewritten'").fetchone() == (
            "new summary",
            None,
            new_hash,
        )


def test_drain_sets_preview_text_on_initial_insert(tmp_path, monkeypatch):
    from brainlayer.drain import drain_once

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    _create_preview_trigger_db(db_path)
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    (queue_dir / "watcher.jsonl").write_text(
        json.dumps(
            {
                "kind": "watcher_chunk",
                "chunk_id": "watcher-preview",
                "content": "Watcher preview text is written during the insert.",
                "created_at": "2026-05-27T04:10:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=1) == 1

    with sqlite3.connect(db_path) as conn:
        preview_text, trigger_hits = conn.execute(
            """
            SELECT c.preview_text, COUNT(p.chunk_id)
            FROM chunks c
            LEFT JOIN preview_trigger_hits p ON p.chunk_id = c.id
            WHERE c.id = 'watcher-preview'
            GROUP BY c.id
            """
        ).fetchone()

    assert preview_text == "Watcher preview text is written during the insert."
    assert trigger_hits == 0


def test_drain_preview_text_uses_content_when_summary_is_blank(tmp_path, monkeypatch):
    from brainlayer.drain import _insert_chunk

    db_path = tmp_path / "brainlayer.db"
    _create_preview_trigger_db(db_path)
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    with _connect_apsw(db_path) as conn:
        _insert_chunk(
            conn,
            {
                "id": "store-preview",
                "content": "Content should win when summary is whitespace.",
                "summary": "\n\t  ",
                "metadata": "{}",
                "source_file": "test",
            },
        )

    with sqlite3.connect(db_path) as conn:
        preview_text, trigger_hits = conn.execute(
            """
            SELECT c.preview_text, COUNT(p.chunk_id)
            FROM chunks c
            LEFT JOIN preview_trigger_hits p ON p.chunk_id = c.id
            WHERE c.id = 'store-preview'
            GROUP BY c.id
            """
        ).fetchone()

    assert preview_text == "Content should win when summary is whitespace."
    assert trigger_hits == 0


def test_drain_seen_merge_does_not_touch_unchanged_tags(tmp_path, monkeypatch):
    from brainlayer.drain import drain_once

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    _create_minimal_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE tag_update_hits (chunk_id TEXT NOT NULL);
            CREATE TRIGGER chunks_tags_update_seen_test
            AFTER UPDATE OF tags ON chunks
            BEGIN
                INSERT INTO tag_update_hits(chunk_id) VALUES (new.id);
            END;
            """
        )
        conn.execute(
            """
            INSERT INTO chunks(id, content, metadata, source_file, content_type, char_count, created_at, tags)
            VALUES ('watcher-seen', 'Same watcher content', '{}', 'watcher', 'assistant_text', 20,
                    '2026-05-27T04:00:00Z', NULL)
            """
        )
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    (queue_dir / "watcher.jsonl").write_text(
        json.dumps(
            {
                "kind": "watcher_chunk",
                "chunk_id": "watcher-seen",
                "content": "Same watcher content",
                "content_type": "assistant_text",
                "created_at": "2026-05-27T04:10:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=1) == 1

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tag_update_hits").fetchone()[0] == 0


def test_drain_seen_merge_compares_tags_semantically(tmp_path, monkeypatch):
    from brainlayer.drain import drain_once

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    _create_minimal_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE tag_update_hits (chunk_id TEXT NOT NULL);
            CREATE TRIGGER chunks_tags_update_seen_test
            AFTER UPDATE OF tags ON chunks
            BEGIN
                INSERT INTO tag_update_hits(chunk_id) VALUES (new.id);
            END;
            """
        )
        conn.execute(
            """
            INSERT INTO chunks(id, content, metadata, source_file, content_type, char_count, created_at, tags)
            VALUES ('watcher-tags', 'Same watcher content', '{}', 'watcher', 'assistant_text', 20,
                    '2026-05-27T04:00:00Z', '["b", "a"]')
            """
        )
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    (queue_dir / "watcher.jsonl").write_text(
        json.dumps(
            {
                "kind": "watcher_chunk",
                "chunk_id": "watcher-tags",
                "content": "Same watcher content",
                "content_type": "assistant_text",
                "tags": ["a", "b"],
                "created_at": "2026-05-27T04:10:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=1) == 1

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tag_update_hits").fetchone()[0] == 0


def test_drain_daemon_serializes_three_concurrent_producers(tmp_path, monkeypatch):
    from brainlayer.drain import drain_once

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    log_path = tmp_path / "drain.log"
    _create_minimal_db(db_path)
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")

    workers = [mp.Process(target=_producer, args=(str(queue_dir), worker, 1000)) for worker in range(3)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=20)
        assert worker.exitcode == 0

    # Poll-until-drained with a generous deadline. The completion signal is the
    # DB end-state (3000 rows + FTS + empty queue), NOT a fixed time window: under
    # machine saturation (many concurrent agents + enrichment competing for the
    # writer) the drain legitimately takes longer. The old 45s cap was even shorter
    # than the ~47s isolation runtime, so it flaked as `assert 2500 == 3000` under
    # load. We stop as soon as the queue is fully consumed (deterministic), with the
    # deadline only as a backstop against a genuine hang.
    deadline = time.monotonic() + 240
    total_drained = 0
    count = fts_count = 0
    while time.monotonic() < deadline:
        total_drained += drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=250, log_path=log_path)
        with _connect_apsw(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM chunks WHERE project = 'arbitration-test'").fetchone()[0]
            fts_count = conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'arbitration'").fetchone()[
                0
            ]
        queue_empty = not list(queue_dir.glob("*.jsonl"))
        if count == 3000 and fts_count == 3000 and queue_empty:
            break
        time.sleep(0.05)

    assert count == 3000, (
        f"drained {count}/3000 chunks before deadline (queue_empty={not list(queue_dir.glob('*.jsonl'))})"
    )
    assert fts_count == 3000, f"FTS indexed {fts_count}/3000"
    assert total_drained == 3000, f"drain_once accumulated {total_drained} (expected 3000)"
    assert not list(queue_dir.glob("*.jsonl"))
    assert "database is locked" not in log_path.read_text(encoding="utf-8").lower()


def test_real_concurrent_writers_keep_interactive_store_searchable_under_sla(tmp_path, monkeypatch):
    """Real drain + hotlane writers must not starve interactive store->search."""
    from brainlayer.drain import drain_once
    from brainlayer.mcp import search_handler, store_handler
    from brainlayer.mcp.search_handler import _search
    from brainlayer.mcp.store_handler import _store
    from brainlayer.queue_io import enqueue_enrichment_updates, enqueue_store
    from brainlayer.vector_store import VectorStore

    hotlane = _load_hotlane_module()
    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    writer_pid_dir = tmp_path / "pidfiles"
    log_path = tmp_path / "drain.log"
    marker = "d3realconcurrencygate42"

    monkeypatch.setenv("BRAINLAYER_ARBITRATED", "1")
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue_dir))
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(writer_pid_dir))
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "1")
    monkeypatch.setenv("BRAINLAYER_STORE_BUSY_BUDGET_MS", "400")
    monkeypatch.setenv("BRAINLAYER_EMBED_TIMEOUT_MS", "1000")
    monkeypatch.setattr(store_handler, "_get_embedding_model", lambda: _FastEmbeddingModel())
    monkeypatch.setattr(search_handler, "_get_embedding_model", lambda: _FastEmbeddingModel())

    store = VectorStore(db_path)
    now = "2026-06-21T20:00:00+00:00"
    try:
        cursor = store.conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        for index in range(48):
            chunk_id = f"hotlane-seed-{index}"
            content = f"hotlane seed backlog item {index}"
            cursor.execute(
                """
                INSERT INTO chunks (
                    id, content, metadata, source_file, project, content_type,
                    value_type, char_count, source, created_at, status
                )
                VALUES (?, ?, '{}', 'brainbar-store', 'arbitration-test', 'note',
                        'high', ?, 'mcp', ?, 'active')
                """,
                (chunk_id, content, len(content), now),
            )
        cursor.execute("COMMIT")
    finally:
        store.close()

    for index in range(8):
        enqueue_store(
            content=f"drain background backlog item {index}",
            memory_type="note",
            project="arbitration-test",
            source="watcher",
            queue_dir=queue_dir,
        )
    enqueue_enrichment_updates(
        [
            {
                "chunk_id": "hotlane-seed-0",
                "content_hash": None,
                "enrichment": {"summary": "low priority enrichment must yield"},
            }
        ],
        queue_dir=queue_dir,
    )

    stop = threading.Event()
    drain_writes = 0
    hotlane_embeddings = 0
    counters_lock = threading.Lock()

    def drain_loop() -> None:
        nonlocal drain_writes
        while not stop.is_set():
            drained = drain_once(
                db_path=db_path,
                queue_dir=queue_dir,
                batch_size=8,
                log_path=log_path,
                embed_fn=lambda _text: [0.0625] * 1024,
            )
            if drained:
                with counters_lock:
                    drain_writes += drained
            time.sleep(0.005)

    def recording_cycle(**kwargs):
        nonlocal hotlane_embeddings
        result = hotlane.run_cycle(**kwargs)
        if result.embedded:
            with counters_lock:
                hotlane_embeddings += result.embedded
        return result

    def hotlane_loop() -> None:
        hotlane.STOP = False
        hotlane.run(
            db_path=db_path,
            interval=0.005,
            recent_limit=4,
            backlog_interval=0.005,
            backlog_batch=4,
            enrich_interval=9999,
            enrich_limit=0,
            enrich_since_hours=24,
            vector_store_cls=VectorStore,
            model_factory=_FastEmbeddingModel,
            cycle_fn=recording_cycle,
            sleep_fn=lambda seconds: time.sleep(min(seconds, 0.005)),
            max_cycles=400,
            queue_dir=queue_dir,
        )

    drain_thread = threading.Thread(target=drain_loop, name="test-drain-loop", daemon=True)
    hotlane_thread = threading.Thread(target=hotlane_loop, name="test-hotlane-loop", daemon=True)
    drain_thread.start()
    hotlane_thread.start()

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with counters_lock:
            high_priority_queue_empty = not any(
                path for path in queue_dir.glob("*.jsonl") if not path.name.startswith("enrichment-")
            )
            if drain_writes >= 8 and hotlane_embeddings > 0 and high_priority_queue_empty:
                break
        time.sleep(0.01)
    with counters_lock:
        assert drain_writes >= 8
        assert hotlane_embeddings > 0
    assert not any(path for path in queue_dir.glob("*.jsonl") if not path.name.startswith("enrichment-"))

    readonly_store = VectorStore(db_path, readonly=True)
    monkeypatch.setattr(search_handler, "_get_vector_store", lambda: readonly_store)

    try:
        roundtrip_started = time.perf_counter()
        store_started = time.perf_counter()
        store_result = asyncio.run(
            _store(
                content=f"{marker} interactive store must be searchable while drain and hotlane write",
                memory_type="note",
                project="arbitration-test",
                tags=["d3", "concurrency"],
                importance=8,
            )
        )
        store_latency = time.perf_counter() - store_started
        assert store_latency < 0.5
        assert store_result[1]["queued"] is True

        search_payload = None
        while time.perf_counter() - roundtrip_started < 2.0:
            search_result = asyncio.run(
                _search(
                    query=marker,
                    project="arbitration-test",
                    source="all",
                    num_results=5,
                    include_operational=True,
                )
            )
            if not isinstance(search_result, tuple):
                error_text = "\n".join(getattr(part, "text", "") for part in getattr(search_result, "content", []))
                search_payload = {"total": 0, "results": [], "error": error_text}
                time.sleep(0.025)
                continue
            search_payload = search_result[1]
            if search_payload.get("total", 0) >= 1 and any(
                marker in (item.get("content", "") + item.get("snippet", ""))
                for item in search_payload.get("results", [])
            ):
                break
            time.sleep(0.025)

        roundtrip_latency = time.perf_counter() - roundtrip_started
        assert search_payload is not None
        assert search_payload.get("total", 0) >= 1, search_payload
        assert any(
            marker in (item.get("content", "") + item.get("snippet", "")) for item in search_payload.get("results", [])
        ), search_payload
        assert roundtrip_latency < 2.0
    finally:
        readonly_store.close()
        stop.set()
        hotlane.STOP = True
        drain_thread.join(timeout=2)
        hotlane_thread.join(timeout=2)


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
        conn.execute("INSERT INTO kg_entities (id, name) VALUES ('person-1', 'Person One')")
        conn.commit()

    queued_path = enqueue_store(
        content="replacement content",
        project="arbitration-test",
        source="../unsafe/source",
        supersedes="old-id",
        entity_id="person-1",
        queue_dir=queue_dir,
    )

    assert queued_path.parent == queue_dir
    assert ".." not in queued_path.name
    assert "/" not in queued_path.name
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", queued_path.name)
    assert drain_once(db_path=db_path, queue_dir=queue_dir, embed_fn=lambda text: [0.1] * 1024) == 1

    with _connect_apsw(db_path) as conn:
        replacement_id = conn.execute("SELECT id FROM chunks WHERE content = 'replacement content'").fetchone()[0]
        superseded_by = conn.execute("SELECT superseded_by FROM chunks WHERE id = 'old-id'").fetchone()[0]
        entity_link = conn.execute("SELECT chunk_id FROM kg_entity_chunks WHERE entity_id = 'person-1'").fetchone()[0]

    assert superseded_by == replacement_id
    assert entity_link == replacement_id


def test_drain_existing_promised_chunk_still_applies_supersedes(tmp_path, monkeypatch):
    """A queued replay after post-store supersede contention must finish lifecycle work."""
    from brainlayer.drain import drain_once

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    _create_minimal_db(db_path)
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO chunks (id, content, metadata, source_file)
            VALUES ('old-id', 'old content', '{}', 'seed')
            """
        )
        conn.execute(
            """
            INSERT INTO chunks (id, content, metadata, source_file)
            VALUES ('manual-new1234', 'replacement content', '{}', 'mcp')
            """
        )
        conn.commit()

    (queue_dir / "store-replay.jsonl").write_text(
        json.dumps(
            {
                "kind": "store_memory",
                "chunk_id": "manual-new1234",
                "content": "replacement content",
                "memory_type": "note",
                "project": "arbitration-test",
                "supersedes": "old-id",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=1) == 1

    with sqlite3.connect(db_path) as conn:
        superseded_by = conn.execute("SELECT superseded_by FROM chunks WHERE id = 'old-id'").fetchone()[0]

    assert superseded_by == "manual-new1234"


def test_drain_store_events_merge_duplicates_and_write_alias(tmp_path):
    from brainlayer.drain import drain_once
    from brainlayer.queue_io import enqueue_store

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    _create_minimal_db(db_path)

    enqueue_store(
        chunk_id="store-a",
        content="Duplicate store memory should merge through the single-writer drain",
        project="arbitration-test",
        tags=["first"],
        importance=3,
        queue_dir=queue_dir,
    )
    enqueue_store(
        chunk_id="store-b",
        content="Duplicate store memory should merge through the single-writer drain",
        project="arbitration-test",
        tags=["second"],
        importance=9,
        queue_dir=queue_dir,
    )

    assert drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=10) == 2

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, seen_count, importance, tags FROM chunks WHERE COALESCE(archived, 0) = 0"
        ).fetchall()
        alias = conn.execute("SELECT old_chunk_id, canonical_chunk_id FROM chunk_id_alias").fetchone()

    assert len(rows) == 1
    canonical_id, seen_count, importance, tags = rows[0]
    assert seen_count == 2
    assert importance == 9.0
    assert tags == '["first", "second"]'
    assert alias[0] != canonical_id
    assert alias[1] == canonical_id


def test_drain_duplicate_store_uses_canonical_for_supersedes_and_entity_link(tmp_path, monkeypatch):
    from brainlayer.drain import drain_once

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    _create_minimal_db(db_path)
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO chunks (id, content, metadata, source_file)
            VALUES ('old-id', 'old content', '{}', 'seed')
            """
        )
        conn.execute("INSERT INTO kg_entities (id, name) VALUES ('person-1', 'Person One')")
        conn.commit()

    duplicate_content = "Duplicate store memory should route references to the canonical row"
    (queue_dir / "store-ordered.jsonl").write_text(
        json.dumps(
            {
                "kind": "store_memory",
                "chunk_id": "store-a",
                "content": duplicate_content,
                "memory_type": "note",
                "project": "arbitration-test",
            }
        )
        + "\n"
        + json.dumps(
            {
                "kind": "store_memory",
                "chunk_id": "store-b",
                "content": duplicate_content,
                "memory_type": "note",
                "project": "arbitration-test",
                "supersedes": "old-id",
                "metadata": {"entity_id": "person-1"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=10) == 2

    with sqlite3.connect(db_path) as conn:
        superseded_by = conn.execute("SELECT superseded_by FROM chunks WHERE id = 'old-id'").fetchone()[0]
        entity_link = conn.execute("SELECT chunk_id FROM kg_entity_chunks WHERE entity_id = 'person-1'").fetchone()[0]

    assert superseded_by == "store-a"
    assert entity_link == "store-a"


def test_drain_watcher_same_id_reposts_increment_seen_count(tmp_path, monkeypatch):
    from brainlayer.drain import drain_once

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    _create_minimal_db(db_path)
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    event = {
        "kind": "watcher_chunk",
        "chunk_id": "watcher-same",
        "content": "Same watcher chunk should count repeat sightings",
        "created_at": "2026-05-16T09:00:00Z",
        "tags": ["watcher"],
    }
    (queue_dir / "watcher.jsonl").write_text(
        json.dumps(event) + "\n" + json.dumps({**event, "created_at": "2026-05-16T10:00:00Z"}) + "\n",
        encoding="utf-8",
    )

    assert drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=10) == 2

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT seen_count, last_seen_at FROM chunks WHERE id = 'watcher-same'").fetchone()
        audit = conn.execute("SELECT chunk_id_dropped, chunk_id_kept, mechanism FROM dedupe_audit").fetchone()

    assert row == (2, "2026-05-16T10:00:00Z")
    assert audit == ("watcher-same", "watcher-same", "sha256_same_id")


def test_drain_watcher_same_id_timestamp_change_merges_originals(tmp_path, monkeypatch):
    from brainlayer.drain import drain_once

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    _create_minimal_db(db_path)
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    first = {
        "kind": "watcher_chunk",
        "chunk_id": "watcher-timestamp",
        "content": "Deploy at 2026-05-16T10:00:00Z",
        "created_at": "2026-05-16T09:00:00Z",
    }
    second = {
        **first,
        "content": "Deploy at 2026-05-17T10:00:00Z",
        "created_at": "2026-05-16T10:00:00Z",
    }
    (queue_dir / "watcher.jsonl").write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )

    assert drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=10) == 2

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT content, seen_count FROM chunks WHERE id = 'watcher-timestamp'").fetchone()
        audit = conn.execute("SELECT chunk_id_dropped, chunk_id_kept, mechanism FROM dedupe_audit").fetchone()

    assert "2026-05-16T10:00:00Z" in row[0]
    assert "2026-05-17T10:00:00Z" in row[0]
    assert row[1] == 2
    assert audit == ("watcher-timestamp", "watcher-timestamp", "same_id_content_merge")


def test_drain_embeds_every_queued_store(tmp_path):
    from brainlayer.drain import drain_once
    from brainlayer.queue_io import enqueue_store

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    _create_minimal_db(db_path)

    for index in range(3):
        enqueue_store(
            content=f"queued semantic memory {index}",
            project="arbitration-test",
            source="mcp",
            queue_dir=queue_dir,
        )

    assert drain_once(db_path=db_path, queue_dir=queue_dir, embed_fn=lambda text: [0.1] * 1024) == 3

    deadline = time.monotonic() + 2
    vector_count = 0
    binary_count = 0
    while time.monotonic() < deadline:
        with _connect_apsw(db_path) as conn:
            vector_count = conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
            binary_count = conn.execute("SELECT COUNT(*) FROM chunk_vectors_binary").fetchone()[0]
        if vector_count == 3 and binary_count == 3:
            break
        time.sleep(0.05)

    assert vector_count == 3
    assert binary_count == 3


def test_drain_ignores_non_object_store_metadata(tmp_path):
    from brainlayer.drain import drain_once

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    _create_minimal_db(db_path)
    (queue_dir / "store.jsonl").write_text(
        json.dumps(
            {
                "kind": "store_memory",
                "chunk_id": "bad-meta",
                "content": "queued memory with bad metadata",
                "memory_type": "note",
                "metadata": "not-a-dict",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert drain_once(db_path=db_path, queue_dir=queue_dir, embed_fn=lambda text: [0.1] * 1024) == 1

    with _connect_apsw(db_path) as conn:
        row = conn.execute("SELECT metadata FROM chunks WHERE id = 'bad-meta'").fetchone()

    assert json.loads(row[0]) == {"memory_type": "note"}


def test_drain_loads_sqlite_vec_for_vec0_tables(tmp_path):
    from brainlayer.drain import drain_once
    from brainlayer.queue_io import enqueue_store

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    _create_vec_db(db_path)

    enqueue_store(
        content="queued memory requiring sqlite vec",
        project="arbitration-test",
        source="mcp",
        queue_dir=queue_dir,
    )

    assert drain_once(db_path=db_path, queue_dir=queue_dir, embed_fn=lambda text: [0.1] * 1024) == 1

    with _connect_apsw(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM chunk_vectors_binary").fetchone()[0] == 1


def test_poison_queue_file_does_not_rollback_good_file(tmp_path):
    from brainlayer.drain import drain_once
    from brainlayer.queue_io import enqueue_store

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    log_path = tmp_path / "drain.log"
    _create_minimal_db(db_path)

    good_path = enqueue_store(
        content="good memory survives poison",
        project="arbitration-test",
        source="mcp",
        queue_dir=queue_dir,
    )
    poison_path = queue_dir / "poison.jsonl"
    poison_path.write_bytes(b"\xff\xff")

    assert drain_once(db_path=db_path, queue_dir=queue_dir, log_path=log_path, embed_fn=lambda text: [0.1] * 1024) == 1

    with _connect_apsw(db_path) as conn:
        stored = conn.execute("SELECT COUNT(*) FROM chunks WHERE content = 'good memory survives poison'").fetchone()[0]

    assert stored == 1
    assert not good_path.exists()
    assert not poison_path.exists()
    assert list(queue_dir.glob("poison.jsonl.bad*"))
    assert "poison" in log_path.read_text(encoding="utf-8").lower()


def test_chunk_id_collision_is_logged_and_dropped(tmp_path):
    from brainlayer.drain import drain_once
    from brainlayer.queue_io import enqueue_store

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    log_path = tmp_path / "drain.log"
    _create_minimal_db(db_path)
    with _connect_apsw(db_path) as conn:
        conn.execute(
            """
            INSERT INTO chunks (id, content, metadata, source_file)
            VALUES ('coll', 'ORIGINAL', '{}', 'seed')
            """
        )

    enqueue_store(
        content="QUEUED REPLACEMENT",
        project="arbitration-test",
        source="mcp",
        queue_dir=queue_dir,
        chunk_id="coll",
    )

    assert drain_once(db_path=db_path, queue_dir=queue_dir, log_path=log_path, embed_fn=lambda text: [0.1] * 1024) == 1

    with _connect_apsw(db_path) as conn:
        content = conn.execute("SELECT content FROM chunks WHERE id = 'coll'").fetchone()[0]
        count = conn.execute("SELECT COUNT(*) FROM chunks WHERE id = 'coll'").fetchone()[0]

    log_text = log_path.read_text(encoding="utf-8").lower()
    assert content == "ORIGINAL"
    assert count == 1
    assert not list(queue_dir.glob("*.jsonl"))
    assert "collided" in log_text
    assert "collisions_dropped=1" in log_text


def test_drain_lock_does_not_depend_on_deletable_lock_file(tmp_path, monkeypatch):
    from brainlayer.drain import drain_once
    from brainlayer.queue_io import enqueue_store

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    _create_minimal_db(db_path)
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")

    enqueue_store(
        content="lock sentinel memory",
        project="arbitration-test",
        source="mcp",
        queue_dir=queue_dir,
    )

    stale_lock = queue_dir / ".drain.lock"
    stale_lock.write_text("stale", encoding="utf-8")
    stale_lock.unlink()

    assert drain_once(db_path=db_path, queue_dir=queue_dir) == 1
    assert not stale_lock.exists()

    with _connect_apsw(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM chunks WHERE content = 'lock sentinel memory'").fetchone()[0]
    assert count == 1


def test_flush_migrates_legacy_pending_stores_idempotently(tmp_path, monkeypatch):
    from brainlayer.cli import flush
    from brainlayer.paths import get_db_path

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    _create_minimal_db(db_path)
    pending_path = db_path.parent / "pending-stores.jsonl"
    pending_path.write_text('{"content":"legacy pending memory","memory_type":"note","project":"arbitration-test"}\n')

    monkeypatch.setenv("BRAINLAYER_DB", str(db_path))
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue_dir))
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    assert get_db_path() == db_path

    flush()
    pending_path.write_text('{"content":"legacy pending memory","memory_type":"note","project":"arbitration-test"}\n')
    flush()

    with _connect_apsw(db_path) as conn:
        rows = conn.execute("SELECT id FROM chunks WHERE content = 'legacy pending memory'").fetchall()

    assert len(rows) == 1


def test_flush_migrates_legacy_pending_stores_under_shared_lock(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from brainlayer.cli import flush

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    _create_minimal_db(db_path)
    pending_path = db_path.parent / "pending-stores.jsonl"
    pending_path.write_text('{"content":"locked pending memory","memory_type":"note","project":"arbitration-test"}\n')
    locked_paths = []

    @contextmanager
    def fake_pending_lock(path):
        locked_paths.append(path)
        yield

    monkeypatch.setenv("BRAINLAYER_DB", str(db_path))
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue_dir))
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    monkeypatch.setattr("brainlayer.mcp.store_handler._pending_store_file_lock", fake_pending_lock)

    flush()

    assert locked_paths == [pending_path]


def test_flush_preserves_malformed_legacy_pending_store_lines(tmp_path, monkeypatch):
    from brainlayer.cli import flush

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    _create_minimal_db(db_path)
    pending_path = db_path.parent / "pending-stores.jsonl"
    valid = '{"content":"valid pending memory","memory_type":"note","project":"arbitration-test"}'
    malformed = '{"content":'
    pending_path.write_text(f"{valid}\n{malformed}\n")

    monkeypatch.setenv("BRAINLAYER_DB", str(db_path))
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue_dir))
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")

    flush()

    remaining = pending_path.read_text().splitlines()
    assert remaining == [malformed]


def test_flush_pending_only_migrates_legacy_pending_stores_without_draining_queue(tmp_path, monkeypatch):
    from brainlayer.cli import flush

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    _create_minimal_db(db_path)
    pending_path = db_path.parent / "pending-stores.jsonl"
    pending_path.write_text('{"content":"pending only memory","memory_type":"note","project":"arbitration-test"}\n')

    def fail_drain_once(**kwargs):
        raise AssertionError("pending-only flush must not drain the unified queue")

    monkeypatch.setenv("BRAINLAYER_DB", str(db_path))
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue_dir))
    monkeypatch.setattr("brainlayer.drain.drain_once", fail_drain_once)

    flush(pending_only=True)

    assert not pending_path.exists()
    assert len(list(queue_dir.glob("pending-*.jsonl"))) == 1
