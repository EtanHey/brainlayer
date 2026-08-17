"""Repair (c): hotlane eligibility uses archived_at + lineage, not archived/status twins."""

from __future__ import annotations

import importlib
import logging
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_hotlane_module():
    importlib.invalidate_caches()
    sys.modules.pop("scripts.hotlane_brainbar_daemon", None)
    return importlib.import_module("scripts.hotlane_brainbar_daemon")


def _raise_if_called(message: str):
    def inner(**_kwargs):
        raise AssertionError(message)

    return inner


def test_hotlane_cycle_runs_enrichment_through_same_writer_store():
    hotlane = _load_hotlane_module()
    writer_store = object()
    calls = []

    result = hotlane.run_cycle(
        store=writer_store,
        embed_fn=lambda _text: [0.0],
        recent_limit=5,
        backlog_batch=0,
        enrich_limit=25,
        enrich_since_hours=8760,
        candidate_chunk_ids_fn=lambda store, *, limit: calls.append(("candidates", store, limit)) or [],
        hot_embed_fn=_raise_if_called("no hot candidates"),
        pending_embed_fn=_raise_if_called("backlog disabled"),
        enrich_fn=lambda store, **kwargs: (
            calls.append(("enrich", store, kwargs)) or SimpleNamespace(attempted=2, enriched=1, skipped=0, failed=0)
        ),
    )

    assert result.embedded == 0
    assert result.enrich_attempted == 2
    assert result.enriched == 1
    assert calls == [
        ("candidates", writer_store, 5),
        ("enrich", writer_store, {"limit": 25, "since_hours": 8760}),
    ]


def test_hotlane_cycle_can_disable_enrichment():
    hotlane = _load_hotlane_module()

    result = hotlane.run_cycle(
        store=object(),
        embed_fn=lambda _text: [0.0],
        recent_limit=5,
        backlog_batch=0,
        enrich_limit=0,
        enrich_since_hours=8760,
        candidate_chunk_ids_fn=lambda _store, *, limit: [],
        hot_embed_fn=lambda **_kwargs: False,
        pending_embed_fn=lambda **_kwargs: 0,
        enrich_fn=_raise_if_called("enrichment disabled"),
    )

    assert result.enrich_attempted == 0
    assert result.enriched == 0


def test_hotlane_default_backlog_batch_drains_pending_embeddings():
    hotlane = _load_hotlane_module()

    assert hotlane.DEFAULT_BACKLOG_BATCH == 4


def test_pending_chunk_query_defers_content_reads_until_after_bounded_id_scan():
    hotlane = _load_hotlane_module()
    executed = []

    class FakeCursor:
        def execute(self, sql, bindings):
            executed.append((sql, bindings))
            if len(executed) == 1:
                if "c.content" in sql:
                    return [("empty", ""), ("valid-1", "one"), ("valid-2", "two")]
                return [
                    ("empty", "2026-08-01T00:00:00Z", 1),
                    ("archived", "2026-08-01T00:00:01Z", 2),
                    ("valid-1", "2026-08-01T00:00:02Z", 3),
                    ("valid-2", "2026-08-01T00:00:03Z", 4),
                    ("valid-3", "2026-08-01T00:00:04Z", 5),
                ]
            return [
                ("valid-3", "three", None, None, None, 0, "active"),
                ("valid-2", "two", None, None, None, 0, "active"),
                ("empty", "", None, None, None, 0, "active"),
                ("archived", "skip me", "2026-08-02T00:00:00Z", None, None, 1, "archived"),
                ("valid-1", "one", None, None, None, 0, "active"),
            ]

    store = SimpleNamespace(conn=SimpleNamespace(cursor=lambda: FakeCursor()))

    assert hotlane._pending_chunk_rows(store, limit=2) == [
        hotlane.EmbedCandidate("valid-1", "one"),
        hotlane.EmbedCandidate("valid-2", "two"),
    ]
    assert "c.content" not in executed[0][0]
    assert "archived_at" not in executed[0][0]
    assert "superseded_by" not in executed[0][0]
    assert "status" not in executed[0][0]
    assert executed[0][1] == (2,)
    assert "WHERE id IN" in executed[1][0]
    assert executed[1][1] == ("empty", "archived", "valid-1", "valid-2", "valid-3")


@pytest.mark.parametrize("empty_created_at", [None, "2026-08-01T00:00:00Z"])
def test_pending_chunk_query_pages_past_a_full_window_of_empty_content(tmp_path, empty_created_at):
    hotlane = _load_hotlane_module()
    conn = sqlite3.connect(tmp_path / "empty-window.db")
    conn.executescript(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT,
            created_at TEXT,
            archived_at TEXT,
            superseded_by TEXT,
            aggregated_into TEXT,
            archived INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active'
        );
        CREATE INDEX idx_chunks_created_at ON chunks(created_at);
        CREATE TABLE chunk_vectors_rowids (id TEXT PRIMARY KEY);
        """
    )
    empty_rows = [(f"empty-{index:03d}", "", empty_created_at) for index in range(4)]
    conn.executemany("INSERT INTO chunks (id, content, created_at) VALUES (?, ?, ?)", empty_rows)
    conn.execute(
        "INSERT INTO chunks (id, content, created_at) VALUES "
        "('valid-after-empty-window', 'must make progress', '2026-08-02T00:00:00Z')"
    )
    try:
        assert hotlane._pending_chunk_rows(SimpleNamespace(conn=conn), limit=1) == [
            hotlane.EmbedCandidate("valid-after-empty-window", "must make progress")
        ]
    finally:
        conn.close()


def test_pending_chunk_query_bounds_pages_when_all_remaining_content_is_empty(tmp_path):
    hotlane = _load_hotlane_module()
    conn = sqlite3.connect(tmp_path / "all-empty.db")
    conn.executescript(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT,
            created_at TEXT,
            archived_at TEXT,
            superseded_by TEXT,
            aggregated_into TEXT,
            archived INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active'
        );
        CREATE INDEX idx_chunks_created_at ON chunks(created_at);
        CREATE TABLE chunk_vectors_rowids (id TEXT PRIMARY KEY);
        """
    )
    conn.executemany(
        "INSERT INTO chunks (id, content, created_at) VALUES (?, '', ?)",
        [(f"empty-{index:03d}", f"2026-08-01T00:{index:03d}:00Z") for index in range(100)],
    )
    statements = []
    conn.set_trace_callback(statements.append)
    try:
        assert hotlane._pending_chunk_rows(SimpleNamespace(conn=conn), limit=1) == []
        id_page_queries = [statement for statement in statements if "SELECT c.id, c.created_at, c.rowid" in statement]
        assert len(id_page_queries) <= 16
    finally:
        conn.close()


def test_pending_chunk_query_resumes_after_page_budget_on_next_cycle(tmp_path):
    hotlane = _load_hotlane_module()
    conn = sqlite3.connect(tmp_path / "resume-after-budget.db")
    conn.executescript(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT,
            created_at TEXT,
            archived_at TEXT,
            superseded_by TEXT,
            aggregated_into TEXT,
            archived INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active'
        );
        CREATE INDEX idx_chunks_created_at ON chunks(created_at);
        CREATE TABLE chunk_vectors_rowids (id TEXT PRIMARY KEY);
        """
    )
    conn.executemany(
        "INSERT INTO chunks (id, content, created_at) VALUES (?, '', ?)",
        [(f"empty-{index:03d}", f"2026-08-01T00:{index:03d}:00Z") for index in range(16)],
    )
    conn.execute(
        "INSERT INTO chunks (id, content, created_at) VALUES "
        "('valid-after-budget', 'must resume', '2026-08-02T00:00:00Z')"
    )
    conn.execute(
        "INSERT INTO chunks (id, content, created_at) VALUES "
        "('next-valid-after-budget', 'must keep position', '2026-08-03T00:00:00Z')"
    )
    state = hotlane.PendingCandidateScanState()
    store = SimpleNamespace(conn=conn)
    try:
        assert hotlane._pending_chunk_rows(store, limit=1, scan_state=state) == []
        assert hotlane._pending_chunk_rows(store, limit=1, scan_state=state) == [
            hotlane.EmbedCandidate("valid-after-budget", "must resume")
        ]
        assert state.active is True
        conn.execute("INSERT INTO chunk_vectors_rowids (id) VALUES ('valid-after-budget')")
        assert hotlane._pending_chunk_rows(store, limit=1, scan_state=state) == [
            hotlane.EmbedCandidate("next-valid-after-budget", "must keep position")
        ]
    finally:
        conn.close()


def test_hot_candidate_query_scans_a_bounded_recent_window_without_forcing_schema_index():
    hotlane = _load_hotlane_module()
    executed = []

    class FakeCursor:
        def execute(self, sql, bindings):
            executed.append((sql, bindings))
            return []

    store = SimpleNamespace(conn=SimpleNamespace(cursor=FakeCursor))

    assert hotlane._candidate_chunk_rows(store, limit=5) == []
    assert "INDEXED BY" not in executed[0][0]
    assert "ORDER BY c.created_at DESC" in executed[0][0]
    assert executed[0][1] == (hotlane.HOT_CANDIDATE_SCAN_LIMIT,)


def test_hot_candidate_query_uses_index_created_by_python_vectorstore(tmp_path):
    from brainlayer.vector_store import VectorStore

    hotlane = _load_hotlane_module()
    store = VectorStore(tmp_path / "brainlayer.db")
    try:
        cursor = store.conn.cursor()
        cursor.executemany(
            """
            INSERT INTO chunks (id, content, metadata, source_file, source, created_at)
            VALUES (?, ?, '{}', 'provider-session', 'claude', ?)
            """,
            [(f"other-{index}", "already handled", f"2026-08-07T00:{index:03d}:00Z") for index in range(300)],
        )
        cursor.execute(
            """
            INSERT INTO chunks (id, content, metadata, source_file, source, created_at)
            VALUES ('recent-brainbar', 'already embedded', '{}', 'brainbar-store', 'mcp', '9999-12-31T23:59:59Z')
            """
        )
        store._upsert_chunk_vector(cursor, "recent-brainbar", [0.0] * 1024)

        plan = [
            str(row[3])
            for row in cursor.execute(
                "EXPLAIN QUERY PLAN " + hotlane.HOT_CANDIDATE_SCAN_SQL,
                (hotlane.HOT_CANDIDATE_SCAN_LIMIT,),
            )
        ]
        assert any("USING INDEX idx_chunks_created" in detail for detail in plan)
        assert not any("USE TEMP B-TREE" in detail for detail in plan)
        assert hotlane._candidate_chunk_rows(store, limit=5) == []
    finally:
        store.close()


def test_hot_candidate_query_uses_native_created_at_index_without_python_index(tmp_path):
    hotlane = _load_hotlane_module()
    conn = sqlite3.connect(tmp_path / "native.db")
    conn.executescript(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT,
            source_file TEXT,
            source TEXT,
            created_at TEXT,
            archived_at TEXT,
            superseded_by TEXT,
            aggregated_into TEXT,
            archived INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active'
        );
        CREATE INDEX idx_chunks_created_at ON chunks(created_at);
        CREATE TABLE chunk_vectors_rowids (id TEXT PRIMARY KEY);
        INSERT INTO chunks (id, content, source_file, source, created_at)
        VALUES ('native-hot', 'pending native chunk', 'brainbar-store', 'mcp', '2026-08-08T00:00:00Z');
        """
    )
    try:
        assert hotlane._candidate_chunk_rows(SimpleNamespace(conn=conn), limit=5) == [
            hotlane.EmbedCandidate("native-hot", "pending native chunk")
        ]
    finally:
        conn.close()


def test_hot_candidate_scanner_pages_past_recent_embedded_window(tmp_path):
    hotlane = _load_hotlane_module()
    conn = sqlite3.connect(tmp_path / "paged.db")
    conn.executescript(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT,
            source_file TEXT,
            source TEXT,
            created_at TEXT,
            archived_at TEXT,
            superseded_by TEXT,
            aggregated_into TEXT,
            archived INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active'
        );
        CREATE TABLE chunk_vectors_rowids (id TEXT PRIMARY KEY);
        INSERT INTO chunks (id, content, source_file, source, created_at)
        VALUES ('older-hot', 'pending hot chunk', 'brainbar-store', 'mcp', '2026-08-01T00:00:00Z');
        """
    )
    newer_rows = [
        (f"newer-{index}", "already embedded", "brainbar-store", "mcp", f"2026-08-08T00:{index:03d}:00Z")
        for index in range(hotlane.HOT_CANDIDATE_SCAN_LIMIT)
    ]
    conn.executemany(
        "INSERT INTO chunks (id, content, source_file, source, created_at) VALUES (?, ?, ?, ?, ?)",
        newer_rows,
    )
    conn.executemany("INSERT INTO chunk_vectors_rowids (id) VALUES (?)", [(row[0],) for row in newer_rows])
    scanner = hotlane.HotCandidateScanner()
    store = SimpleNamespace(conn=conn)
    try:
        assert scanner(store, limit=5) == []
        assert scanner(store, limit=5) == [hotlane.EmbedCandidate("older-hot", "pending hot chunk")]
    finally:
        conn.close()


def test_hot_candidate_scanner_does_not_skip_unreturned_page_candidates(tmp_path):
    hotlane = _load_hotlane_module()
    conn = sqlite3.connect(tmp_path / "page-capacity.db")
    conn.executescript(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT,
            source_file TEXT,
            source TEXT,
            created_at TEXT,
            archived_at TEXT,
            superseded_by TEXT,
            aggregated_into TEXT,
            archived INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active'
        );
        CREATE TABLE chunk_vectors_rowids (id TEXT PRIMARY KEY);
        """
    )
    page_rows = [(f"page-{index}", "pending", "brainbar-store", "mcp", str(index)) for index in range(128)]
    head_rows = [(f"head-{index}", "embedded", "brainbar-store", "mcp", str(index)) for index in range(127)]
    head_rows.append(("head-hot", "pending head", "brainbar-store", "mcp", "latest"))
    conn.executemany(
        "INSERT INTO chunks (id, content, source_file, source, created_at) VALUES (?, ?, ?, ?, ?)",
        page_rows + head_rows,
    )
    conn.executemany("INSERT INTO chunk_vectors_rowids (id) VALUES (?)", [(row[0],) for row in head_rows[:-1]])
    scanner = hotlane.HotCandidateScanner()
    store = SimpleNamespace(conn=conn)
    try:
        first = scanner(store, limit=2)
        assert first == [
            hotlane.EmbedCandidate("head-hot", "pending head"),
            hotlane.EmbedCandidate("page-127", "pending"),
        ]
        conn.execute("INSERT INTO chunk_vectors_rowids (id) VALUES ('head-hot')")
        assert scanner(store, limit=2)[0] == hotlane.EmbedCandidate("page-127", "pending")
        conn.execute("INSERT INTO chunk_vectors_rowids (id) VALUES ('page-127')")
        assert scanner(store, limit=2) == [
            hotlane.EmbedCandidate("page-126", "pending"),
            hotlane.EmbedCandidate("page-125", "pending"),
        ]
    finally:
        conn.close()


def test_hot_candidate_scanner_catches_rows_pushed_below_moving_head(tmp_path):
    hotlane = _load_hotlane_module()
    conn = sqlite3.connect(tmp_path / "moving-head.db")
    conn.executescript(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT,
            source_file TEXT,
            source TEXT,
            created_at TEXT,
            archived_at TEXT,
            superseded_by TEXT,
            aggregated_into TEXT,
            archived INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active'
        );
        CREATE TABLE chunk_vectors_rowids (id TEXT PRIMARY KEY);
        """
    )
    initial_rows = [(f"initial-{index}", "embedded", "brainbar-store", "mcp", str(index)) for index in range(128)]
    conn.executemany(
        "INSERT INTO chunks (id, content, source_file, source, created_at) VALUES (?, ?, ?, ?, ?)", initial_rows
    )
    conn.executemany("INSERT INTO chunk_vectors_rowids (id) VALUES (?)", [(row[0],) for row in initial_rows])
    scanner = hotlane.HotCandidateScanner()
    store = SimpleNamespace(conn=conn)
    try:
        assert scanner(store, limit=5) == []
        conn.execute(
            "INSERT INTO chunks (id, content, source_file, source, created_at) VALUES "
            "('gap-hot', 'must catch up', 'brainbar-store', 'mcp', 'gap')"
        )
        newer_rows = [(f"new-{index}", "embedded", "brainbar-store", "mcp", str(index)) for index in range(128)]
        conn.executemany(
            "INSERT INTO chunks (id, content, source_file, source, created_at) VALUES (?, ?, ?, ?, ?)", newer_rows
        )
        conn.executemany("INSERT INTO chunk_vectors_rowids (id) VALUES (?)", [(row[0],) for row in newer_rows])
        assert scanner(store, limit=5) == [hotlane.EmbedCandidate("gap-hot", "must catch up")]
    finally:
        conn.close()


def test_hot_candidate_scanner_deduplicates_head_and_forward_catchup(tmp_path):
    hotlane = _load_hotlane_module()
    conn = sqlite3.connect(tmp_path / "head-forward.db")
    conn.executescript(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT,
            source_file TEXT,
            source TEXT,
            created_at TEXT,
            archived_at TEXT,
            superseded_by TEXT,
            aggregated_into TEXT,
            archived INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active'
        );
        CREATE TABLE chunk_vectors_rowids (id TEXT PRIMARY KEY);
        """
    )
    baseline_rows = [(f"baseline-{index}", "embedded", "brainbar-store", "mcp", str(index)) for index in range(1000)]
    conn.executemany(
        "INSERT INTO chunks (id, content, source_file, source, created_at) VALUES (?, ?, ?, ?, ?)", baseline_rows
    )
    conn.executemany("INSERT INTO chunk_vectors_rowids (id) VALUES (?)", [(row[0],) for row in baseline_rows])
    scanner = hotlane.HotCandidateScanner()
    store = SimpleNamespace(conn=conn)
    try:
        assert scanner(store, limit=5) == []
        conn.execute(
            "INSERT INTO chunks (id, content, source_file, source, created_at) VALUES "
            "('new-hot', 'only once', 'brainbar-store', 'mcp', 'after')"
        )
        assert scanner(store, limit=5) == [hotlane.EmbedCandidate("new-hot", "only once")]
        newer_rows = [(f"later-{index}", "embedded", "brainbar-store", "mcp", str(index)) for index in range(128)]
        conn.executemany(
            "INSERT INTO chunks (id, content, source_file, source, created_at) VALUES (?, ?, ?, ?, ?)", newer_rows
        )
        conn.executemany("INSERT INTO chunk_vectors_rowids (id) VALUES (?)", [(row[0],) for row in newer_rows])
        assert scanner(store, limit=5) == [hotlane.EmbedCandidate("new-hot", "only once")]
        conn.execute(
            "INSERT INTO chunks (id, content, source_file, source, created_at) VALUES "
            "('later-hot', 'must progress', 'brainbar-store', 'mcp', 'later')"
        )
        final_rows = [(f"final-{index}", "embedded", "brainbar-store", "mcp", str(index)) for index in range(128)]
        conn.executemany(
            "INSERT INTO chunks (id, content, source_file, source, created_at) VALUES (?, ?, ?, ?, ?)", final_rows
        )
        conn.executemany("INSERT INTO chunk_vectors_rowids (id) VALUES (?)", [(row[0],) for row in final_rows])
        assert scanner(store, limit=5) == [
            hotlane.EmbedCandidate("new-hot", "only once"),
            hotlane.EmbedCandidate("later-hot", "must progress"),
        ]
    finally:
        conn.close()


def test_hot_candidate_scanner_retries_startup_head_candidate_after_displacement(tmp_path):
    hotlane = _load_hotlane_module()
    conn = sqlite3.connect(tmp_path / "startup-head.db")
    conn.executescript(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT,
            source_file TEXT,
            source TEXT,
            created_at TEXT,
            archived_at TEXT,
            superseded_by TEXT,
            aggregated_into TEXT,
            archived INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active'
        );
        CREATE TABLE chunk_vectors_rowids (id TEXT PRIMARY KEY);
        """
    )
    baseline_rows = [(f"base-{index}", "embedded", "brainbar-store", "mcp", str(index)) for index in range(999)]
    conn.executemany(
        "INSERT INTO chunks (id, content, source_file, source, created_at) VALUES (?, ?, ?, ?, ?)", baseline_rows
    )
    conn.executemany("INSERT INTO chunk_vectors_rowids (id) VALUES (?)", [(row[0],) for row in baseline_rows])
    conn.execute(
        "INSERT INTO chunks (id, content, source_file, source, created_at) VALUES "
        "('startup-hot', 'retry startup', 'brainbar-store', 'mcp', 'latest')"
    )
    scanner = hotlane.HotCandidateScanner()
    store = SimpleNamespace(conn=conn)
    try:
        assert scanner(store, limit=5) == [hotlane.EmbedCandidate("startup-hot", "retry startup")]
        newer_rows = [(f"after-{index}", "embedded", "brainbar-store", "mcp", str(index)) for index in range(128)]
        conn.executemany(
            "INSERT INTO chunks (id, content, source_file, source, created_at) VALUES (?, ?, ?, ?, ?)", newer_rows
        )
        conn.executemany("INSERT INTO chunk_vectors_rowids (id) VALUES (?)", [(row[0],) for row in newer_rows])
        assert scanner(store, limit=5) == [hotlane.EmbedCandidate("startup-hot", "retry startup")]
    finally:
        conn.close()


def test_hot_candidate_scanner_does_not_advance_past_retry_capacity(tmp_path):
    hotlane = _load_hotlane_module()
    conn = sqlite3.connect(tmp_path / "retry-capacity.db")
    conn.executescript(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT,
            source_file TEXT,
            source TEXT,
            created_at TEXT,
            archived_at TEXT,
            superseded_by TEXT,
            aggregated_into TEXT,
            archived INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active'
        );
        CREATE TABLE chunk_vectors_rowids (id TEXT PRIMARY KEY);
        """
    )
    retry_rows = [
        (f"retry-{index}", "retry", "brainbar-store", "mcp", str(index))
        for index in range(hotlane.MAX_HOT_CANDIDATE_RETRIES - 2)
    ]
    new_rows = [(f"new-{index}", "new", "brainbar-store", "mcp", str(index)) for index in range(3)]
    conn.executemany(
        "INSERT INTO chunks (id, content, source_file, source, created_at) VALUES (?, ?, ?, ?, ?)",
        retry_rows + new_rows,
    )
    scanner = hotlane.HotCandidateScanner()
    for row in retry_rows:
        scanner._retries[row[0]] = hotlane.EmbedCandidate(row[0], row[1])
    store = SimpleNamespace(conn=conn)
    try:
        first = scanner(store, limit=5)
        assert len(first) == 4
        assert [candidate.chunk_id for candidate in first[-2:]] == ["new-2", "new-1"]
        conn.executemany(
            "INSERT INTO chunk_vectors_rowids (id) VALUES (?)", [(candidate.chunk_id,) for candidate in first]
        )
        assert hotlane.EmbedCandidate("new-0", "new") in scanner(store, limit=5)
    finally:
        conn.close()


def test_hotlane_run_threads_model_batch_embedder_to_backlog_cycle():
    hotlane = _load_hotlane_module()
    received_batch_fns = []

    class FakeStore:
        def close(self):
            pass

    class FakeModel:
        def embed_query(self, _text):
            return [0.0]

        def embed_texts(self, texts):
            return [[0.0] * 1024 for _text in texts]

    def fake_cycle(**kwargs):
        received_batch_fns.append(kwargs.get("embed_batch_fn"))
        return hotlane.CycleResult()

    hotlane.run(
        db_path=Path("/tmp/unused.db"),
        interval=0.25,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=hotlane.DEFAULT_BACKLOG_BATCH,
        enrich_interval=10.0,
        enrich_limit=0,
        enrich_since_hours=8760,
        vector_store_cls=lambda _path: FakeStore(),
        model_factory=FakeModel,
        cycle_fn=fake_cycle,
        queue_depth_fn=lambda _queue_dir: 0,
        high_priority_queue_depth_fn=lambda _queue_dir: 0,
        time_fn=iter([0.0, 100.0]).__next__,
        sleep_fn=lambda _seconds: None,
        max_cycles=1,
    )

    assert len(received_batch_fns) == 1
    assert received_batch_fns[0].__func__ is FakeModel.embed_texts


def test_hotlane_run_uses_document_embeddings_for_stored_chunks():
    hotlane = _load_hotlane_module()
    received_embed_fns = []

    class FakeStore:
        def close(self):
            pass

    class FakeModel:
        def embed_query(self, _text):
            return [1.0]

        def embed_texts(self, texts):
            return [[2.0] for _text in texts]

    def fake_cycle(**kwargs):
        received_embed_fns.append(kwargs["embed_fn"])
        return hotlane.CycleResult()

    hotlane.run(
        db_path=Path("/tmp/unused.db"),
        interval=0.25,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=hotlane.DEFAULT_BACKLOG_BATCH,
        enrich_interval=10.0,
        enrich_limit=0,
        enrich_since_hours=8760,
        vector_store_cls=lambda _path: FakeStore(),
        model_factory=FakeModel,
        cycle_fn=fake_cycle,
        queue_depth_fn=lambda _queue_dir: 0,
        high_priority_queue_depth_fn=lambda _queue_dir: 0,
        time_fn=iter([0.0, 100.0]).__next__,
        sleep_fn=lambda _seconds: None,
        max_cycles=1,
    )

    assert len(received_embed_fns) == 1
    assert received_embed_fns[0]("stored chunk text") == [2.0]


def test_hotlane_run_schedules_backlog_on_first_cycle():
    hotlane = _load_hotlane_module()
    scheduled_backlog_batches = []

    class FakeStore:
        def close(self):
            pass

    def fake_cycle(**kwargs):
        scheduled_backlog_batches.append(kwargs["backlog_batch"])
        return hotlane.CycleResult()

    hotlane.run(
        db_path=Path("/tmp/unused.db"),
        interval=0.25,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=hotlane.DEFAULT_BACKLOG_BATCH,
        enrich_interval=10.0,
        enrich_limit=0,
        enrich_since_hours=8760,
        vector_store_cls=lambda _path: FakeStore(),
        model_factory=lambda: SimpleNamespace(embed_query=lambda _text: [0.0]),
        cycle_fn=fake_cycle,
        queue_depth_fn=lambda _queue_dir: 0,
        high_priority_queue_depth_fn=lambda _queue_dir: 0,
        time_fn=iter([100.0, 100.0]).__next__,
        sleep_fn=lambda _seconds: None,
        max_cycles=1,
    )

    assert scheduled_backlog_batches == [hotlane.DEFAULT_BACKLOG_BATCH]


def test_open_store_readonly_accepts_one_argument_factory():
    hotlane = _load_hotlane_module()
    opened_paths = []
    store = object()

    def one_argument_factory(path):
        opened_paths.append(path)
        return store

    assert hotlane._open_store(one_argument_factory, Path("/tmp/brainlayer.db"), readonly=True) is store
    assert opened_paths == [Path("/tmp/brainlayer.db")]


def test_open_store_readonly_does_not_swallow_constructor_type_errors():
    hotlane = _load_hotlane_module()

    class BrokenStore:
        def __init__(self, _path, *, readonly=False):
            raise TypeError("constructor bug unrelated to readonly")

    with pytest.raises(TypeError, match="constructor bug unrelated to readonly"):
        hotlane._open_store(BrokenStore, Path("/tmp/brainlayer.db"), readonly=True)


def test_split_cycle_bootstraps_missing_db_before_readonly_open(tmp_path):
    hotlane = _load_hotlane_module()
    db_path = tmp_path / "missing-brainlayer.db"
    opened_modes = []

    class FakeStore:
        def __init__(self, path, *, readonly=False):
            opened_modes.append(readonly)
            if readonly and not path.exists():
                raise RuntimeError("readonly sqlite open cannot create the database")
            if not readonly:
                path.touch()

        def close(self):
            pass

    result = hotlane._run_split_cycle(
        db_path=db_path,
        vector_store_cls=FakeStore,
        embed_fn=lambda _text: [0.0],
        recent_limit=1,
        backlog_batch=0,
        enrich_limit=0,
        enrich_since_hours=8760,
        candidate_rows_fn=lambda _store, *, limit: [],
        pending_rows_fn=lambda _store, *, limit: [],
    )

    assert result.embedded == 0
    assert opened_modes == [False, True]


def test_write_embedded_vectors_skips_when_content_changed_after_snapshot():
    hotlane = _load_hotlane_module()
    events = []

    class FakeCursor:
        def execute(self, sql, params=()):
            events.append(("sql", sql.strip().splitlines()[0], params))
            if sql.strip().startswith("SELECT 1"):
                assert params == ("chunk-1", "old content")
                return SimpleNamespace(fetchone=lambda: None)
            return []

    class FakeConn:
        def cursor(self):
            return FakeCursor()

    class FakeStore:
        db_path = Path("/tmp/brainlayer.db")
        conn = FakeConn()

        def _upsert_chunk_vector(self, _cursor, chunk_id, embedding):
            events.append(("upsert", chunk_id, embedding))

    count = hotlane._write_embedded_vectors(
        FakeStore(),
        [hotlane.EmbeddedVector("chunk-1", "old content", [0.5])],
    )

    assert count == 0
    assert ("upsert", "chunk-1", [0.5]) not in events


def test_write_embedded_vectors_commits_each_vector_separately(monkeypatch):
    hotlane = _load_hotlane_module()
    statements = []
    sleeps = []
    monkeypatch.setattr(hotlane, "_sleep", lambda seconds: sleeps.append(seconds))

    class FakeCursor:
        def execute(self, sql, params=()):
            statement = " ".join(sql.split())
            statements.append((statement, params))
            if statement.startswith("SELECT 1"):
                return SimpleNamespace(fetchone=lambda: (1,))
            return []

    class FakeConn:
        def cursor(self):
            return FakeCursor()

    class FakeStore:
        db_path = Path("/tmp/brainlayer.db")
        conn = FakeConn()

        def _upsert_chunk_vector(self, _cursor, _chunk_id, _embedding):
            pass

    count = hotlane._write_embedded_vectors(
        FakeStore(),
        [
            hotlane.EmbeddedVector("chunk-1", "content one", [0.1]),
            hotlane.EmbeddedVector("chunk-2", "content two", [0.2]),
        ],
    )

    assert count == 2
    assert sum(statement == "BEGIN IMMEDIATE" for statement, _params in statements) == 2
    assert sum(statement == "COMMIT" for statement, _params in statements) == 2
    assert sleeps == [hotlane.VECTOR_WRITE_YIELD_SECONDS]


def test_write_embedded_vectors_clears_search_cache_after_partial_commit(monkeypatch):
    hotlane = _load_hotlane_module()
    cleared = []
    upserted = []

    class FakeCursor:
        def execute(self, sql, params=()):
            statement = " ".join(sql.split())
            if statement.startswith("SELECT 1"):
                return SimpleNamespace(fetchone=lambda: (1,))
            return []

    class FakeConn:
        def cursor(self):
            return FakeCursor()

    class FakeStore:
        db_path = Path("/tmp/brainlayer.db")
        conn = FakeConn()

        def _upsert_chunk_vector(self, _cursor, chunk_id, _embedding):
            upserted.append(chunk_id)
            if chunk_id == "chunk-2":
                raise RuntimeError("second vector failed")

    monkeypatch.setattr(
        "brainlayer.search_repo.clear_hybrid_search_cache",
        lambda db_path: cleared.append(db_path),
    )

    with pytest.raises(RuntimeError, match="second vector failed"):
        hotlane._write_embedded_vectors(
            FakeStore(),
            [
                hotlane.EmbeddedVector("chunk-1", "content one", [0.1]),
                hotlane.EmbeddedVector("chunk-2", "content two", [0.2]),
            ],
        )

    assert upserted == ["chunk-1", "chunk-2"]
    assert cleared == [Path("/tmp/brainlayer.db")]


def test_write_embedded_vectors_path_upserts_without_vectorstore_writer_pidfile(tmp_path, monkeypatch):
    hotlane = _load_hotlane_module()
    from brainlayer.vector_store import VectorStore

    db_path = tmp_path / "brainlayer.db"
    pidfile_dir = tmp_path / "writer-pids"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))

    store = VectorStore(db_path)
    cursor = store.conn.cursor()
    cursor.execute(
        """
        INSERT INTO chunks (id, content, metadata, source_file, source)
        VALUES ('chunk-1', 'stable content', '{}', 'brainbar-store', 'mcp')
        """
    )
    store.close()
    assert list(pidfile_dir.glob("brainlayer-writer-*")) == []

    count = hotlane._write_embedded_vectors(
        db_path,
        [hotlane.EmbeddedVector("chunk-1", "stable content", [0.25] * 1024)],
    )

    assert count == 1
    assert list(pidfile_dir.glob("brainlayer-writer-*")) == []

    readonly_store = VectorStore(db_path, readonly=True)
    try:
        cursor = readonly_store.conn.cursor()
        assert cursor.execute("SELECT COUNT(*) FROM chunk_vectors WHERE chunk_id = 'chunk-1'").fetchone()[0] == 1
        assert cursor.execute("SELECT COUNT(*) FROM chunk_vectors_binary WHERE chunk_id = 'chunk-1'").fetchone()[0] == 1
    finally:
        readonly_store.close()


def test_split_cycle_embeds_all_hot_candidates_before_writer_revalidation(tmp_path):
    hotlane = _load_hotlane_module()
    db_path = tmp_path / "brainlayer.db"
    db_path.touch()
    vectors_seen = []

    class FakeStore:
        def close(self):
            pass

    def write_vectors_fn(_store, vectors):
        vectors_seen.extend(vectors)
        return sum(1 for vector in vectors if vector.chunk_id == "hot-fresh")

    result = hotlane._run_split_cycle(
        db_path=db_path,
        vector_store_cls=lambda _path, readonly=False: FakeStore(),
        embed_fn=lambda text: [float(len(text))],
        recent_limit=2,
        backlog_batch=0,
        enrich_limit=0,
        enrich_since_hours=8760,
        candidate_rows_fn=lambda _store, *, limit: [
            hotlane.EmbedCandidate("hot-stale", "stale content"),
            hotlane.EmbedCandidate("hot-fresh", "fresh content"),
        ][:limit],
        pending_rows_fn=lambda _store, *, limit: [],
        write_vectors_fn=write_vectors_fn,
    )

    assert [vector.chunk_id for vector in vectors_seen] == ["hot-stale", "hot-fresh"]
    assert result.embedded == 1


def test_hotlane_run_advances_enrich_timer_before_failed_cycle():
    hotlane = _load_hotlane_module()
    scheduled_enrich_limits = []

    class FakeStore:
        def close(self):
            pass

    def fake_cycle(**kwargs):
        scheduled_enrich_limits.append(kwargs["enrich_limit"])
        if len(scheduled_enrich_limits) == 1:
            raise RuntimeError("gemini transient failure")
        return hotlane.CycleResult()

    hotlane.run(
        db_path=Path("/tmp/unused.db"),
        interval=0.25,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=0,
        enrich_interval=10.0,
        enrich_limit=25,
        enrich_since_hours=8760,
        vector_store_cls=lambda _path: FakeStore(),
        model_factory=lambda: SimpleNamespace(embed_query=lambda _text: [0.0]),
        cycle_fn=fake_cycle,
        queue_depth_fn=lambda _queue_dir: 0,
        high_priority_queue_depth_fn=lambda _queue_dir: 0,
        time_fn=iter([0.0, 100.0, 101.0]).__next__,
        sleep_fn=lambda _seconds: None,
        max_cycles=2,
    )

    assert scheduled_enrich_limits == [25, 0]


def test_hotlane_run_requests_cpu_embedding_device_by_default():
    hotlane = _load_hotlane_module()
    requested_devices = []

    def device_aware_model_factory(*, device=None):
        requested_devices.append(device)
        return SimpleNamespace(embed_query=lambda _text: [0.0])

    hotlane.run(
        db_path=Path("/tmp/unused.db"),
        interval=0.25,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=0,
        enrich_interval=10.0,
        enrich_limit=0,
        enrich_since_hours=8760,
        model_factory=device_aware_model_factory,
        max_cycles=0,
    )

    assert requested_devices == ["cpu"]


def test_hotlane_run_preserves_positional_only_model_factory():
    hotlane = _load_hotlane_module()

    def positional_only_model_factory(device="cpu", /):
        return SimpleNamespace(embed_query=lambda _text: [0.0])

    hotlane.run(
        db_path=Path("/tmp/unused.db"),
        interval=0.25,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=0,
        enrich_interval=10.0,
        enrich_limit=0,
        enrich_since_hours=8760,
        model_factory=positional_only_model_factory,
        max_cycles=0,
    )


def test_hotlane_run_disables_enrichment_after_daily_cap():
    hotlane = _load_hotlane_module()
    scheduled_enrich_limits = []

    class FakeStore:
        def close(self):
            pass

    def fake_cycle(**kwargs):
        scheduled_enrich_limits.append(kwargs["enrich_limit"])
        if len(scheduled_enrich_limits) == 1:
            return hotlane.CycleResult(enrich_attempted=1, enrich_failed=1, enrich_daily_cap_reached=True)
        return hotlane.CycleResult()

    hotlane.run(
        db_path=Path("/tmp/unused.db"),
        interval=0.25,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=0,
        enrich_interval=10.0,
        enrich_limit=25,
        enrich_since_hours=8760,
        vector_store_cls=lambda _path: FakeStore(),
        model_factory=lambda: SimpleNamespace(embed_query=lambda _text: [0.0]),
        cycle_fn=fake_cycle,
        queue_depth_fn=lambda _queue_dir: 0,
        high_priority_queue_depth_fn=lambda _queue_dir: 0,
        time_fn=iter([0.0, 100.0, 111.0]).__next__,
        sleep_fn=lambda _seconds: None,
        max_cycles=2,
    )

    assert scheduled_enrich_limits == [25, 0]


def test_hotlane_run_opens_and_closes_writer_store_each_cycle(tmp_path):
    hotlane = _load_hotlane_module()
    events = []

    class FakeStore:
        def __init__(self, path):
            events.append(("open", path))

        def close(self):
            events.append(("close", None))

    def fake_cycle(**_kwargs):
        return hotlane.CycleResult()

    hotlane.run(
        db_path=tmp_path / "brainlayer.db",
        interval=0.25,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=0,
        enrich_interval=10.0,
        enrich_limit=0,
        enrich_since_hours=8760,
        vector_store_cls=FakeStore,
        model_factory=lambda: SimpleNamespace(embed_query=lambda _text: [0.0]),
        cycle_fn=fake_cycle,
        queue_depth_fn=lambda _queue_dir: 0,
        high_priority_queue_depth_fn=lambda _queue_dir: 0,
        time_fn=iter([0.0, 100.0, 101.0]).__next__,
        sleep_fn=lambda _seconds: None,
        max_cycles=2,
    )

    assert [event[0] for event in events] == ["open", "close", "open", "close"]


def test_hotlane_run_continues_embedding_during_enrichment_only_queue_backlog(tmp_path):
    hotlane = _load_hotlane_module()
    opened = []
    cycle_calls = []
    sleeps = []
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    (queue_dir / "enrichment-first.jsonl").write_text("{}\n")
    (queue_dir / "queue-enrichment-legacy.jsonl").write_text("{}\n")

    class FakeStore:
        def __init__(self, path):
            opened.append(path)

        def close(self):
            pass

    hotlane.run(
        db_path=tmp_path / "brainlayer.db",
        interval=0.25,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=hotlane.DEFAULT_BACKLOG_BATCH,
        enrich_interval=10.0,
        enrich_limit=hotlane.DEFAULT_HOTLANE_ENRICH_LIMIT,
        enrich_since_hours=8760,
        vector_store_cls=FakeStore,
        model_factory=lambda: SimpleNamespace(embed_query=lambda _text: [0.0]),
        cycle_fn=lambda **kwargs: cycle_calls.append(kwargs) or hotlane.CycleResult(),
        time_fn=iter([0.0, 100.0]).__next__,
        sleep_fn=sleeps.append,
        max_cycles=1,
        queue_dir=queue_dir,
    )

    assert opened == [tmp_path / "brainlayer.db"]
    assert len(cycle_calls) == 1
    assert cycle_calls[0]["backlog_batch"] == hotlane.DEFAULT_BACKLOG_BATCH
    assert cycle_calls[0]["enrich_limit"] == 0
    assert sleeps == [0.25]


def test_hotlane_run_logs_high_priority_backpressure_once_per_blocked_state(tmp_path, caplog):
    hotlane = _load_hotlane_module()
    caplog.set_level(logging.INFO, logger=hotlane.LOGGER.name)
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    high_priority_event = queue_dir / "store-pending.jsonl"
    high_priority_event.write_text("{}\n")
    (queue_dir / "enrichment-pending.jsonl").write_text("{}\n")
    cycle_calls = []
    sleep_count = 0

    class FakeStore:
        def close(self):
            pass

    def update_queue_after_cycle(_seconds):
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count == 2:
            high_priority_event.unlink()
        elif sleep_count == 3:
            high_priority_event.write_text("{}\n")

    hotlane.run(
        db_path=tmp_path / "brainlayer.db",
        interval=0.25,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=hotlane.DEFAULT_BACKLOG_BATCH,
        enrich_interval=10.0,
        enrich_limit=hotlane.DEFAULT_HOTLANE_ENRICH_LIMIT,
        enrich_since_hours=8760,
        vector_store_cls=lambda _path: FakeStore(),
        model_factory=lambda: SimpleNamespace(embed_query=lambda _text: [0.0]),
        cycle_fn=lambda **kwargs: cycle_calls.append(kwargs) or hotlane.CycleResult(),
        time_fn=iter([0.0, 100.0, 101.0, 102.0, 103.0, 104.0]).__next__,
        sleep_fn=update_queue_after_cycle,
        max_cycles=5,
        queue_dir=queue_dir,
    )

    yield_logs = [
        record
        for record in caplog.records
        if record.getMessage() == "durable high-priority queue has backlog; suppressing hot embedding and enrichment"
    ]
    assert len(yield_logs) == 2
    assert len(cycle_calls) == 2
    assert cycle_calls[0]["recent_limit"] == 0
    assert cycle_calls[0]["backlog_batch"] == hotlane.DEFAULT_BACKLOG_BATCH
    assert cycle_calls[0]["enrich_limit"] == 0
    assert cycle_calls[1]["recent_limit"] == 5
    assert cycle_calls[1]["backlog_batch"] == 0
    assert cycle_calls[1]["enrich_limit"] == 0


def test_hotlane_run_yields_all_writer_work_when_backlog_embedding_is_disabled(tmp_path):
    hotlane = _load_hotlane_module()
    opened = []
    cycle_calls = []
    sleeps = []

    class FakeStore:
        def __init__(self, path):
            opened.append(path)

        def close(self):
            pass

    hotlane.run(
        db_path=tmp_path / "brainlayer.db",
        interval=0.25,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=0,
        enrich_interval=10.0,
        enrich_limit=hotlane.DEFAULT_HOTLANE_ENRICH_LIMIT,
        enrich_since_hours=8760,
        vector_store_cls=FakeStore,
        model_factory=lambda: SimpleNamespace(embed_query=lambda _text: [0.0]),
        cycle_fn=lambda **kwargs: cycle_calls.append(kwargs) or hotlane.CycleResult(),
        time_fn=iter([0.0, 100.0]).__next__,
        sleep_fn=sleeps.append,
        max_cycles=1,
        queue_depth_fn=lambda _queue_dir: 3,
        high_priority_queue_depth_fn=lambda _queue_dir: 1,
    )

    assert opened == []
    assert cycle_calls == []
    assert sleeps == [0.25]


def test_hotlane_run_reserves_due_backlog_slice_during_high_priority_queue_backlog(tmp_path):
    hotlane = _load_hotlane_module()
    cycle_calls = []
    sleeps = []

    class FakeStore:
        def close(self):
            pass

    hotlane.run(
        db_path=tmp_path / "brainlayer.db",
        interval=0.25,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=hotlane.DEFAULT_BACKLOG_BATCH,
        enrich_interval=10.0,
        enrich_limit=hotlane.DEFAULT_HOTLANE_ENRICH_LIMIT,
        enrich_since_hours=8760,
        vector_store_cls=lambda _path: FakeStore(),
        model_factory=lambda: SimpleNamespace(embed_query=lambda _text: [0.0]),
        cycle_fn=lambda **kwargs: cycle_calls.append(kwargs) or hotlane.CycleResult(),
        time_fn=iter([100.0, 100.0, 101.0]).__next__,
        sleep_fn=sleeps.append,
        max_cycles=2,
        queue_depth_fn=lambda _queue_dir: 3,
        high_priority_queue_depth_fn=lambda _queue_dir: 1,
    )

    assert len(cycle_calls) == 1
    assert cycle_calls[0]["recent_limit"] == 0
    assert cycle_calls[0]["backlog_batch"] == hotlane.DEFAULT_BACKLOG_BATCH
    assert cycle_calls[0]["enrich_limit"] == 0
    assert sleeps == [0.25, 0.25]


def test_hotlane_run_repeats_reserved_backlog_slice_during_continuous_pressure(tmp_path):
    hotlane = _load_hotlane_module()
    cycle_calls = []
    sleeps = []

    class FakeStore:
        def close(self):
            pass

    hotlane.run(
        db_path=tmp_path / "brainlayer.db",
        interval=0.25,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=hotlane.DEFAULT_BACKLOG_BATCH,
        enrich_interval=10.0,
        enrich_limit=hotlane.DEFAULT_HOTLANE_ENRICH_LIMIT,
        enrich_since_hours=8760,
        vector_store_cls=lambda _path: FakeStore(),
        model_factory=lambda: SimpleNamespace(embed_query=lambda _text: [0.0]),
        cycle_fn=lambda **kwargs: cycle_calls.append(kwargs) or hotlane.CycleResult(),
        time_fn=iter([0.0, 100.0, 105.0, 110.0, 115.0]).__next__,
        sleep_fn=sleeps.append,
        max_cycles=4,
        queue_depth_fn=lambda _queue_dir: 3,
        high_priority_queue_depth_fn=lambda _queue_dir: 1,
    )

    assert len(cycle_calls) == 2
    assert all(call["recent_limit"] == 0 for call in cycle_calls)
    assert all(call["backlog_batch"] == hotlane.DEFAULT_BACKLOG_BATCH for call in cycle_calls)
    assert all(call["enrich_limit"] == 0 for call in cycle_calls)
    assert sleeps == [0.25] * 4


def test_hotlane_run_logs_reserved_slice_once_during_continuous_pressure(tmp_path, caplog):
    hotlane = _load_hotlane_module()
    caplog.set_level(logging.INFO, logger=hotlane.LOGGER.name)

    class FakeStore:
        def close(self):
            pass

    hotlane.run(
        db_path=tmp_path / "brainlayer.db",
        interval=0.25,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=hotlane.DEFAULT_BACKLOG_BATCH,
        enrich_interval=10.0,
        enrich_limit=hotlane.DEFAULT_HOTLANE_ENRICH_LIMIT,
        enrich_since_hours=8760,
        vector_store_cls=lambda _path: FakeStore(),
        model_factory=lambda: SimpleNamespace(embed_query=lambda _text: [0.0]),
        cycle_fn=lambda **_kwargs: hotlane.CycleResult(),
        time_fn=iter([0.0, 100.0, 105.0, 110.0, 115.0]).__next__,
        sleep_fn=lambda _seconds: None,
        max_cycles=4,
        queue_depth_fn=lambda _queue_dir: 3,
        high_priority_queue_depth_fn=lambda _queue_dir: 1,
    )

    reserved_logs = [
        record
        for record in caplog.records
        if record.getMessage()
        == (
            "durable high-priority queue has backlog; reserving backlog embedding slice "
            f"batch={hotlane.DEFAULT_BACKLOG_BATCH}"
        )
    ]
    assert len(reserved_logs) == 1


def test_hotlane_run_skips_default_hot_embedding_during_queue_backlog(tmp_path, monkeypatch):
    hotlane = _load_hotlane_module()
    split_calls = []
    sleeps = []

    def fake_split_cycle(**kwargs):
        split_calls.append(kwargs)
        return hotlane.CycleResult(embedded=1)

    monkeypatch.setattr(hotlane, "_run_split_cycle", fake_split_cycle)

    hotlane.run(
        db_path=tmp_path / "brainlayer.db",
        interval=0.25,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=hotlane.DEFAULT_BACKLOG_BATCH,
        enrich_interval=10.0,
        enrich_limit=hotlane.DEFAULT_HOTLANE_ENRICH_LIMIT,
        enrich_since_hours=8760,
        vector_store_cls=lambda _path, readonly=False: None,
        model_factory=lambda: SimpleNamespace(embed_query=lambda _text: [0.0]),
        time_fn=iter([0.0, 100.0]).__next__,
        sleep_fn=sleeps.append,
        max_cycles=1,
        queue_depth_fn=lambda _queue_dir: 3,
        high_priority_queue_depth_fn=lambda _queue_dir: 1,
    )

    assert len(split_calls) == 1
    assert split_calls[0]["recent_limit"] == 0
    assert split_calls[0]["backlog_batch"] == hotlane.DEFAULT_BACKLOG_BATCH
    assert split_calls[0]["enrich_limit"] == 0
    assert sleeps == [0.25]


def test_hotlane_run_caps_backlog_batch_at_priority_gate_limit(tmp_path):
    hotlane = _load_hotlane_module()
    scheduled_backlog_batches = []

    class FakeStore:
        def close(self):
            pass

    hotlane.run(
        db_path=tmp_path / "brainlayer.db",
        interval=0.25,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=128,
        enrich_interval=10.0,
        enrich_limit=0,
        enrich_since_hours=8760,
        vector_store_cls=lambda _path: FakeStore(),
        model_factory=lambda: SimpleNamespace(embed_query=lambda _text: [0.0]),
        cycle_fn=lambda **kwargs: scheduled_backlog_batches.append(kwargs["backlog_batch"]) or hotlane.CycleResult(),
        queue_depth_fn=lambda _queue_dir: 0,
        high_priority_queue_depth_fn=lambda _queue_dir: 0,
        time_fn=iter([100.0, 100.0]).__next__,
        sleep_fn=lambda _seconds: None,
        max_cycles=1,
    )

    assert scheduled_backlog_batches == [16]


def test_hotlane_split_cycle_writes_vectors_without_opening_vectorstore_writer(tmp_path):
    hotlane = _load_hotlane_module()
    events = []
    db_path = tmp_path / "brainlayer.db"
    db_path.touch()

    class FakeCursor:
        def __init__(self, readonly):
            self.readonly = readonly

        def execute(self, sql, params=()):
            if self.readonly:
                if sql == hotlane.HOT_CANDIDATE_SCAN_SQL:
                    return [("hot-1", "hot content", "brainbar-store", "mcp", None, None, None, 0, "active", None)]
                if "SELECT c.id, c.created_at, c.rowid" in sql:
                    return [("pending-1", "2026-08-01T00:00:00Z", 1)]
                return [("pending-1", "pending content", None, None, None, 0, "active")]
            events.append(("sql", sql.strip().splitlines()[0]))
            if sql.strip().startswith("SELECT 1"):
                return SimpleNamespace(fetchone=lambda: (1,))
            return []

    class FakeConn:
        def __init__(self, readonly):
            self.readonly = readonly

        def cursor(self):
            return FakeCursor(self.readonly)

    class FakeStore:
        def __init__(self, path, readonly=False):
            self.db_path = path
            self.conn = FakeConn(readonly)
            events.append(("open", readonly))

        def _upsert_chunk_vector(self, _cursor, chunk_id, embedding):
            events.append(("upsert", chunk_id, embedding))

        def close(self):
            events.append(("close", None))

    def write_vectors_fn(target, vectors):
        events.append(("write_target", target))
        for vector in vectors:
            events.append(("upsert", vector.chunk_id, vector.embedding))
        return len(vectors)

    result = hotlane._run_split_cycle(
        db_path=db_path,
        vector_store_cls=FakeStore,
        embed_fn=lambda text: events.append(("embed", text)) or [1.0],
        embed_batch_fn=lambda texts: events.append(("embed_batch", tuple(texts))) or [[2.0] for _ in texts],
        recent_limit=5,
        backlog_batch=4,
        enrich_limit=0,
        enrich_since_hours=8760,
        write_vectors_fn=write_vectors_fn,
    )

    assert result.embedded == 2
    assert events.index(("embed", "hot content")) < events.index(("write_target", db_path))
    assert events.index(("embed", "pending content")) < events.index(("write_target", db_path))
    assert ("open", False) not in events
    assert ("upsert", "hot-1", [1.0]) in events
    assert ("upsert", "pending-1", [1.0]) in events


def test_hotlane_split_cycle_falls_through_recent_candidates_after_embed_failure(tmp_path):
    hotlane = _load_hotlane_module()
    events = []
    db_path = tmp_path / "brainlayer.db"
    db_path.touch()

    class FakeCursor:
        def __init__(self, readonly):
            self.readonly = readonly

        def execute(self, sql, params=()):
            if self.readonly:
                if sql == hotlane.HOT_CANDIDATE_SCAN_SQL:
                    return [
                        ("hot-bad", "bad content", "brainbar-store", "mcp", None, None, None, 0, "active", None),
                        ("hot-good", "good content", "brainbar-store", "mcp", None, None, None, 0, "active", None),
                    ]
                return []
            events.append(("sql", sql.strip().splitlines()[0]))
            if sql.strip().startswith("SELECT 1"):
                return SimpleNamespace(fetchone=lambda: (1,))
            return []

    class FakeConn:
        def __init__(self, readonly):
            self.readonly = readonly

        def cursor(self):
            return FakeCursor(self.readonly)

    class FakeStore:
        def __init__(self, path, readonly=False):
            self.db_path = path
            self.conn = FakeConn(readonly)
            events.append(("open", readonly))

        def _upsert_chunk_vector(self, _cursor, chunk_id, embedding):
            events.append(("upsert", chunk_id, embedding))

        def close(self):
            events.append(("close", None))

    def embed_fn(text):
        events.append(("embed", text))
        if text == "bad content":
            raise RuntimeError("transient embed failure")
        return [3.0]

    def write_vectors_fn(_target, vectors):
        for vector in vectors:
            events.append(("upsert", vector.chunk_id, vector.embedding))
        return len(vectors)

    result = hotlane._run_split_cycle(
        db_path=db_path,
        vector_store_cls=FakeStore,
        embed_fn=embed_fn,
        recent_limit=5,
        backlog_batch=0,
        enrich_limit=0,
        enrich_since_hours=8760,
        write_vectors_fn=write_vectors_fn,
    )

    assert result.embedded == 1
    assert ("embed", "bad content") in events
    assert ("embed", "good content") in events
    assert ("open", False) not in events
    assert ("upsert", "hot-good", [3.0]) in events
    assert ("upsert", "hot-bad", [3.0]) not in events


def test_hotlane_split_cycle_does_not_open_writer_when_no_embedding_or_enrichment_work(tmp_path):
    hotlane = _load_hotlane_module()
    events = []
    db_path = tmp_path / "brainlayer.db"
    db_path.touch()

    class FakeCursor:
        def execute(self, _sql, _params=()):
            return []

    class FakeConn:
        def cursor(self):
            return FakeCursor()

    class FakeStore:
        def __init__(self, _path, readonly=False):
            events.append(("open", readonly))
            self.conn = FakeConn()

        def close(self):
            events.append(("close", None))

    result = hotlane._run_split_cycle(
        db_path=db_path,
        vector_store_cls=FakeStore,
        embed_fn=lambda _text: [1.0],
        recent_limit=5,
        backlog_batch=4,
        enrich_limit=0,
        enrich_since_hours=8760,
    )

    assert result == hotlane.CycleResult()
    assert events == [("open", True), ("close", None)]
