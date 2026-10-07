"""Compatibility tests for historical batch checkpoints and local result replay."""

import json
import sqlite3

import apsw

import brainlayer.cloud_backfill as cloud_backfill
from brainlayer.vector_store import VectorStore


def _insert_unenriched_chunk(
    store: VectorStore, chunk_id: str, content: str, content_type: str = "assistant_text"
) -> None:
    """Insert a minimal unenriched chunk eligible for export."""
    cursor = store.conn.cursor()
    cursor.execute(
        """
        INSERT INTO chunks (id, content, metadata, source_file, project, content_type, char_count, source, sender)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            chunk_id,
            content,
            "{}",
            "test.jsonl",
            "test-project",
            content_type,
            len(content),
            "claude_code",
            None,
        ),
    )


def test_checkpoint_sidecar_db_migrates_legacy_rows_and_keeps_new_writes_off_main_db(tmp_path, monkeypatch):
    """Checkpoint bookkeeping should live in the sidecar DB, not the main content DB."""
    checkpoint_db = tmp_path / "enrichment_checkpoints.db"
    monkeypatch.setattr(cloud_backfill, "CHECKPOINT_DB_PATH", checkpoint_db)

    store = VectorStore(tmp_path / "brainlayer.db")
    try:
        # Simulate legacy rows that were previously stored in the main DB.
        cloud_backfill._ensure_checkpoint_table_in_conn(store.conn)
        store.conn.cursor().execute(
            f"""
            INSERT INTO {cloud_backfill.CHECKPOINT_TABLE}
            (batch_id, backend, model, status, chunk_count, jsonl_path, submitted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-batch",
                "gemini",
                "models/gemini-2.5-flash",
                "submitted",
                500,
                "/tmp/legacy.jsonl",
                "2026-03-14T00:00:00+00:00",
            ),
        )

        cloud_backfill.ensure_checkpoint_table(store)
        cloud_backfill.save_checkpoint(
            store,
            batch_id="new-batch",
            backend="gemini",
            model="models/gemini-2.5-flash",
            status="submitted",
            chunk_count=500,
            jsonl_path="/tmp/new.jsonl",
            submitted_at="2026-03-14T00:01:00+00:00",
        )

        checkpoint_conn = apsw.Connection(str(checkpoint_db))
        try:
            rows = list(
                checkpoint_conn.cursor().execute(
                    f"SELECT batch_id, status, jsonl_path FROM {cloud_backfill.CHECKPOINT_TABLE} ORDER BY batch_id"
                )
            )
        finally:
            checkpoint_conn.close()

        assert rows == [
            ("legacy-batch", "submitted", "/tmp/legacy.jsonl"),
            ("new-batch", "submitted", "/tmp/new.jsonl"),
        ]

        # New writes should not keep using the legacy table in the main DB.
        main_rows = list(
            store.conn.cursor().execute(
                f"SELECT batch_id FROM {cloud_backfill.CHECKPOINT_TABLE} WHERE batch_id = ?",
                ("new-batch",),
            )
        )
        assert main_rows == []
    finally:
        store.close()


def test_get_unsubmitted_export_files_skips_paths_already_checkpointed(tmp_path, monkeypatch):
    """Existing JSONLs should be filtered by stable checkpoint state."""
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    checkpoint_db = tmp_path / "enrichment_checkpoints.db"
    monkeypatch.setattr(cloud_backfill, "EXPORT_DIR", export_dir)
    monkeypatch.setattr(cloud_backfill, "CHECKPOINT_DB_PATH", checkpoint_db)

    file_a = export_dir / "batch_001.jsonl"
    file_b = export_dir / "batch_002.jsonl"
    file_c = export_dir / "batch_003.jsonl"
    for path in (file_a, file_b, file_c):
        path.write_text("{}\n")

    cloud_backfill.save_checkpoint(
        None,
        batch_id="submitted-batch",
        backend="gemini",
        model="models/gemini-2.5-flash",
        status="submitted",
        chunk_count=500,
        jsonl_path=str(file_a),
    )
    cloud_backfill.save_checkpoint(
        None,
        batch_id="expired-batch",
        backend="gemini",
        model="models/gemini-2.5-flash",
        status="expired",
        chunk_count=500,
        jsonl_path=str(file_b),
    )
    cloud_backfill.save_checkpoint(
        None,
        batch_id="failed-batch",
        backend="gemini",
        model="models/gemini-2.5-flash",
        status="failed",
        chunk_count=500,
        jsonl_path=str(file_c),
    )

    remaining = cloud_backfill.get_unsubmitted_export_files(export_dir)
    assert remaining == [file_c]


def test_open_backfill_store_falls_back_to_read_only_when_vectorstore_is_locked(tmp_path, monkeypatch):
    """submit-only style runs should still open the DB for reads when VectorStore init is locked."""
    db_path = tmp_path / "brainlayer.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE chunks (id TEXT, enriched_at TEXT, intent TEXT)")
        conn.commit()
    finally:
        conn.close()

    def locked_vector_store(_db_path):
        raise apsw.BusyError("database is locked")

    monkeypatch.setattr(cloud_backfill, "VectorStore", locked_vector_store)

    store = cloud_backfill.open_backfill_store(db_path, allow_read_only_fallback=True)
    try:
        assert isinstance(store, cloud_backfill.ReadOnlyBackfillStore)
    finally:
        store.close()


def test_get_pending_jobs_is_scoped_to_the_selected_db(tmp_path, monkeypatch):
    """Pending jobs for one DB must not leak into another DB's resume path."""
    monkeypatch.setattr(cloud_backfill, "CHECKPOINT_DB_PATH", tmp_path / "shared-sidecar.db", raising=False)

    db_dir_a = tmp_path / "db-a"
    db_dir_b = tmp_path / "db-b"
    db_dir_a.mkdir()
    db_dir_b.mkdir()

    store_a = VectorStore(db_dir_a / "brainlayer.db")
    store_b = VectorStore(db_dir_b / "brainlayer.db")
    try:
        cloud_backfill.save_checkpoint(
            store_a,
            batch_id="batch-a",
            backend="gemini",
            model="models/gemini-2.5-flash",
            status="submitted",
            chunk_count=10,
            jsonl_path="/tmp/a.jsonl",
        )

        assert [job["batch_id"] for job in cloud_backfill.get_pending_jobs(store_a)] == ["batch-a"]
        assert cloud_backfill.get_pending_jobs(store_b) == []
    finally:
        store_a.close()
        store_b.close()


def test_import_results_commits_canonical_fields_for_unenriched_chunks(tmp_path, monkeypatch):
    """Fresh chunks should receive canonical enrichment fields, not preview-only fields."""
    store = VectorStore(tmp_path / "brainlayer.db")
    try:
        _insert_unenriched_chunk(
            store,
            "chunk-1",
            "brainlayer rollout notes from the remote enrichment import path.",
        )

        monkeypatch.setattr(cloud_backfill, "save_checkpoint", lambda *args, **kwargs: None)

        results = [
            {
                "key": "chunk-1",
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": json.dumps(
                                            {
                                                "summary": "Remote enrichment imported successfully.",
                                                "tags": ["brainlayer", "kg"],
                                                "importance": 7,
                                                "intent": "implementing",
                                                "primary_symbols": ["scripts/cloud_backfill.py"],
                                                "resolved_query": "How does remote batch import keep KG data in sync?",
                                                "epistemic_level": "validated",
                                                "debt_impact": "resolution",
                                                "external_deps": [],
                                            }
                                        )
                                    }
                                ]
                            }
                        }
                    ]
                },
            }
        ]

        counts = cloud_backfill.import_results(store, results, "batch-1")
        row = (
            store.conn.cursor()
            .execute(
                "SELECT summary, summary_v2, enrichment_version, enriched_at, enrich_status FROM chunks WHERE id = ?",
                ("chunk-1",),
            )
            .fetchone()
        )

        assert counts == {"success": 1, "failed": 0, "skipped": 0}
        assert row[0] == "Remote enrichment imported successfully."
        assert row[1] is None
        assert row[2] == "r82-hybrid-taxonomy"
        assert row[3] is not None
        assert row[4] == "success"
    finally:
        store.close()


def test_import_results_commits_enrichment_for_eligible_backlog_chunks(tmp_path, monkeypatch):
    """Batch-drain imports must clear the same eligible set as realtime enrichment."""
    store = VectorStore(tmp_path / "brainlayer.db")
    try:
        _insert_unenriched_chunk(
            store,
            "chunk-drain-1",
            "Batch drain should commit canonical enrichment for this long eligible chunk.",
        )

        monkeypatch.setattr(cloud_backfill, "save_checkpoint", lambda *args, **kwargs: None)

        assert [chunk["id"] for chunk in store.get_enrichment_candidates(limit=10)] == ["chunk-drain-1"]

        results = [
            {
                "key": "chunk-drain-1",
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": json.dumps(
                                            {
                                                "summary": "Canonical batch drain summary.",
                                                "tags": ["brainlayer", "batch-drain"],
                                                "importance": 8,
                                                "intent": "implementing",
                                                "primary_symbols": ["src/brainlayer/cloud_backfill.py"],
                                                "resolved_query": "How does batch drain clear eligible chunks?",
                                                "epistemic_level": "validated",
                                                "debt_impact": "resolution",
                                                "external_deps": ["Gemini Batch API"],
                                                "entities": [
                                                    {
                                                        "name": "BrainLayer",
                                                        "type": "project",
                                                        "salience": 0.9,
                                                        "role": "system_under_test",
                                                    }
                                                ],
                                            }
                                        )
                                    }
                                ]
                            }
                        }
                    ]
                },
            }
        ]

        counts = cloud_backfill.import_results(store, results, "batch-drain-1")

        row = (
            store.conn.cursor()
            .execute(
                """
                SELECT summary, tags, importance, intent, enriched_at, enrich_status, summary_v2
                FROM chunks WHERE id = ?
                """,
                ("chunk-drain-1",),
            )
            .fetchone()
        )

        assert counts == {"success": 1, "failed": 0, "skipped": 0}
        assert row[0] == "Canonical batch drain summary."
        assert json.loads(row[1]) == ["brainlayer", "batch-drain"]
        assert row[2] == 8
        assert row[3] == "implementing"
        assert row[4] is not None
        assert row[5] == "success"
        assert row[6] is None
        assert store.get_enrichment_candidates(limit=10) == []
    finally:
        store.close()


def test_import_results_writes_summary_v2_for_legacy_chunks(tmp_path, monkeypatch):
    """Legacy enriched chunks should receive preview summaries without overwriting summary."""
    store = VectorStore(tmp_path / "brainlayer.db")
    try:
        cursor = store.conn.cursor()
        cursor.execute(
            """
            INSERT INTO chunks (
                id, content, metadata, source_file, project, content_type, char_count,
                source, summary, enriched_at, created_at
            ) VALUES (?, ?, '{}', 'test.jsonl', 'test-project', 'assistant_text', ?, 'claude_code', ?, ?, ?)
            """,
            (
                "legacy-1",
                "Legacy chunk that already has an old summary and needs a preview v2 summary.",
                73,
                "Old summary",
                "2026-04-01T00:00:00+00:00",
                "2026-04-01T00:00:00+00:00",
            ),
        )

        monkeypatch.setattr(cloud_backfill, "save_checkpoint", lambda *args, **kwargs: None)

        results = [
            {
                "key": "legacy-1",
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": json.dumps(
                                            {
                                                "summary": "Improved preview summary",
                                                "tags": ["brainlayer"],
                                                "importance": 8,
                                                "intent": "implementing",
                                                "primary_symbols": [],
                                                "resolved_query": "What changed in the legacy chunk?",
                                                "epistemic_level": "validated",
                                                "debt_impact": "low",
                                                "external_deps": [],
                                            }
                                        )
                                    }
                                ]
                            }
                        }
                    ]
                },
            }
        ]

        counts = cloud_backfill.import_results(store, results, "batch-legacy")
        row = cursor.execute(
            "SELECT summary, summary_v2, enrichment_version, enriched_at FROM chunks WHERE id = ?",
            ("legacy-1",),
        ).fetchone()

        assert counts == {"success": 1, "failed": 0, "skipped": 0}
        assert row == (
            "Old summary",
            "Improved preview summary",
            "2.0",
            "2026-04-01T00:00:00+00:00",
        )
    finally:
        store.close()


def test_import_results_keeps_chunk_retryable_when_parse_fails(tmp_path, monkeypatch):
    """Parse failures should leave preview fields unset so the chunk can retry later."""
    store = VectorStore(tmp_path / "brainlayer.db")
    try:
        _insert_unenriched_chunk(
            store,
            "chunk-1",
            "brainlayer rollout notes from the remote enrichment import path.",
        )

        monkeypatch.setattr(cloud_backfill, "save_checkpoint", lambda *args, **kwargs: None)

        results = [
            {
                "key": "chunk-1",
                "response": {"candidates": [{"content": {"parts": [{"text": "not valid json"}]}}]},
            }
        ]

        counts = cloud_backfill.import_results(store, results, "batch-1")
        row = (
            store.conn.cursor()
            .execute(
                "SELECT summary, summary_v2, enrichment_version, enriched_at FROM chunks WHERE id = ?",
                ("chunk-1",),
            )
            .fetchone()
        )

        assert counts == {"success": 0, "failed": 1, "skipped": 0}
        assert row == (None, None, "1.0", None)
    finally:
        store.close()
