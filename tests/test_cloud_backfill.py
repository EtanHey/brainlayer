"""Regression tests for cloud_backfill batch export helpers."""

import json
import sqlite3
from pathlib import Path

import apsw
import pytest

from brainlayer.vector_store import VectorStore
from scripts import cloud_backfill


def _insert_unenriched_chunk(store: VectorStore, chunk_id: str, content: str, content_type: str = "assistant_text") -> None:
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


def test_export_unenriched_chunks_disables_thinking(tmp_path, monkeypatch):
    """Batch export JSONL must force thinkingBudget=0 for every request."""
    export_dir = tmp_path / "exports"
    monkeypatch.setattr(cloud_backfill, "EXPORT_DIR", export_dir)

    store = VectorStore(tmp_path / "backfill.db")
    try:
        _insert_unenriched_chunk(
            store,
            "chunk-1",
            "This chunk is long enough to be exported without sanitization in the test fixture.",
        )

        jsonl_files = cloud_backfill.export_unenriched_chunks(
            store,
            max_chunks=1,
            content_types=["assistant_text"],
            no_sanitize=True,
        )

        assert len(jsonl_files) == 1
        line = json.loads(jsonl_files[0].read_text().splitlines()[0])
        assert line["key"] == "chunk-1"
        assert line["request"]["generationConfig"]["responseMimeType"] == "application/json"
        assert line["request"]["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 0
    finally:
        store.close()


def test_estimate_batch_cost_uses_discounted_batch_rates():
    """Batch usage cost helper should apply the documented 50% discount."""
    cost = cloud_backfill.estimate_batch_cost_usd(1_000_000, 2_000_000)

    assert cost == pytest.approx(0.675)
    assert cloud_backfill.estimate_batch_cost_usd(0, 0) == 0


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


def test_submit_only_reuses_existing_exports_without_reexporting(tmp_path, monkeypatch):
    """submit-only should use the existing batch files and only submit the remaining ones."""
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    checkpoint_db = tmp_path / "enrichment_checkpoints.db"
    monkeypatch.setattr(cloud_backfill, "EXPORT_DIR", export_dir)
    monkeypatch.setattr(cloud_backfill, "CHECKPOINT_DB_PATH", checkpoint_db)

    submitted_file = export_dir / "batch_001.jsonl"
    remaining_file = export_dir / "batch_002.jsonl"
    submitted_file.write_text("{}\n")
    remaining_file.write_text("{}\n")

    cloud_backfill.save_checkpoint(
        None,
        batch_id="submitted-batch",
        backend="gemini",
        model="models/gemini-2.5-flash",
        status="submitted",
        chunk_count=500,
        jsonl_path=str(submitted_file),
    )

    submitted_paths: list[Path] = []

    def fake_export(*args, **kwargs):  # pragma: no cover - should not be called
        raise AssertionError("submit-only should reuse existing export files, not regenerate them")

    def fake_submit(jsonl_path, model, store):
        submitted_paths.append(Path(jsonl_path))
        return f"job-{Path(jsonl_path).stem}"

    monkeypatch.setattr(cloud_backfill, "export_unenriched_chunks", fake_export)
    monkeypatch.setattr(cloud_backfill, "submit_gemini_batch", fake_submit)
    monkeypatch.setattr(cloud_backfill.time, "sleep", lambda *_args, **_kwargs: None)

    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)
    store.close()

    cloud_backfill.run_full_backfill(db_path, submit_only=True)

    assert submitted_paths == [remaining_file]


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


def test_submit_only_exports_new_chunks_when_old_files_are_checkpointed(tmp_path, monkeypatch):
    """submit-only should export new unenriched chunks instead of returning early."""
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    checkpoint_db = tmp_path / "enrichment_checkpoints.db"
    monkeypatch.setattr(cloud_backfill, "EXPORT_DIR", export_dir)
    monkeypatch.setattr(cloud_backfill, "CHECKPOINT_DB_PATH", checkpoint_db, raising=False)

    old_file = export_dir / "batch_001.jsonl"
    old_file.write_text("{}\n")
    new_file = export_dir / "batch_002.jsonl"

    cloud_backfill.save_checkpoint(
        None,
        batch_id="imported-batch",
        backend="gemini",
        model="models/gemini-2.5-flash",
        status="imported",
        chunk_count=500,
        jsonl_path=str(old_file),
    )

    exported_paths: list[Path] = []
    submitted_paths: list[Path] = []

    def fake_export(*args, **kwargs):
        exported_paths.append(new_file)
        new_file.write_text("{}\n")
        return [new_file]

    def fake_submit(jsonl_path, model, store):
        submitted_paths.append(Path(jsonl_path))
        return f"job-{Path(jsonl_path).stem}"

    monkeypatch.setattr(cloud_backfill, "export_unenriched_chunks", fake_export)
    monkeypatch.setattr(cloud_backfill, "submit_gemini_batch", fake_submit)
    monkeypatch.setattr(cloud_backfill.time, "sleep", lambda *_args, **_kwargs: None)

    store = VectorStore(tmp_path / "brainlayer.db")
    try:
        _insert_unenriched_chunk(store, "chunk-1", "new unenriched content that still needs export")
    finally:
        store.close()

    cloud_backfill.run_full_backfill(tmp_path / "brainlayer.db", submit_only=True)

    assert exported_paths == [new_file]
    assert submitted_paths == [new_file]
