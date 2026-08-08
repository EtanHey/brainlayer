from __future__ import annotations

import json
from pathlib import Path

import apsw
import pytest
from typer.testing import CliRunner

import brainlayer.runtime_store as runtime_store_module
from brainlayer.runtime_store import (
    OfflineMigrator,
    ReadonlyStore,
    RuntimeStoreModeError,
    SchemaFingerprintMismatch,
    WriterRuntimeStore,
    open_writer_store,
)
from brainlayer.vector_store import VectorStore


def test_migrate_store_command_requires_explicit_copy_path(tmp_path, monkeypatch):
    from brainlayer.cli import app

    canonical = tmp_path / "canonical" / "brainlayer.db"
    copy_path = tmp_path / "copy" / "brainlayer.db"
    monkeypatch.setattr("brainlayer.runtime_store._canonical_db_path", lambda: canonical)

    missing = CliRunner().invoke(app, ["migrate-store"])
    assert missing.exit_code != 0

    migrated = CliRunner().invoke(app, ["migrate-store", str(copy_path)])
    assert migrated.exit_code == 0, migrated.output
    assert copy_path.exists()

    refused = CliRunner().invoke(app, ["migrate-store", str(canonical)])
    assert refused.exit_code == 1
    assert "canonical" in refused.output.lower()


def test_chromadb_migrate_command_requires_explicit_offline_path(tmp_path, monkeypatch):
    from brainlayer.cli import app

    canonical = tmp_path / "canonical" / "brainlayer.db"
    monkeypatch.setattr("brainlayer.runtime_store._canonical_db_path", lambda: canonical)

    missing = CliRunner().invoke(app, ["migrate"])
    assert missing.exit_code != 0

    refused = CliRunner().invoke(app, ["migrate", str(canonical)])
    assert refused.exit_code == 1
    assert "canonical" in refused.output.lower()
    assert not canonical.exists()


def test_repair_fts_command_requires_explicit_offline_copy(tmp_path, monkeypatch):
    from brainlayer.cli import app

    canonical = tmp_path / "canonical" / "brainlayer.db"
    copy_path = tmp_path / "copy" / "brainlayer.db"
    monkeypatch.setattr("brainlayer.runtime_store._canonical_db_path", lambda: canonical)
    OfflineMigrator(copy_path).close()

    missing = CliRunner().invoke(app, ["repair-fts"])
    assert missing.exit_code != 0

    repaired = CliRunner().invoke(app, ["repair-fts", str(copy_path)])
    assert repaired.exit_code == 0, repaired.output

    refused = CliRunner().invoke(app, ["repair-fts", str(canonical)])
    assert refused.exit_code == 1
    assert "canonical" in refused.output.lower()


def _bootstrap(db_path: Path) -> None:
    store = VectorStore(db_path)
    store.close()


def _telemetry_events(log_path: Path, operation: str) -> list[dict]:
    if not log_path.exists():
        return []
    return [
        event
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if (event := json.loads(line)).get("operation") == operation
    ]


def test_readonly_store_uses_readonly_connection_without_legacy_init(tmp_path, monkeypatch):
    db_path = tmp_path / "readonly.db"
    _bootstrap(db_path)

    def fail_legacy_init(self):
        raise AssertionError("readonly store must not run legacy initialization")

    monkeypatch.setattr(VectorStore, "_init_db_with_retry", fail_legacy_init)

    with ReadonlyStore(db_path) as store:
        assert store._readonly is True
        assert store.conn.readonly("main") is True
        with pytest.raises(apsw.ReadOnlyError):
            store.conn.cursor().execute(
                "INSERT INTO chunks (id, content, metadata, source_file) VALUES ('x','x','{}','x')"
            )


def test_runtime_store_opens_existing_schema_without_legacy_init(tmp_path, monkeypatch):
    db_path = tmp_path / "runtime.db"
    _bootstrap(db_path)

    def fail_legacy_init(self):
        raise AssertionError("runtime store must not run legacy initialization")

    monkeypatch.setattr(VectorStore, "_init_db_with_retry", fail_legacy_init)

    with WriterRuntimeStore(db_path) as store:
        assert store._readonly is False
        assert store.conn.readonly("main") is False
        assert len(store.schema_fingerprint) == 64


def test_runtime_writer_defaults_to_synchronous_normal(tmp_path, monkeypatch):
    db_path = tmp_path / "runtime-sync.db"
    monkeypatch.delenv("BRAINLAYER_WRITE_SYNCHRONOUS", raising=False)
    _bootstrap(db_path)

    with WriterRuntimeStore(db_path) as store:
        assert store.conn.cursor().execute("PRAGMA synchronous").fetchone()[0] == 1


def test_runtime_store_closes_connection_when_extension_setup_fails(tmp_path, monkeypatch):
    db_path = tmp_path / "extension-failure.db"
    _bootstrap(db_path)
    real_connection = runtime_store_module.apsw.Connection
    closed: list[bool] = []

    class TrackedConnection:
        def __init__(self, *args, **kwargs):
            self._connection = real_connection(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._connection, name)

        def close(self):
            closed.append(True)
            return self._connection.close()

    monkeypatch.setattr(runtime_store_module.apsw, "Connection", TrackedConnection)
    monkeypatch.setattr(
        runtime_store_module,
        "_load_vector_extension",
        lambda _conn: (_ for _ in ()).throw(RuntimeError("extension load failed")),
    )

    with pytest.raises(RuntimeError, match="extension load failed"):
        WriterRuntimeStore(db_path)

    assert closed == [True]


def test_runtime_store_missing_database_fails_closed_without_creating_it(tmp_path):
    db_path = tmp_path / "missing" / "runtime.db"

    with pytest.raises(SchemaFingerprintMismatch, match="does not exist"):
        WriterRuntimeStore(db_path)

    assert not db_path.exists()
    assert not db_path.parent.exists()


def test_runtime_store_schema_mismatch_fails_closed_and_releases_pidfile(tmp_path, monkeypatch):
    db_path = tmp_path / "stale.db"
    pidfile_dir = tmp_path / "pidfiles"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))
    _bootstrap(db_path)

    conn = apsw.Connection(str(db_path))
    conn.cursor().execute("DROP TRIGGER chunks_fts_insert")
    conn.close()

    with pytest.raises(SchemaFingerprintMismatch, match="chunks_fts_insert") as exc_info:
        WriterRuntimeStore(db_path)

    assert exc_info.value.expected_fingerprint
    assert exc_info.value.actual_fingerprint
    assert list(pidfile_dir.glob("*.pid")) == []


def test_runtime_store_rejects_same_name_trigger_with_wrong_definition(tmp_path):
    db_path = tmp_path / "wrong-trigger.db"
    _bootstrap(db_path)
    conn = apsw.Connection(str(db_path))
    cursor = conn.cursor()
    cursor.execute("DROP TRIGGER chunks_fts_insert")
    cursor.execute("CREATE TRIGGER chunks_fts_insert AFTER INSERT ON chunks BEGIN SELECT 1; END")
    conn.close()

    with pytest.raises(SchemaFingerprintMismatch, match="definition mismatches: chunks_fts_insert"):
        WriterRuntimeStore(db_path)


def test_runtime_open_telemetry_has_fingerprint_and_no_scan_or_mutation_statements(tmp_path, monkeypatch):
    db_path = tmp_path / "telemetry.db"
    log_path = tmp_path / "runtime-open.jsonl"
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY", "1")
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY_PATH", str(log_path))
    monkeypatch.setenv("BRAINLAYER_WRITER_HEARTBEAT_DIR", str(tmp_path / "heartbeats"))
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY_FTS_SAMPLE_TTL_SECONDS", "0")
    _bootstrap(db_path)
    log_path.unlink(missing_ok=True)

    with WriterRuntimeStore(db_path) as store:
        fingerprint = store.schema_fingerprint

    finished = [event for event in _telemetry_events(log_path, "runtime_open") if event["event"] == "txn_finished"]
    assert len(finished) == 1
    assert finished[0]["outcome"] == "completed"
    assert finished[0]["schema_fingerprint"] == fingerprint
    assert finished[0]["duration_ms"] < 100
    assert finished[0]["fts_segments_before"] == {}
    assert finished[0]["fts_segments"] == {}
    statements = finished[0]["statements"]
    assert statements
    normalized = [statement["normalized_sql"].upper() for statement in statements]
    forbidden = ("CREATE ", "DROP ", "ALTER ", "INSERT ", "UPDATE ", "DELETE ", "COUNT(", "OPTIMIZE")
    assert not any(token in statement for statement in normalized for token in forbidden)
    corpus_tables = ("FROM CHUNKS ", "FROM CHUNKS_FTS", "FROM CHUNK_FTS_ROWIDS")
    assert not any(table in statement for statement in normalized for table in corpus_tables)
    assert all(
        statement["fullscan_steps"] == 0
        or "SQLITE_SCHEMA" in statement["normalized_sql"].upper()
        or statement["normalized_sql"].upper().startswith("PRAGMA TABLE_INFO")
        for statement in statements
    )


def test_open_writer_store_defaults_new_and_legacy_flag_rolls_back(tmp_path, monkeypatch):
    db_path = tmp_path / "factory.db"
    _bootstrap(db_path)

    monkeypatch.delenv("BRAINLAYER_RUNTIME_STORE", raising=False)
    with open_writer_store(db_path) as store:
        assert type(store) is WriterRuntimeStore

    monkeypatch.setenv("BRAINLAYER_RUNTIME_STORE", "legacy")
    with open_writer_store(db_path) as store:
        assert type(store) is VectorStore


def test_open_writer_store_rejects_unknown_mode(tmp_path, monkeypatch):
    db_path = tmp_path / "factory.db"
    _bootstrap(db_path)
    monkeypatch.setenv("BRAINLAYER_RUNTIME_STORE", "surprise")

    with pytest.raises(RuntimeStoreModeError, match="BRAINLAYER_RUNTIME_STORE"):
        open_writer_store(db_path)


def test_offline_migrator_allows_copy_and_refuses_canonical(tmp_path, monkeypatch):
    canonical = tmp_path / "canonical" / "brainlayer.db"
    copy_path = tmp_path / "copy" / "brainlayer.db"
    monkeypatch.setattr("brainlayer.runtime_store._canonical_db_path", lambda: canonical)

    with OfflineMigrator(copy_path):
        pass
    assert copy_path.exists()

    with pytest.raises(PermissionError, match="canonical"):
        OfflineMigrator(canonical)
    assert not canonical.exists()


def test_offline_migrator_refuses_configured_database_path(tmp_path, monkeypatch):
    configured = tmp_path / "configured-live.db"
    monkeypatch.setenv("BRAINLAYER_DB", str(configured))

    with pytest.raises(PermissionError, match="canonical"):
        OfflineMigrator(configured)

    assert not configured.exists()


def test_offline_migrator_canonical_requires_gated_swap_override(tmp_path, monkeypatch):
    canonical = tmp_path / "canonical" / "brainlayer.db"
    monkeypatch.setattr("brainlayer.runtime_store._canonical_db_path", lambda: canonical)
    monkeypatch.setenv("BRAINLAYER_OFFLINE_MIGRATOR_GATED_SWAP", "1")

    with OfflineMigrator(canonical):
        pass

    assert canonical.exists()


def test_offline_migrator_opens_the_once_resolved_copy_path(tmp_path, monkeypatch):
    canonical = tmp_path / "canonical" / "brainlayer.db"
    copy_path = tmp_path / "copy" / "brainlayer.db"
    link_path = tmp_path / "copy-link.db"
    monkeypatch.setattr("brainlayer.runtime_store._canonical_db_path", lambda: canonical)
    OfflineMigrator(copy_path).close()
    link_path.symlink_to(copy_path)

    with OfflineMigrator(link_path) as store:
        assert store.db_path == copy_path.resolve()
