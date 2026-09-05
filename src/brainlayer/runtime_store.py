"""Explicit read, runtime-write, and offline-migration store entrypoints."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import apsw
import sqlite_vec

from .vector_store import (
    _CONNECTION_HOOK_STATE,
    VectorStore,
    _configure_writer_pragmas,
    _write_busy_timeout_ms,
)
from .writer_telemetry import start_writer_span

logger = logging.getLogger(__name__)

RUNTIME_SCHEMA_CONTRACT_VERSION = 1

_REQUIRED_SCHEMA_OBJECTS = {
    "agent_profiles": "table",
    "chunk_fts_rowids": "table",
    "chunk_id_alias": "table",
    "chunk_tags": "table",
    "chunk_vectors": "table",
    "chunks": "table",
    "chunks_fts": "table",
    "chunks_fts_operational": "table",
    "chunks_fts_trigram": "table",
    "dedupe_audit": "table",
    "entity_facts": "table",
    "health_events": "table",
    "kg_entities": "table",
    "kg_entity_chunks": "table",
    "kg_relations": "table",
    "operations": "table",
    "schema_migrations": "table",
    "session_context": "table",
    "session_enrichments": "table",
    "tag_tombstones": "table",
    "chunk_tags_delete": "trigger",
    "chunk_tags_insert": "trigger",
    "chunk_tags_update": "trigger",
    "chunk_tags_update_clear": "trigger",
    "chunks_fts_delete": "trigger",
    "chunks_fts_insert": "trigger",
    "chunks_fts_operational_insert": "trigger",
    "chunks_fts_trigram_delete": "trigger",
    "chunks_fts_trigram_insert": "trigger",
    "chunks_fts_update": "trigger",
    "idx_chunk_id_alias_canonical": "index",
    "idx_chunks_content_hash": "index",
    "idx_chunks_dedupe_hash": "index",
    "idx_chunks_simhash_band_0": "index",
    "idx_chunks_simhash_band_1": "index",
    "idx_chunks_simhash_band_2": "index",
    "idx_chunks_simhash_band_3": "index",
}

_OPTIONAL_SCHEMA_OBJECTS = {
    "chunk_vectors_binary",
}

_REQUIRED_CHUNK_COLUMNS = {
    "aggregated_into",
    "archived",
    "archived_at",
    "brick_id",
    "char_count",
    "chunk_origin",
    "content",
    "content_class",
    "content_hash",
    "content_type",
    "conversation_id",
    "created_at",
    "decay_score",
    "dedupe_hash",
    "enrich_status",
    "enriched_at",
    "enrichment_backend",
    "enrichment_model",
    "enrichment_version",
    "epistemic_level",
    "external_deps",
    "half_life_days",
    "id",
    "importance",
    "ingested_at",
    "intent",
    "key_facts",
    "language",
    "last_retrieved",
    "last_seen_at",
    "metadata",
    "position",
    "primary_symbols",
    "project",
    "provenance_class",
    "raw_entities_json",
    "resolved_queries",
    "resolved_query",
    "retrieval_count",
    "seen_count",
    "sender",
    "sentiment_label",
    "sentiment_score",
    "sentiment_signals",
    "simhash",
    "simhash_band_0",
    "simhash_band_1",
    "simhash_band_2",
    "simhash_band_3",
    "source",
    "source_file",
    "source_project_id",
    "source_uri",
    "status",
    "summary",
    "summary_v2",
    "superseded_by",
    "tag_confidence",
    "tags",
    "topic_cluster",
    "value_type",
    "version_scope",
}

_INTEGER_CHUNK_COLUMNS = {
    "archived",
    "char_count",
    "ingested_at",
    "position",
    "retrieval_count",
    "seen_count",
}
_REAL_CHUNK_COLUMNS = {
    "decay_score",
    "half_life_days",
    "importance",
    "last_retrieved",
    "sentiment_score",
    "tag_confidence",
}
_NOT_NULL_CHUNK_COLUMNS = {"content", "metadata", "source_file"}
_PRIMARY_KEY_CHUNK_COLUMNS = {"id"}
_CHUNK_COLUMN_DEFAULTS: dict[str, frozenset[str | None]] = {
    "archived": frozenset({"0"}),
    "chunk_origin": frozenset({"'unknown'"}),
    "content_class": frozenset({"'knowledge'"}),
    "created_at": frozenset({None, "strftime('%Y-%m-%dT%H:%M:%fZ','now')"}),
    "decay_score": frozenset({"1.0"}),
    "enrichment_version": frozenset({"'1.0'"}),
    "half_life_days": frozenset({"30.0"}),
    "last_retrieved": frozenset({"NULL"}),
    "retrieval_count": frozenset({"0"}),
    "seen_count": frozenset({"1"}),
    "status": frozenset({"'active'"}),
}

# Exact normalized definitions for runtime-critical triggers, indexes, and
# virtual tables. chunks_fts deliberately accepts both schemas present in the
# current canonical DB (prefix index) and a freshly migrated DB (no prefix).
_REQUIRED_SQL_HASHES: dict[str, frozenset[str]] = {
    "chunk_tags_delete": frozenset({"afe039dd59665608da1d2dfca5693524739713b32ec94d820d9be3e91d91f34d"}),
    "chunk_tags_insert": frozenset({"b0da90e12220eec074a72e34efc961f350365965b0566d92d493ade15da2aae6"}),
    "chunk_tags_update": frozenset({"0aa51ed4303432c7a1dad25a99a48dd39975f75ac844727ad9bf5df2ff801150"}),
    "chunk_tags_update_clear": frozenset({"334ac88d3dec680d9078e9e735ecd67ef96ec97d0b24c292d4ae5c2256b83d1c"}),
    "chunk_vectors": frozenset({"75f11e949563f6f93baeaea6c8de8a68d4931b48581808ef06b43099a2eb4c90"}),
    "chunks_fts": frozenset(
        {
            "0b5b4a3dafe4921c2272b1c7536254cd8ef1e69388ac928266b48083f5702034",
            "9192e74ca4e3bd65bcaf3f85c596693d61460c549ec87df44aba644b588612dd",
        }
    ),
    "chunks_fts_delete": frozenset({"78d8809fd5864f9c8a8f66bd9547b19872476b838c50dc74bf5dd8a36d1f79b1"}),
    "chunks_fts_insert": frozenset({"8d6d5ab6d16a03a8b56dc4ba89918c0d07da5c69639f7fcbd33e29e7ea3a29c0"}),
    "chunks_fts_operational": frozenset({"d9541e259b0bf038d1ac5e155951f5637ac9c67f3a5d674919438b644183c85b"}),
    "chunks_fts_operational_insert": frozenset({"ba5b681dcc5c4734e39ba0bd5693562653c134b9a75e7887d961ec85428c8e81"}),
    "chunks_fts_trigram": frozenset({"86456656a1ec54a6623009d19c7e3da670af08a1474aef81f99e4fe42e46dbab"}),
    "chunks_fts_trigram_delete": frozenset({"95b91ed7a93f772fef45020a035f287c9e9c05f23e9464ce824e16517784c127"}),
    "chunks_fts_trigram_insert": frozenset({"7bc292a92499f33e38e381c2ba9dc315f57cd20232cb284d7429623cdccf6db8"}),
    "chunks_fts_update": frozenset({"04a43eb7d91f2463fdfd8c0c98fd2ef2a0f64e2fabc2468a9ab7bef687365986"}),
    "idx_chunk_id_alias_canonical": frozenset({"f14ffbac95a091c3329f3fb8d98bb98b6759d56c41a9c1a5725944ebcc9b45e5"}),
    "idx_chunks_content_hash": frozenset({"972b62d3b45116689ab7f38a361de9a98ea109ce58ba724367480e621df9895b"}),
    "idx_chunks_dedupe_hash": frozenset({"c66667f3761edc89a9e968e7d184fc8b2eacf05b74012a560aad4b21881940c3"}),
    "idx_chunks_simhash_band_0": frozenset({"07c03830b2443d490c6e9997599ab7849c49743729bfa2a1ddc0ae642fbcafd3"}),
    "idx_chunks_simhash_band_1": frozenset({"3f8901a6bf1dc70f01f8f8b17b2fc68215f45c062a633a48e6832326340899f3"}),
    "idx_chunks_simhash_band_2": frozenset({"1fcdaedbce66e519aa5e25ff55fde96dcc48db507011dd6ec92eb85b030378c2"}),
    "idx_chunks_simhash_band_3": frozenset({"ef9832d08371d0f7afd5e72f868b716e1cb84afd849664d5ff9e0d8d74741e57"}),
}


def _chunk_column_contract(name: str) -> tuple[str, bool, tuple[str | None, ...], bool]:
    if name in _INTEGER_CHUNK_COLUMNS:
        column_type = "INTEGER"
    elif name in _REAL_CHUNK_COLUMNS:
        column_type = "REAL"
    else:
        column_type = "TEXT"
    allowed_defaults = _CHUNK_COLUMN_DEFAULTS.get(name, frozenset({None}))
    return (
        column_type,
        name in _NOT_NULL_CHUNK_COLUMNS,
        tuple(sorted(allowed_defaults, key=lambda value: "" if value is None else value)),
        name in _PRIMARY_KEY_CHUNK_COLUMNS,
    )


def _normalize_schema_sql(sql: str | None) -> str:
    return " ".join(str(sql or "").split()).lower()


def _schema_sql_hash(sql: str | None) -> str:
    return hashlib.sha256(_normalize_schema_sql(sql).encode("utf-8")).hexdigest()


def _contract_payload() -> dict[str, Any]:
    return {
        "version": RUNTIME_SCHEMA_CONTRACT_VERSION,
        "objects": sorted(_REQUIRED_SCHEMA_OBJECTS.items()),
        "chunks_columns": [(name, *_chunk_column_contract(name)) for name in sorted(_REQUIRED_CHUNK_COLUMNS)],
        "sql_hashes": [(name, sorted(hashes)) for name, hashes in sorted(_REQUIRED_SQL_HASHES.items())],
    }


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


EXPECTED_RUNTIME_SCHEMA_FINGERPRINT = _fingerprint(_contract_payload())


class SchemaFingerprintMismatch(RuntimeError):
    """Raised when a runtime store cannot prove the expected schema contract."""

    def __init__(
        self,
        message: str,
        *,
        expected_fingerprint: str = EXPECTED_RUNTIME_SCHEMA_FINGERPRINT,
        actual_fingerprint: str,
    ) -> None:
        super().__init__(message)
        self.expected_fingerprint = expected_fingerprint
        self.actual_fingerprint = actual_fingerprint


class RuntimeStoreModeError(ValueError):
    """Raised for an invalid runtime-store rollback selector."""


def _canonical_db_path() -> Path:
    from .paths import get_db_path

    return get_db_path()


def resolve_offline_database_path(db_path: Path) -> Path:
    """Resolve an offline target once and reject the configured live database."""
    path = Path(db_path).expanduser().resolve()
    canonical = _canonical_db_path().expanduser().resolve()
    if path == canonical and os.environ.get("BRAINLAYER_OFFLINE_MIGRATOR_GATED_SWAP") != "1":
        raise PermissionError(
            "offline migration refuses the configured canonical BrainLayer database; migrate and repair an "
            "explicit copy, then use the gated atomic-swap phase"
        )
    return path


def _mismatch_fingerprint(
    *,
    objects: dict[str, str],
    chunk_columns: dict[str, tuple[str, bool, str | None, bool]],
    sql_hashes: dict[str, str],
) -> str:
    return _fingerprint(
        {
            "version": RUNTIME_SCHEMA_CONTRACT_VERSION,
            "objects": sorted(objects.items()),
            "chunks_columns": sorted((name, *metadata) for name, metadata in chunk_columns.items()),
            "sql_hashes": sorted(sql_hashes.items()),
        }
    )


def _validate_runtime_schema(cursor: apsw.Cursor) -> tuple[str, dict[str, str], set[str]]:
    object_names = sorted(_REQUIRED_SCHEMA_OBJECTS.keys() | _OPTIONAL_SCHEMA_OBJECTS)
    placeholders = ",".join("?" for _ in object_names)
    schema_rows = {
        row[0]: (row[1], row[2])
        for row in cursor.execute(
            f"SELECT name, type, sql FROM sqlite_schema WHERE name IN ({placeholders})",  # noqa: S608
            object_names,
        )
    }
    actual_objects = {name: object_type for name, (object_type, _sql) in schema_rows.items()}
    chunk_column_rows = {
        row[1]: (str(row[2]).upper(), bool(row[3]), row[4], bool(row[5]))
        for row in cursor.execute("PRAGMA table_info(chunks)")
    }
    chunk_columns = set(chunk_column_rows)
    actual_sql_hashes = {
        name: _schema_sql_hash(schema_rows[name][1]) for name in _REQUIRED_SQL_HASHES if name in schema_rows
    }
    required_actual_objects = {
        name: object_type for name, object_type in actual_objects.items() if name in _REQUIRED_SCHEMA_OBJECTS
    }
    required_actual_columns = {
        name: metadata for name, metadata in chunk_column_rows.items() if name in _REQUIRED_CHUNK_COLUMNS
    }
    actual_fingerprint = _mismatch_fingerprint(
        objects=required_actual_objects,
        chunk_columns=required_actual_columns,
        sql_hashes=actual_sql_hashes,
    )

    missing_objects = sorted(
        name for name, expected_type in _REQUIRED_SCHEMA_OBJECTS.items() if actual_objects.get(name) != expected_type
    )
    missing_columns = sorted(_REQUIRED_CHUNK_COLUMNS - chunk_columns)
    column_definition_mismatches = sorted(
        name
        for name in _REQUIRED_CHUNK_COLUMNS & chunk_columns
        if (
            chunk_column_rows[name][0] != _chunk_column_contract(name)[0]
            or chunk_column_rows[name][1] != _chunk_column_contract(name)[1]
            or chunk_column_rows[name][2] not in _chunk_column_contract(name)[2]
            or chunk_column_rows[name][3] != _chunk_column_contract(name)[3]
        )
    )
    sql_definition_mismatches = sorted(
        name
        for name, allowed_hashes in _REQUIRED_SQL_HASHES.items()
        if name in actual_sql_hashes and actual_sql_hashes[name] not in allowed_hashes
    )
    if missing_objects or missing_columns or column_definition_mismatches or sql_definition_mismatches:
        details: list[str] = []
        if missing_objects:
            details.append(f"missing or wrong-type objects: {', '.join(missing_objects)}")
        if missing_columns:
            details.append(f"missing chunks columns: {', '.join(missing_columns)}")
        if column_definition_mismatches:
            details.append(f"chunks column definition mismatches: {', '.join(column_definition_mismatches)}")
        if sql_definition_mismatches:
            details.append(f"definition mismatches: {', '.join(sql_definition_mismatches)}")
        raise SchemaFingerprintMismatch(
            "runtime schema fingerprint mismatch; " + "; ".join(details),
            actual_fingerprint=actual_fingerprint,
        )
    return EXPECTED_RUNTIME_SCHEMA_FINGERPRINT, actual_objects, chunk_columns


@contextmanager
def _without_connection_maintenance_hooks():
    previous = getattr(_CONNECTION_HOOK_STATE, "skip_maintenance", None)
    _CONNECTION_HOOK_STATE.skip_maintenance = True
    try:
        yield
    finally:
        if previous is None:
            delattr(_CONNECTION_HOOK_STATE, "skip_maintenance")
        else:
            _CONNECTION_HOOK_STATE.skip_maintenance = previous


def _load_vector_extension(conn: apsw.Connection) -> None:
    conn.enableloadextension(True)
    try:
        conn.loadextension(sqlite_vec.loadable_path())
    finally:
        conn.enableloadextension(False)


class ReadonlyStore(VectorStore):
    """Existing-database reader that never initializes or migrates schema."""

    def __init__(self, db_path: Path):
        path = Path(db_path)
        if not path.exists():
            actual = _fingerprint({"version": RUNTIME_SCHEMA_CONTRACT_VERSION, "missing_database": True})
            raise SchemaFingerprintMismatch(
                f"readonly database does not exist: {path}",
                actual_fingerprint=actual,
            )
        super().__init__(path, readonly=True)
        try:
            fingerprint, _objects, _columns = _validate_runtime_schema(self.conn.cursor())
            self.schema_fingerprint = fingerprint
        except BaseException:
            self.close()
            raise


class WriterRuntimeStore(VectorStore):
    """Existing-database writer with a bounded, schema-only open path."""

    def __init__(self, db_path: Path, *, on_connection: Callable[[apsw.Connection], None] | None = None):
        """Open the writer.

        ``on_connection`` is handed the connection the moment it exists, BEFORE the schema
        probe runs any SQL on it. Opening the store is a phase with no cap of its own -- the
        `brainlayer index` watchdog uses this to attach `sqlite3_interrupt` so a schema probe
        that runs long can actually be aborted, instead of only alarmed.
        """
        db_path = Path(db_path)
        if not db_path.exists():
            actual = _fingerprint({"version": RUNTIME_SCHEMA_CONTRACT_VERSION, "missing_database": True})
            raise SchemaFingerprintMismatch(
                f"runtime database does not exist: {db_path}",
                actual_fingerprint=actual,
            )

        self._initialize_instance_state(db_path, readonly=False, create_parent=False)
        if self._readonly:
            actual = _fingerprint({"version": RUNTIME_SCHEMA_CONTRACT_VERSION, "readonly_database": True})
            raise SchemaFingerprintMismatch(
                f"runtime database is not writable: {db_path}",
                actual_fingerprint=actual,
            )

        self._acquire_writer_pidfile()
        try:
            self._init_runtime_db(on_connection=on_connection)
        except Exception:
            self._release_writer_pidfile()
            raise

    def _init_runtime_db(self, *, on_connection: Callable[[apsw.Connection], None] | None = None) -> None:
        with _without_connection_maintenance_hooks():
            self.conn = apsw.Connection(str(self.db_path), flags=apsw.SQLITE_OPEN_READWRITE)
        if on_connection is not None:
            # Before any SQL below, so the whole probe window is interruptible. A failure here
            # must not cost us the store -- but it must not be quiet either: the caller loses
            # its ability to abort this open, which is exactly the fail-open shape the hook
            # exists to remove. Loud, and the caller's own hook is expected to record it.
            try:
                on_connection(self.conn)
            except Exception as exc:
                logger.error(
                    "on_connection hook failed; this open is NOT interruptible: %s: %s",
                    type(exc).__name__,
                    exc,
                )
        try:
            self.conn.setbusytimeout(_write_busy_timeout_ms())
            _load_vector_extension(self.conn)
            _configure_writer_pragmas(self.conn)
        except BaseException:
            self.conn.close()
            raise

        span = start_writer_span(
            self.conn,
            db_path=self.db_path,
            producer="vector_store",
            lane="runtime",
            operation="runtime_open",
            span_kind="writer_operation",
            transaction_mode="schema_probe_only",
            sample_fts=False,
        )
        try:
            cursor = self.conn.cursor()
            fingerprint, objects, chunk_columns = _validate_runtime_schema(cursor)
            self.schema_fingerprint = fingerprint
            span.add_metadata(
                schema_fingerprint=fingerprint,
                schema_contract_version=RUNTIME_SCHEMA_CONTRACT_VERSION,
            )
            self._schema_user_version = cursor.execute("PRAGMA user_version").fetchone()[0]
            self._has_chunk_origin = "chunk_origin" in chunk_columns
            self._has_content_class = "content_class" in chunk_columns
            self._has_provenance_class = "provenance_class" in chunk_columns
            self._has_source_class = "source_class" in chunk_columns
            self._has_superseded_by = "superseded_by" in chunk_columns
            self._has_invalid_at = "invalid_at" in chunk_columns
            self._binary_index_available = "chunk_vectors_binary" in objects
            self._trigram_fts_available = "chunks_fts_trigram" in objects
            self._chunk_tags_available = "chunk_tags" in objects
            self._local = threading.local()
            span.finish("completed")
        except BaseException as exc:
            actual = getattr(exc, "actual_fingerprint", None)
            if actual:
                span.add_metadata(
                    schema_fingerprint=actual,
                    schema_contract_version=RUNTIME_SCHEMA_CONTRACT_VERSION,
                )
            span.finish("error", error=f"{type(exc).__name__}: {exc}")
            self.conn.close()
            raise


class OfflineMigrator(VectorStore):
    """Legacy schema/repair store restricted to an explicit offline copy."""

    def __init__(self, db_path: Path):
        path = resolve_offline_database_path(Path(db_path))
        super().__init__(path)


def open_writer_store(
    db_path: Path,
    *,
    on_connection: Callable[[apsw.Connection], None] | None = None,
) -> WriterRuntimeStore | VectorStore:
    """Open the default runtime writer, or the guarded legacy rollback path.

    ``on_connection`` receives the writer connection as early as the path allows, so a caller
    holding a deadline can interrupt a long open. The runtime path hands it over before the
    schema probe; the legacy path can only hand it over once construction returns.
    """
    mode = os.environ.get("BRAINLAYER_RUNTIME_STORE", "runtime").strip().lower()
    if mode == "runtime":
        return WriterRuntimeStore(Path(db_path), on_connection=on_connection)
    if mode == "legacy":
        store = VectorStore(Path(db_path))
        if on_connection is not None:
            try:
                on_connection(store.conn)
            except Exception as exc:
                logger.error(
                    "on_connection hook failed; this open is NOT interruptible: %s: %s",
                    type(exc).__name__,
                    exc,
                )
        return store
    raise RuntimeStoreModeError("BRAINLAYER_RUNTIME_STORE must be 'runtime' (default) or 'legacy' for rollback")
