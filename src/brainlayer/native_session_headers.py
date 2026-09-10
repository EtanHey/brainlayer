"""Independent physical catalog for provider session headers.

Stored fields are provider-recorded metadata. They do not authenticate a sender,
human, or original author.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .raw_source_archive import RawSourceArchive

MAX_HEADER_RECORD_BYTES = 256 * 1024
MAX_CONTEXT_BYTES = 32 * 1024
MAX_RECEIPT_BYTES = 64 * 1024
MAX_SCALAR_BYTES = 4 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RECEIPT_KEYS = {
    "manifest_sha256",
    "manifest_path",
    "source_path",
    "provider",
    "sha256",
    "size",
    "source_stat",
}
_PAYLOAD_STRINGS = ("id", "session_id", "parent_thread_id", "agent_path", "agent_nickname", "agent_role")
_SPAWN_STRINGS = ("parent_thread_id", "agent_path", "agent_nickname", "agent_role")
_GENERATION_TABLE = "native_source_generations"
_OCCURRENCE_TABLE = "native_session_header_occurrences"


class NativeHeaderCatalogError(ValueError):
    pass


class NativeSourceGenerationConflict(NativeHeaderCatalogError):
    pass


@dataclass(frozen=True)
class NativeHeaderCatalogResult:
    generation_key: str
    occurrence_key: str
    disposition: str
    generation_status: str
    session_id: str | None
    inserted: bool


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


@contextmanager
def _savepoint(conn: Any) -> Iterator[None]:
    conn.execute("SAVEPOINT native_header_catalog")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK TO native_header_catalog")
        conn.execute("RELEASE native_header_catalog")
        raise
    conn.execute("RELEASE native_header_catalog")


def ensure_native_session_headers_schema(conn: Any) -> None:
    generation_exists = _table_exists(conn, _GENERATION_TABLE)
    occurrence_exists = _table_exists(conn, _OCCURRENCE_TABLE)
    if generation_exists != occurrence_exists:
        raise NativeHeaderCatalogError("native header catalog schema is incomplete")
    if not generation_exists:
        conn.execute(
            """CREATE TABLE native_source_generations (
        generation_key TEXT PRIMARY KEY, provider TEXT NOT NULL, source_file TEXT NOT NULL,
        watcher_generation INTEGER NOT NULL, root_manifest_sha256 TEXT NOT NULL,
        root_receipt_json TEXT NOT NULL, current_manifest_sha256 TEXT NOT NULL,
        current_receipt_json TEXT NOT NULL, status TEXT NOT NULL,
        canonical_occurrence_key TEXT, first_seen_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        UNIQUE(provider, source_file, watcher_generation, root_manifest_sha256),
        CHECK(watcher_generation >= 0), CHECK(status IN ('active','conflicted','invalid')))"""
        )
        conn.execute(
            """CREATE TABLE native_session_header_occurrences (
        occurrence_key TEXT PRIMARY KEY, generation_key TEXT NOT NULL,
        disposition TEXT NOT NULL, manifest_sha256 TEXT NOT NULL,
        record_start_offset INTEGER NOT NULL, record_end_offset INTEGER NOT NULL,
        raw_record_hash TEXT NOT NULL, session_id TEXT, header_context_json TEXT,
        header_context_sha256 TEXT, validation_error TEXT, observed_at TEXT NOT NULL,
        UNIQUE(generation_key, record_start_offset, record_end_offset, raw_record_hash),
        FOREIGN KEY(generation_key) REFERENCES native_source_generations(generation_key),
        CHECK(disposition IN ('canonical','repeat','conflict','invalid')))"""
        )
        conn.execute(
            "CREATE INDEX idx_native_source_generations_source "
            "ON native_source_generations(provider,source_file,watcher_generation,status,generation_key)"
        )
        conn.execute(
            "CREATE INDEX idx_native_session_headers_generation "
            "ON native_session_header_occurrences(generation_key,disposition,record_start_offset,occurrence_key)"
        )
        conn.execute(
            "CREATE UNIQUE INDEX idx_native_session_headers_canonical "
            "ON native_session_header_occurrences(generation_key) "
            "WHERE disposition = 'canonical'"
        )
        return
    _validate_native_header_catalog_schema(conn)


def _table_exists(conn: Any, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table,),
        ).fetchone()
        is not None
    )


def _table_columns(conn: Any, table: str) -> list[tuple[str, str, int, Any, int]]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [(str(row[1]), str(row[2]).upper(), int(row[3]), row[4], int(row[5])) for row in rows]


def _table_sql(conn: Any, table: str) -> str:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    if row is None or not isinstance(row[0], str):
        raise NativeHeaderCatalogError(f"native header catalog table missing: {table}")
    return re.sub(r"\s+", "", row[0].lower())


def _require_unique_constraint(conn: Any, table: str, columns: tuple[str, ...]) -> None:
    for row in conn.execute(f"PRAGMA index_list({table})").fetchall():
        if int(row[2]) != 1 or str(row[3]) != "u":
            continue
        name = str(row[1])
        actual = tuple(str(item[2]) for item in conn.execute(f"PRAGMA index_info({name})").fetchall())
        if actual == columns:
            return
    raise NativeHeaderCatalogError(f"native header catalog unique constraint mismatch: {table} {columns!r}")


def _require_index(
    conn: Any,
    table: str,
    name: str,
    columns: tuple[str, ...],
    *,
    unique: bool,
    partial: bool,
    sql: str | None = None,
) -> None:
    row = next((item for item in conn.execute(f"PRAGMA index_list({table})").fetchall() if item[1] == name), None)
    if row is None:
        raise NativeHeaderCatalogError(f"native header catalog index missing: {name}")
    if bool(row[2]) != unique or bool(row[4]) != partial:
        raise NativeHeaderCatalogError(f"native header catalog index flags mismatch: {name}")
    actual = tuple(str(item[2]) for item in conn.execute(f"PRAGMA index_info({name})").fetchall())
    if actual != columns:
        raise NativeHeaderCatalogError(f"native header catalog index columns mismatch: {name}")
    if sql is not None:
        definition = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name=?",
            (name,),
        ).fetchone()
        compact = re.sub(r"\s+", "", str(definition[0]).lower()) if definition else ""
        if compact != re.sub(r"\s+", "", sql.lower()):
            raise NativeHeaderCatalogError(f"native header catalog index predicate mismatch: {name}")


def _validate_native_header_catalog_schema(conn: Any) -> None:
    generation_columns = [
        ("generation_key", "TEXT", 0, None, 1),
        ("provider", "TEXT", 1, None, 0),
        ("source_file", "TEXT", 1, None, 0),
        ("watcher_generation", "INTEGER", 1, None, 0),
        ("root_manifest_sha256", "TEXT", 1, None, 0),
        ("root_receipt_json", "TEXT", 1, None, 0),
        ("current_manifest_sha256", "TEXT", 1, None, 0),
        ("current_receipt_json", "TEXT", 1, None, 0),
        ("status", "TEXT", 1, None, 0),
        ("canonical_occurrence_key", "TEXT", 0, None, 0),
        ("first_seen_at", "TEXT", 1, None, 0),
        ("updated_at", "TEXT", 1, None, 0),
    ]
    occurrence_columns = [
        ("occurrence_key", "TEXT", 0, None, 1),
        ("generation_key", "TEXT", 1, None, 0),
        ("disposition", "TEXT", 1, None, 0),
        ("manifest_sha256", "TEXT", 1, None, 0),
        ("record_start_offset", "INTEGER", 1, None, 0),
        ("record_end_offset", "INTEGER", 1, None, 0),
        ("raw_record_hash", "TEXT", 1, None, 0),
        ("session_id", "TEXT", 0, None, 0),
        ("header_context_json", "TEXT", 0, None, 0),
        ("header_context_sha256", "TEXT", 0, None, 0),
        ("validation_error", "TEXT", 0, None, 0),
        ("observed_at", "TEXT", 1, None, 0),
    ]
    if _table_columns(conn, _GENERATION_TABLE) != generation_columns:
        raise NativeHeaderCatalogError("native header catalog generation columns mismatch")
    if _table_columns(conn, _OCCURRENCE_TABLE) != occurrence_columns:
        raise NativeHeaderCatalogError("native header catalog occurrence columns mismatch")
    generation_sql = _table_sql(conn, _GENERATION_TABLE)
    occurrence_sql = _table_sql(conn, _OCCURRENCE_TABLE)
    for expression in ("check(watcher_generation>=0)", "check(statusin('active','conflicted','invalid'))"):
        if expression not in generation_sql:
            raise NativeHeaderCatalogError("native header catalog generation constraints mismatch")
    if "check(dispositionin('canonical','repeat','conflict','invalid'))" not in occurrence_sql:
        raise NativeHeaderCatalogError("native header catalog occurrence constraints mismatch")
    foreign_keys = [
        (str(row[2]), str(row[3]), str(row[4]), str(row[5]), str(row[6]))
        for row in conn.execute(f"PRAGMA foreign_key_list({_OCCURRENCE_TABLE})").fetchall()
    ]
    if foreign_keys != [("native_source_generations", "generation_key", "generation_key", "NO ACTION", "NO ACTION")]:
        raise NativeHeaderCatalogError("native header catalog foreign key mismatch")
    _require_unique_constraint(
        conn,
        _GENERATION_TABLE,
        ("provider", "source_file", "watcher_generation", "root_manifest_sha256"),
    )
    _require_unique_constraint(
        conn,
        _OCCURRENCE_TABLE,
        ("generation_key", "record_start_offset", "record_end_offset", "raw_record_hash"),
    )
    _require_index(
        conn,
        _GENERATION_TABLE,
        "idx_native_source_generations_source",
        ("provider", "source_file", "watcher_generation", "status", "generation_key"),
        unique=False,
        partial=False,
    )
    _require_index(
        conn,
        _OCCURRENCE_TABLE,
        "idx_native_session_headers_generation",
        ("generation_key", "disposition", "record_start_offset", "occurrence_key"),
        unique=False,
        partial=False,
    )
    _require_index(
        conn,
        _OCCURRENCE_TABLE,
        "idx_native_session_headers_canonical",
        ("generation_key",),
        unique=True,
        partial=True,
        sql=(
            "CREATE UNIQUE INDEX idx_native_session_headers_canonical "
            "ON native_session_header_occurrences(generation_key) "
            "WHERE disposition = 'canonical'"
        ),
    )


def _verified_receipt(
    archive: RawSourceArchive,
    receipt: Any,
    provider: str,
    source_file: str,
) -> str:
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_KEYS:
        raise NativeHeaderCatalogError("raw archive receipt has invalid shape")
    manifest_sha = receipt.get("manifest_sha256")
    expected_manifest = archive.manifests / f"{manifest_sha}.json"
    if (
        not isinstance(manifest_sha, str)
        or _SHA256.fullmatch(manifest_sha) is None
        or receipt.get("provider") != provider
        or receipt.get("source_path") != source_file
        or receipt.get("manifest_path") != str(expected_manifest)
    ):
        raise NativeHeaderCatalogError("raw archive receipt identity mismatch")
    try:
        raw = _canonical(receipt)
    except (TypeError, ValueError) as error:
        raise NativeHeaderCatalogError("raw archive receipt is not canonical JSON data") from error
    if len(raw.encode()) > MAX_RECEIPT_BYTES:
        raise NativeHeaderCatalogError("raw archive receipt exceeds byte cap")
    with archive.open_snapshot(receipt):
        pass
    return raw


def _generation_key(provider: str, source_file: str, watcher_generation: int, root_manifest: str) -> str:
    return hashlib.sha256(
        _canonical(["native-source-generation-v1", provider, source_file, watcher_generation, root_manifest]).encode()
    ).hexdigest()


def _reserve(
    conn: Any,
    archive: RawSourceArchive,
    *,
    provider: str,
    source_file: str,
    watcher_generation: int,
    root_receipt: dict[str, Any],
    current_receipt: dict[str, Any],
) -> str:
    if provider != "codex":
        raise NativeHeaderCatalogError("native session header provider must be codex")
    if source_file != str(Path(source_file).absolute()):
        raise NativeHeaderCatalogError("source_file must be absolute")
    if type(watcher_generation) is not int or watcher_generation < 0:
        raise NativeHeaderCatalogError("watcher_generation must be a nonnegative integer")
    root_json = _verified_receipt(archive, root_receipt, provider, source_file)
    current_json = _verified_receipt(archive, current_receipt, provider, source_file)
    if not archive.is_prefix(root_receipt, current_receipt):
        raise NativeSourceGenerationConflict("current receipt does not extend generation root")
    key = _generation_key(provider, source_file, watcher_generation, root_receipt["manifest_sha256"])
    row = conn.execute(
        "SELECT provider,source_file,watcher_generation,root_manifest_sha256,root_receipt_json,"
        "current_receipt_json FROM native_source_generations WHERE generation_key=?",
        (key,),
    ).fetchone()
    now = datetime.now(timezone.utc).isoformat()
    if row is None:
        conn.execute(
            """INSERT INTO native_source_generations (
                generation_key, provider, source_file, watcher_generation,
                root_manifest_sha256, root_receipt_json, current_manifest_sha256,
                current_receipt_json, status, canonical_occurrence_key,
                first_seen_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                key,
                provider,
                source_file,
                watcher_generation,
                root_receipt["manifest_sha256"],
                root_json,
                current_receipt["manifest_sha256"],
                current_json,
                "active",
                None,
                now,
                now,
            ),
        )
        return key
    if tuple(row[:5]) != (provider, source_file, watcher_generation, root_receipt["manifest_sha256"], root_json):
        raise NativeSourceGenerationConflict("stored generation root differs")
    stored_current = json.loads(row[5])
    if current_json == row[5] or archive.is_prefix(current_receipt, stored_current):
        return key
    if not archive.is_prefix(stored_current, current_receipt):
        raise NativeSourceGenerationConflict("current receipt diverges from stored generation")
    conn.execute(
        "UPDATE native_source_generations SET current_manifest_sha256=?,current_receipt_json=?,updated_at=? "
        "WHERE generation_key=?",
        (current_receipt["manifest_sha256"], current_json, now, key),
    )
    return key


def reserve_native_source_generation(
    conn: Any,
    archive: RawSourceArchive,
    *,
    provider: str,
    source_file: str,
    watcher_generation: int,
    root_receipt: dict[str, Any],
    current_receipt: dict[str, Any],
) -> str:
    ensure_native_session_headers_schema(conn)
    with _savepoint(conn):
        return _reserve(
            conn,
            archive,
            provider=provider,
            source_file=source_file,
            watcher_generation=watcher_generation,
            root_receipt=root_receipt,
            current_receipt=current_receipt,
        )


def _record_bytes(
    archive: RawSourceArchive,
    receipt: dict[str, Any],
    start: int,
    end: int,
    digest: str,
) -> bytes:
    if (
        type(start) is not int
        or type(end) is not int
        or start < 0
        or end <= start
        or end - start > MAX_HEADER_RECORD_BYTES
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or type(receipt.get("size")) is not int
        or end > receipt["size"]
    ):
        raise NativeHeaderCatalogError("invalid physical header bounds or digest")
    with archive.open_snapshot(receipt) as source:
        if start:
            source.seek(start - 1)
            if source.read(1) != b"\n":
                raise NativeHeaderCatalogError("header does not begin at an LF record boundary")
        source.seek(start)
        raw = source.read(end - start)
    if not raw.endswith(b"\n") or hashlib.sha256(raw).hexdigest() != digest:
        raise NativeHeaderCatalogError("header bytes do not match LF bounds and digest")
    return raw


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _context(raw: bytes) -> tuple[dict[str, Any] | None, str | None, str | None]:
    try:
        line = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite {value}")),
        )
        if not isinstance(line, dict) or line.get("type") != "session_meta":
            raise ValueError("not_session_meta")
        payload = line.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("invalid_payload")
        context: dict[str, Any] = {}

        def add_string(pointer: str, container: dict[str, Any], field: str) -> None:
            if field not in container or container[field] is None:
                return
            value = container[field]
            if not isinstance(value, str) or not value or "\x00" in value or len(value.encode()) > MAX_SCALAR_BYTES:
                raise ValueError(f"invalid_{field}")
            context[pointer] = value

        for field in _PAYLOAD_STRINGS:
            add_string(f"/payload/{field}", payload, field)
        session_id = context.get("/payload/id")
        if not isinstance(session_id, str) or str(uuid.UUID(session_id)) != session_id.lower():
            raise ValueError("invalid_session_id")
        source = payload.get("source")
        if source is not None:
            if not isinstance(source, dict):
                raise ValueError("invalid_source")
            subagent = source.get("subagent")
            if subagent is not None:
                if not isinstance(subagent, dict):
                    raise ValueError("invalid_subagent")
                spawn = subagent.get("thread_spawn")
                if spawn is not None:
                    if not isinstance(spawn, dict):
                        raise ValueError("invalid_thread_spawn")
                    for field in _SPAWN_STRINGS:
                        add_string(f"/payload/source/subagent/thread_spawn/{field}", spawn, field)
                    if "depth" in spawn:
                        depth = spawn["depth"]
                        if type(depth) is not int or not 0 <= depth <= 128:
                            raise ValueError("invalid_depth")
                        context["/payload/source/subagent/thread_spawn/depth"] = depth
        encoded = _canonical(context)
        if len(encoded.encode()) > MAX_CONTEXT_BYTES:
            raise ValueError("context_byte_cap")
        return context, None, session_id
    except (UnicodeDecodeError, ValueError, TypeError, OverflowError) as error:
        return None, str(error)[:128] or type(error).__name__, None


def catalog_native_session_header(
    conn: Any,
    archive: RawSourceArchive,
    *,
    provider: str,
    source_file: str,
    watcher_generation: int,
    root_receipt: dict[str, Any],
    occurrence_receipt: dict[str, Any],
    record_start_offset: int,
    record_end_offset: int,
    raw_record_hash: str,
) -> NativeHeaderCatalogResult:
    ensure_native_session_headers_schema(conn)
    _verified_receipt(archive, occurrence_receipt, provider, source_file)
    raw = _record_bytes(archive, occurrence_receipt, record_start_offset, record_end_offset, raw_record_hash)
    context, error, session_id = _context(raw)
    context_json = _canonical(context) if context is not None else None
    context_sha = hashlib.sha256(context_json.encode()).hexdigest() if context_json is not None else None
    with _savepoint(conn):
        generation_key = _reserve(
            conn,
            archive,
            provider=provider,
            source_file=source_file,
            watcher_generation=watcher_generation,
            root_receipt=root_receipt,
            current_receipt=occurrence_receipt,
        )
        occurrence_key = hashlib.sha256(
            _canonical(
                [
                    "native-session-header-occurrence-v1",
                    generation_key,
                    record_start_offset,
                    record_end_offset,
                    raw_record_hash,
                ]
            ).encode()
        ).hexdigest()
        existing = conn.execute(
            "SELECT disposition,session_id,header_context_json,header_context_sha256,validation_error "
            "FROM native_session_header_occurrences WHERE occurrence_key=?",
            (occurrence_key,),
        ).fetchone()
        if existing is not None:
            status = conn.execute(
                "SELECT status FROM native_source_generations WHERE generation_key=?",
                (generation_key,),
            ).fetchone()[0]
            if tuple(existing[1:]) != (session_id, context_json, context_sha, error):
                raise NativeHeaderCatalogError("stored physical header occurrence differs")
            return NativeHeaderCatalogResult(generation_key, occurrence_key, existing[0], status, session_id, False)
        generation = conn.execute(
            "SELECT status,canonical_occurrence_key FROM native_source_generations WHERE generation_key=?",
            (generation_key,),
        ).fetchone()
        if error is not None:
            disposition, next_status = "invalid", "invalid"
        elif generation[1] is None:
            disposition = "canonical" if generation[0] == "active" else "conflict"
            next_status = generation[0]
        else:
            canonical_sha = conn.execute(
                "SELECT header_context_sha256 FROM native_session_header_occurrences WHERE occurrence_key=?",
                (generation[1],),
            ).fetchone()[0]
            disposition = "repeat" if canonical_sha == context_sha else "conflict"
            next_status = generation[0] if disposition == "repeat" or generation[0] == "invalid" else "conflicted"
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT INTO native_session_header_occurrences (
                occurrence_key, generation_key, disposition, manifest_sha256,
                record_start_offset, record_end_offset, raw_record_hash,
                session_id, header_context_json, header_context_sha256,
                validation_error, observed_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                occurrence_key,
                generation_key,
                disposition,
                occurrence_receipt["manifest_sha256"],
                record_start_offset,
                record_end_offset,
                raw_record_hash,
                session_id,
                context_json,
                context_sha,
                error,
                now,
            ),
        )
        canonical_key = occurrence_key if disposition == "canonical" else generation[1]
        conn.execute(
            "UPDATE native_source_generations SET status=?,canonical_occurrence_key=?,updated_at=? "
            "WHERE generation_key=?",
            (next_status, canonical_key, now, generation_key),
        )
        return NativeHeaderCatalogResult(generation_key, occurrence_key, disposition, next_status, session_id, True)


def native_source_generation_status(conn: Any, generation_key: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT provider,source_file,watcher_generation,root_manifest_sha256,"
        "current_manifest_sha256,status,canonical_occurrence_key "
        "FROM native_source_generations WHERE generation_key=?",
        (generation_key,),
    ).fetchone()
    if row is None:
        return None
    keys = (
        "provider",
        "source_file",
        "watcher_generation",
        "root_manifest_sha256",
        "current_manifest_sha256",
        "status",
        "canonical_occurrence_key",
    )
    return dict(zip(keys, row, strict=True))
