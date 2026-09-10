"""Standalone catalog of declared native record linkage; not wired to ingestion yet."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

MAX_DECLARED_HEADERS_BYTES = 32_768
MAX_NATIVE_UUID_LOOKUP_LIMIT = 8
MAX_PRESERVED_SEGMENT_BYTES = 4_096
_SUPPORTED_PROVIDER = "claude"
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
_DECLARED_TOP_LEVEL_HEADERS = (
    ("type", "/type"),
    ("uuid", "/uuid"),
    ("parentUuid", "/parentUuid"),
    ("logicalParentUuid", "/logicalParentUuid"),
    ("preservedSegment", "/preservedSegment"),
)


class UnsupportedNativeRecordProvider(ValueError):
    pass


class NativeRecordLinkConflict(ValueError):
    pass


@dataclass(frozen=True)
class NativeRecordLinkResult:
    record_key: str
    inserted: bool


def native_record_payload_from_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Return exact Claude structural evidence, or None when its namespace is unavailable."""
    if entry.get("_provider") != _SUPPORTED_PROVIDER:
        return None
    session_id = entry.get("sessionId")
    source_file = entry.get("_source_file")
    start_offset = entry.get("_line_start_offset")
    end_offset = entry.get("_line_end_offset")
    raw_record_hash = entry.get("_raw_record_hash")
    if (
        not isinstance(session_id, str)
        or not session_id
        or not isinstance(source_file, str)
        or not source_file
        or type(start_offset) is not int
        or type(end_offset) is not int
        or start_offset < 0
        or end_offset <= start_offset
        or not isinstance(raw_record_hash, str)
        or not _SHA256_RE.fullmatch(raw_record_hash)
    ):
        return None

    declared_headers = {pointer: entry[key] for key, pointer in _DECLARED_TOP_LEVEL_HEADERS if key in entry}
    compact_metadata = entry.get("compactMetadata")
    if isinstance(compact_metadata, dict) and "preservedSegment" in compact_metadata:
        declared_headers["/compactMetadata/preservedSegment"] = compact_metadata["preservedSegment"]
    return {
        "provider": _SUPPORTED_PROVIDER,
        "session_id": session_id,
        "source_file": source_file,
        "record_start_offset": start_offset,
        "record_end_offset": end_offset,
        "raw_record_hash": raw_record_hash,
        "declared_headers": declared_headers,
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _parent_state(headers: dict[str, Any], pointer: str) -> tuple[str, str | None]:
    if pointer not in headers:
        return "missing", None
    value = headers[pointer]
    if value is None:
        return "null", None
    if isinstance(value, str):
        return "string", value
    return "invalid", None


def _validated_headers(
    headers: Any,
) -> tuple[str, str | None, str, str | None, str, str | None, str | None, str | None, int]:
    """Return bounded pointer-qualified headers and opaque preserved-segment evidence."""
    if not isinstance(headers, dict):
        raise ValueError("declared_headers must be a JSON object keyed by JSON Pointer")
    for pointer in headers:
        if not isinstance(pointer, str) or not pointer.startswith("/"):
            raise ValueError("declared_headers keys must be JSON Pointers")
    uuid = headers.get("/uuid")
    native_uuid = uuid if isinstance(uuid, str) and uuid else None
    parent_state, parent_value = _parent_state(headers, "/parentUuid")
    logical_state, logical_value = _parent_state(headers, "/logicalParentUuid")
    segment = headers.get("/compactMetadata/preservedSegment")
    stored_headers = {key: value for key, value in headers.items() if key != "/compactMetadata/preservedSegment"}
    headers_json = _canonical_json(stored_headers)
    if len(headers_json.encode("utf-8")) > MAX_DECLARED_HEADERS_BYTES:
        raise ValueError("declared_headers exceeds byte cap")
    segment_json = None
    segment_sha256 = None
    omitted = 0
    if "/compactMetadata/preservedSegment" in headers:
        candidate = _canonical_json(segment)
        segment_sha256 = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        if len(candidate.encode("utf-8")) > MAX_PRESERVED_SEGMENT_BYTES:
            omitted = 1
        else:
            segment_json = candidate
    return (
        headers_json,
        native_uuid,
        parent_state,
        parent_value,
        logical_state,
        logical_value,
        segment_json,
        segment_sha256,
        omitted,
    )


def ensure_native_record_links_schema(conn: Any) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS native_record_links (
            record_key TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            session_id TEXT NOT NULL,
            source_file TEXT NOT NULL,
            record_start_offset INTEGER NOT NULL,
            record_end_offset INTEGER NOT NULL,
            raw_record_hash TEXT NOT NULL,
            native_uuid TEXT,
            parent_uuid_state TEXT NOT NULL,
            parent_uuid_value TEXT,
            logical_parent_uuid_state TEXT NOT NULL,
            logical_parent_uuid_value TEXT,
            declared_headers_json TEXT NOT NULL,
            preserved_segment_json TEXT,
            preserved_segment_sha256 TEXT,
            preserved_segment_omitted INTEGER NOT NULL DEFAULT 0
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_native_record_links_uuid "
        "ON native_record_links("
        "provider, session_id, native_uuid, source_file, record_start_offset, record_key)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_native_record_links_physical "
        "ON native_record_links(provider, session_id, source_file, record_start_offset, "
        "record_end_offset, raw_record_hash)"
    )


def catalog_native_record(
    conn: Any,
    *,
    provider: str,
    session_id: str,
    source_file: str,
    record_start_offset: int,
    record_end_offset: int,
    raw_record_hash: str,
    declared_headers: dict[str, Any],
) -> NativeRecordLinkResult:
    if provider != _SUPPORTED_PROVIDER:
        raise UnsupportedNativeRecordProvider(f"unsupported provider: {provider!r}")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("explicit non-empty session_id required")
    if not isinstance(source_file, str) or not source_file:
        raise ValueError("source_file locator required")
    if (
        type(record_start_offset) is not int
        or type(record_end_offset) is not int
        or record_start_offset < 0
        or record_end_offset <= record_start_offset
    ):
        raise ValueError("invalid record byte offsets")
    if not isinstance(raw_record_hash, str) or not _SHA256_RE.fullmatch(raw_record_hash):
        raise ValueError("raw_record_hash must be an exact SHA-256 hex digest")
    raw_record_hash = raw_record_hash.lower()
    try:
        (
            headers_json,
            native_uuid,
            parent_state,
            parent_value,
            logical_state,
            logical_value,
            segment_json,
            segment_sha256,
            omitted,
        ) = _validated_headers(declared_headers)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid native record headers") from exc
    identity = [provider, session_id, source_file, record_start_offset, record_end_offset, raw_record_hash]
    record_key = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    incoming = (
        provider,
        session_id,
        source_file,
        record_start_offset,
        record_end_offset,
        raw_record_hash,
        native_uuid,
        parent_state,
        parent_value,
        logical_state,
        logical_value,
        headers_json,
        segment_json,
        segment_sha256,
        omitted,
    )
    existing = conn.execute(
        "SELECT provider, session_id, source_file, record_start_offset, record_end_offset, raw_record_hash, "
        "native_uuid, parent_uuid_state, parent_uuid_value, logical_parent_uuid_state, logical_parent_uuid_value, "
        "declared_headers_json, preserved_segment_json, preserved_segment_sha256, preserved_segment_omitted "
        "FROM native_record_links WHERE record_key = ?",
        (record_key,),
    ).fetchone()
    if existing is not None:
        if tuple(existing) != incoming:
            raise NativeRecordLinkConflict("conflicting declared metadata for one physical native record")
        return NativeRecordLinkResult(record_key=record_key, inserted=False)
    conn.execute(
        "INSERT INTO native_record_links (record_key, provider, session_id, source_file, record_start_offset, "
        "record_end_offset, raw_record_hash, native_uuid, parent_uuid_state, parent_uuid_value, "
        "logical_parent_uuid_state, logical_parent_uuid_value, declared_headers_json, preserved_segment_json, "
        "preserved_segment_sha256, preserved_segment_omitted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (record_key, *incoming),
    )
    return NativeRecordLinkResult(record_key=record_key, inserted=True)


def find_native_uuid_records(
    conn: Any,
    *,
    provider: str,
    session_id: str,
    native_uuid: str,
    source_file: str | None = None,
    limit: int = 2,
) -> list[dict[str, Any]]:
    """Return a bounded, ordered ambiguity sample; two rows signal a non-unique UUID."""
    if provider != _SUPPORTED_PROVIDER:
        raise UnsupportedNativeRecordProvider(f"unsupported provider: {provider!r}")
    if type(limit) is not int or not 1 <= limit <= MAX_NATIVE_UUID_LOOKUP_LIMIT:
        raise ValueError(f"limit must be 1..{MAX_NATIVE_UUID_LOOKUP_LIMIT}")
    if source_file is not None and (not isinstance(source_file, str) or not source_file):
        raise ValueError("source_file must be a non-empty locator when supplied")
    query = (
        "SELECT record_key, source_file, record_start_offset, record_end_offset, raw_record_hash "
        "FROM native_record_links WHERE provider = ? AND session_id = ? AND native_uuid = ?"
    )
    bindings: list[Any] = [provider, session_id, native_uuid]
    if source_file is not None:
        query += " AND source_file = ?"
        bindings.append(source_file)
    query += " ORDER BY source_file, record_start_offset, record_key LIMIT ?"
    bindings.append(limit)
    rows = conn.execute(query, tuple(bindings)).fetchall()
    return [
        {
            "record_key": row[0],
            "source_file": row[1],
            "record_start_offset": row[2],
            "record_end_offset": row[3],
            "raw_record_hash": row[4],
        }
        for row in rows
    ]
