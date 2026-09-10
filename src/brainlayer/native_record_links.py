"""Standalone catalog of declared native record linkage; not wired to ingestion yet."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any

MAX_DECLARED_HEADERS_BYTES = 32_768
MAX_NATIVE_UUID_LOOKUP_LIMIT = 8
MAX_PRESERVED_SEGMENT_BYTES = 4_096
MAX_TARGET_REPORT_BYTES = 256 * 1024
TARGET_EXTRACTION_VERSION = 1
TARGET_SCOPE = "supported_tool_calls_v1"
_SUPPORTED_PROVIDERS = frozenset({"claude", "codex"})
_CHAIN_PROVIDER = "claude"
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


def native_record_entry_is_admitted(entry: dict[str, Any]) -> bool:
    """Apply the existing source/class denylist before structural catalog admission."""
    from .ingest_denylist import (
        BRAINLAYER_INGEST_DENYLIST_ENV,
        MEMORY_READER_ATTRIBUTIONS,
        is_denylisted,
    )

    source_file = entry.get("_source_file")
    if not isinstance(source_file, str) or not source_file:
        return False
    if BRAINLAYER_INGEST_DENYLIST_ENV in os.environ:
        return not is_denylisted(source_file)

    raw_attribution = entry.get("attributionAgent")
    attribution = raw_attribution.strip().casefold().replace("_", "-") if isinstance(raw_attribution, str) else ""
    if attribution in MEMORY_READER_ATTRIBUTIONS:
        return False
    return not is_denylisted(
        source_file,
        unknown_subagent_is_denylisted=not bool(attribution),
    )


def native_record_payload_from_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Return exact Claude structural evidence, or None when its namespace is unavailable."""
    if entry.get("_provider") != _CHAIN_PROVIDER:
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
        "provider": _CHAIN_PROVIDER,
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


def _validated_target_report(report: Any) -> tuple[int, str, str]:
    """Accept only the bounded, body-free output contract of native_targets."""
    if report is None:
        return 0, "[]", "[]"
    if not isinstance(report, dict) or set(report) != {"version", "scope", "targets", "diagnostics"}:
        raise ValueError("target report has unsupported fields")
    if (
        type(report["version"]) is not int
        or report["version"] != TARGET_EXTRACTION_VERSION
        or report["scope"] != TARGET_SCOPE
    ):
        raise ValueError("unsupported target report version or scope")
    targets, diagnostics = report["targets"], report["diagnostics"]
    if not isinstance(targets, list) or not isinstance(diagnostics, list):
        raise ValueError("target report targets and diagnostics must be lists")
    if len(targets) > 25 or len(diagnostics) > 32:
        raise ValueError("target report exceeds helper list cap")
    target_keys = {
        "tool_name",
        "target",
        "target_kind",
        "declared_unverified",
        "call_id",
        "raw_container_pointer",
        "decoded_pointer",
        "pointer_kind",
    }
    required_string_keys = {"tool_name", "target", "target_kind", "decoded_pointer", "pointer_kind"}
    nullable_string_keys = {"call_id", "raw_container_pointer"}
    for target in targets:
        if not isinstance(target, dict) or set(target) != target_keys or target.get("declared_unverified") is not True:
            raise ValueError("target report target has unsupported fields")
        for key in required_string_keys:
            value = target[key]
            if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 4_096:
                raise ValueError("target report target contains invalid required value")
        for key in nullable_string_keys:
            value = target[key]
            if value is not None and (not isinstance(value, str) or not value or len(value.encode("utf-8")) > 4_096):
                raise ValueError("target report target contains invalid nullable value")
        if target["target_kind"] not in {"agent_handle", "task_handle", "surface"}:
            raise ValueError("target report target has unsupported target kind")
        if target["pointer_kind"] not in {"direct_raw_json", "decoded_json_string"}:
            raise ValueError("target report target has unsupported pointer kind")
    if any(not isinstance(item, str) or not item or len(item.encode("utf-8")) > 4_096 for item in diagnostics):
        raise ValueError("target report contains invalid diagnostic")
    targets_json, diagnostics_json = _canonical_json(targets), _canonical_json(diagnostics)
    if len((targets_json + diagnostics_json).encode("utf-8")) > MAX_TARGET_REPORT_BYTES:
        raise ValueError("target report exceeds byte cap")
    return TARGET_EXTRACTION_VERSION, targets_json, diagnostics_json


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
            preserved_segment_omitted INTEGER NOT NULL DEFAULT 0,
            target_extraction_version INTEGER NOT NULL DEFAULT 0,
            declared_targets_json TEXT NOT NULL DEFAULT '[]',
            target_diagnostics_json TEXT NOT NULL DEFAULT '[]'
        )"""
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(native_record_links)")}
    for name, definition in (
        ("target_extraction_version", "INTEGER NOT NULL DEFAULT 0"),
        ("declared_targets_json", "TEXT NOT NULL DEFAULT '[]'"),
        ("target_diagnostics_json", "TEXT NOT NULL DEFAULT '[]'"),
    ):
        if name not in columns:
            conn.execute(f"ALTER TABLE native_record_links ADD COLUMN {name} {definition}")
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
    target_report: dict[str, Any] | None = None,
) -> NativeRecordLinkResult:
    if provider not in _SUPPORTED_PROVIDERS:
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
        headers = _validated_headers(declared_headers)
        derived = _validated_target_report(target_report)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid native record headers or target report") from exc
    identity = [provider, session_id, source_file, record_start_offset, record_end_offset, raw_record_hash]
    record_key = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
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
    ) = headers
    core = (
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
        "SELECT provider,session_id,source_file,record_start_offset,record_end_offset,raw_record_hash,"
        "native_uuid,parent_uuid_state,parent_uuid_value,logical_parent_uuid_state,logical_parent_uuid_value,"
        "declared_headers_json,preserved_segment_json,preserved_segment_sha256,preserved_segment_omitted,"
        "target_extraction_version,declared_targets_json,target_diagnostics_json FROM native_record_links WHERE record_key=?",
        (record_key,),
    ).fetchone()
    if existing is not None:
        if tuple(existing[:15]) != core:
            raise NativeRecordLinkConflict("conflicting declared metadata for one physical native record")
        existing_version = int(existing[15])
        if target_report is None:
            return NativeRecordLinkResult(record_key=record_key, inserted=False)
        if existing_version > TARGET_EXTRACTION_VERSION:
            raise NativeRecordLinkConflict("refusing to overwrite newer target extraction version")
        if existing_version == TARGET_EXTRACTION_VERSION:
            if tuple(existing[15:]) != derived:
                raise NativeRecordLinkConflict("conflicting target evidence for one physical native record")
            return NativeRecordLinkResult(record_key=record_key, inserted=False)
        conn.execute(
            "UPDATE native_record_links SET target_extraction_version=?,declared_targets_json=?,target_diagnostics_json=? WHERE record_key=?",
            (*derived, record_key),
        )
        return NativeRecordLinkResult(record_key=record_key, inserted=False)
    conn.execute(
        "INSERT INTO native_record_links (record_key,provider,session_id,source_file,record_start_offset,record_end_offset,raw_record_hash,native_uuid,parent_uuid_state,parent_uuid_value,logical_parent_uuid_state,logical_parent_uuid_value,declared_headers_json,preserved_segment_json,preserved_segment_sha256,preserved_segment_omitted,target_extraction_version,declared_targets_json,target_diagnostics_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (record_key, *core, *derived),
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
    if provider != _CHAIN_PROVIDER:
        raise UnsupportedNativeRecordProvider("native UUID chain lookup supports Claude only")
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
