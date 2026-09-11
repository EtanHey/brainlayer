from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any


def _canonical_context_maps(raw: Any) -> list[dict[str, Any]]:
    """Return distinct declared context maps without interpreting their fields."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    by_json: dict[str, dict[str, Any]] = {}
    for value in raw:
        if not isinstance(value, dict):
            continue
        try:
            canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            continue
        by_json[canonical] = value
    return [by_json[key] for key in sorted(by_json)]


def canonical_record_context_json(existing: Any, incoming: Any = None) -> str:
    """Encode the union of complete declared maps as a stable JSON array."""
    maps = _canonical_context_maps(existing) + _canonical_context_maps(incoming)
    unique = _canonical_context_maps(maps)
    return json.dumps(unique, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def ensure_chunk_source_records(conn: Any, *, migrate: bool = True) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS chunk_source_records (
        record_key TEXT PRIMARY KEY, canonical_chunk_id TEXT NOT NULL,
        source_file TEXT NOT NULL, record_start_offset INTEGER NOT NULL,
        record_end_offset INTEGER NOT NULL, raw_record_hash TEXT NOT NULL,
        session_id TEXT, observed_at TEXT NOT NULL,
        record_context_json TEXT NOT NULL DEFAULT '[]',
        native_generation_key TEXT,
        native_provider TEXT,
        native_watcher_generation INTEGER,
        native_record_role TEXT,
        native_occurrence_key TEXT,
        native_record_manifest_sha256 TEXT,
        FOREIGN KEY(canonical_chunk_id) REFERENCES chunks(id) ON DELETE CASCADE)""")
    if migrate:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(chunk_source_records)")}
        if "record_context_json" not in columns:
            conn.execute("ALTER TABLE chunk_source_records ADD COLUMN record_context_json TEXT NOT NULL DEFAULT '[]'")
        for column, definition in (
            ("native_generation_key", "TEXT"),
            ("native_provider", "TEXT"),
            ("native_watcher_generation", "INTEGER"),
            ("native_record_role", "TEXT"),
            ("native_occurrence_key", "TEXT"),
            ("native_record_manifest_sha256", "TEXT"),
        ):
            if column not in columns:
                conn.execute(f"ALTER TABLE chunk_source_records ADD COLUMN {column} {definition}")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunk_source_records_physical "
            "ON chunk_source_records("
            "source_file, record_start_offset, record_end_offset, raw_record_hash, session_id, record_key"
            ")"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunk_source_records_native_binding "
            "ON chunk_source_records(native_generation_key, native_provider, native_watcher_generation, record_key)"
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunk_source_records_canonical ON chunk_source_records(canonical_chunk_id)"
    )


def record_watcher_source(conn: Any, *, canonical_chunk_id: str, source_file: str, metadata: dict[str, Any]) -> None:
    start, end, digest = (
        metadata.get("_line_start_offset"),
        metadata.get("_line_end_offset"),
        metadata.get("_raw_record_hash"),
    )
    if (
        type(start) is not int
        or type(end) is not int
        or start < 0
        or end <= start
        or not isinstance(digest, str)
        or len(digest) != 64
    ):
        return
    try:
        int(digest, 16)
    except ValueError:
        return
    context_json = canonical_record_context_json(metadata.get("record_context"))
    key = hashlib.sha256(
        json.dumps([canonical_chunk_id, source_file, start, end, digest], separators=(",", ":")).encode()
    ).hexdigest()
    row = conn.execute("SELECT record_context_json FROM chunk_source_records WHERE record_key = ?", (key,)).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO chunk_source_records (record_key, canonical_chunk_id, source_file, record_start_offset, record_end_offset, raw_record_hash, session_id, observed_at, record_context_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                canonical_chunk_id,
                source_file,
                start,
                end,
                digest,
                metadata.get("session_id") if isinstance(metadata.get("session_id"), str) else None,
                datetime.now(timezone.utc).isoformat(),
                context_json,
            ),
        )
    else:
        merged = canonical_record_context_json(row[0], context_json)
        if merged != row[0]:
            conn.execute("UPDATE chunk_source_records SET record_context_json = ? WHERE record_key = ?", (merged, key))
