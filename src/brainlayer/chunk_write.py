"""Canonical chunk insert contract.

Every production writer must persist the same required columns. Writer-specific
fields (conversation/position/sender, watcher offsets) stay optional.
Enrichment-only fields are never required at insert time.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .agent_provenance import resolve_source_class
from .chunk_origin import detect_chunk_origin
from .content_class import classify_content_class, normalize_content_class
from .dedupe import compute_dedupe_fields

CANONICAL_INSERT_COLUMNS = frozenset(
    {
        "id",
        "content",
        "metadata",
        "source_file",
        "project",
        "content_type",
        "value_type",
        "char_count",
        "source",
        "created_at",
        "chunk_origin",
        "content_hash",
        "ingested_at",
        "seen_count",
        "last_seen_at",
        "content_class",
        "preview_text",
        "brick_id",
        "source_uri",
        "status",
        "dedupe_hash",
        "simhash",
        "simhash_band_0",
        "simhash_band_1",
        "simhash_band_2",
        "simhash_band_3",
    }
)

WRITER_SPECIFIC_COLUMNS = frozenset(
    {
        "conversation_id",
        "position",
        "sender",
        "source_end_offset",
        "source_last_queued_at",
        "source_class",
        "provenance_class",
        "tags",
        "importance",
        "half_life_days",
        "topic_cluster",
    }
)

ENRICHMENT_ONLY_COLUMNS = frozenset(
    {
        "summary",
        "intent",
        "key_facts",
        "resolved_query",
        "resolved_queries",
        "enriched_at",
        "enrich_status",
        "enrichment_model",
        "enrichment_backend",
        "enrichment_version",
        "epistemic_level",
        "version_scope",
        "debt_impact",
        "external_deps",
        "sentiment_label",
        "sentiment_score",
        "sentiment_signals",
        "primary_symbols",
        "raw_entities_json",
        "tag_confidence",
        "summary_v2",
    }
)

_ISO_CANDIDATE_COLUMNS = (
    "created_at",
    "last_seen_at",
    "archived_at",
    "enriched_at",
    "valid_from",
    "invalid_at",
    "sys_period_start",
    "sys_period_end",
)


def _now_iso(now: datetime | None = None) -> str:
    stamp = now or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _preview_text(values: Mapping[str, Any]) -> str:
    summary = str(values.get("summary") or "").strip()
    content = str(values.get("content") or "").strip()
    source = summary or content
    return source.replace("\n", " ").replace("\r", " ").replace("\t", " ")[:220]


def _json_text(value: Any, *, default: str = "{}") -> str:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    text = str(value)
    return text if text else default


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _table_columns(conn: Any, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in rows}


def prepare_canonical_insert(
    values: Mapping[str, Any],
    *,
    now: datetime | None = None,
    columns: Iterable[str] | None = None,
    ingested_at: int | None = None,
) -> dict[str, Any]:
    """Fill required insert columns. Does not write."""
    from .timestamp_iso import normalize_timestamp

    content = str(values.get("content") or "")
    if not content.strip():
        raise ValueError("content must be non-empty")
    chunk_id = str(values.get("id") or "").strip()
    if not chunk_id:
        raise ValueError("id is required")

    created_at = normalize_timestamp(values.get("created_at")) or _now_iso(now)
    last_seen_at = normalize_timestamp(values.get("last_seen_at")) or created_at
    source_file = values.get("source_file") or "unknown"
    source = values.get("source") or "unknown"
    tags = values.get("tags")
    content_class = normalize_content_class(values.get("content_class"))
    if values.get("content_class") in (None, ""):
        content_class = classify_content_class(
            content,
            content_type=values.get("content_type"),
            tags=tags,
            source=source,
            source_file=str(source_file),
            project=values.get("project"),
        )
    fields = compute_dedupe_fields(content, created_at)
    resolved_origin = detect_chunk_origin(content, values.get("chunk_origin"))
    source_class = resolve_source_class(
        str(source_file),
        supplied_source_class=values.get("source_class"),
        provenance_class=values.get("provenance_class"),
        source=str(source) if source is not None else None,
    )
    epoch = int(ingested_at if ingested_at is not None else time.time())
    if values.get("ingested_at") is not None:
        try:
            epoch = int(float(values["ingested_at"]))
        except (TypeError, ValueError):
            epoch = int(time.time())

    prepared: dict[str, Any] = dict(values)
    prepared.update(
        {
            "id": chunk_id,
            "content": content,
            "metadata": _json_text(values.get("metadata")),
            "source_file": source_file,
            "project": values.get("project"),
            "content_type": values.get("content_type") or "note",
            "value_type": values.get("value_type") or "high",
            "char_count": int(values.get("char_count") or len(content)),
            "source": source,
            "created_at": created_at,
            "chunk_origin": resolved_origin,
            "content_hash": values.get("content_hash") or _content_hash(content),
            "ingested_at": epoch,
            "seen_count": int(values.get("seen_count") or 1),
            "last_seen_at": last_seen_at,
            "content_class": content_class,
            "preview_text": str(values.get("preview_text") or "").strip() or _preview_text(values),
            "brick_id": values.get("brick_id") or chunk_id,
            "source_uri": values.get("source_uri") or source_file,
            "status": values.get("status") or "active",
            "dedupe_hash": values.get("dedupe_hash") or fields.dedupe_hash,
            "simhash": values.get("simhash") or fields.simhash,
            "simhash_band_0": values.get("simhash_band_0") or fields.bands[0],
            "simhash_band_1": values.get("simhash_band_1") or fields.bands[1],
            "simhash_band_2": values.get("simhash_band_2") or fields.bands[2],
            "simhash_band_3": values.get("simhash_band_3") or fields.bands[3],
        }
    )
    if source_class is not None:
        prepared["source_class"] = source_class
    if prepared.get("tags") is not None and not isinstance(prepared["tags"], str):
        prepared["tags"] = json.dumps(prepared["tags"])
    if "valid_from" not in prepared or prepared.get("valid_from") in (None, ""):
        prepared["valid_from"] = created_at
    if "sys_period_start" not in prepared or prepared.get("sys_period_start") in (None, ""):
        prepared["sys_period_start"] = _now_iso(now)
    if "sys_period_end" not in prepared or prepared.get("sys_period_end") in (None, ""):
        prepared["sys_period_end"] = "9999-12-31T23:59:59.999999Z"
    for column in _ISO_CANDIDATE_COLUMNS:
        if column in prepared and prepared[column] not in (None, ""):
            prepared[column] = normalize_timestamp(prepared[column]) or prepared[column]
    if columns is not None:
        allowed = set(columns)
        prepared = {key: value for key, value in prepared.items() if key in allowed}
    return prepared


def insert_canonical_chunk(
    conn: Any,
    values: Mapping[str, Any],
    *,
    on_conflict: str = "ignore",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Insert one chunk row using the canonical column set."""
    columns = _table_columns(conn, "chunks")
    row = prepare_canonical_insert(values, now=now, columns=columns)
    names = [key for key in row if key in columns]
    placeholders = ", ".join("?" for _ in names)
    column_sql = ", ".join(names)
    if on_conflict == "ignore":
        sql = f"INSERT OR IGNORE INTO chunks ({column_sql}) VALUES ({placeholders})"
    else:
        sql = f"INSERT INTO chunks ({column_sql}) VALUES ({placeholders})"
    conn.execute(sql, [row[name] for name in names])
    return row
