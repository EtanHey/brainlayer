"""Backfill chunk_origin provenance for legacy BrainLayer rows."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .chunk_origin import (
    CHUNK_ORIGIN_GEMINI_FLASH_LITE,
    CHUNK_ORIGIN_GROQ,
    CHUNK_ORIGIN_MANUAL,
    CHUNK_ORIGIN_MLX,
    CHUNK_ORIGIN_OLLAMA,
    CHUNK_ORIGIN_UNKNOWN,
)


@dataclass
class ChunkOriginBackfillResult:
    scanned: int = 0
    updated: int = 0
    batches: int = 0
    checkpoints: int = 0
    inferred: dict[str, int] = field(default_factory=dict)


def _text_contains(value: Any, needle: str) -> bool:
    return needle in str(value or "").casefold()


def _metadata_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _known_backend_origin(value: Any, *, gemini_origin: str) -> str | None:
    normalized = str(value or "").casefold()
    if "gemini-2.5-flash-lite" in normalized:
        return gemini_origin
    if "groq" in normalized:
        return CHUNK_ORIGIN_GROQ
    if "ollama" in normalized:
        return CHUNK_ORIGIN_OLLAMA
    if "mlx" in normalized:
        return CHUNK_ORIGIN_MLX
    return None


def infer_chunk_origin(row: sqlite3.Row, *, gemini_origin: str = CHUNK_ORIGIN_GEMINI_FLASH_LITE) -> str | None:
    """Infer a legacy row's provenance only when there is a direct signal."""
    chunk_id = str(row["id"] or "")
    source = str(row["source"] or "")
    source_file = str(row["source_file"] or "")
    metadata = row["metadata"]
    enrichment_version = str(row["enrichment_version"] or "")
    summary_v2 = str(row["summary_v2"] or "")

    if chunk_id.startswith("manual-") or source == CHUNK_ORIGIN_MANUAL or "manual" in source_file.casefold():
        return CHUNK_ORIGIN_MANUAL

    metadata_obj = _metadata_object(metadata)
    for key in ("chunk_origin", "origin", "backend", "enrichment_backend", "enrichment_model", "model"):
        origin = _known_backend_origin(metadata_obj.get(key), gemini_origin=gemini_origin)
        if origin:
            return origin

    if (
        metadata_obj.get("prompt_version") == "r81"
        or enrichment_version == "r81"
        or _text_contains(summary_v2, "prompt=r81")
    ):
        return gemini_origin

    return None


def _coerce_metadata_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def backfill_chunk_origin_provenance(
    db_path: Path,
    *,
    apply: bool = False,
    batch_size: int = 5000,
    checkpoint_every: int = 5,
    gemini_origin: str = CHUNK_ORIGIN_GEMINI_FLASH_LITE,
) -> ChunkOriginBackfillResult:
    """Backfill unknown chunk_origin rows in bounded batches.

    The backfill deliberately updates only rows with direct provenance signals.
    Operators should stop enrichment/drain workers or coordinate a quiet window
    before running with apply=True against the live DB.
    """
    batch_size = max(1, int(batch_size))
    checkpoint_every = max(1, int(checkpoint_every))

    result = ChunkOriginBackfillResult()
    inferred = Counter()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
        if "chunk_origin" not in columns:
            raise RuntimeError("chunks.chunk_origin column is missing")

        optional_columns = {
            "source": "NULL AS source",
            "source_file": "NULL AS source_file",
            "metadata": "NULL AS metadata",
            "enrichment_version": "NULL AS enrichment_version",
            "summary_v2": "NULL AS summary_v2",
        }
        select_parts = ["rowid", "id"]
        for column, fallback in optional_columns.items():
            select_parts.append(column if column in columns else fallback)

        last_rowid = 0
        while True:
            rows = list(
                conn.execute(
                    f"""
                    SELECT {", ".join(select_parts)}
                    FROM chunks
                    WHERE rowid > ?
                      AND COALESCE(chunk_origin, ?) = ?
                    ORDER BY rowid
                    LIMIT ?
                    """,
                    (last_rowid, CHUNK_ORIGIN_UNKNOWN, CHUNK_ORIGIN_UNKNOWN, batch_size),
                )
            )
            if not rows:
                break

            result.batches += 1
            updates: list[tuple[str, int]] = []
            for row in rows:
                last_rowid = int(row["rowid"])
                result.scanned += 1
                inferred_origin = infer_chunk_origin(
                    {
                        key: _coerce_metadata_text(row[key])
                        for key in ("id", "source", "source_file", "metadata", "enrichment_version", "summary_v2")
                    },
                    gemini_origin=gemini_origin,
                )
                if inferred_origin:
                    inferred[inferred_origin] += 1
                    updates.append((inferred_origin, int(row["rowid"])))

            if apply and updates:
                before_changes = conn.total_changes
                conn.executemany(
                    """
                    UPDATE chunks
                    SET chunk_origin = ?
                    WHERE rowid = ?
                      AND COALESCE(chunk_origin, ?) = ?
                    """,
                    [(origin, rowid, CHUNK_ORIGIN_UNKNOWN, CHUNK_ORIGIN_UNKNOWN) for origin, rowid in updates],
                )
                result.updated += conn.total_changes - before_changes
                conn.commit()
                if result.batches % checkpoint_every == 0:
                    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                    result.checkpoints += 1
            elif apply:
                conn.commit()

        if apply and result.batches and result.batches % checkpoint_every != 0:
            conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            result.checkpoints += 1
        result.inferred = dict(inferred)
        return result
    finally:
        conn.close()
