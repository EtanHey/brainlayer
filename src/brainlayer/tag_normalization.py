"""Tag normalization and tombstoning utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import apsw

_VALID_TAXONOMY_TAGS: frozenset[str] | None = None


@dataclass(frozen=True)
class TagTombstoneResult:
    tombstoned: int
    updated_chunks: int


def valid_taxonomy_tags() -> frozenset[str]:
    global _VALID_TAXONOMY_TAGS
    if _VALID_TAXONOMY_TAGS is None:
        taxonomy_path = Path(__file__).resolve().parent / "taxonomy.json"
        data = json.loads(taxonomy_path.read_text(encoding="utf-8"))
        labels: set[str] = set()
        for category in data.get("categories", {}).values():
            if isinstance(category, dict):
                labels.update(str(label).strip().lower() for label in category.get("labels", {}) if str(label).strip())
        _VALID_TAXONOMY_TAGS = frozenset(labels)
    return _VALID_TAXONOMY_TAGS


def ensure_tag_tombstone_schema(conn: apsw.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tag_tombstones (
            tag TEXT PRIMARY KEY,
            reason TEXT NOT NULL,
            occurrence_count INTEGER NOT NULL,
            tombstoned_at TEXT NOT NULL
        )
        """
    )


def _loads_tags(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        decoded = raw
    else:
        try:
            decoded = json.loads(str(raw))
        except json.JSONDecodeError:
            return []
    if not isinstance(decoded, list):
        return []
    return [str(tag).strip() for tag in decoded if str(tag).strip()]


def _batches(values: list[str], size: int = 500) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def tombstone_singleton_tags(conn: apsw.Connection) -> TagTombstoneResult:
    """Tombstone non-taxonomy tags that occur once and remove them from chunks."""
    ensure_tag_tombstone_schema(conn)
    taxonomy_tags = valid_taxonomy_tags()
    rows = list(
        conn.execute(
            """
            SELECT tag, COUNT(DISTINCT chunk_id) AS occurrences
            FROM chunk_tags
            GROUP BY tag
            HAVING occurrences = 1
            """
        )
    )
    tombstones = [str(tag) for tag, occurrences in rows if str(tag).strip().lower() not in taxonomy_tags]
    if not tombstones:
        return TagTombstoneResult(tombstoned=0, updated_chunks=0)

    now = datetime.now(timezone.utc).isoformat()
    for tag in tombstones:
        conn.execute(
            """
            INSERT OR REPLACE INTO tag_tombstones(tag, reason, occurrence_count, tombstoned_at)
            VALUES (?, 'singleton-non-taxonomy', 1, ?)
            """,
            (tag, now),
        )

    tombstone_set = set(tombstones)
    chunk_ids: list[str] = []
    seen_chunk_ids: set[str] = set()
    for batch in _batches(tombstones):
        for row in conn.execute(
            f"""
            SELECT DISTINCT chunk_id
            FROM chunk_tags
            WHERE tag IN ({", ".join("?" for _ in batch)})
            """,
            batch,
        ):
            chunk_id = str(row[0])
            if chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                chunk_ids.append(chunk_id)

    updated = 0
    for chunk_id in chunk_ids:
        row = conn.execute("SELECT tags FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        if row is None:
            continue
        tags = _loads_tags(row[0])
        filtered = [tag for tag in tags if tag not in tombstone_set]
        if filtered == tags:
            continue
        conn.execute(
            "UPDATE chunks SET tags = ? WHERE id = ?",
            (json.dumps(filtered, ensure_ascii=True, separators=(",", ":")), chunk_id),
        )
        updated += 1

    return TagTombstoneResult(tombstoned=len(tombstones), updated_chunks=updated)
