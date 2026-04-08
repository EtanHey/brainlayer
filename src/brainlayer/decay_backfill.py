import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import apsw

from .vector_store import VectorStore

PINNED_TAGS = {"architecture", "decision", "correction"}


def _parse_created_at(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    return None


def importance_to_half_life(importance: float | int | None) -> float:
    if importance is None:
        return 30.0
    score = min(max(float(importance), 1.0), 10.0)
    if score <= 5.0:
        return round(7.0 + ((score - 1.0) * (23.0 / 4.0)), 2)
    return round(30.0 + ((score - 5.0) * 12.0), 2)


def _is_pinned_tag_set(tags_raw: Any) -> int:
    if not tags_raw:
        return 0
    try:
        tags = json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw
    except json.JSONDecodeError:
        return 0
    if not isinstance(tags, list):
        return 0
    return int(any(str(tag) in PINNED_TAGS for tag in tags))


def backfill_decay_fields(db_path: str | Path, *, dry_run: bool = False) -> dict[str, int]:
    bootstrap_store = VectorStore(Path(db_path))
    bootstrap_store.close()

    connection = apsw.Connection(str(db_path))
    connection.setbusytimeout(30_000)
    cursor = connection.cursor()
    rows = list(
        cursor.execute(
            """
            SELECT id, importance, tags, created_at
            FROM chunks
            WHERE last_retrieved IS NULL
            """
        )
    )

    updates = []
    for chunk_id, importance, tags, created_at in rows:
        updates.append(
            (
                importance_to_half_life(importance),
                _parse_created_at(created_at),
                _is_pinned_tag_set(tags),
                chunk_id,
            )
        )

    if not dry_run and updates:
        cursor.executemany(
            """
            UPDATE chunks
            SET half_life_days = ?,
                last_retrieved = ?,
                pinned = ?
            WHERE id = ?
            """,
            updates,
        )

    return {"updated_rows": len(updates)}
