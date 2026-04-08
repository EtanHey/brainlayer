import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import apsw

ARCHIVE_THRESHOLD = 0.1


def _parse_reference_timestamp(value: Any, fallback: float) -> float:
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return fallback
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    return fallback


def run_decay_job(
    db_path: str | Path,
    *,
    now: float | None = None,
    dry_run: bool = False,
    batch_size: int = 10_000,
) -> dict[str, float | int | bool]:
    started_at = time.perf_counter()
    current_time = time.time() if now is None else now
    connection = apsw.Connection(str(db_path))
    connection.setbusytimeout(30_000)
    cursor = connection.cursor()

    last_rowid = 0
    batch_number = 0
    rows_processed = 0
    archived_rows = 0
    pinned_rows = 0
    total_decay = 0.0

    while True:
        batch_rowids = [
            row[0]
            for row in cursor.execute(
                """
                SELECT rowid
                FROM chunks
                WHERE archived = 0 AND rowid > ?
                ORDER BY rowid
                LIMIT ?
                """,
                (last_rowid, batch_size),
            )
        ]
        if not batch_rowids:
            break
        last_rowid = batch_rowids[-1]

        placeholders = ",".join("?" for _ in batch_rowids)
        batch_sql = (
            """
            WITH batch AS (
                SELECT
                    rowid,
                    id,
                    pinned,
                    CASE
                        WHEN pinned = 1 THEN 1.0
                        ELSE MAX(
                            0.05,
                            pow(
                                1.0 + (
                                    MAX(? - COALESCE(last_retrieved, unixepoch(created_at), ?), 0.0) / 86400.0
                                ) / (
                                    9.0 * (
                                        COALESCE(half_life_days, 30.0) * (
                                            1.0 + ln(1.0 + COALESCE(retrieval_count, 0)) * 0.3
                                        )
                                    )
                                ),
                                -1.0
                            )
                        )
                    END AS new_decay_score
                FROM chunks
                WHERE rowid IN ("""
            + placeholders
            + """)
            )
        """
        )
        if dry_run:
            updated_rows = list(
                cursor.execute(
                    batch_sql
                    + """
                    SELECT rowid, new_decay_score,
                           CASE WHEN pinned = 1 THEN 0 WHEN new_decay_score < ? THEN 1 ELSE 0 END AS archived,
                           pinned
                    FROM batch
                    """,
                    (current_time, current_time, *batch_rowids, ARCHIVE_THRESHOLD),
                )
            )
        else:
            updated_rows = list(
                cursor.execute(
                    batch_sql
                    + """
                    UPDATE chunks
                    SET decay_score = (SELECT new_decay_score FROM batch WHERE batch.rowid = chunks.rowid),
                        archived = CASE
                            WHEN (SELECT pinned FROM batch WHERE batch.rowid = chunks.rowid) = 1 THEN 0
                            WHEN (SELECT new_decay_score FROM batch WHERE batch.rowid = chunks.rowid) < ? THEN 1
                            ELSE 0
                        END,
                        archived_at = CASE
                            WHEN (SELECT pinned FROM batch WHERE batch.rowid = chunks.rowid) = 1 THEN NULL
                            WHEN (SELECT new_decay_score FROM batch WHERE batch.rowid = chunks.rowid) < ? THEN ?
                            ELSE NULL
                        END
                    WHERE rowid IN (SELECT rowid FROM batch)
                    RETURNING rowid, decay_score, archived, pinned
                    """,
                    (current_time, current_time, *batch_rowids, ARCHIVE_THRESHOLD, ARCHIVE_THRESHOLD, current_time),
                )
            )
        rows_processed += len(updated_rows)
        pinned_rows += sum(int(bool(row[3])) for row in updated_rows)
        archived_rows += sum(int(row[2]) for row in updated_rows)
        total_decay += sum(float(row[1]) for row in updated_rows)
        batch_number += 1
        if not dry_run and batch_number % 3 == 0:
            cursor.execute("PRAGMA wal_checkpoint(PASSIVE)")

    duration_seconds = time.perf_counter() - started_at
    average_decay = (total_decay / rows_processed) if rows_processed else 0.0
    return {
        "dry_run": dry_run,
        "rows_processed": rows_processed,
        "archived_rows": archived_rows,
        "pinned_rows": pinned_rows,
        "average_decay": average_decay,
        "duration_seconds": duration_seconds,
    }
