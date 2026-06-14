"""Phase-1 bitemporal chunk schema and helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

SENTINEL_SYS_PERIOD_END = "9999-12-31T23:59:59.999999Z"
LEGACY_SYS_PERIOD_START = "0001-01-01T00:00:00.000000Z"

_TEMPORAL_COLUMNS = [
    ("content_hash", "TEXT"),
    ("valid_from", "TEXT"),
    ("invalid_at", "TEXT"),
    ("sys_period_start", f"TEXT DEFAULT '{LEGACY_SYS_PERIOD_START}'"),
    ("sys_period_end", f"TEXT DEFAULT '{SENTINEL_SYS_PERIOD_END}'"),
]


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _column_defs(cursor: Any, table: str) -> list[tuple[str, str]]:
    return [(str(row[1]), str(row[2]) if row[2] else "TEXT") for row in cursor.execute(f"PRAGMA table_info({table})")]


def _table_columns(cursor: Any, table: str) -> set[str]:
    return {str(row[1]) for row in cursor.execute(f"PRAGMA table_info({table})")}


def _ensure_temporal_columns(cursor: Any) -> None:
    existing = _table_columns(cursor, "chunks")
    for column, column_type in _TEMPORAL_COLUMNS:
        if column not in existing:
            cursor.execute(f"ALTER TABLE chunks ADD COLUMN {_quote_ident(column)} {column_type}")
            existing.add(column)


def _ensure_history_table(cursor: Any) -> None:
    chunk_columns = _column_defs(cursor, "chunks")
    column_sql = ", ".join(f"{_quote_ident(name)} {column_type}" for name, column_type in chunk_columns)
    cursor.execute(f"CREATE TABLE IF NOT EXISTS _chunks_history ({column_sql})")

    history_columns = _table_columns(cursor, "_chunks_history")
    for name, column_type in chunk_columns:
        if name not in history_columns:
            cursor.execute(f"ALTER TABLE _chunks_history ADD COLUMN {_quote_ident(name)} {column_type}")
            history_columns.add(name)


def _ensure_clock_table(cursor: Any) -> None:
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS _brainlayer_bitemporal_clock (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            wall_second TEXT NOT NULL,
            tick INTEGER NOT NULL
        )
        """
    )
    cursor.execute(
        """
        INSERT OR IGNORE INTO _brainlayer_bitemporal_clock(id, wall_second, tick)
        VALUES (1, strftime('%Y-%m-%dT%H:%M:%S','now'), 0)
        """
    )


def _history_insert_sql(cursor: Any) -> str:
    columns = [name for name, _ in _column_defs(cursor, "chunks")]
    column_sql = ", ".join(_quote_ident(name) for name in columns)
    values = []
    for name in columns:
        if name == "sys_period_end":
            values.append(
                """
                (
                    SELECT wall_second || printf('.%06dZ', tick)
                    FROM _brainlayer_bitemporal_clock
                    WHERE id = 1
                )
                """
            )
        else:
            values.append(f"OLD.{_quote_ident(name)}")
    value_sql = ", ".join(values)
    return f"INSERT INTO _chunks_history ({column_sql}) VALUES ({value_sql});"


def _clock_tick_sql() -> str:
    return """
        UPDATE _brainlayer_bitemporal_clock
        SET
            tick = CASE
                WHEN wall_second = strftime('%Y-%m-%dT%H:%M:%S','now') THEN tick + 1
                ELSE 0
            END,
            wall_second = strftime('%Y-%m-%dT%H:%M:%S','now')
        WHERE id = 1;
    """


def _install_triggers(cursor: Any) -> None:
    history_insert = _history_insert_sql(cursor)
    clock_tick = _clock_tick_sql()
    cursor.execute("DROP TRIGGER IF EXISTS chunks_bitemporal_update")
    cursor.execute("DROP TRIGGER IF EXISTS chunks_bitemporal_delete")
    cursor.execute(
        f"""
        CREATE TRIGGER chunks_bitemporal_update
        AFTER UPDATE ON chunks
        BEGIN
            {clock_tick}
            {history_insert}
        END
        """
    )
    cursor.execute(
        f"""
        CREATE TRIGGER chunks_bitemporal_delete
        AFTER DELETE ON chunks
        BEGIN
            {clock_tick}
            {history_insert}
        END
        """
    )


def apply_bitemporal_migration(conn: Any) -> None:
    """Idempotently install Phase-1 bitemporal schema on an existing DB.

    The migration owns a short `BEGIN IMMEDIATE` write transaction. It adds only
    nullable/defaulted columns and metadata tables; it does not backfill legacy
    row hashes or migrate live data.
    """
    cursor = conn.cursor()
    transaction_started = False
    try:
        cursor.execute("BEGIN IMMEDIATE")
        transaction_started = True
        _ensure_temporal_columns(cursor)
        _ensure_history_table(cursor)
        _ensure_clock_table(cursor)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_current_active ON chunks(created_at, id) WHERE invalid_at IS NULL"
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_history_period
            ON _chunks_history(id, sys_period_start, sys_period_end)
            """
        )
        _install_triggers(cursor)
        cursor.execute("COMMIT")
        transaction_started = False
    except Exception:
        if transaction_started:
            cursor.execute("ROLLBACK")
        raise


def supersede_chunk(conn: Any, chunk_id: str, *, invalid_at: str | None = None) -> bool:
    """Mark a chunk invalid without deleting it.

    The old row image is copied to `_chunks_history` by the UPDATE trigger.
    """
    now = invalid_at or datetime.now(timezone.utc).isoformat()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE chunks
        SET invalid_at = COALESCE(invalid_at, ?),
            sys_period_end = CASE
                WHEN sys_period_end IS NULL OR sys_period_end = ? THEN ?
                ELSE sys_period_end
            END
        WHERE id = ?
        """,
        (now, SENTINEL_SYS_PERIOD_END, now, chunk_id),
    )
    return conn.changes() > 0
