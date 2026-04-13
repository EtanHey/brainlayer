"""Shared WAL checkpoint helpers for BrainLayer maintenance commands."""

from __future__ import annotations

import os
import sqlite3

from .paths import get_db_path

_VALID_CHECKPOINT_MODES = {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}


def resolve_db_path() -> str | None:
    """Return the configured DB path if it exists."""
    db_path = str(get_db_path())
    return db_path if os.path.exists(db_path) else None


def get_wal_size(db_path: str) -> int:
    """Return WAL file size in bytes, or 0 if the WAL file does not exist."""
    try:
        return os.path.getsize(f"{db_path}-wal")
    except OSError:
        return 0


def format_size(size_bytes: int) -> str:
    """Render a byte count as a human-readable string."""
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


def checkpoint(db_path: str, mode: str = "TRUNCATE") -> tuple[int, int, int]:
    """Run WAL checkpoint and return (busy, log_pages, checkpointed_pages)."""
    mode = mode.upper()
    if mode not in _VALID_CHECKPOINT_MODES:
        raise ValueError(f"Invalid checkpoint mode: {mode}")

    conn = sqlite3.connect(db_path, timeout=10)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        result = conn.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
        return result
    finally:
        conn.close()


def run_wal_checkpoint(mode: str = "TRUNCATE") -> dict[str, object]:
    """Execute a checkpoint and return structured results."""
    db_path = resolve_db_path()
    if not db_path:
        raise FileNotFoundError("no database found")

    wal_before = get_wal_size(db_path)
    busy, log_pages, checkpointed_pages = checkpoint(db_path, mode)
    wal_after = get_wal_size(db_path)

    return {
        "db": db_path,
        "mode": mode,
        "wal_before_bytes": wal_before,
        "wal_after_bytes": wal_after,
        "wal_before": format_size(wal_before),
        "wal_after": format_size(wal_after),
        "busy": busy,
        "log_pages": log_pages,
        "checkpointed_pages": checkpointed_pages,
    }
