"""Connection-local WAL recycle policy for APSW and sqlite3 writers.

The 256 MB journal_size_limit trims retained allocation when the WAL resets;
it cannot cap an active transaction or frames still needed by a pinned reader.
Keep the legacy 10,000-page autocheckpoint default (~41 MB at 4 KiB/page) to
amortize checkpoint work across ingestion batches. It attempts PASSIVE backfill,
not guaranteed reclamation: readers can defer a reset. BrainBar shares the size
limit but uses 1,000 pages for interactive commits. Python retains its deployment
overrides (0 disables either
limit/autocheckpoint). Readonly connections must never call this helper.
"""

import os
import sqlite3

import apsw


def _nonnegative_env(name: str, default: int) -> int:
    try:
        return max(int(os.environ.get(name, str(default))), 0)
    except (TypeError, ValueError):
        return default


def configure_writer_wal_policy(conn: apsw.Connection | sqlite3.Connection) -> None:
    """Apply on every writer open; these PRAGMAs do not persist across connections."""
    limit = _nonnegative_env("BRAINLAYER_WAL_SIZE_LIMIT_BYTES", 256_000_000)
    pages = _nonnegative_env("BRAINLAYER_WAL_AUTOCHECKPOINT", 10_000)
    conn.execute(f"PRAGMA journal_size_limit = {limit if limit > 0 else -1}").fetchone()
    conn.execute(f"PRAGMA wal_autocheckpoint = {pages}").fetchone()
