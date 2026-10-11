"""Shared, read-only selection of T3's known SQLite source files."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .alarm import raise_alarm

DEFAULT_T3_STATE_DB = Path.home() / ".t3/userdata/state.sqlite"


def _tables(path: Path) -> set[str]:
    connection = sqlite3.connect(
        f"{path.absolute().as_uri()}?mode=ro&immutable=0", uri=True, timeout=1, isolation_level=None
    )
    try:
        return {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        connection.close()


def _version(tables: set[str]) -> int:
    return 2 if any(name.startswith("orchestration_v2_projection_") for name in tables) else 1


def resolve_t3_sources(
    state_db: str | Path | None = None,
    projection_version: int | None = None,
    *,
    tolerate_legacy_failure: bool = False,
) -> list[tuple[Path, int]]:
    """Prefer V2 and retain legacy projections; explicit paths auto-detect tables.

    Only ingestion may tolerate an unavailable legacy probe when V2 is present:
    it must then attempt that source and report the failure in aggregate health.
    """
    if projection_version not in {None, 1, 2}:
        raise ValueError("T3 projection_version must be 1 or 2")
    root = Path.home() / ".t3/userdata"
    modern, legacy = root / "statev2.sqlite", root / "state.sqlite"
    sources = [(modern, projection_version or 2)] if state_db is None and modern.is_file() else []
    selected = Path(state_db).expanduser() if state_db is not None else legacy
    try:
        if state_db is not None:
            version = projection_version or (_version(_tables(selected)) if selected.is_file() else 1)
            return [(selected, version)]
        if legacy.is_file():
            tables = _tables(legacy)
            if (
                any(name.startswith("orchestration_v2_projection_") for name in tables)
                or tables & {"projection_threads", "provider_session_runtime"}
                or not sources
            ):
                sources.append((legacy, projection_version or _version(tables)))
    except sqlite3.Error as exc:
        if sources and tolerate_legacy_failure:
            return sources + [(legacy, projection_version or 1)]
        raise_alarm(
            "t3_source_unavailable",
            "could not inspect T3 source read-only",
            {"state_db_path": str(selected), "error": str(exc)},
        )
    return sources or [(legacy, projection_version or 1)]
