"""Shared, read-only selection of T3's known SQLite source files."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .alarm import raise_alarm

DEFAULT_T3_STATE_DB = Path.home() / ".t3/userdata/state.sqlite"


def resolve_t3_sources(
    state_db: str | Path | None = None, projection_version: int | None = None
) -> list[tuple[Path, int]]:
    """Prefer V2, retain legacy projections, and preserve explicit selection.

    Schema validation belongs to the selected reader. A present V2 file is never
    silently replaced with legacy data when its schema is broken.
    """
    if projection_version not in {None, 1, 2}:
        raise ValueError("T3 projection_version must be 1 or 2")
    if state_db is not None:
        return [(Path(state_db).expanduser(), projection_version or 1)]
    root = Path.home() / ".t3/userdata"
    modern, legacy = root / "statev2.sqlite", root / "state.sqlite"
    sources = [(modern, projection_version or 2)] if modern.is_file() else []
    if legacy.is_file():
        connection = None
        try:
            connection = sqlite3.connect(f"{legacy.absolute().as_uri()}?mode=ro&immutable=0", uri=True, timeout=1)
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        except sqlite3.Error as exc:
            raise_alarm(
                "t3_source_unavailable",
                "could not inspect T3 source read-only",
                {"state_db_path": str(legacy), "error": str(exc)},
            )
        finally:
            if connection is not None:
                connection.close()
        if "orchestration_v2_projection_threads" in tables:
            sources.append((legacy, projection_version or 2))
        elif tables & {"projection_threads", "provider_session_runtime"} or not sources:
            sources.append((legacy, projection_version or 1))
    return sources or [(legacy, projection_version or 1)]
