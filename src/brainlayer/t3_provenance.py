"""Read-only discriminator for Codex sessions initiated by the T3 Code app."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path

from .alarm import raise_alarm

T3_APP_SESSION = "t3-app-session"
DEFAULT_T3_STATE_DB = Path.home() / ".t3" / "userdata" / "state.sqlite"
_CODEX_SESSION_ID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)
_REQUIRED_RUNTIME_COLUMNS = frozenset({"thread_id", "provider_name", "resume_cursor_json"})


def codex_session_id_from_source(source_file: str | Path) -> str | None:
    """Return the final UUID in a Codex JSONL filename, if present."""
    matches = _CODEX_SESSION_ID_RE.findall(Path(source_file).stem)
    return matches[-1].lower() if matches else None


def is_t3_app_initiated_codex_session(
    source_file: str | Path,
    *,
    state_db: str | Path | None = None,
) -> bool:
    """Return whether a Codex transcript is explicitly linked by T3 runtime state.

    A missing T3 database means there is no local T3 app installation to link
    against. An existing database with a changed or unreadable schema is fatal:
    silently treating those sessions as ordinary Codex would reintroduce the
    provenance collision this module prevents.
    """
    session_id = codex_session_id_from_source(source_file)
    if session_id is None:
        return False

    path = Path(state_db or os.environ.get("BRAINLAYER_T3_STATE_DB", DEFAULT_T3_STATE_DB)).expanduser()
    if not path.exists():
        return False

    return session_id in t3_app_codex_session_ids(path)


def t3_app_codex_session_ids(state_db: str | Path = DEFAULT_T3_STATE_DB) -> set[str]:
    """Return Codex session IDs explicitly linked by T3 runtime cursors."""
    path = Path(state_db).expanduser()

    try:
        connection = sqlite3.connect(f"{path.absolute().as_uri()}?mode=ro&immutable=0", uri=True, timeout=1.0)
    except sqlite3.Error as exc:
        raise_alarm(
            "t3_runtime_unavailable",
            "could not open T3 runtime state read-only",
            {"path": str(path), "error": str(exc)},
        )

    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(provider_session_runtime)")}
        missing = sorted(_REQUIRED_RUNTIME_COLUMNS - columns)
        if missing:
            raise_alarm(
                "t3_runtime_schema_drift",
                "provider_session_runtime no longer exposes the required T3 linkage columns",
                {"path": str(path), "missing_columns": missing},
            )

        session_ids: set[str] = set()
        for provider_name, resume_cursor_json in connection.execute(
            "SELECT provider_name, resume_cursor_json FROM provider_session_runtime WHERE provider_name = ?", ("codex",)
        ):
            try:
                resume_cursor = json.loads(resume_cursor_json)
            except (TypeError, json.JSONDecodeError) as exc:
                raise_alarm(
                    "t3_runtime_linkage_invalid",
                    "provider_session_runtime.resume_cursor_json is not valid JSON",
                    {"path": str(path), "provider_name": provider_name, "error": str(exc)},
                )
            thread_id = resume_cursor.get("threadId") if isinstance(resume_cursor, dict) else None
            if isinstance(thread_id, str) and _CODEX_SESSION_ID_RE.fullmatch(thread_id):
                session_ids.add(thread_id.lower())
        return session_ids
    except sqlite3.Error as exc:
        raise_alarm(
            "t3_runtime_schema_drift",
            "could not query provider_session_runtime for T3 provenance",
            {"path": str(path), "error": str(exc)},
        )
    finally:
        connection.close()
