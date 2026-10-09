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
_V2_TABLE = "orchestration_v2_projection_provider_threads"
_HEADER_LIMIT = 1024 * 1024


def _projection_version(value: int | None) -> int:
    selected = str(value if value is not None else os.environ.get("BRAINLAYER_T3_PROJECTION_VERSION", "1"))
    if selected not in {"1", "2"}:
        raise_alarm("t3_runtime_selection_invalid", "T3 projection version must be 1 or 2")
    return int(selected)


def codex_header_identity(source_file: str | Path, cache: dict | None = None) -> tuple[str, bool]:
    """Read bounded session_meta identity; never infer a fork identity from its filename."""
    path = Path(source_file)
    try:
        stat = path.stat()
        key = (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if cache is not None and key in cache:
            return cache[key]
        with path.open("rb") as handle:
            line = handle.readline(_HEADER_LIMIT + 1)
        if len(line) > _HEADER_LIMIT or not line.endswith(b"\n"):
            raise ValueError("session header is incomplete or exceeds the bounded read")
        record = json.loads(line)
        payload = record.get("payload") if isinstance(record, dict) else None
        if record.get("type") != "session_meta" or not isinstance(payload, dict):
            raise ValueError("first record must be session_meta")
        session_id = payload.get("id")
        if not isinstance(session_id, str) or not _CODEX_SESSION_ID_RE.fullmatch(session_id):
            raise ValueError("session_meta.id must be a UUID")
        alternate = payload.get("session_id")
        if alternate is not None and (not isinstance(alternate, str) or alternate.lower() != session_id.lower()):
            raise ValueError("session_meta identity fields disagree")
        result = (session_id.lower(), payload.get("originator") == "T3 Code")
        if cache is not None:
            cache[key] = result
        return result
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise_alarm(
            "t3_session_identity_invalid",
            "could not establish bounded Codex session identity",
            {"path": str(path), "error": str(exc)},
        )


def codex_session_id_from_source(source_file: str | Path) -> str | None:
    """Return the final UUID in a Codex JSONL filename, if present."""
    matches = _CODEX_SESSION_ID_RE.findall(Path(source_file).stem)
    return matches[-1].lower() if matches else None


def is_t3_app_initiated_codex_session(
    source_file: str | Path,
    *,
    state_db: str | Path | None = None,
    linked_session_ids: set[str] | None = None,
    projection_version: int | None = None,
    session_identity_cache: dict | None = None,
) -> bool:
    """Return whether a Codex transcript is explicitly linked by T3 runtime state.

    A missing T3 database means there is no local T3 app installation to link
    against. An existing database with a changed or unreadable schema is fatal:
    silently treating those sessions as ordinary Codex would reintroduce the
    provenance collision this module prevents.
    """
    version = _projection_version(projection_version)
    session_id, t3_origin = (
        codex_header_identity(source_file, session_identity_cache)
        if version == 2
        else (codex_session_id_from_source(source_file), False)
    )
    if session_id is None:
        return False

    if linked_session_ids is None:
        path = Path(state_db or os.environ.get("BRAINLAYER_T3_STATE_DB", DEFAULT_T3_STATE_DB)).expanduser()
        if not path.exists() and version == 1:
            return False
        linked_session_ids = t3_app_codex_session_ids(path, projection_version=version)
    linked = session_id in linked_session_ids
    if t3_origin and not linked:
        raise_alarm(
            "t3_runtime_linkage_pending",
            "T3-origin session has no selected projection linkage",
            {"path": str(source_file)},
        )
    return linked


def t3_app_codex_session_ids(
    state_db: str | Path = DEFAULT_T3_STATE_DB, *, projection_version: int | None = None
) -> set[str]:
    """Return Codex session IDs explicitly linked by T3 runtime cursors."""
    path = Path(state_db).expanduser()
    version = _projection_version(projection_version)

    try:
        connection = sqlite3.connect(f"{path.absolute().as_uri()}?mode=ro&immutable=0", uri=True, timeout=1.0)
    except sqlite3.Error as exc:
        raise_alarm(
            "t3_runtime_unavailable",
            "could not open T3 runtime state read-only",
            {"path": str(path), "error": str(exc)},
        )

    try:
        if version == 2:
            columns = {row[1] for row in connection.execute(f"PRAGMA table_info({_V2_TABLE})")}
            if not {"provider", "payload_json", "provider_thread_id"} <= columns:
                raise_alarm(
                    "t3_runtime_schema_drift",
                    "selected V2 provider projection lacks linkage columns",
                    {"path": str(path)},
                )
            session_ids: set[str] = set()
            for provider, raw in connection.execute(
                f"SELECT provider, payload_json FROM {_V2_TABLE} WHERE provider = ?", ("codex",)
            ):
                try:
                    payload = json.loads(raw)
                    if not isinstance(payload, dict):
                        raise ValueError("provider payload must be an object")
                    for field in ("nativeThreadRef", "nativeConversationHeadRef"):
                        ref = payload.get(field)
                        if ref is None:
                            continue
                        if not isinstance(ref, dict) or ref.get("driver") != provider:
                            raise ValueError("native reference must name its structural provider")
                        identity = ref.get("nativeId")
                        if identity is None:
                            continue
                        if not isinstance(identity, str) or not _CODEX_SESSION_ID_RE.fullmatch(identity):
                            raise ValueError("Codex native reference must contain a UUID")
                        session_ids.add(identity.lower())
                except (TypeError, ValueError) as exc:
                    raise_alarm(
                        "t3_runtime_linkage_invalid",
                        "selected V2 provider linkage is malformed",
                        {"path": str(path), "error": str(exc)},
                    )
            return session_ids
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
