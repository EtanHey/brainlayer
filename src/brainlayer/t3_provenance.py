"""Read-only discriminator for Codex sessions initiated by the T3 Code app."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from . import t3_linkage_alarm
from .alarm import BrainLayerAlarm, build_alarm, raise_alarm
from .t3_sources import DEFAULT_T3_STATE_DB as DEFAULT_T3_STATE_DB
from .t3_sources import resolve_t3_sources

T3_APP_SESSION = "t3-app-session"
# A T3-origin Codex session T3 never linked within the hold window: indexed, labelled, no thread link.
T3_UNLINKED = "t3-unlinked"
# How long a T3-origin Codex session is held for T3 to link it, counted from the session's own start.
# Evidence (2026-11-19): on the M4, 0 of 87 sessions that ever raised t3_runtime_linkage_pending
# (16 Oct - 19 Nov) were linked later, and no linked session ever raised it, so T3 links before
# the watcher's first read (one 30 s poll); on the M1, 0 of 8 in 7 days. 24 h is far above that.
DEFAULT_T3_LINKAGE_HOLD_S = 24 * 3600.0
T3_LINKAGE_HOLD_ENV = "BRAINLAYER_T3_LINKAGE_HOLD_S"
# Set by the flush callback on an entry it defers this flush while its session waits for T3 linkage:
# the hold's start (epoch seconds). The watcher reads it to report held_for_linkage instead of lag.
T3_LINKAGE_HOLD_KEY = "_t3_linkage_hold_started_at"
_CODEX_SESSION_ID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)
_REQUIRED_RUNTIME_COLUMNS = frozenset({"thread_id", "provider_name", "resume_cursor_json"})
_V2_TABLE = "orchestration_v2_projection_provider_threads"
_HEADER_LIMIT = 1024 * 1024


def _raise_read_only_alarm(code: str, message: str, context: dict | None = None) -> NoReturn:
    raise build_alarm(code, message, context)


def _projection_version(value: int | None) -> int | None:
    selected = str(value if value is not None else os.environ.get("BRAINLAYER_T3_PROJECTION_VERSION", ""))
    if not selected:
        return None
    if selected not in {"1", "2"}:
        (_raise_read_only_alarm if t3_linkage_alarm.read_only_active() else raise_alarm)(
            "t3_runtime_selection_invalid", "T3 projection version must be 1 or 2"
        )
    return int(selected)


@dataclass(frozen=True)
class CompactCodexHeader:
    identity: tuple[str, bool] | None
    size: int
    error: str | None = None


def compact_codex_header(record, size):
    """Validate once, retaining neither instructions nor the source payload."""
    try:
        return CompactCodexHeader(_validate_codex_header(record), size)
    except (ValueError, TypeError, AttributeError) as error:
        return CompactCodexHeader(None, size, str(error))


def _validate_codex_header(record):
    payload = record.get("payload") if isinstance(record, dict) else None
    if record.get("type") != "session_meta" or not isinstance(payload, dict):
        raise ValueError("first record must be session_meta")
    session_id = payload.get("id")
    if not isinstance(session_id, str) or not _CODEX_SESSION_ID_RE.fullmatch(session_id):
        raise ValueError("session_meta.id must be a UUID")
    alternate = payload.get("session_id")
    if alternate is not None:
        if not isinstance(alternate, str) or not _CODEX_SESSION_ID_RE.fullmatch(alternate):
            raise ValueError("session_meta.session_id must be a UUID")
        if alternate.lower() != session_id.lower():
            parent = payload.get("parent_thread_id")
            if not isinstance(parent, str) or not _CODEX_SESSION_ID_RE.fullmatch(parent):
                raise ValueError("fork parent_thread_id must be a UUID")
            if parent.lower() != alternate.lower():
                # Nested subagents retain the root session_id, but name their
                # immediate parent both here and in the structural source.
                source = payload.get("source")
                subagent = source.get("subagent") if isinstance(source, dict) else None
                spawn = subagent.get("thread_spawn") if isinstance(subagent, dict) else None
                recorded_parent = spawn.get("parent_thread_id") if isinstance(spawn, dict) else None
                depth = spawn.get("depth") if isinstance(spawn, dict) else None
                if (
                    not isinstance(recorded_parent, str)
                    or recorded_parent.lower() != parent.lower()
                    or type(depth) is not int
                    or depth < 2
                ):
                    raise ValueError("session_meta identity fields disagree")
    return (session_id.lower(), payload.get("originator") == "T3 Code")


def codex_header_identity(
    source_file: str | Path, cache: dict | None = None, header: tuple[dict, int] | CompactCodexHeader | None = None
) -> tuple[str, bool]:
    """Read bounded session_meta identity; never infer a fork identity from its filename."""
    path = Path(source_file)
    try:
        key = None
        if isinstance(header, CompactCodexHeader):
            if header.error or header.identity is None or not 0 < header.size <= _HEADER_LIMIT:
                raise ValueError(header.error or "session header exceeds the bounded read")
            return header.identity
        if header is None:
            stat = path.stat()
            key = (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if cache is not None and key in cache:
                return cache[key]
            with path.open("rb") as handle:
                line = handle.readline(_HEADER_LIMIT + 1)
            if len(line) > _HEADER_LIMIT or not line.endswith(b"\n"):
                raise ValueError("session header is incomplete or exceeds the bounded read")
            record = json.loads(line)
        else:
            # Already decoded FIRST newline-terminated record, with its actual byte size.
            # Validation below is identical to the file reader; this never chooses a file key.
            record, size = header
            if type(size) is not int or not 0 < size <= _HEADER_LIMIT:
                raise ValueError("session header exceeds the bounded read")
        result = _validate_codex_header(record)
        if cache is not None and key is not None:
            cache[key] = result
        return result
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        (_raise_read_only_alarm if t3_linkage_alarm.read_only_active() else raise_alarm)(
            "t3_session_identity_invalid",
            "could not establish bounded Codex session identity",
            {"path": str(path), "error": str(exc)},
        )


def t3_linkage_hold_window_s() -> float:
    """The hold window in seconds; a missing, non-finite or negative value is the default."""
    raw = os.environ.get(T3_LINKAGE_HOLD_ENV)
    try:
        value = float(raw) if raw is not None else DEFAULT_T3_LINKAGE_HOLD_S
    except ValueError:
        value = math.nan
    return value if math.isfinite(value) and value >= 0 else DEFAULT_T3_LINKAGE_HOLD_S


def codex_session_started_at(session_id: str) -> float | None:
    """Epoch seconds embedded in a UUIDv7 Codex session id; None for any other UUID version."""
    digits = session_id.replace("-", "").lower()
    if len(digits) != 32 or digits[12] != "7" or not _CODEX_SESSION_ID_RE.fullmatch(session_id):
        return None
    return int(digits[:12], 16) / 1000.0


def t3_linkage_hold_started_at(
    source_file: str | Path, *, session_header: CompactCodexHeader | None = None, cache: dict | None = None
) -> float:
    """When a session's linkage hold began: the session's own start, read from its header id.

    T3 links a session it adopts within seconds of starting it, so the start is the clock. It lives
    in the data, so a restart, a copied file or a touched mtime cannot restart the hold. A header
    without a UUIDv7 id falls back to the file's mtime, which is never earlier than the start, so
    the fallback only ever holds longer.
    """
    started = None
    try:
        with t3_linkage_alarm.read_only():
            session_id, _ = codex_header_identity(source_file, cache, session_header)
        started = codex_session_started_at(session_id)
    except BrainLayerAlarm:
        started = None
    if started is None:
        try:
            started = Path(source_file).stat().st_mtime
        except OSError:
            started = time.time()
    return started


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
    session_header: tuple[dict, int] | CompactCodexHeader | None = None,
) -> bool:
    """Return whether a Codex transcript is explicitly linked by T3 runtime state.

    A missing T3 database means there is no local T3 app installation to link
    against. An existing database with a changed or unreadable schema is fatal:
    silently treating those sessions as ordinary Codex would reintroduce the
    provenance collision this module prevents.
    """
    version = _projection_version(projection_version)
    selected_path = state_db or os.environ.get("BRAINLAYER_T3_STATE_DB")
    selection_key = ("t3-source-selection", str(selected_path), version)
    if version == 1 and linked_session_ids is not None:
        # Source-only fallback must not inspect the runtime that just failed.
        sources = []
    elif session_identity_cache is not None and selection_key in session_identity_cache:
        sources = session_identity_cache[selection_key]
    else:
        sources = resolve_t3_sources(selected_path, version)
        if session_identity_cache is not None:
            session_identity_cache[selection_key] = sources
    session_id, t3_origin = (
        codex_header_identity(source_file, session_identity_cache, session_header)
        if any(v == 2 for _, v in sources)
        else (codex_session_id_from_source(source_file), False)
    )
    if session_id is None:
        return False

    if linked_session_ids is None:
        if not any(path.exists() for path, _ in sources) and version != 2:
            return False
        linked_session_ids = t3_app_codex_session_ids(selected_path, projection_version=version)
    linked = session_id in linked_session_ids
    if t3_origin:
        t3_linkage_alarm.observe(source_file, session_id, sources, pending=not linked)
    return linked


def t3_app_codex_session_ids(state_db: str | Path | None = None, *, projection_version: int | None = None) -> set[str]:
    """Union linkage from the same default sources used by thread ingestion."""
    version = _projection_version(projection_version)
    sources = resolve_t3_sources(state_db or os.environ.get("BRAINLAYER_T3_STATE_DB"), version)
    return set().union(*(_resolved_source_codex_session_ids(path, selected) for path, selected in sources))


def _resolved_source_codex_session_ids(path: Path, version: int) -> set[str]:
    session_ids = _source_codex_session_ids(path, version)
    t3_linkage_alarm.runtime_resolved(path, version)
    return session_ids


def _runtime_alarm(path: Path, version: int, code: str, message: str, context: dict) -> NoReturn:
    # Fatal on every flush; CRITICAL emission is deduplicated per source (#1402).
    t3_linkage_alarm.raise_runtime_alarm(path, version, code, message, context)


def _source_codex_session_ids(path: Path, version: int) -> set[str]:

    try:
        connection = sqlite3.connect(f"{path.absolute().as_uri()}?mode=ro&immutable=0", uri=True, timeout=1.0)
    except sqlite3.Error as exc:
        _runtime_alarm(
            path,
            version,
            "t3_runtime_unavailable",
            "could not open T3 runtime state read-only",
            {"path": str(path), "error": str(exc)},
        )

    try:
        if version == 2:
            columns = {row[1] for row in connection.execute(f"PRAGMA table_info({_V2_TABLE})")}
            if not {"provider", "payload_json", "provider_thread_id"} <= columns:
                _runtime_alarm(
                    path,
                    version,
                    "t3_runtime_schema_drift",
                    "selected V2 provider projection lacks linkage columns",
                    {"path": str(path)},
                )
            session_ids: set[str] = set()
            driver_column = "driver" if "driver" in columns else "NULL"
            for stored_driver, raw in connection.execute(f"SELECT {driver_column}, payload_json FROM {_V2_TABLE}"):
                try:
                    payload = json.loads(raw)
                    if not isinstance(payload, dict):
                        raise ValueError("provider payload must be an object")
                    payload_driver = payload.get("driver")
                    driver = stored_driver or payload_driver
                    if not isinstance(driver, str) or not driver:
                        raise ValueError("provider linkage must declare its structural driver")
                    if stored_driver and payload_driver and stored_driver != payload_driver:
                        raise ValueError("provider projection and payload drivers disagree")
                    if driver != "codex":
                        continue
                    for field in ("nativeThreadRef", "nativeConversationHeadRef"):
                        ref = payload.get(field)
                        if ref is None:
                            continue
                        if not isinstance(ref, dict) or ref.get("driver") != driver:
                            raise ValueError("native reference must name its structural provider")
                        identity = ref.get("nativeId")
                        if identity is None:
                            continue
                        if not isinstance(identity, str) or not _CODEX_SESSION_ID_RE.fullmatch(identity):
                            raise ValueError("Codex native reference must contain a UUID")
                        session_ids.add(identity.lower())
                except (TypeError, ValueError) as exc:
                    _runtime_alarm(
                        path,
                        version,
                        "t3_runtime_linkage_invalid",
                        "selected V2 provider linkage is malformed",
                        {"path": str(path), "error": str(exc)},
                    )
            return session_ids
        columns = {row[1] for row in connection.execute("PRAGMA table_info(provider_session_runtime)")}
        missing = sorted(_REQUIRED_RUNTIME_COLUMNS - columns)
        if missing:
            _runtime_alarm(
                path,
                version,
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
                _runtime_alarm(
                    path,
                    version,
                    "t3_runtime_linkage_invalid",
                    "provider_session_runtime.resume_cursor_json is not valid JSON",
                    {"path": str(path), "provider_name": provider_name, "error": str(exc)},
                )
            thread_id = resume_cursor.get("threadId") if isinstance(resume_cursor, dict) else None
            if isinstance(thread_id, str) and _CODEX_SESSION_ID_RE.fullmatch(thread_id):
                session_ids.add(thread_id.lower())
        return session_ids
    except sqlite3.Error as exc:
        _runtime_alarm(
            path,
            version,
            "t3_runtime_schema_drift",
            "could not query provider_session_runtime for T3 provenance",
            {"path": str(path), "error": str(exc)},
        )
    finally:
        connection.close()
