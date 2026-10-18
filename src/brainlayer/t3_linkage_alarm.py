"""Durable, per-source pending-linkage status and hourly alarm reminders.

Only emission backs off: every pending observation still raises BrainLayerAlarm.
Status lives beside the resolved BrainLayer DB, never inside the transcript or DB.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path

from . import alarm
from .paths import resolve_db_path

logger = logging.getLogger(__name__)
REMINDER_SECONDS = 3600
_CACHE: dict[Path, tuple[int, dict]] = {}


def status_directory() -> Path:
    return resolve_db_path().parent / "t3-linkage-status"


def observe(source_file: str | Path, session_id: str, sources: list, *, pending: bool) -> None:
    """Record transitions; suppress repeats only after status was persisted."""
    source = str(Path(source_file).resolve())
    directory = status_directory()
    key = hashlib.sha256(source.encode()).hexdigest()
    path = directory / f"{key}.json"
    fingerprint = hashlib.sha256(
        json.dumps([session_id, [(str(p.resolve()), v) for p, v in sources]]).encode()
    ).hexdigest()
    now = time.time()
    state = "pending" if pending else "resolved"
    entry = _CACHE.get(path)
    try:
        cached = entry[1] if entry and path.stat().st_mtime_ns == entry[0] else None
    except OSError:
        cached = None
    if cached and cached["fingerprint"] == fingerprint and cached["state"] == state:
        if not pending or 0 <= now - cached["last_alert_at"] < REMINDER_SECONDS:
            if pending:
                raise alarm.build_alarm(
                    "t3_runtime_linkage_pending",
                    "T3-origin session has no selected projection linkage",
                    {"path": str(source_file)},
                )
            return
    event = alarm.build_alarm(
        "t3_runtime_linkage_pending", "T3-origin session has no selected projection linkage", {"path": str(source_file)}
    )
    emit = pending
    try:
        if not pending and not path.exists():
            return
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"{key}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                previous = json.loads(path.read_text())
                if not isinstance(previous, dict):
                    raise ValueError("status must be an object")
                same = previous.get("fingerprint") == fingerprint and previous.get("state") == state
                last_alert = float(previous.get("last_alert_at", 0))
            except FileNotFoundError:
                previous, same, last_alert = {}, False, 0
            except (ValueError, TypeError) as exc:
                logger.warning("Unreadable T3 linkage alarm status %s: %s", path, exc)
                previous, same, last_alert = {}, False, 0
            emit = pending and (not same or not 0 <= now - last_alert < REMINDER_SECONDS)
            record = (
                dict(previous)
                if same
                else {
                    "code": event.code,
                    "path": source,
                    "fingerprint": fingerprint,
                    "state": state,
                    "first_seen_at": now,
                }
            )
            if emit:
                record["last_alert_at"] = now
            if not same or emit:
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as handle:
                        temporary = Path(handle.name)
                        json.dump(record, handle, sort_keys=True)
                        handle.flush()
                        os.fsync(handle.fileno())
                    temporary.replace(path)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            _CACHE[path] = (path.stat().st_mtime_ns, record)
    except OSError:
        # No durable receipt means no suppression; never pretend a failed status write worked.
        logger.error("Could not persist T3 linkage alarm status %s", path, exc_info=True)
        emit = pending
    if pending:
        if emit:
            alarm.emit_alarm(event)
        raise event
