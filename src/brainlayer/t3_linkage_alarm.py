"""Durable, per-source T3 alarm status with alert-on-change and hourly reminders.

Only emission backs off: every observation of a T3 alarm still raises BrainLayerAlarm.
Status lives beside the resolved BrainLayer DB, never inside the transcript or DB, and
records only code/path/fingerprint/timestamps -- never alarm context or runtime rows.
Resolved status files (and their locks) are removed, so recurrence re-alerts.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, NoReturn

from . import alarm
from .paths import resolve_db_path

logger = logging.getLogger(__name__)
REMINDER_SECONDS = 3600
_CACHE: dict[Path, tuple[int, dict]] = {}


_READ_ONLY = ContextVar("t3_alarm_read_only", default=False)


def read_only_active():
    return _READ_ONLY.get()


@contextmanager
def read_only():
    """Keep fatal decisions, without changing alarm files or emitting telemetry."""
    token = _READ_ONLY.set(True)
    try:
        yield
    finally:
        _READ_ONLY.reset(token)


def status_directory() -> Path:
    return resolve_db_path().parent / "t3-linkage-status"


def _status_path(key_material: str) -> Path:
    return status_directory() / f"{hashlib.sha256(key_material.encode()).hexdigest()}.json"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def observe(source_file: str | Path, session_id: str, sources: list, *, pending: bool) -> None:
    """Record pending-linkage transitions for one transcript."""
    source = str(Path(source_file).resolve())
    path = _status_path(source)
    if not pending:
        _resolve(path)
        return
    fingerprint = _digest([session_id, [(str(p.resolve()), v) for p, v in sources]])
    event = alarm.build_alarm(
        "t3_runtime_linkage_pending", "T3-origin session has no selected projection linkage", {"path": str(source_file)}
    )
    _raise_deduplicated(path, source, fingerprint, event)


def raise_runtime_alarm(
    runtime_path: str | Path, version: int, code: str, message: str, context: dict[str, Any]
) -> NoReturn:
    """Raise a T3 runtime alarm (unavailable/schema_drift/linkage_invalid) with deduplicated emission."""
    source = str(Path(runtime_path).resolve())
    event = alarm.build_alarm(code, message, context)
    _raise_deduplicated(
        _status_path(f"t3-runtime:{version}:{source}"), source, _digest([code, message, context]), event
    )


def runtime_resolved(runtime_path: str | Path, version: int) -> None:
    _resolve(_status_path(f"t3-runtime:{version}:{Path(runtime_path).resolve()}"))


def _resolve(path: Path) -> None:
    if read_only_active():
        return
    _CACHE.pop(path, None)
    if not path.exists():
        return
    lock_path = path.with_suffix(".lock")
    try:
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Removing the lock inode can at worst let a racing writer skip suppression once;
            # suppression always needs a persisted same-state receipt, so no alarm is silenced.
            path.unlink(missing_ok=True)
            lock_path.unlink(missing_ok=True)
    except BlockingIOError:
        logger.info("T3 alarm status %s is busy; resolution cleanup deferred", path)
    except OSError as exc:
        logger.warning("Could not clear resolved T3 alarm status %s: %s", path, exc)


def _raise_deduplicated(path: Path, source: str, fingerprint: str, event: alarm.BrainLayerAlarm) -> NoReturn:
    """Always raise; emit only on change, hourly, or when no durable receipt exists."""
    if read_only_active():
        raise event
    now = time.time()
    entry = _CACHE.get(path)
    try:
        cached = entry[1] if entry and path.stat().st_mtime_ns == entry[0] else None
    except OSError:
        cached = None
    if (
        cached
        and cached["fingerprint"] == fingerprint
        and cached["state"] == "pending"
        and 0 <= now - cached["last_alert_at"] < REMINDER_SECONDS
    ):
        _count(now, "suppressed_unchanged")
        raise event
    emit = True
    reason = "changed"
    directory = path.parent
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                previous = json.loads(path.read_text())
                if not isinstance(previous, dict):
                    raise ValueError("status must be an object")
                same = previous.get("fingerprint") == fingerprint and previous.get("state") == "pending"
                last_alert = float(previous.get("last_alert_at", 0))
            except FileNotFoundError:
                previous, same, last_alert = {}, False, 0
            except (ValueError, TypeError) as exc:
                logger.warning("Unreadable T3 linkage alarm status %s: %s", path, exc)
                previous, same, last_alert = {}, False, 0
            emit = not same or not 0 <= now - last_alert < REMINDER_SECONDS
            reason = "hourly_reminder" if same else "changed"
            record = (
                dict(previous)
                if same
                else {
                    "code": event.code,
                    "path": source,
                    "fingerprint": fingerprint,
                    "state": "pending",
                    "first_seen_at": now,
                }
            )
            if emit:
                record["last_alert_at"] = now
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
    except BlockingIOError:
        # Another process holds this source's status; without our own receipt, stay loud.
        logger.info("T3 alarm status %s is busy; emitting %s without suppression", path, event.code)
        emit = True
        reason = "lock_or_persist_failed"
    except OSError:
        # No durable receipt means no suppression; never pretend a failed status write worked.
        logger.error("Could not persist T3 linkage alarm status %s", path, exc_info=True)
        emit = True
        reason = "lock_or_persist_failed"
    _count(now, reason if emit else "suppressed_unchanged")
    if emit:
        event.context.update(emission_reason=reason, source_fingerprint=fingerprint)
        alarm.emit_alarm(event)
    raise event


def _counter_directory() -> Path:
    return status_directory().with_name("t3-linkage-counts")


_COUNTER_KEYS = ("observed", "emitted", "suppressed_unchanged", "emitted_because_lock_or_persist_failed")


def _read_counter(path: Path, hour: int) -> dict[str, int]:
    zero = {"hour": hour, **dict.fromkeys(_COUNTER_KEYS, 0)}
    try:
        row = json.loads(path.read_text())
        if not isinstance(row, dict) or row.get("hour") != hour:
            raise ValueError("invalid bucket hour")
        if any(type(row.get(key)) is not int or row[key] < 0 for key in _COUNTER_KEYS):
            raise ValueError("invalid bucket counters")
        return {key: row[key] for key in zero}
    except FileNotFoundError:
        return zero
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Unreadable T3 alarm counter bucket %s; treating as zero: %s", path, exc)
        return zero


def _count(now: float, reason: str) -> None:
    """Update bounded hourly diagnostics under a shared lock.

    Only a new bucket is fsynced; crash loss of subsequent increments is acceptable.
    Retain at most seven days, including the current hour. Counter failures never
    change alarm emission or suppression.
    """
    hour = int(now // REMINDER_SECONDS)
    increments = {
        "observed": 1,
        "emitted": int(reason != "suppressed_unchanged"),
        "suppressed_unchanged": int(reason == "suppressed_unchanged"),
        "emitted_because_lock_or_persist_failed": int(reason == "lock_or_persist_failed"),
    }
    try:
        directory = _counter_directory()
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "counts.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            target = directory / f"{hour}.json"
            new_bucket = not target.exists()
            row = _read_counter(target, hour)
            for key, value in increments.items():
                row[key] += value
            with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as handle:
                temporary = Path(handle.name)
                try:
                    json.dump(row, handle)
                    handle.flush()
                    if new_bucket:
                        os.fsync(handle.fileno())
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
            for path in directory.glob("*.json"):
                if path.stem.isdecimal() and int(path.stem) <= hour - 7 * 24:
                    path.unlink(missing_ok=True)
    except BlockingIOError:
        logger.debug("T3 alarm counters are busy; diagnostic increment skipped")
    except (OSError, ValueError, TypeError, KeyError):
        logger.error("Could not persist T3 alarm counters", exc_info=True)


def alarm_counts() -> list[dict[str, int]]:
    """Read diagnostics-only totals for the current and preceding 167 hours.

    Unreadable buckets are logged once per read and treated as zero. Reads never
    modify receipts or alarm state; expired buckets are ignored even without writes.
    No BrainBar or health caller consumes these totals yet.
    """
    hour = int(time.time() // REMINDER_SECONDS)
    rows = []
    for path in _counter_directory().glob("*.json"):
        if path.stem.isdecimal() and hour - 168 < int(path.stem) <= hour:
            rows.append(_read_counter(path, int(path.stem)))
    return sorted(rows, key=lambda row: row["hour"])
