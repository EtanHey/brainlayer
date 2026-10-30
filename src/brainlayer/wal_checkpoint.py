"""Shared WAL checkpoint helpers for BrainLayer maintenance commands."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .paths import get_db_path, supervisor_socket_path

_VALID_CHECKPOINT_MODES = {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}
_DEFAULT_GUARD_TIMEOUT_SECONDS = 10.0
_GUARD_POLL_SECONDS = 0.1


class CheckpointGuardTimeout(TimeoutError):
    """Raised when a live checkpoint owner outlasts the acquisition deadline."""


def checkpoint_lock_path(db_path: str | Path) -> Path:
    """Return the cross-process guard used by checkpoints and recovery."""
    resolved = str(Path(db_path).expanduser().resolve())
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    return Path("/tmp") / f"brainlayer-wal-checkpoint-{digest}.lock"


def _bounded_guard_timeout(value: object) -> float:
    if value is None:
        return _DEFAULT_GUARD_TIMEOUT_SECONDS
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return _DEFAULT_GUARD_TIMEOUT_SECONDS
    if not math.isfinite(parsed):
        return _DEFAULT_GUARD_TIMEOUT_SECONDS
    return max(0.0, parsed)


def _guard_timeout_seconds() -> float:
    return _bounded_guard_timeout(os.environ.get("BRAINLAYER_CHECKPOINT_GUARD_TIMEOUT_SECONDS"))


@contextmanager
def checkpoint_guard(
    db_path: str | Path,
    *,
    blocking: bool = True,
    timeout_seconds: float | None = None,
) -> Iterator[bool]:
    """Serialize checkpoints with bounded wait; let recovery defer immediately."""
    lock_path = checkpoint_lock_path(db_path)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    try:
        if blocking:
            timeout = _guard_timeout_seconds() if timeout_seconds is None else _bounded_guard_timeout(timeout_seconds)
            deadline = time.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except BlockingIOError:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise CheckpointGuardTimeout(
                            f"checkpoint guard acquisition timed out after {timeout:.1f}s for {db_path}"
                        )
                    time.sleep(min(_GUARD_POLL_SECONDS, remaining))
        else:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                pass
        yield acquired
    finally:
        if acquired:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


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


def checkpoint(
    db_path: str,
    mode: str = "TRUNCATE",
    *,
    guard_timeout_seconds: float | None = None,
    busy_timeout_seconds: float = 10.0,
) -> tuple[int, int, int]:
    """Run WAL checkpoint and return (busy, log_pages, checkpointed_pages)."""
    mode = mode.upper()
    if mode not in _VALID_CHECKPOINT_MODES:
        raise ValueError(f"Invalid checkpoint mode: {mode}")
    if not math.isfinite(busy_timeout_seconds) or busy_timeout_seconds < 0:
        raise ValueError("busy_timeout_seconds must be finite and nonnegative")

    with checkpoint_guard(db_path, timeout_seconds=guard_timeout_seconds) as acquired:
        if not acquired:  # pragma: no cover - bounded acquisition resolves or raises
            raise RuntimeError("checkpoint guard was not acquired")
        conn = sqlite3.connect(db_path, timeout=busy_timeout_seconds)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            result = conn.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
            return result
        finally:
            conn.close()


def run_wal_checkpoint(
    mode: str = "TRUNCATE",
    *,
    retry_busy: bool = False,
    max_attempts: int = 8,
    retry_base_seconds: float = 1.0,
    retry_max_seconds: float = 30.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Execute a checkpoint and return structured results."""
    db_path = resolve_db_path()
    if not db_path:
        raise FileNotFoundError("no database found")

    mode = mode.upper()
    wal_before = get_wal_size(db_path)
    attempts = 0
    max_attempts = max(1, int(max_attempts))
    delay = max(0.0, retry_base_seconds)
    while True:
        attempts += 1
        busy, log_pages, checkpointed_pages = checkpoint(db_path, mode)
        if not busy or mode != "TRUNCATE" or not retry_busy or attempts >= max_attempts:
            break
        sleep_fn(delay)
        delay = min(max(delay * 2, retry_base_seconds), max(retry_max_seconds, retry_base_seconds))
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
        "attempts": attempts,
    }


def _fallback_defer_reason(directory):
    from .supervisor_guard import component_gate

    staging = Path(
        os.environ.get("BRAINLAYER_BACKUP_STAGING_DIR") or Path.home() / ".local/share/brainlayer/backups"
    ).expanduser()
    for path, reason in (
        (directory / "calendar/control.lock", "calendar-active"),
        (staging / ".backup.lock", "snapshot-active"),
    ):
        try:
            fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
        except FileNotFoundError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return reason
        finally:
            os.close(fd)
    try:
        with component_gate(directory):
            pass
    except BlockingIOError:
        return "heavy-maintenance-hold"
    # Probe and release: calendar/backup admission must never fail behind fallback I/O.
    # A step starting after a clean probe is safe: SQLite serializes checkpoints with
    # its own reader/writer locks (including backup snapshots); it may return BUSY.
    return ""


def periodic_checkpoint(*, high_water_bytes=256_000_000, busy_timeout_seconds=0.1, now=None):
    """One independent tick. Busy wait is short; SQLite's actual copy I/O is not time-bounded."""
    from . import job_alerts
    from .drive_credentials import _atomic_write

    started = time.monotonic()
    now = time.time() if now is None else now
    db = resolve_db_path()
    if db is None:
        raise FileNotFoundError("no database found")
    state_path = Path(db).with_name("wal-fallback.json")
    state_error = ""
    try:
        previous = json.loads(state_path.read_text())
        if not isinstance(previous, dict):
            raise ValueError("expected object")
        since = previous.get("high_water_since")
        if since is not None and (type(since) not in (int, float) or not math.isfinite(since)):
            raise ValueError("invalid high-water timestamp")
        for key in ("attempted_at", "finished_at"):
            if key in previous:
                datetime.fromisoformat(previous[key])
    except FileNotFoundError:
        previous = {}
    except (OSError, ValueError, TypeError) as exc:
        previous, state_error = {}, f"fallback state unreadable or invalid ({type(exc).__name__})"
    before = get_wal_size(db)
    since = previous.get("high_water_since")
    since = (now if since is None else since) if before >= high_water_bytes else None
    unfinished = previous.get("attempted_at") and not previous.get("finished_at")
    result = dict(
        attempted_at=datetime.fromtimestamp(now, UTC).isoformat(),
        db=db,
        mode="TRUNCATE",
        wal_before_bytes=before,
        attempts=0,
        busy=None,
        log_pages=None,
        checkpointed_pages=None,
        outcome="below-high-water",
        reason="",
        holders=None,
        high_water_since=since,
    )
    # Durable intent survives SIGTERM/SIGKILL inside SQLite. The next tick reports a
    # timeout rather than silently restarting the 1800 s high-water escalation clock.
    _atomic_write(state_path, result)
    if state_error:
        result.update(outcome="error", reason=state_error)
    elif unfinished:
        result.update(
            outcome="timeout",
            reason="previous checkpoint timed out or was interrupted",
            timed_out_at=previous["attempted_at"],
        )
    elif before >= high_water_bytes:
        result["outcome"] = "deferred"
        try:
            result["reason"] = _fallback_defer_reason(supervisor_socket_path().parent)
            # Conservative admission cap, not a throughput promise: huge backfills can
            # lose all progress when the 30 s supervisor budget expires. Leave those to
            # the calendar's longer checkpoint budget and surface them immediately.
            if not result["reason"] and before > 1_000_000_000:
                result["reason"] = "wal-too-large"
            if not result["reason"]:
                result["reason"] = "checkpoint-guard-busy"
                result["attempts"] = 1
                _atomic_write(state_path, result)
                busy, pages, copied = checkpoint(
                    db, "TRUNCATE", guard_timeout_seconds=0, busy_timeout_seconds=busy_timeout_seconds
                )
                result.update(
                    busy=busy,
                    log_pages=pages,
                    checkpointed_pages=copied,
                    reason="sqlite-busy" if busy else "",
                    outcome="deferred" if busy else "reclaimed",
                )
        except (BlockingIOError, CheckpointGuardTimeout):
            pass  # The independent scheduler retries on its next tick; no in-tick retry or process control.
        except (OSError, sqlite3.Error) as exc:
            result.update(outcome="error", reason=f"{type(exc).__name__}: {exc}")
    after = get_wal_size(db)
    since = (now if since is None else since) if after >= high_water_bytes else None
    age = max(0, now - since) if since is not None else 0
    result.update(wal_after_bytes=after, high_water_since=since, high_water_age_seconds=age)
    if result["outcome"] == "reclaimed" and after >= high_water_bytes:
        result.update(outcome="deferred", reason="still-high-water")
    if age >= 1800:
        try:
            holders = subprocess.run(
                ["lsof", "-nP", "-F", "pc", "--", db, f"{db}-wal", f"{db}-shm"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            result["holders"] = dict(
                returncode=holders.returncode, output=holders.stdout[:8192], error=holders.stderr[:1024]
            )
        except (OSError, subprocess.SubprocessError) as exc:
            result["holders"] = dict(unmeasurable=type(exc).__name__)
    result.update(finished_at=datetime.now(UTC).isoformat(), duration_seconds=time.monotonic() - started)
    _atomic_write(state_path, result)
    reason = None
    if age >= 1800 or result["outcome"] in {"error", "timeout"} or result["reason"] == "wal-too-large":
        detail = {
            "sqlite-busy": "checkpoint blocked by active readers or writers",
            "calendar-active": "maintenance calendar is active",
            "snapshot-active": "backup is active",
            "heavy-maintenance-hold": "maintenance hold is active",
            "checkpoint-guard-busy": "another checkpoint is active",
            "wal-too-large": "too large for a safe attempt within 30 s; needs the calendar checkpoint",
        }.get(result["reason"], result["reason"] or "still above high water")
        reason = (
            f"BrainLayer WAL is {format_size(after)}; high water for {age / 60:.0f} min. "
            f"{detail}. See wal-fallback.json for progress and diagnostics."
        )
    job_alerts.report("wal-checkpoint-fallback", reason, path=job_alerts.alert_path(db_path=Path(db)))
    return result


if __name__ == "__main__":
    print(json.dumps(periodic_checkpoint(), sort_keys=True), flush=True)
