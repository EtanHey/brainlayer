"""Shared WAL checkpoint helpers for BrainLayer maintenance commands."""

from __future__ import annotations

import fcntl
import hashlib
import math
import os
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from .paths import get_db_path

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
) -> tuple[int, int, int]:
    """Run WAL checkpoint and return (busy, log_pages, checkpointed_pages)."""
    mode = mode.upper()
    if mode not in _VALID_CHECKPOINT_MODES:
        raise ValueError(f"Invalid checkpoint mode: {mode}")

    with checkpoint_guard(db_path, timeout_seconds=guard_timeout_seconds) as acquired:
        if not acquired:  # pragma: no cover - bounded acquisition resolves or raises
            raise RuntimeError("checkpoint guard was not acquired")
        conn = sqlite3.connect(db_path, timeout=10)
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
