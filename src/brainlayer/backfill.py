"""Idempotent time-window filtering for transcript watcher backfills."""

from __future__ import annotations

import errno
import fcntl
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from .watcher_bridge import FlushWatermarks


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_backfill_window(since: str | None, until: str | None) -> tuple[datetime, datetime]:
    """Parse a required half-open UTC interval and reject empty ranges."""
    if not since or not until:
        raise ValueError("--since and --until must be provided together")
    try:
        parsed_since = _parse_utc(since)
        parsed_until = _parse_utc(until)
    except (ValueError, OverflowError) as exc:
        raise ValueError("--since and --until must be ISO 8601 timestamps") from exc
    if parsed_since >= parsed_until:
        raise ValueError("--since must be earlier than --until")
    return parsed_since, parsed_until


def window_registry_suffix(since: datetime, until: datetime) -> str:
    """Return a stable filesystem-safe name for a backfill interval."""

    def format_timestamp(value: datetime) -> str:
        base = f"{value:%Y%m%dT%H%M%S}"
        fraction = f"{value.microsecond:06d}" if value.microsecond else ""
        return f"{base}{fraction}Z"

    return f"{format_timestamp(since)}-{format_timestamp(until)}"


class BackfillAlreadyRunning(RuntimeError):
    """Raised when another process owns the same registry-scoped backfill."""


@contextmanager
def backfill_run_lock(registry_path: str | Path) -> Iterator[None]:
    """Serialize scan, enqueue, and offset persistence for one registry."""
    registry = Path(registry_path).expanduser()
    registry.parent.mkdir(parents=True, exist_ok=True)
    lock_path = registry.with_name(f"{registry.name}.backfill.lock")
    with lock_path.open("a+b") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                raise
            raise BackfillAlreadyRunning(f"another backfill is using registry {registry}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _adjacent_pair_indices(parts: tuple[str, ...], first: str, second: str) -> list[int]:
    return [index for index in range(len(parts) - 1) if parts[index : index + 2] == (first, second)]


def is_legacy_excluded_path(path: str | Path) -> bool:
    """Identify roots blocked by the blanket denylist retired in July 2026."""
    parts = Path(path).expanduser().parts
    if _adjacent_pair_indices(parts, ".codex", "sessions"):
        return True
    if _adjacent_pair_indices(parts, ".gemini", "sessions"):
        return True
    for cursor_index, part in enumerate(parts):
        if part == ".cursor" and "agent-transcripts" in parts[cursor_index + 1 :]:
            return True
    for claude_index in _adjacent_pair_indices(parts, ".claude", "projects"):
        if "subagents" in parts[claude_index + 3 :]:
            return True
    return False


class WindowedFlush:
    """Filter normalized watcher entries while confirming every scanned offset."""

    def __init__(
        self,
        downstream: Callable[[list[dict]], dict[str, int] | None],
        *,
        since: datetime,
        until: datetime,
        source_predicate: Callable[[str | Path], bool] | None = None,
    ) -> None:
        self.downstream = downstream
        self.since = since
        self.until = until
        self.source_predicate = source_predicate
        self.scanned_entries = 0
        self.matched_entries = 0
        self.inserted_chunks = 0

    def _matches(self, entry: dict) -> bool:
        source_file = entry.get("_source_file")
        if self.source_predicate and (not isinstance(source_file, str) or not self.source_predicate(source_file)):
            return False
        if entry.get("_timestamp_synthesized") is True:
            return False
        timestamp = entry.get("timestamp")
        if not isinstance(timestamp, str):
            return False
        try:
            parsed = _parse_utc(timestamp)
        except (ValueError, OverflowError):
            return False
        return self.since <= parsed < self.until

    def __call__(self, entries: list[dict]) -> FlushWatermarks | None:
        match_flags = [self._matches(entry) for entry in entries]
        matched = [entry for entry, matches in zip(entries, match_flags, strict=True) if matches]
        downstream_result = self.downstream(matched) if matched else FlushWatermarks()
        self.scanned_entries += len(entries)
        self.matched_entries += len(matched)
        if downstream_result is None:
            return None
        watermarks = dict(downstream_result or {})
        by_source: dict[str, list[tuple[int, bool]]] = {}
        for entry, matches in zip(entries, match_flags, strict=True):
            source_file = entry.get("_source_file")
            offset = entry.get("_line_end_offset")
            if isinstance(source_file, str) and isinstance(offset, int):
                by_source.setdefault(source_file, []).append((offset, matches))
        for source_file, source_entries in by_source.items():
            confirmed = int(watermarks.get(source_file, 0))
            for offset, matches in sorted(source_entries):
                if offset <= confirmed:
                    continue
                if matches:
                    break
                confirmed = offset
            if confirmed > 0:
                watermarks[source_file] = confirmed

        inserted = int(getattr(downstream_result, "inserted", len(matched)))
        downstream_skipped = int(getattr(downstream_result, "skipped", 0))
        self.inserted_chunks += inserted
        return FlushWatermarks(
            watermarks,
            inserted=inserted,
            skipped=downstream_skipped + len(entries) - len(matched),
        )
