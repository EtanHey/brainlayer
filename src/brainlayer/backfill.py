"""Idempotent time-window filtering for transcript watcher backfills."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

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


def _contains_ordered(parts: tuple[str, ...], expected: tuple[str, ...]) -> bool:
    position = 0
    for part in parts:
        if part == expected[position]:
            position += 1
            if position == len(expected):
                return True
    return False


def is_legacy_excluded_path(path: str | Path) -> bool:
    """Identify roots blocked by the blanket denylist retired in July 2026."""
    parts = Path(path).expanduser().parts
    return any(
        _contains_ordered(parts, expected)
        for expected in (
            (".claude", "projects", "subagents"),
            (".codex", "sessions"),
            (".cursor", "agent-transcripts"),
            (".gemini", "sessions"),
        )
    )


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
