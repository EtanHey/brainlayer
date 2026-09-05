"""Real-time JSONL file watcher for BrainLayer.

Watches ~/.claude/projects/ for new and modified .jsonl files,
tail-follows them from last known offset, and queues parsed lines
for batch insertion into the BrainLayer database.

Architecture (from R47 research):
  1. Directory watcher detects new .jsonl files
  2. Per-file tailer reads from stored offset, buffers partial lines
  3. BatchIndexer accumulates parsed lines and flushes periodically
  4. Offset registry persists progress to survive restarts

This is the Python watchdog prototype. The production version will use
Swift DispatchSource kqueue in BrainBar for sub-1ms notification latency.
"""

import errno
import hashlib
import json
import logging
import math
import os
import sqlite3
import stat
import tempfile
import threading
import time
from collections.abc import Iterator
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised via the Windows lock seam
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt
except ImportError:  # pragma: no cover - unavailable on POSIX
    msvcrt = None  # type: ignore[assignment]

from .alarm import BrainLayerAlarm, raise_alarm
from .ingest_denylist import _match_parts as match_glob_parts
from .ingest_denylist import is_denylisted, is_directory_denylisted

logger = logging.getLogger(__name__)

_WATCH_MAX_FILE_BYTES_ENV = "BRAINLAYER_WATCH_MAX_FILE_BYTES"
_DEFAULT_WATCH_MAX_FILE_BYTES = 100 * 1024 * 1024
_WATCH_MAX_RECORD_BYTES_ENV = "BRAINLAYER_WATCH_MAX_RECORD_BYTES"
_DEFAULT_WATCH_MAX_RECORD_BYTES = 128 * 1024 * 1024
_WATCH_READ_CHUNK_BYTES = 64 * 1024
_MAX_HEALTH_FAILURE_DETAILS = 100
_MAX_HEALTH_QUARANTINE_DETAILS = 100


_OFFSET_PRUNE_RETRY_ENV = "BRAINLAYER_WATCHER_OFFSET_PRUNE_RETRY_S"
_DEFAULT_OFFSET_PRUNE_RETRY_S = 900.0

MIN_WATCH_POLL_INTERVAL_S = 30.0


def enforce_min_poll_interval(poll_interval_s: float) -> float:
    """Hold `brainlayer watch --poll` to the R3 >=30s floor, loudly.

    The floor was asserted three ways against values that cannot be violated -- the CLI
    option default and the `--poll` argument in both repo plists -- and against nothing at
    the one boundary a value actually arrives through. That gap is not theoretical: the
    installed ~/Library/LaunchAgents/com.brainlayer.watch.plist still passes `--poll 1.0`,
    so re-enabling the label without re-running scripts/launchd/install.sh watch hands this
    function the exact configuration R3 exists to remove.

    Clamp rather than exit: `watch` runs under launchd KeepAlive, so refusing a stale plist
    turns it into a restart loop that ingests nothing -- trading a CPU burn for a total
    ingestion outage. Clamping keeps the watcher serving at the floor and puts the
    violation in watch.err.log at WARNING, where the operator can see it and fix the source.

    isfinite before the bounds test, for the same reason as
    `_watch_offset_prune_retry_interval_s`: `float("nan") < 30.0` is False, so nan would
    sail past a bare `< MIN` check into `Event.wait(nan)`, and `inf` would park the poll
    loop forever -- both silent, both worse than the value they replaced.
    """
    if math.isfinite(poll_interval_s) and poll_interval_s >= MIN_WATCH_POLL_INTERVAL_S:
        return poll_interval_s
    logger.warning(
        "--poll %r violates the R3 >=%.0fs batching constraint; clamping to %.0fs. "
        "Something still asks this watcher to poll sub-30s -- most likely a stale "
        "~/Library/LaunchAgents/com.brainlayer.watch.plist; re-run "
        "scripts/launchd/install.sh watch to fix it at the source.",
        poll_interval_s,
        MIN_WATCH_POLL_INTERVAL_S,
        MIN_WATCH_POLL_INTERVAL_S,
    )
    return MIN_WATCH_POLL_INTERVAL_S


def _watch_offset_prune_retry_interval_s() -> float:
    """Seconds to wait before re-attempting a prune that could not complete."""
    raw_value = os.environ.get(_OFFSET_PRUNE_RETRY_ENV, str(_DEFAULT_OFFSET_PRUNE_RETRY_S))
    try:
        parsed_value = float(raw_value)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using default %s",
            _OFFSET_PRUNE_RETRY_ENV,
            raw_value,
            _DEFAULT_OFFSET_PRUNE_RETRY_S,
        )
        return _DEFAULT_OFFSET_PRUNE_RETRY_S
    # isfinite before the sign test: float("nan") and float("inf") survive both
    # ValueError and `<= 0`, and `monotonic() - attempt >= nan/inf` is never true, which
    # would silently disable the retry timer for the life of the process.
    if not math.isfinite(parsed_value) or parsed_value <= 0:
        logger.warning(
            "%s must be a positive finite number; using default %s",
            _OFFSET_PRUNE_RETRY_ENV,
            _DEFAULT_OFFSET_PRUNE_RETRY_S,
        )
        return _DEFAULT_OFFSET_PRUNE_RETRY_S
    return parsed_value


def _watch_read_window_bytes() -> int:
    """Load the per-file, per-poll read window from the legacy environment name."""
    raw_value = os.environ.get(_WATCH_MAX_FILE_BYTES_ENV, str(_DEFAULT_WATCH_MAX_FILE_BYTES))
    try:
        parsed_value = int(raw_value)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using default %d",
            _WATCH_MAX_FILE_BYTES_ENV,
            raw_value,
            _DEFAULT_WATCH_MAX_FILE_BYTES,
        )
        return _DEFAULT_WATCH_MAX_FILE_BYTES
    if parsed_value <= 0:
        logger.warning(
            "Invalid %s=%r; using default %d",
            _WATCH_MAX_FILE_BYTES_ENV,
            raw_value,
            _DEFAULT_WATCH_MAX_FILE_BYTES,
        )
        return _DEFAULT_WATCH_MAX_FILE_BYTES
    return parsed_value


def _watch_max_record_bytes() -> int:
    """Load the hard in-memory ceiling for one JSONL record."""
    raw_value = os.environ.get(_WATCH_MAX_RECORD_BYTES_ENV, str(_DEFAULT_WATCH_MAX_RECORD_BYTES))
    try:
        parsed_value = int(raw_value)
    except ValueError:
        parsed_value = 0
    if parsed_value <= 0:
        logger.warning(
            "Invalid %s=%r; using default %d",
            _WATCH_MAX_RECORD_BYTES_ENV,
            raw_value,
            _DEFAULT_WATCH_MAX_RECORD_BYTES,
        )
        return _DEFAULT_WATCH_MAX_RECORD_BYTES
    return parsed_value


_RECURSIVE_JSONL_GLOB = "**/*.jsonl"


class _JSONLPatternMatcher:
    """Decides whether one DirEntry under `base` is a file that matches `glob_pattern`.

    The default `**/*.jsonl` short-circuits on the suffix and never builds a parts tuple; any
    other pattern is matched on the path relative to base with the denylist's `**`-aware part
    matcher. A regular file is required either way (DirEntry.is_file follows symlinks, as
    pathlib's did).
    """

    def __init__(self, base: str, glob_pattern: str) -> None:
        self.default_pattern = glob_pattern == _RECURSIVE_JSONL_GLOB
        self.pattern_parts = tuple(part for part in glob_pattern.split("/") if part)
        self.suffix_only = bool(self.pattern_parts) and self.pattern_parts[-1] == "*.jsonl"
        self.relative_start = len(base.rstrip(os.sep)) + 1

    def __call__(self, entry: os.DirEntry) -> bool:
        if self.suffix_only and not entry.name.endswith(".jsonl"):
            return False
        if not entry.is_file():
            return False
        if self.default_pattern:
            return True
        return match_glob_parts(tuple(entry.path[self.relative_start :].split(os.sep)), self.pattern_parts)


def _iter_jsonl_files(
    base: Path,
    glob_pattern: str,
    skip_dir: Callable[[str], bool] | None = None,
) -> Iterator[tuple[str, Callable[[], os.stat_result]]]:
    """Yield (path, stat) for every regular file under base that matches glob_pattern.

    An os.scandir walk in place of pathlib.glob, for every root. Semantics kept from the
    glob: directory symlinks are not descended (pathlib's recurse_symlinks default), file
    symlinks count if their target is a regular file, dotfiles match, matching is
    case-sensitive, and the stat handed back follows the symlink. Files are matched against
    the pattern relative to base with the same `**`-aware part matcher the denylist uses, so
    cursor's `**/agent-transcripts/**/*.jsonl` root takes the same path as the default
    `**/*.jsonl` -- which short-circuits on the suffix and never builds a parts tuple. One
    narrowing, on purpose: pathlib would step into a directory symlink whose name matched
    a literal pattern segment; this walk never follows directory symlinks at all.

    `skip_dir(path)` is asked before descending into any directory (including base) and a
    True answer prunes the whole subtree. The watcher passes the denylist's subtree verdict:
    the deployed `~/.cursor/**/agent-transcripts/**` covers 4,315 of the 5,086 directories
    under ~/.cursor/projects, and the walk was reading every one of them on every poll --
    about 1s of kernel time per poll on the efficiency cores -- to discover 4,302 files that
    the per-file denylist then discarded. Measured with the pruning, the same roots read
    ~470 directories.

    The stat is returned as a callable so the caller can apply the per-file denylist first
    and skip the stat entirely for a file it will not track.
    """
    base_str = str(base)
    matches = _JSONLPatternMatcher(base_str, glob_pattern)
    # No early `return` before the first yield: a generator that can also "return None" reads
    # as a possible non-sequence to static analysis at the unpacking call site.
    pending = [] if skip_dir is not None and skip_dir(base_str) else [base_str]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if skip_dir is None or not skip_dir(entry.path):
                                pending.append(entry.path)
                        elif matches(entry):
                            yield entry.path, entry.stat
                    except OSError:
                        continue
        except OSError:
            continue


# The health probe counts realtime_watcher rows written since the current health window
# opened. The original single statement,
#     COUNT(*) WHERE source = 'realtime_watcher'
#              AND COALESCE(ingested_at, CAST(strftime('%s', created_at) AS INTEGER)) >= ?
# could only be served by idx_chunks_source, so it visited every realtime_watcher row in the
# table (451,000 on the real DB, 0.33s CPU) on every poll to count the ~50 written in the last
# minute. The two statements below are the two branches of that COALESCE, each on the range
# index that fits it; the `+column` markers keep the planner off the source index. The
# created_at text floor is a day earlier than the window so a legacy timestamp written with a
# non-UTC offset still lands inside the index range; the strftime term then applies the
# exact original comparison to those few rows.
_REALTIME_INSERTS_SINCE_SQL = """
    SELECT COUNT(*) FROM chunks
    WHERE +source = 'realtime_watcher'
      AND ingested_at >= ?
"""
_REALTIME_LEGACY_CREATED_SINCE_SQL = """
    SELECT COUNT(*) FROM chunks
    WHERE +source = 'realtime_watcher'
      AND +ingested_at IS NULL
      AND created_at >= ?
      AND CAST(strftime('%s', created_at) AS INTEGER) >= ?
"""
_CREATED_AT_FLOOR_SLACK_S = 24 * 60 * 60


def realtime_insert_probe_statements(window_start: int) -> list[tuple[str, tuple[Any, ...]]]:
    """The (sql, params) pairs whose counts sum to realtime_watcher rows since window_start."""
    created_at_floor = datetime.fromtimestamp(window_start - _CREATED_AT_FLOOR_SLACK_S, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    return [
        (_REALTIME_INSERTS_SINCE_SQL, (window_start,)),
        (_REALTIME_LEGACY_CREATED_SINCE_SQL, (created_at_floor, window_start)),
    ]


@dataclass(frozen=True)
class WatchRoot:
    provider: str
    path: Path | str
    glob_pattern: str = _RECURSIVE_JSONL_GLOB

    @property
    def resolved_path(self) -> Path:
        return Path(self.path).expanduser()


def default_watch_roots(home: Path | None = None) -> list[WatchRoot]:
    root = home or Path.home()
    return [
        WatchRoot("claude", root / ".claude" / "projects"),
        WatchRoot("codex", root / ".codex" / "sessions"),
        WatchRoot("cursor", root / ".cursor" / "sessions"),
        WatchRoot(
            "cursor-agent-transcripts",
            root / ".cursor" / "projects",
            "**/agent-transcripts/**/*.jsonl",
        ),
        WatchRoot("gemini", root / ".gemini" / "sessions"),
    ]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("text", "content", "message"):
            text = _content_to_text(content.get(key))
            if text:
                return text
        return ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") not in {None, "text", "input_text", "output_text"}:
                continue
            text = _content_to_text(item)
            if text:
                parts.append(text)
        return " ".join(parts)
    return ""


def _mapping_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def normalize_provider_entry(entry: dict[str, Any], provider: str) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None

    entry_type = entry.get("type")
    if entry_type in {"user", "assistant"} and isinstance(entry.get("message"), dict):
        normalized = dict(entry)
        normalized["_provider"] = provider
        return normalized

    payload = entry.get("payload")
    payload_entry = payload if isinstance(payload, dict) else None
    if provider == "codex" and entry_type == "response_item":
        if not payload_entry or payload_entry.get("type") != "message":
            return None
        candidate = {**payload_entry, "timestamp": entry.get("timestamp")}
    elif payload_entry:
        candidate = {**payload_entry, "timestamp": entry.get("timestamp") or payload_entry.get("timestamp")}
    else:
        candidate = entry

    role = (
        candidate.get("role")
        or candidate.get("sender")
        or candidate.get("speaker")
        or _mapping_value(candidate.get("author")).get("role")
        or _mapping_value(candidate.get("message")).get("role")
    )
    role = str(role or "").lower()
    if role in {"model", "gemini", "ai", "bot"}:
        role = "assistant"
    if role not in {"user", "assistant"}:
        return None

    text = _content_to_text(candidate.get("content") or candidate.get("text") or candidate.get("message"))
    if not text:
        return None

    return {
        "type": role,
        "message": {"role": role, "content": [{"type": "text", "text": text}]},
        "timestamp": candidate.get("timestamp")
        or candidate.get("created_at")
        or datetime.now(timezone.utc).isoformat(),
        "_provider": provider,
    }


class CoverageWatchdog:
    def __init__(
        self,
        *,
        coverage_ratio_threshold: float = 0.25,
        lag_threshold_bytes: int = 1_048_576,
        alert_after_s: float = 300.0,
        now_fn: Callable[[], float] = time.monotonic,
    ):
        self.coverage_ratio_threshold = coverage_ratio_threshold
        self.lag_threshold_bytes = lag_threshold_bytes
        self.alert_after_s = alert_after_s
        self.now_fn = now_fn
        self._coverage_bad_since: float | None = None
        self._lag_bad_since: float | None = None

    def evaluate(
        self,
        *,
        active_entries_per_minute: float,
        realtime_inserts_per_minute: float,
        max_offset_lag_bytes: int,
    ) -> dict[str, Any]:
        now = self.now_fn()
        reasons = []
        coverage_bad = (
            active_entries_per_minute > 0
            and realtime_inserts_per_minute / active_entries_per_minute < self.coverage_ratio_threshold
        )
        if coverage_bad:
            self._coverage_bad_since = now if self._coverage_bad_since is None else self._coverage_bad_since
            if now - self._coverage_bad_since >= self.alert_after_s:
                reasons.append("coverage_drop")
        else:
            self._coverage_bad_since = None

        lag_bad = max_offset_lag_bytes > self.lag_threshold_bytes
        if lag_bad:
            self._lag_bad_since = now if self._lag_bad_since is None else self._lag_bad_since
            if now - self._lag_bad_since >= self.alert_after_s:
                reasons.append("offset_lag")
        else:
            self._lag_bad_since = None

        return {"alerting": bool(reasons), "alert_reasons": reasons}


# ── Offset Registry ──────────────────────────────────────────────────────────


_OFFSET_TOMBSTONES_KEY = "__brainlayer_offset_tombstones__"
_OFFSET_TOMBSTONE_RETENTION_S = 24 * 60 * 60


def _lock_offset_registry_file(lock_file) -> None:
    """Acquire an exclusive advisory lock on POSIX or Windows."""
    if fcntl is not None:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        return
    if msvcrt is None:
        raise OSError("no supported offset-registry file lock is available")
    lock_file.seek(0, os.SEEK_END)
    if lock_file.tell() == 0:
        lock_file.write(b"\0")
        lock_file.flush()
    lock_file.seek(0)
    while True:
        try:
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError as error:
            if error.errno not in {errno.EACCES, errno.EAGAIN}:
                raise
            time.sleep(0.05)


def _unlock_offset_registry_file(lock_file) -> None:
    """Release the platform-specific advisory registry lock."""
    lock_file.seek(0)
    if fcntl is not None:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    elif msvcrt is not None:
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


class OffsetRegistry:
    """Persists file read offsets so we resume after restart.

    Stored as JSON: {filepath: {offset, inode, mtime}}
    Atomic writes via tmpfile + rename.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, dict] = {}
        self._removed: dict[str, dict[str, int | float]] = {}
        self._dirty_paths: set[str] = set()
        self._dirty = False
        self._last_prune_complete = True
        self._load()

    @staticmethod
    def _sanitize_tombstones(raw: object) -> dict[str, dict[str, int | float]]:
        """Load legacy timestamps and validated generation-aware tombstones."""
        if not isinstance(raw, dict):
            return {}
        sanitized: dict[str, dict[str, int | float]] = {}
        for filepath, raw_tombstone in raw.items():
            if not isinstance(filepath, str):
                continue
            if isinstance(raw_tombstone, (int, float)) and not isinstance(raw_tombstone, bool):
                removed_at = float(raw_tombstone)
                generation = 0
                inode = 0
            elif isinstance(raw_tombstone, dict):
                raw_removed_at = raw_tombstone.get("removed_at")
                raw_generation = raw_tombstone.get("generation", 0)
                raw_inode = raw_tombstone.get("inode", 0)
                if not isinstance(raw_removed_at, (int, float)) or isinstance(raw_removed_at, bool):
                    continue
                if not isinstance(raw_generation, int) or isinstance(raw_generation, bool) or raw_generation < 0:
                    continue
                if not isinstance(raw_inode, int) or isinstance(raw_inode, bool) or raw_inode < 0:
                    continue
                removed_at = float(raw_removed_at)
                generation = raw_generation
                inode = raw_inode
            else:
                continue
            if math.isfinite(removed_at):
                sanitized[filepath] = {
                    "removed_at": removed_at,
                    "generation": generation,
                    "inode": inode,
                }
        return sanitized

    @staticmethod
    def _entry_generation(entry: object) -> int:
        """Return a validated rewind generation for one registry entry."""
        if not isinstance(entry, dict):
            return 0
        generation = entry.get("generation", 0)
        if isinstance(generation, int) and not isinstance(generation, bool) and generation >= 0:
            return generation
        return 0

    def _load(self):
        try:
            with open(self.path) as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                loaded = {}
            tombstones = loaded.pop(_OFFSET_TOMBSTONES_KEY, {})
            self._data = loaded
            self._removed = self._sanitize_tombstones(tombstones)
        except (OSError, json.JSONDecodeError):
            self._data = {}
            self._removed = {}

    def get(self, filepath: str) -> tuple[int, int]:
        """Return (offset, inode) for a file. (0, 0) if unknown."""
        entry = self._data.get(filepath, {})
        return entry.get("offset", 0), entry.get("inode", 0)

    def generation(self, filepath: str) -> int:
        """Return the current rewind generation for a file."""
        return self._entry_generation(self._data.get(filepath))

    def set(self, filepath: str, offset: int, inode: int, *, generation: int | None = None):
        """Update offset for a file, optionally preserving its validated generation."""
        current_generation = self._entry_generation(self._data.get(filepath))
        preserve_generation = generation is not None
        if generation is None:
            generation = current_generation
        elif not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
            raise ValueError("generation must be a non-negative integer")
        elif generation < current_generation:
            return
        tombstone = self._removed.get(filepath)
        valid_inode = isinstance(inode, int) and not isinstance(inode, bool) and inode > 0
        if tombstone is not None and valid_inode:
            if preserve_generation and generation <= int(tombstone["generation"]):
                return
            if not preserve_generation:
                generation = max(generation, int(tombstone["generation"]) + 1, time.time_ns())
        self._data[filepath] = {
            "offset": offset,
            "inode": inode,
            "mtime": time.time(),
            "generation": generation,
        }
        if tombstone is None or valid_inode:
            self._removed.pop(filepath, None)
        self._dirty_paths.add(filepath)
        self._dirty = True

    def flush(self) -> bool:
        """Atomically write to disk. Returns True on success."""
        if not self._dirty:
            return True
        tmp_path = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self.path.with_name(f"{self.path.name}.lock")
            with lock_path.open("a+b") as lock_file:
                _lock_offset_registry_file(lock_file)
                try:
                    try:
                        with self.path.open() as registry_file:
                            disk_data = json.load(registry_file)
                    except FileNotFoundError:
                        disk_data = {}
                    except (OSError, UnicodeError, json.JSONDecodeError) as error:
                        logger.warning("Failed to read offset registry before merge: %s", error)
                        return False
                    if not isinstance(disk_data, dict):
                        logger.warning("Failed to read offset registry before merge: expected a JSON object")
                        return False

                    disk_tombstones = disk_data.pop(_OFFSET_TOMBSTONES_KEY, {})
                    tombstones = self._sanitize_tombstones(disk_tombstones)
                    for filepath, local_tombstone in self._removed.items():
                        disk_tombstone = tombstones.get(filepath)
                        if disk_tombstone is None or (
                            int(local_tombstone["generation"]),
                            float(local_tombstone["removed_at"]),
                        ) > (
                            int(disk_tombstone["generation"]),
                            float(disk_tombstone["removed_at"]),
                        ):
                            tombstones[filepath] = local_tombstone

                    merged = dict(disk_data)
                    for filepath in self._dirty_paths:
                        local_entry = self._data.get(filepath)
                        if not isinstance(local_entry, dict):
                            continue
                        local_mtime = local_entry.get("mtime", 0)
                        tombstone = tombstones.get(filepath)
                        if tombstone is not None:
                            local_generation = self._entry_generation(local_entry)
                            if local_generation <= int(tombstone["generation"]):
                                merged.pop(filepath, None)
                                continue
                        disk_entry = disk_data.get(filepath)
                        if not isinstance(disk_entry, dict):
                            merged[filepath] = local_entry
                            tombstones.pop(filepath, None)
                            continue
                        local_generation = self._entry_generation(local_entry)
                        disk_generation = self._entry_generation(disk_entry)
                        if local_generation > disk_generation:
                            merged[filepath] = local_entry
                            tombstones.pop(filepath, None)
                            continue
                        if local_generation < disk_generation:
                            continue
                        same_inode = local_entry.get("inode", 0) == disk_entry.get("inode", 0)
                        if same_inode:
                            if local_entry.get("offset", 0) >= disk_entry.get("offset", 0):
                                merged[filepath] = local_entry
                                tombstones.pop(filepath, None)
                        elif local_mtime >= disk_entry.get("mtime", 0):
                            merged[filepath] = local_entry
                            tombstones.pop(filepath, None)
                    for filepath, tombstone in list(tombstones.items()):
                        disk_entry = merged.get(filepath)
                        if isinstance(disk_entry, dict):
                            newer_generation = self._entry_generation(disk_entry) > int(tombstone["generation"])
                            if newer_generation:
                                tombstones.pop(filepath, None)
                                continue
                        merged.pop(filepath, None)
                    tombstone_cutoff = time.time() - _OFFSET_TOMBSTONE_RETENTION_S
                    tombstones = {
                        filepath: tombstone
                        for filepath, tombstone in tombstones.items()
                        if float(tombstone["removed_at"]) >= tombstone_cutoff
                    }

                    fd, tmp_path = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
                    with os.fdopen(fd, "w") as registry_file:
                        persisted = dict(merged)
                        if tombstones:
                            persisted[_OFFSET_TOMBSTONES_KEY] = tombstones
                        json.dump(persisted, registry_file)
                        registry_file.flush()
                        os.fsync(registry_file.fileno())
                    os.replace(tmp_path, self.path)
                    directory_fd = os.open(self.path.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                    self._data = merged
                    self._removed = tombstones
                    self._dirty_paths.clear()
                finally:
                    _unlock_offset_registry_file(lock_file)
            self._dirty = False
            return True
        except OSError as e:
            logger.warning("Failed to flush offset registry: %s", e)
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            return False

    def remove(self, filepath: str):
        """Remove tracking for a file."""
        removed_entry = self._data.pop(filepath, None)
        existing_tombstone = self._removed.get(filepath)
        entry_generation = self._entry_generation(removed_entry)
        existing_generation = int(existing_tombstone["generation"]) if existing_tombstone else 0
        inode = removed_entry.get("inode", 0) if isinstance(removed_entry, dict) else 0
        if not isinstance(inode, int) or isinstance(inode, bool) or inode < 0:
            inode = 0
        self._removed[filepath] = {
            "removed_at": time.time(),
            "generation": max(entry_generation + 1, existing_generation, time.time_ns()),
            "inode": inode or (int(existing_tombstone["inode"]) if existing_tombstone else 0),
        }
        self._dirty_paths.discard(filepath)
        self._dirty = True

    def mark_rewind(self, filepath: str, inode: int) -> int:
        """Start a newer offset generation so a confirmed rewind may move backward."""
        current_generation = self._entry_generation(self._data.get(filepath))
        tombstone = self._removed.get(filepath)
        tombstone_generation = int(tombstone["generation"]) if tombstone else 0
        generation = max(current_generation + 1, tombstone_generation + 1, time.time_ns())
        self._data[filepath] = {
            "offset": 0,
            "inode": inode,
            "mtime": time.time(),
            "generation": generation,
        }
        self._removed.pop(filepath, None)
        self._dirty_paths.add(filepath)
        self._dirty = True
        return generation

    @staticmethod
    def _has_unavailable_symlink_ancestor(candidate: Path, root: Path) -> bool:
        """Return True when a path crosses a symlink whose target is unavailable."""
        ancestor = candidate
        while ancestor != root and ancestor.is_relative_to(root):
            try:
                ancestor_mode = ancestor.lstat().st_mode
            except FileNotFoundError:
                ancestor = ancestor.parent
                continue
            except OSError:
                return True
            if stat.S_ISLNK(ancestor_mode):
                try:
                    target_mode = ancestor.stat().st_mode
                except OSError:
                    return True
                target_type_matches = stat.S_ISREG(target_mode) if ancestor == candidate else stat.S_ISDIR(target_mode)
                if not target_type_matches:
                    return True
            ancestor = ancestor.parent
        return False

    @staticmethod
    def _has_live_parent_evidence(candidate: Path, live_parent_dirs: AbstractSet[Path]) -> bool:
        """Require a live transcript in the tracked file's containing directory."""
        parent = candidate.parent
        try:
            if not parent.is_dir():
                return False
        except OSError:
            return False
        return parent in live_parent_dirs

    @property
    def last_prune_complete(self) -> bool:
        """Whether the last prune safely evaluated every tracked root."""
        return self._last_prune_complete

    def has_stale_entries_under(self, new_dirs: AbstractSet[str], live_files: list[str]) -> bool:
        """Whether a tracked entry that is not a live file has its parent on the path to one of new_dirs.

        That is the one shape a prune could act on that a newly appeared directory can create:
        a returning volume brings its directories back with the deleted files' registry entries
        still pointing under them. The relation is the one `prune_missing_files` itself uses as
        live evidence -- `_has_live_parent_evidence` accepts the entry's parent being ANY
        ancestor of a live file -- so a volume that remounts with live files only in a
        subdirectory of the stale entry's parent (new dir `r/sub`, stale entry `r/gone.jsonl`)
        is prunable and must re-trigger too. String prefixes, not pathlib, for the same reason
        as the caller; new_dirs is small (the directories that appeared since the last poll).
        """
        live = set(live_files)
        candidates = tuple(new_dirs)
        for filepath in self._data:
            if filepath in live:
                continue
            parent = filepath.rpartition(os.sep)[0]
            prefix = f"{parent}{os.sep}"
            if any(directory == parent or directory.startswith(prefix) for directory in candidates):
                return True
        return False

    def prune_missing_files(
        self,
        active_roots: list[Path],
        active_files: list[str | Path] | None = None,
    ) -> int:
        """Drop deleted offsets only when a sibling file proves the root is mounted."""
        self._last_prune_complete = True
        live_files: list[Path] = []
        candidates = active_files if active_files is not None else list(self._data)
        for filepath in candidates:
            candidate = Path(os.path.abspath(os.path.expanduser(str(filepath))))
            try:
                if candidate.is_file():
                    live_files.append(candidate)
            except OSError:
                continue
        live_parent_dirs = {parent for live_file in live_files for parent in live_file.parents}

        root_availability: dict[Path, bool] = {}
        for root in active_roots:
            candidate_root = Path(os.path.abspath(os.path.expanduser(str(root))))
            try:
                root_availability[candidate_root] = candidate_root.is_dir() and candidate_root in live_parent_dirs
            except OSError:
                root_availability[candidate_root] = False

        missing: list[str] = []
        for filepath in self._data:
            candidate = Path(os.path.abspath(os.path.expanduser(filepath)))
            matching_roots = [root for root in root_availability if candidate == root or candidate.is_relative_to(root)]
            if not matching_roots:
                continue
            most_specific_depth = max(len(root.parts) for root in matching_roots)
            most_specific_roots = [root for root in matching_roots if len(root.parts) == most_specific_depth]
            if not any(root_availability[root] for root in most_specific_roots):
                self._last_prune_complete = False
                continue
            if any(self._has_unavailable_symlink_ancestor(candidate, root) for root in most_specific_roots):
                self._last_prune_complete = False
                continue
            if not self._has_live_parent_evidence(candidate, live_parent_dirs):
                self._last_prune_complete = False
                continue
            try:
                candidate_mode = candidate.stat().st_mode
            except FileNotFoundError:
                missing.append(filepath)
            except OSError:
                self._last_prune_complete = False
                continue
            else:
                if not stat.S_ISREG(candidate_mode):
                    missing.append(filepath)
        for filepath in missing:
            self.remove(filepath)
        return len(missing)


# ── JSONL Tailer ─────────────────────────────────────────────────────────────


class OversizedJSONLRecordError(ValueError):
    """A single JSONL record exceeded the configured in-memory safety ceiling."""


class JSONLTailer:
    """Tail-follows a single JSONL file from a stored offset.

    Handles partial writes: buffers incomplete lines until a newline arrives.
    Validates JSON before yielding and stops before corrupt records so callers
    can surface the failure without checkpointing unparsed bytes.
    Detects file rewinds (checkpoint restore) when file shrinks.
    """

    def __init__(self, filepath: str, offset: int = 0, max_record_bytes: int | None = None):
        self.filepath = filepath
        self.offset = offset
        self.max_record_bytes = max_record_bytes
        self._buffer = b""
        self.rewound = False  # Set to True when rewind detected
        self.rewind_old_offset = 0
        self.rewind_new_offset = 0
        self.last_error: OSError | json.JSONDecodeError | UnicodeDecodeError | OversizedJSONLRecordError | None = None
        self.failed_record: bytes | None = None
        self.observed_inode = self.get_inode()

    def restore_snapshot(self, snapshot: tuple[int, bytes]) -> None:
        """Put the tailer back to a (offset, buffer) pair captured before a failed read.

        The watcher used to reach in and unpack straight onto `offset` and `_buffer`; owning
        the two assignments here keeps the buffer private and gives static analysis a typed
        parameter instead of a `tuple | None` it cannot narrow (DeepSource on #781).
        """
        self.offset, self._buffer = snapshot

    def check_rewind(self) -> bool:
        """Check if file has shrunk (checkpoint restore). Returns True if rewound."""
        try:
            file_size = os.path.getsize(self.filepath)
        except OSError:
            return False

        effective_offset = self.offset + len(self._buffer)
        if file_size < effective_offset:
            self.rewind_old_offset = effective_offset
            self.rewind_new_offset = file_size
            self.offset = 0  # Reset to start of file
            self._buffer = b""
            self.rewound = True
            logger.warning(
                "Rewind detected: %s shrank from %d to %d",
                self.filepath,
                self.rewind_old_offset,
                self.rewind_new_offset,
            )
            return True
        return False

    def read_new_lines(
        self,
        max_lines: int | None = None,
        max_bytes: int | None = None,
    ) -> list[dict]:
        """Read a bounded window of new bytes and return complete parsed JSON dicts."""
        self.last_error = None
        self.failed_record = None
        # Check for rewind before reading
        self.check_rewind()

        if self.max_record_bytes is not None and self._partial_record_bytes() > self.max_record_bytes:
            return self.read_buffered_lines(max_lines=max_lines)

        try:
            with open(self.filepath, "rb") as f:
                f.seek(self.offset + len(self._buffer))
                remaining_bytes = max_bytes if max_bytes is not None and max_bytes > 0 else None
                complete_lines = self._buffer.count(b"\n")
                current_record_bytes = self._partial_record_bytes()
                combined = bytearray(self._buffer)
                while remaining_bytes is None or remaining_bytes > 0:
                    if max_lines is not None and complete_lines >= max_lines:
                        break
                    chunk_bytes = _WATCH_READ_CHUNK_BYTES
                    if remaining_bytes is not None:
                        chunk_bytes = min(chunk_bytes, remaining_bytes)
                    if self.max_record_bytes is not None:
                        record_capacity = self.max_record_bytes - current_record_bytes + 1
                        if record_capacity <= 0:
                            break
                        chunk_bytes = min(chunk_bytes, record_capacity)
                    new_data = f.read(chunk_bytes)
                    if not new_data:
                        break
                    combined.extend(new_data)
                    if remaining_bytes is not None:
                        remaining_bytes -= len(new_data)

                    cursor = 0
                    while True:
                        newline_index = new_data.find(b"\n", cursor)
                        if newline_index < 0:
                            current_record_bytes += len(new_data) - cursor
                            break
                        current_record_bytes += newline_index - cursor
                        complete_lines += 1
                        current_record_bytes = 0
                        cursor = newline_index + 1
                self._buffer = bytes(combined)
        except OSError as error:
            self.last_error = error
            return []

        if b"\n" not in self._buffer and not self._buffer:
            return []
        return self.read_buffered_lines(max_lines=max_lines)

    def has_complete_buffered_line(self) -> bool:
        """Return whether an already-read complete record is waiting to be emitted."""
        return b"\n" in self._buffer

    def read_buffered_lines(self, max_lines: int | None = None) -> list[dict]:
        """Parse complete buffered records without reading more bytes from disk."""
        lines = []
        self.last_error = None
        self.failed_record = None
        consumed_bytes = 0
        starting_offset = self.offset

        while True:
            if max_lines is not None and len(lines) >= max_lines:
                break
            nl_idx = self._buffer.find(b"\n", consumed_bytes)
            if nl_idx < 0:
                break
            line_data = self._buffer[consumed_bytes:nl_idx]

            if not line_data.strip():
                consumed_bytes = nl_idx + 1
                continue

            if self.max_record_bytes is not None and len(line_data) > self.max_record_bytes:
                self.last_error = OversizedJSONLRecordError(f"JSONL record exceeds {self.max_record_bytes} bytes")
                break

            try:
                parsed = json.loads(line_data)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                self.last_error = error
                break

            line_end_offset = starting_offset + nl_idx + 1
            consumed_bytes = nl_idx + 1
            if isinstance(parsed, dict):
                parsed["_line_end_offset"] = line_end_offset
                lines.append(parsed)

        if consumed_bytes:
            self._buffer = self._buffer[consumed_bytes:]
            self.offset = starting_offset + consumed_bytes

        if self.last_error is not None and b"\n" in self._buffer:
            failed_end = self._buffer.index(b"\n") + 1
            self.failed_record = self._buffer[:failed_end]
        elif self.max_record_bytes is not None and self._partial_record_bytes() > self.max_record_bytes:
            self.last_error = OversizedJSONLRecordError(f"JSONL record exceeds {self.max_record_bytes} bytes")

        return lines

    def discard_failed_record(self) -> tuple[int, int, bytes] | None:
        """Advance over a failed complete record after the caller durably quarantines it."""
        if self.failed_record is None or not self._buffer.startswith(self.failed_record):
            return None
        start_offset = self.offset
        record = self.failed_record
        self._buffer = self._buffer[len(record) :]
        self.offset += len(record)
        self.failed_record = None
        self.last_error = None
        return start_offset, self.offset, record

    def _partial_record_bytes(self) -> int:
        last_newline = self._buffer.rfind(b"\n")
        return len(self._buffer) if last_newline < 0 else len(self._buffer) - last_newline - 1

    def get_inode(self) -> int:
        """Return the inode of the file, or 0 if not accessible."""
        try:
            return os.stat(self.filepath).st_ino
        except OSError:
            return 0


# ── Batch Indexer ────────────────────────────────────────────────────────────


class BatchIndexer:
    """Accumulates parsed JSONL lines and flushes to a callback.

    Flushes when batch_size lines accumulate or flush_interval_ms elapses.
    Thread-safe: multiple tailers can feed into one indexer.
    """

    def __init__(
        self,
        on_flush: Callable[[list[dict]], dict[str, int] | None],
        batch_size: int = 10,
        flush_interval_ms: int = 100,
        on_confirm_batch: Callable[[dict[str, int], list[dict]], None] | None = None,
    ):
        self.on_flush = on_flush
        self.batch_size = batch_size
        self.flush_interval_ms = flush_interval_ms
        self.on_confirm_batch = on_confirm_batch
        self._buffer: list[dict] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()
        self.total_flushed = 0
        self.total_outputs = 0
        self.total_failed_inputs = 0
        self._flush_failures = 0

    def add(self, items: list[dict]):
        """Add parsed lines to the buffer."""
        if not items:
            return
        with self._lock:
            self._buffer.extend(items)
            if len(self._buffer) >= self.batch_size:
                self._do_flush()

    def tick(self):
        """Check if flush interval has elapsed. Call periodically."""
        with self._lock:
            if not self._buffer:
                return
            elapsed_ms = (time.monotonic() - self._last_flush) * 1000
            if elapsed_ms >= self.flush_interval_ms:
                self._do_flush()

    def flush(self):
        """Force flush remaining items."""
        with self._lock:
            if self._buffer:
                self._do_flush()

    def has_buffered_source(self, filepath: str) -> bool:
        """Return whether unconfirmed buffered entries came from filepath."""
        with self._lock:
            return any(item.get("_source_file") == filepath for item in self._buffer)

    def _do_flush(self):
        """Internal flush — must be called with _lock held."""
        batch = self._buffer
        self._last_flush = time.monotonic()
        count = len(batch)
        try:
            result = self.on_flush(batch)
            deferred_entries = self._deferred_entries(batch, result)
            watermarks = self._confirmed_watermarks(batch, result, deferred_entries)
            if watermarks and self.on_confirm_batch:
                self.on_confirm_batch(watermarks, batch)
            self._buffer = deferred_entries
            self._flush_failures = 0
            self.total_flushed += count - len(deferred_entries)
            self.total_outputs += getattr(result, "inserted", count)
        except Exception as e:
            logger.error("Batch flush failed (%d items), retaining in buffer: %s", count, e)
            self.total_failed_inputs += self._isolate_failed_flush(batch, e)

    def retained_failed_input_count(self) -> int:
        """Return currently retained flush-failed inputs without counting retry attempts."""
        with self._lock:
            return len(self._buffer) if self._flush_failures > 0 else 0

    def _deferred_entries(self, batch: list[dict], result: dict[str, int] | None) -> list[dict]:
        requested = getattr(result, "deferred_entries", None)
        if not requested:
            return []
        batch_ids = {id(item) for item in batch}
        requested_ids = {id(item) for item in requested}
        if not requested_ids.issubset(batch_ids):
            raise ValueError("flush callback deferred an entry outside the current batch")
        return [item for item in batch if id(item) in requested_ids]

    def _confirmed_watermarks(
        self,
        batch: list[dict],
        result: dict[str, int] | None,
        deferred_entries: list[dict] | None = None,
    ) -> dict[str, int]:
        if isinstance(result, dict):
            deferred_sources = {
                str(item["_source_file"])
                for item in deferred_entries or []
                if isinstance(item.get("_source_file"), str)
            }
            return {
                str(source_file): int(offset)
                for source_file, offset in result.items()
                if str(source_file) not in deferred_sources
            }
        return {}

    def _flush_retain_limit(self) -> int:
        try:
            return max(1, int(os.environ.get("BRAINLAYER_WATCHER_FLUSH_RETAIN_LIMIT", "3")))
        except ValueError:
            return 3

    def _quarantine_entries(self, entries: list[dict], reason: Exception) -> None:
        quarantine_dir = Path(
            os.environ.get("BRAINLAYER_WATCHER_QUARANTINE_DIR", "~/.brainlayer/quarantine")
        ).expanduser()
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        path = quarantine_dir / f"watcher-flush-{int(time.time() * 1000)}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps({"reason": str(reason), "entry": entry}, sort_keys=True) + "\n")
        logger.critical("Quarantined %d watcher flush entries at %s after repeated failures", len(entries), path)

    def _isolate_failed_flush(self, batch: list[dict], reason: Exception) -> int:
        if len(batch) <= 1:
            self._flush_failures += 1
            if self._flush_failures >= self._flush_retain_limit():
                self._quarantine_entries(batch, reason)
                self._buffer = []
                self._flush_failures = 0
            return len(batch)

        retained: list[dict] = []
        confirmed_items: list[dict] = []
        confirmed: dict[str, int] = {}
        outputs = 0
        failed_inputs = 0
        for item in batch:
            try:
                result = self.on_flush([item])
            except Exception:
                retained.append(item)
                failed_inputs += 1
                continue
            deferred_entries = self._deferred_entries([item], result)
            item_watermarks = self._confirmed_watermarks([item], result, deferred_entries)
            for source_file, offset in item_watermarks.items():
                confirmed[source_file] = max(confirmed.get(source_file, 0), offset)
            retained.extend(deferred_entries)
            if deferred_entries:
                outputs += getattr(result, "inserted", 0)
                continue
            confirmed_items.append(item)
            outputs += getattr(result, "inserted", 1)

        if confirmed and self.on_confirm_batch:
            self.on_confirm_batch(confirmed, confirmed_items)
        self.total_outputs += outputs
        self.total_flushed += len(batch) - len(retained)
        self._buffer = retained
        self._flush_failures = self._flush_failures + 1 if failed_inputs else 0
        return failed_inputs


# ── JSONL Watcher ────────────────────────────────────────────────────────────


class JSONLWatcher:
    """Watches a directory tree for .jsonl files and tail-follows them.

    Uses polling (os.scandir) rather than watchdog for simplicity in the
    prototype. The Swift DispatchSource version will use kqueue for sub-ms
    notification.

    Usage:
        watcher = JSONLWatcher(
            watch_dir="~/.claude/projects/",
            registry_path="~/.local/share/brainlayer/offsets.json",
            on_flush=my_callback,
        )
        watcher.start()  # blocking
    """

    def __init__(
        self,
        watch_dir: str | Path | None = None,
        registry_path: str | Path | None = None,
        on_flush: Callable[[list[dict]], dict[str, int] | None] | None = None,
        on_rewind: Callable[[str, str, int, int], None] | None = None,
        on_tick: Callable[[], None] | None = None,
        watch_roots: list[WatchRoot] | None = None,
        db_path: str | Path | None = None,
        # >=30s batching, per the R3 constraint: a discovery pass over the real corpus
        # measures ~1s and a steady-state poll ~2.4s, so the old 1.0s default meant the
        # loop never slept. 30s is the constraint's floor and the lowest-latency value that
        # satisfies it. (60s was tried to clear the <5% idle-CPU soak gate; it did not --
        # the shipping code measures 6.41% -- so the extra latency bought nothing and the
        # gate is left failing honestly rather than tuned around.)
        # Not clamped here: the in-process tests drive poll_once through a real loop at
        # 0.01-0.05s and a 30s floor in the constructor would stall the suite, not the
        # burn. The floor is enforced at the boundary a deployed value crosses -- the
        # `watch` command -- by enforce_min_poll_interval() above.
        poll_interval_s: float = MIN_WATCH_POLL_INTERVAL_S,
        batch_size: int = 10,
        flush_interval_ms: int = 100,
        registry_flush_interval_s: float = 5.0,
        health_path: str | Path | None = None,
        coverage_watchdog: CoverageWatchdog | None = None,
        max_lines_per_file: int = 100,
    ):
        if watch_roots is not None:
            self.watch_roots = [WatchRoot(root.provider, root.path, root.glob_pattern) for root in watch_roots]
        elif watch_dir is not None:
            self.watch_roots = [WatchRoot("claude", Path(watch_dir).expanduser())]
        else:
            self.watch_roots = default_watch_roots()
        self.watch_dir = self.watch_roots[0].resolved_path if self.watch_roots else Path(".")
        self.on_rewind = on_rewind
        self.on_tick = on_tick
        registry = (
            Path(registry_path).expanduser() if registry_path else Path.home() / ".local/share/brainlayer/offsets.json"
        )
        self.registry = OffsetRegistry(registry)
        self.indexer = BatchIndexer(
            on_flush=on_flush or (lambda _items: None),
            batch_size=batch_size,
            flush_interval_ms=flush_interval_ms,
            on_confirm_batch=self._advance_confirmed_batch,
        )
        self.poll_interval_s = poll_interval_s
        self.registry_flush_interval_s = registry_flush_interval_s
        self.max_lines_per_file = max(1, max_lines_per_file)
        self.max_read_bytes_per_file = _watch_read_window_bytes()
        self.max_record_bytes = _watch_max_record_bytes()
        self._tailers: dict[str, JSONLTailer] = {}
        self._file_providers: dict[str, str] = {}
        # (mtime, size, inode) per file, so an unchanged file costs nothing on the next
        # poll. Discovery already stats every file, so this is free to collect. The inode is
        # load-bearing, not decoration: rotation is only detected inside _ensure_tailer /
        # read_new_lines, which a skip never reaches, so without it a file replaced at the
        # same path with the same size and mtime would be skipped forever and its content
        # never ingested.
        self._current_file_stats: dict[str, tuple[float, int, int]] = {}
        self._observed_file_stats: dict[str, tuple[float, int, int]] = {}
        # Denylist verdicts memoised for the duration of one poll. is_denylisted() expands
        # globs and can read file content to attribute subagents; a warm sweep of the real
        # corpus costs ~0.8s, and poll_once was evaluating it four times per file. Cleared
        # every poll so a changed BRAINLAYER_INGEST_DENYLIST is still picked up promptly.
        self._denylist_memo: dict[str, bool] = {}
        self._denylist_dir_memo: dict[str, bool] = {}
        self._file_ingestion_failures: dict[str, dict[str, Any]] = {}
        self._quarantined_record_count_total = 0
        self._quarantined_records: list[dict[str, Any]] = []
        self._pending_quarantined_offsets: dict[str, list[tuple[int, int, int, int]]] = {}
        self._stop = threading.Event()
        self._last_registry_flush = time.monotonic()
        self.health_path = Path(health_path).expanduser() if health_path else None
        self.db_path = Path(db_path).expanduser() if db_path else None
        self.coverage_watchdog = coverage_watchdog or CoverageWatchdog()
        self._health_window_started = time.monotonic()
        self._health_window_started_epoch = time.time()
        self._health_entries_seen = 0
        self._health_output_at_start = 0
        self.poll_count = 0
        self.last_poll_made_progress = False
        self._offset_prune_complete = False
        # An incomplete prune must back off rather than re-scan the registry every poll.
        # `last_prune_complete` is permanently False whenever the registry holds entries
        # whose parent directory is gone (8,744 of 21,529 on this machine), so gating only
        # on that flag meant a 10-12s full-filesystem scan on every poll, forever, pruning
        # nothing. Retrying on a timer keeps the volume-returns-later behaviour at ~0 cost.
        self.offset_prune_retry_interval_s = _watch_offset_prune_retry_interval_s()
        self._last_offset_prune_attempt = float("-inf")
        self._last_prune_parent_dirs: frozenset[str] | None = None

    def _advance_confirmed_offsets(self, confirmations: dict[str, tuple[int, int, int]]) -> None:
        for filepath, (offset, source_inode, source_generation) in confirmations.items():
            current_offset, _current_inode = self.registry.get(filepath)
            if offset >= current_offset:
                self.registry.set(filepath, offset, source_inode, generation=source_generation)
                self._advance_quarantined_offsets(filepath, source_inode, source_generation)

    def _advance_confirmed_batch(self, watermarks: dict[str, int], batch: list[dict]) -> None:
        """Confirm only offsets produced by the file generation currently being tailed."""
        current_confirmations: dict[str, tuple[int, int, int]] = {}
        for filepath, reported_offset in watermarks.items():
            tailer = self._tailers.get(filepath)
            if tailer is None:
                continue
            current_inode = tailer.observed_inode
            current_generation = self.registry.generation(filepath)
            eligible_offsets = [
                item["_line_end_offset"]
                for item in batch
                if item.get("_source_file") == filepath
                and item.get("_source_inode") == current_inode
                and item.get("_source_generation") == current_generation
                and isinstance(item.get("_line_end_offset"), int)
                and item["_line_end_offset"] <= reported_offset
            ]
            if eligible_offsets:
                current_confirmations[filepath] = (max(eligible_offsets), current_inode, current_generation)
        self._advance_confirmed_offsets(current_confirmations)

    def _advance_quarantined_offsets(self, filepath: str, source_inode: int, source_generation: int) -> None:
        """Advance over consecutive durably quarantined records after prior bytes confirm."""
        pending = self._pending_quarantined_offsets.get(filepath)
        if not pending:
            return
        current_offset, current_inode = self.registry.get(filepath)
        current_generation = self.registry.generation(filepath)
        if current_generation != source_generation or current_inode not in (0, source_inode):
            return
        remaining: list[tuple[int, int, int, int]] = []
        for start_offset, end_offset, pending_inode, pending_generation in sorted(pending):
            if (pending_inode, pending_generation) != (source_inode, source_generation):
                remaining.append((start_offset, end_offset, pending_inode, pending_generation))
                continue
            if current_offset < start_offset:
                remaining.append((start_offset, end_offset, pending_inode, pending_generation))
                continue
            if end_offset > current_offset:
                self.registry.set(filepath, end_offset, pending_inode, generation=pending_generation)
                current_offset = end_offset
        if remaining:
            self._pending_quarantined_offsets[filepath] = remaining
        else:
            self._pending_quarantined_offsets.pop(filepath, None)

    def provider_for_file(self, filepath: str) -> str:
        if self._denylisted(filepath):
            return "unknown"

        provider = self._file_providers.get(filepath)
        if provider:
            return provider

        path = Path(filepath).expanduser().resolve(strict=False)
        for root in self.watch_roots:
            root_path = root.resolved_path.resolve(strict=False)
            if path == root_path or path.is_relative_to(root_path):
                return root.provider
        return "unknown"

    def _discover_jsonl_files(self) -> list[str]:
        """Find all .jsonl files under each watched project, including nested session artifacts."""
        discovered: list[tuple[float, str, str]] = []
        self._file_providers = {}
        self._current_file_stats = {}
        self._denylist_memo = {}
        self._denylist_dir_memo = {}
        for root in self.watch_roots:
            root_path = root.resolved_path
            if not root_path.exists():
                continue
            try:
                bases = [root_path]
                if root.provider == "claude" and root.glob_pattern == "**/*.jsonl":
                    bases = [path for path in root_path.iterdir() if path.is_dir()]
                for base in bases:
                    for path, stat_file in _iter_jsonl_files(base, root.glob_pattern, skip_dir=self._denylisted_dir):
                        if self._denylisted(path):
                            continue
                        try:
                            stat_result = stat_file()
                            mtime = stat_result.st_mtime
                        except OSError as e:
                            logger.debug("Skipping JSONL file during discovery after stat failure: %s: %s", path, e)
                            continue
                        self._current_file_stats[path] = (mtime, stat_result.st_size, stat_result.st_ino)
                        discovered.append((mtime, path, root.provider))
            except OSError:
                continue
        discovered.sort(key=lambda item: item[0], reverse=True)
        files = [path for _mtime, path, provider in discovered]
        self._file_providers = {path: provider for _mtime, path, provider in discovered}
        return files

    def _denylisted(self, filepath: str) -> bool:
        """Memoised is_denylisted() for one poll cycle. See _denylist_memo."""
        cached = self._denylist_memo.get(filepath)
        if cached is None:
            cached = is_denylisted(filepath)
            self._denylist_memo[filepath] = cached
        return cached

    def _denylisted_dir(self, dirpath: str) -> bool:
        """Memoised is_directory_denylisted() for one poll cycle; prunes discovery subtrees."""
        cached = self._denylist_dir_memo.get(dirpath)
        if cached is None:
            cached = is_directory_denylisted(dirpath)
            self._denylist_dir_memo[dirpath] = cached
        return cached

    def _can_skip_unchanged(self, filepath: str) -> bool:
        """Return whether this poll may skip a file whose bytes cannot have moved.

        Discovery already stats every file, so comparing (mtime, size) against the poll
        that last drained the file is free. Without this, every poll opened and read all
        ~12,700 tracked files; with it, an idle corpus costs only the discovery walk.

        The gate is deliberately conservative -- it refuses to skip whenever the file
        might still owe us bytes, because a false skip stalls a session silently:
          * no tailer yet, so the file has never been read at all;
          * the tailer is behind the file's size (a read capped by max_lines_per_file or
            max_read_bytes_per_file leaves the stat unchanged but the tailer short);
          * a complete line is still buffered from the previous read;
          * quarantined offsets or a recorded ingestion failure are still pending.
        """
        previous = self._observed_file_stats.get(filepath)
        current = self._current_file_stats.get(filepath)
        if previous is None or current is None or previous != current:
            return False
        tailer = self._tailers.get(filepath)
        _mtime, size, inode = current
        # One expression, evaluated in order: `tailer is not None` guards every attribute
        # access after it, exactly as the sequential early-returns did.
        #
        # `offset == size` and not `>= size`: a tailer AHEAD of EOF believes it read more
        # bytes than the file holds, which is the signature of a truncation. Skipping there
        # bypasses check_rewind -- the checkpoint-restore path that soft-archives reverted
        # chunks. Only a tailer exactly at EOF has provably nothing left to read.
        return (
            tailer is not None
            and tailer.offset == size
            and tailer.observed_inode in (0, inode)
            and not tailer.has_complete_buffered_line()
            and not self._pending_quarantined_offsets.get(filepath)
            and filepath not in self._file_ingestion_failures
        )

    def _normalize_lines(self, filepath: str, new_lines: list[dict]) -> list[dict]:
        provider = self.provider_for_file(filepath)
        normalized = []
        for line in new_lines:
            entry = normalize_provider_entry(line, provider)
            if not entry and provider == "claude":
                entry = dict(line)
            if not entry:
                continue
            entry["_source_file"] = filepath
            entry["_provider"] = provider
            if isinstance(line.get("_line_end_offset"), int):
                entry["_line_end_offset"] = line["_line_end_offset"]
            normalized.append(entry)
        return normalized

    def _checkpoint_discarded_progress(
        self,
        filepath: str,
        read_start_offset: int,
        read_end_offset: int,
        normalized_lines: list[dict],
        source_inode: int,
        source_generation: int,
    ) -> None:
        """Confirm intentionally discarded bytes without crossing indexable work."""
        required_confirmed_offset = max(
            (line["_line_end_offset"] for line in normalized_lines if isinstance(line.get("_line_end_offset"), int)),
            default=read_start_offset,
        )
        if read_end_offset <= required_confirmed_offset:
            return
        if self.indexer.has_buffered_source(filepath):
            return
        confirmed_offset, _confirmed_inode = self.registry.get(filepath)
        if confirmed_offset < required_confirmed_offset:
            return
        self._advance_confirmed_offsets(
            {
                filepath: (
                    read_end_offset,
                    source_inode,
                    source_generation,
                )
            }
        )

    def _record_file_ingestion_failure(
        self,
        filepath: str,
        error: BaseException,
        extra_context: dict[str, Any] | None = None,
    ) -> None:
        """Raise once per distinct failure and retain it on the health surface."""
        tailer = self._tailers.get(filepath)
        confirmed_offset, confirmed_inode = self.registry.get(filepath)
        try:
            file_size = os.path.getsize(filepath)
        except OSError:
            file_size = None
        context = {
            "file_path": filepath,
            "error_type": type(error).__name__,
            "error": str(error),
            "confirmed_offset": confirmed_offset,
            "confirmed_inode": confirmed_inode,
            "read_offset": tailer.offset if tailer else confirmed_offset,
            "file_size_bytes": file_size,
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }
        if extra_context:
            context.update(extra_context)
        fingerprint = (
            context["error_type"],
            context["error"],
            context["confirmed_offset"],
            context["read_offset"],
            context.get("disposition"),
            context.get("quarantine_path"),
        )
        previous = self._file_ingestion_failures.get(filepath)
        self._file_ingestion_failures[filepath] = {**context, "_fingerprint": fingerprint}
        if previous and previous.get("_fingerprint") == fingerprint:
            return
        try:
            raise_alarm(
                "watcher_file_ingestion_failed",
                "watcher deferred a JSONL file because bytes could not be safely parsed",
                context,
            )
        except BrainLayerAlarm as alarm:
            logger.error("Watcher file-ingestion alarm emitted without stopping watcher: %s", alarm)

    def _clear_file_ingestion_failure(self, filepath: str) -> None:
        self._file_ingestion_failures.pop(filepath, None)

    def _quarantine_failed_record(
        self,
        filepath: str,
        tailer: JSONLTailer,
        source_inode: int,
        source_generation: int,
    ) -> bool:
        """Durably preserve one malformed record, alarm, and advance over only those bytes."""
        if tailer.failed_record is None or not isinstance(
            tailer.last_error,
            (json.JSONDecodeError, UnicodeDecodeError),
        ):
            return False

        record = tailer.failed_record
        start_offset = tailer.offset
        end_offset = start_offset + len(record)
        digest = hashlib.sha256(record).hexdigest()
        quarantine_dir = Path(
            os.environ.get("BRAINLAYER_WATCHER_QUARANTINE_DIR", "~/.brainlayer/quarantine")
        ).expanduser()
        quarantine_path = quarantine_dir / (
            f"watcher-parse-{Path(filepath).stem}-{start_offset}-{digest[:16]}.jsonl.bad"
        )
        temp_path: Path | None = None
        try:
            quarantine_dir.mkdir(parents=True, exist_ok=True)
            if not quarantine_path.exists():
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=quarantine_dir,
                    prefix=f".{quarantine_path.name}.",
                    delete=False,
                ) as handle:
                    temp_path = Path(handle.name)
                    handle.write(record)
                    handle.flush()
                    os.fsync(handle.fileno())
                temp_path.replace(quarantine_path)
                temp_path = None
                directory_fd = os.open(quarantine_dir, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            elif quarantine_path.read_bytes() != record:
                raise OSError(f"quarantine collision at {quarantine_path}")
        except OSError as error:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            tailer.last_error = error
            tailer.failed_record = None
            return False

        parse_error = tailer.last_error
        event = {
            "file_path": filepath,
            "start_offset": start_offset,
            "end_offset": end_offset,
            "record_bytes": len(record),
            "sha256": digest,
            "quarantine_path": str(quarantine_path),
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._quarantined_record_count_total += 1
        self._quarantined_records.append(event)
        self._quarantined_records = self._quarantined_records[-_MAX_HEALTH_QUARANTINE_DETAILS:]
        self._record_file_ingestion_failure(
            filepath,
            parse_error,
            {
                "disposition": "quarantined",
                "quarantine_path": str(quarantine_path),
                "quarantined_start_offset": start_offset,
                "quarantined_end_offset": end_offset,
            },
        )
        discarded = tailer.discard_failed_record()
        if discarded is None:
            raise RuntimeError("quarantined record no longer matches the tailer buffer")
        self._pending_quarantined_offsets.setdefault(filepath, []).append(
            (start_offset, end_offset, source_inode, source_generation)
        )
        self._advance_quarantined_offsets(filepath, source_inode, source_generation)
        return True

    def _max_offset_lag_bytes(self, files: list[str]) -> int:
        max_lag = 0
        for filepath in files:
            # Discovery stat'd every one of these moments ago; re-statting the whole corpus
            # here would double the syscall cost of every poll for no new information.
            cached = self._current_file_stats.get(filepath)
            if cached is not None:
                size = cached[1]
            else:
                try:
                    size = os.path.getsize(filepath)
                except OSError:
                    continue
            tailer = self._tailers.get(filepath)
            offset = tailer.offset if tailer else self.registry.get(filepath)[0]
            max_lag = max(max_lag, max(size - offset, 0))
        return max_lag

    def _db_realtime_inserts_since_window_start(self) -> int | None:
        if not self.db_path:
            return None
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=1)
            try:
                window_started = int(self._health_window_started_epoch)
                # Liveness first. The drain writes one row per watcher chunk it persists,
                # the table is indexed on ingested_at, and whenever it holds rows for this
                # window they are the answer. The chunk count used to be computed before
                # this check and then discarded whenever liveness won -- 0.33s of every
                # poll on the real DB, spent on a number nobody read.
                if conn.execute(
                    """
                    SELECT 1 FROM sqlite_master
                    WHERE type = 'table'
                      AND name = 'watcher_liveness_events'
                    """
                ).fetchone():
                    liveness_row = conn.execute(
                        """
                        SELECT COUNT(*) FROM watcher_liveness_events
                        WHERE ingested_at >= ?
                        """,
                        (window_started,),
                    ).fetchone()
                    liveness_count = int(liveness_row[0]) if liveness_row else 0
                    if liveness_count > 0:
                        return liveness_count
                chunk_count = 0
                for sql, params in realtime_insert_probe_statements(window_started):
                    row = conn.execute(sql, params).fetchone()
                    chunk_count += int(row[0]) if row else 0
                return chunk_count
            finally:
                conn.close()
        except sqlite3.Error:
            return None

    def _write_health_snapshot(self, files: list[str]):
        if not self.health_path:
            return

        now = time.monotonic()
        elapsed = max(now - self._health_window_started, 1.0)
        normalized_entries_per_min = self._health_entries_seen / elapsed * 60.0
        outputs_per_min = (self.indexer.total_outputs - self._health_output_at_start) / elapsed * 60.0
        failed_flush_inputs_per_min = self.indexer.retained_failed_input_count() / elapsed * 60.0
        active_entries_per_min = outputs_per_min + failed_flush_inputs_per_min
        db_inserts = self._db_realtime_inserts_since_window_start()
        db_probe_failed = self.db_path is not None and db_inserts is None
        if db_inserts is not None:
            db_inserts_per_min = db_inserts / elapsed * 60.0
        elif self.db_path is not None:
            db_inserts_per_min = None
        else:
            db_inserts_per_min = outputs_per_min
        watchdog_inserts_per_min = 0.0 if db_probe_failed else db_inserts_per_min
        max_lag = self._max_offset_lag_bytes(files)
        watchdog = self.coverage_watchdog.evaluate(
            active_entries_per_minute=active_entries_per_min,
            realtime_inserts_per_minute=watchdog_inserts_per_min,
            max_offset_lag_bytes=max_lag,
        )
        all_failure_payloads = [
            {key: value for key, value in failure.items() if key != "_fingerprint"}
            for _filepath, failure in sorted(self._file_ingestion_failures.items())
        ]
        failure_payloads = all_failure_payloads[:_MAX_HEALTH_FAILURE_DETAILS]
        failure_overflow_count = len(all_failure_payloads) - len(failure_payloads)
        alert_reasons = list(watchdog.get("alert_reasons", []))
        if all_failure_payloads and "file_ingestion_failure" not in alert_reasons:
            alert_reasons.append("file_ingestion_failure")
        if self._quarantined_record_count_total and "quarantined_record" not in alert_reasons:
            alert_reasons.append("quarantined_record")
        watchdog = {
            **watchdog,
            "alerting": bool(watchdog.get("alerting"))
            or bool(all_failure_payloads)
            or bool(self._quarantined_record_count_total),
            "alert_reasons": alert_reasons,
        }
        coverage_degraded = (
            active_entries_per_min > 0
            and watchdog_inserts_per_min / active_entries_per_min < self.coverage_watchdog.coverage_ratio_threshold
        )
        durable_writes_per_min = watchdog_inserts_per_min
        zero_write_degraded = coverage_degraded and durable_writes_per_min <= 0
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "poll_count": self.poll_count,
            "providers": sorted({root.provider for root in self.watch_roots}),
            "files_tracked": len(files),
            "active_jsonl_entries_per_minute": active_entries_per_min,
            "normalized_jsonl_entries_per_minute": normalized_entries_per_min,
            "db_probe_failed": db_probe_failed,
            "db_realtime_inserts_per_minute": db_inserts_per_min,
            "failed_flush_inputs_per_minute": failed_flush_inputs_per_min,
            "watcher_chunks_output_per_minute": outputs_per_min,
            "max_offset_lag_bytes": max_lag,
            "file_ingestion_failure_count": len(all_failure_payloads),
            "file_ingestion_failures": failure_payloads,
            "file_ingestion_failures_overflow_count": failure_overflow_count,
            "quarantined_record_count_total": self._quarantined_record_count_total,
            "quarantined_records": list(self._quarantined_records),
            **watchdog,
        }
        try:
            self.health_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.health_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            tmp_path.replace(self.health_path)
        except OSError:
            logger.debug("Failed to write watcher health snapshot", exc_info=True)

        if (
            watchdog.get("alerting") is True
            and "coverage_drop" in watchdog.get("alert_reasons", [])
            and active_entries_per_min > 0
            and durable_writes_per_min <= 0
        ):
            try:
                raise_alarm(
                    "watcher_zero_writes_while_active",
                    "watcher accepted indexable JSONL input but produced zero durable realtime writes",
                    {
                        "active_jsonl_entries_per_minute": active_entries_per_min,
                        "normalized_jsonl_entries_per_minute": normalized_entries_per_min,
                        "db_probe_failed": db_probe_failed,
                        "db_realtime_inserts_per_minute": db_inserts_per_min,
                        "durable_writes_per_minute": durable_writes_per_min,
                        "failed_flush_inputs_per_minute": failed_flush_inputs_per_min,
                        "files_tracked": len(files),
                        "max_offset_lag_bytes": max_lag,
                        "providers": payload["providers"],
                        "watcher_chunks_output_per_minute": outputs_per_min,
                        "watchdog_reasons": watchdog.get("alert_reasons", []),
                    },
                )
            except BrainLayerAlarm as alarm:
                logger.error("Watcher health alarm emitted without stopping watcher: %s", alarm)

        if elapsed >= 60.0 and not zero_write_degraded:
            self._health_window_started = now
            self._health_window_started_epoch = time.time()
            self._health_entries_seen = 0
            self._health_output_at_start = self.indexer.total_outputs

    def _ensure_tailer(self, filepath: str) -> JSONLTailer:
        """Get or create a tailer for a file, respecting stored offsets."""
        if filepath in self._tailers:
            tailer = self._tailers[filepath]
            # Check inode hasn't changed (file replaced)
            current_inode = tailer.get_inode()
            stored_offset, stored_inode = self.registry.get(filepath)
            inode_changed = current_inode != 0 and (
                (stored_inode != 0 and current_inode != stored_inode)
                or (tailer.observed_inode != 0 and current_inode != tailer.observed_inode)
            )
            if inode_changed:
                # File was replaced — reset offset
                self._pending_quarantined_offsets.pop(filepath, None)
                self.registry.mark_rewind(filepath, current_inode)
                tailer = JSONLTailer(filepath, offset=0, max_record_bytes=self.max_record_bytes)
                self._tailers[filepath] = tailer
            return tailer

        stored_offset, stored_inode = self.registry.get(filepath)
        tailer = JSONLTailer(filepath, offset=stored_offset, max_record_bytes=self.max_record_bytes)

        # Verify inode matches
        current_inode = tailer.get_inode()
        if stored_inode != 0 and current_inode != 0 and current_inode != stored_inode:
            self._pending_quarantined_offsets.pop(filepath, None)
            self.registry.mark_rewind(filepath, current_inode)
            tailer = JSONLTailer(filepath, offset=0, max_record_bytes=self.max_record_bytes)

        self._tailers[filepath] = tailer
        return tailer

    def _handle_rewind(
        self,
        filepath: str,
        old_offset: int,
        new_offset: int,
        inode: int,
    ) -> None:
        """Persist a rewind and notify archival consumers."""
        session_id = Path(filepath).stem
        self._pending_quarantined_offsets.pop(filepath, None)
        self.registry.mark_rewind(filepath, inode)
        logger.warning(
            "Checkpoint restore: %s (offset %d → %d)",
            session_id,
            old_offset,
            new_offset,
        )
        try:
            from .telemetry import emit

            emit(
                "brainlayer-watcher",
                {
                    "_type": "rewind_detected",
                    "session_id": session_id,
                    "file_path": filepath,
                    "old_offset": old_offset,
                    "new_offset": new_offset,
                },
            )
        except Exception:
            pass

        if self.on_rewind:
            try:
                self.on_rewind(filepath, session_id, old_offset, new_offset)
            except Exception as e:
                logger.error("Rewind callback failed: %s", e)

    def _maybe_prune_offsets(self, files: list[str]) -> None:
        """Prune deleted offsets, retrying an incomplete prune without re-scanning every poll.

        An incomplete prune retries when the evidence that blocked it might have changed --
        a new parent directory among the discovered files can supply the live-parent proof
        a previously unmounted root was missing -- and otherwise backs off onto a timer.

        Both halves are load-bearing. Without the change-detector a returning volume would
        wait out the full interval (pinned by
        test_poll_retries_pruning_after_unavailable_startup_root). Without the timer, a
        registry holding entries whose roots are gone re-scans the whole filesystem on every
        poll forever: measured at 10-12s per poll, pruning nothing after the first pass.
        """
        if self._offset_prune_complete:
            return
        # str.rpartition, not os.path.dirname or Path.parent: this runs over every
        # discovered file on every poll. Measured on 12,000 paths -- rpartition 0.8ms,
        # os.path.dirname 2.4ms, Path(p).parent 24.9ms. Complying with the pathlib
        # lint here would have cost 22ms per poll in the loop this PR just optimised.
        parent_dirs = frozenset(filepath.rpartition(os.sep)[0] for filepath in files)
        timer_elapsed = time.monotonic() - self._last_offset_prune_attempt >= self.offset_prune_retry_interval_s
        if not timer_elapsed and self._last_prune_parent_dirs is not None:
            # A changed parent-dir set used to re-trigger the scan on its own. On a live machine
            # that is every new session directory -- another agent opening a worktree fired it
            # inside the first two minutes of the 600s soak, at 7.4 CPU-seconds on the
            # efficiency cores, more than eight steady polls. The only change that can make a
            # blocked entry prunable is a NEW directory the registry already holds an entry under
            # that is not among the live files (the returning volume). Anything else waits for
            # the timer -- and that includes SHRINKAGE of the set, deliberately: when a directory
            # loses its last live file, nothing under it can be pruned (the prune requires live
            # evidence in the entry's directory tree, which just disappeared), so a scan now
            # would find exactly what the next 900 s tick finds. An orphan created that way
            # waits at most one timer interval. The remembered set moves forward either way, so
            # each new directory is judged once, not on every poll until the timer fires.
            new_dirs = parent_dirs - self._last_prune_parent_dirs
            self._last_prune_parent_dirs = parent_dirs
            if not new_dirs or not self.registry.has_stale_entries_under(new_dirs, files):
                return

        self._last_offset_prune_attempt = time.monotonic()
        self._last_prune_parent_dirs = parent_dirs
        pruned = self.registry.prune_missing_files(
            [root.resolved_path for root in self.watch_roots],
            files,
        )
        if pruned:
            logger.info("Pruned %d deleted files from the offset registry", pruned)
        self._offset_prune_complete = self.registry.flush() and self.registry.last_prune_complete
        if not self._offset_prune_complete:
            logger.info(
                "Offset prune incomplete (entries whose roots are not currently mounted); retrying in %.0fs",
                self.offset_prune_retry_interval_s,
            )

    def poll_once(self) -> int:
        """Run one poll cycle. Returns number of new lines found."""
        total_new = 0
        files: list[str] = []
        self.poll_count += 1
        self.last_poll_made_progress = False

        try:
            files = self._discover_jsonl_files()
            live_files = {filepath for filepath in files if not self._denylisted(filepath)}
            self._file_ingestion_failures = {
                filepath: failure
                for filepath, failure in self._file_ingestion_failures.items()
                if filepath in live_files
            }
            self._pending_quarantined_offsets = {
                filepath: pending
                for filepath, pending in self._pending_quarantined_offsets.items()
                if filepath in live_files
            }
            self._maybe_prune_offsets(files)

            for filepath in list(self._tailers):
                if self._denylisted(filepath):
                    self._tailers.pop(filepath, None)
                    self._file_providers.pop(filepath, None)
                    self._pending_quarantined_offsets.pop(filepath, None)
                    self.registry.remove(filepath)

            for filepath in files:
                if self._denylisted(filepath):
                    self._tailers.pop(filepath, None)
                    self._file_providers.pop(filepath, None)
                    self._pending_quarantined_offsets.pop(filepath, None)
                    self.registry.remove(filepath)
                    continue
                if self._can_skip_unchanged(filepath):
                    continue
                tailer: JSONLTailer | None = None
                tailer_snapshot: tuple[int, bytes] | None = None
                source_inode = 0
                source_generation = 0
                read_accepted = False
                try:
                    tailer = self._tailers.get(filepath)
                    drain_buffer = False
                    if tailer is not None and tailer.has_complete_buffered_line():
                        _registry_offset, registry_inode = self.registry.get(filepath)
                        current_inode = tailer.get_inode()
                        inode_changed = current_inode != 0 and (
                            (registry_inode != 0 and registry_inode != current_inode)
                            or (tailer.observed_inode != 0 and tailer.observed_inode != current_inode)
                        )
                        if not inode_changed and not tailer.check_rewind():
                            drain_buffer = True
                        elif tailer.rewound:
                            self._handle_rewind(
                                filepath,
                                tailer.rewind_old_offset,
                                tailer.rewind_new_offset,
                                tailer.get_inode(),
                            )
                            tailer.rewound = False

                    if drain_buffer:
                        read_start_offset = tailer.offset
                        tailer_snapshot = (tailer.offset, tailer._buffer)
                        source_inode = tailer.observed_inode
                        source_generation = self.registry.generation(filepath)
                        new_lines = tailer.read_buffered_lines(max_lines=self.max_lines_per_file)
                    else:
                        tailer = self._ensure_tailer(filepath)
                        read_start_offset = tailer.offset
                        tailer_snapshot = (tailer.offset, tailer._buffer)
                        source_inode = tailer.observed_inode
                        source_generation = self.registry.generation(filepath)
                        new_lines = tailer.read_new_lines(
                            max_lines=self.max_lines_per_file,
                            max_bytes=self.max_read_bytes_per_file,
                        )

                    # Handle rewind detection (checkpoint restore)
                    if tailer.rewound:
                        read_start_offset = 0
                        self._handle_rewind(
                            filepath,
                            tailer.rewind_old_offset,
                            tailer.rewind_new_offset,
                            tailer.get_inode(),
                        )
                        tailer.rewound = False  # Reset flag
                        source_inode = tailer.observed_inode
                        source_generation = self.registry.generation(filepath)

                    normalized_lines = self._normalize_lines(filepath, new_lines) if new_lines else []
                    if normalized_lines:
                        for line in normalized_lines:
                            line["_source_inode"] = source_inode
                            line["_source_generation"] = source_generation
                        self.indexer.add(normalized_lines)
                        read_accepted = True
                        self._health_entries_seen += len(normalized_lines)
                        total_new += len(normalized_lines)
                    self._checkpoint_discarded_progress(
                        filepath,
                        read_start_offset,
                        tailer.offset,
                        normalized_lines,
                        source_inode,
                        source_generation,
                    )
                    if tailer.last_error is not None:
                        if not self._quarantine_failed_record(
                            filepath,
                            tailer,
                            source_inode,
                            source_generation,
                        ):
                            self._record_file_ingestion_failure(filepath, tailer.last_error)
                    else:
                        self._clear_file_ingestion_failure(filepath)
                    if tailer_snapshot is not None and (
                        tailer.offset != tailer_snapshot[0] or tailer._buffer != tailer_snapshot[1]
                    ):
                        self.last_poll_made_progress = True
                    read_accepted = True
                except Exception as error:
                    if tailer is not None and tailer_snapshot is not None and not read_accepted:
                        tailer.restore_snapshot(tailer_snapshot)
                        tailer.last_error = error
                        tailer.failed_record = None
                    logger.exception("Poll file error: %s", filepath)
                    self._record_file_ingestion_failure(filepath, error)

            self.indexer.tick()
            if self.on_tick:
                try:
                    self.on_tick()
                except Exception:
                    logger.exception("Watcher tick callback failed")
            return total_new
        finally:
            # Periodic registry flush
            now = time.monotonic()
            if now - self._last_registry_flush >= self.registry_flush_interval_s:
                self.registry.flush()
                self._last_registry_flush = now

            # Remember what we saw this pass so the next poll can skip files whose bytes
            # cannot have moved. Safe to record unconditionally: _can_skip_unchanged still
            # refuses when the tailer is behind the recorded size.
            self._observed_file_stats = dict(self._current_file_stats)

            self._write_health_snapshot(files)

    def start(self):
        """Start the watcher loop (blocking). Call stop() from another thread."""
        logger.info("JSONL watcher started: %s", self.watch_dir)
        start_time = time.monotonic()

        # Emit startup telemetry
        try:
            from .telemetry import emit_watcher_error, emit_watcher_heartbeat, emit_watcher_startup

            initial_files = self._discover_jsonl_files()
            emit_watcher_startup(
                sessions_watched=len(initial_files),
                watcher_pid=os.getpid(),
            )
        except Exception:
            pass  # Telemetry must never block startup

        heartbeat_interval_s = 60.0
        last_heartbeat = time.monotonic()
        last_error_fingerprint: str | None = None
        repeated_error_count = 0
        try:
            livelock_threshold = int(os.environ.get("BRAINLAYER_WATCHER_LIVELOCK_ERROR_THRESHOLD", "5"))
        except ValueError:
            livelock_threshold = 5

        while not self._stop.is_set():
            try:
                self.poll_once()
                last_error_fingerprint = None
                repeated_error_count = 0
            except Exception as e:
                logger.error("Poll cycle error: %s", e)
                fingerprint = f"{type(e).__name__}:{e}"
                if fingerprint == last_error_fingerprint:
                    repeated_error_count += 1
                else:
                    last_error_fingerprint = fingerprint
                    repeated_error_count = 1
                try:
                    emit_watcher_error("poll_cycle", str(e))
                except Exception:
                    pass
                if repeated_error_count > livelock_threshold:
                    logger.critical(
                        "Watcher live-locked on repeated poll error (%d): %s",
                        repeated_error_count,
                        fingerprint,
                    )
                    try:
                        from .telemetry import emit

                        emit(
                            "brainlayer-watcher",
                            {
                                "_type": "watcher_live_locked",
                                "error": fingerprint,
                                "repeated_error_count": repeated_error_count,
                            },
                        )
                    except Exception:
                        pass

            # Periodic heartbeat
            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_interval_s:
                try:
                    emit_watcher_heartbeat(
                        sessions_tracked=len(self._tailers),
                        chunks_indexed_total=self.indexer.total_flushed,
                        uptime_seconds=now - start_time,
                    )
                except Exception:
                    pass
                last_heartbeat = now
                logger.info(
                    "Watcher alive: %d sessions tracked, %d chunks indexed",
                    len(self._tailers),
                    self.indexer.total_flushed,
                )

            self._stop.wait(self.poll_interval_s)

        # Final flush
        self.indexer.flush()
        self.registry.flush()
        logger.info("JSONL watcher stopped. Total flushed: %d", self.indexer.total_flushed)

    def stop(self):
        """Signal the watcher to stop."""
        self._stop.set()
