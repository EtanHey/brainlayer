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

import json
import logging
import os
import sqlite3
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WatchRoot:
    provider: str
    path: Path | str

    @property
    def resolved_path(self) -> Path:
        return Path(self.path).expanduser()


def default_watch_roots(home: Path | None = None) -> list[WatchRoot]:
    root = home or Path.home()
    return [
        WatchRoot("claude", root / ".claude" / "projects"),
        WatchRoot("codex", root / ".codex" / "sessions"),
        WatchRoot("cursor", root / ".cursor" / "sessions"),
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


class OffsetRegistry:
    """Persists file read offsets so we resume after restart.

    Stored as JSON: {filepath: {offset, inode, mtime}}
    Atomic writes via tmpfile + rename.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, dict] = {}
        self._dirty = False
        self._load()

    def _load(self):
        try:
            with open(self.path) as f:
                self._data = json.load(f)
        except (OSError, json.JSONDecodeError):
            self._data = {}

    def get(self, filepath: str) -> tuple[int, int]:
        """Return (offset, inode) for a file. (0, 0) if unknown."""
        entry = self._data.get(filepath, {})
        return entry.get("offset", 0), entry.get("inode", 0)

    def set(self, filepath: str, offset: int, inode: int):
        """Update offset for a file."""
        self._data[filepath] = {
            "offset": offset,
            "inode": inode,
            "mtime": time.time(),
        }
        self._dirty = True

    def flush(self) -> bool:
        """Atomically write to disk. Returns True on success."""
        if not self._dirty:
            return True
        tmp_path = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(self._data, f)
            os.rename(tmp_path, str(self.path))
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
        self._data.pop(filepath, None)
        self._dirty = True


# ── JSONL Tailer ─────────────────────────────────────────────────────────────


class JSONLTailer:
    """Tail-follows a single JSONL file from a stored offset.

    Handles partial writes: buffers incomplete lines until a newline arrives.
    Validates JSON before yielding — skips corrupt lines silently.
    Detects file rewinds (checkpoint restore) when file shrinks.
    """

    def __init__(self, filepath: str, offset: int = 0):
        self.filepath = filepath
        self.offset = offset
        self._buffer = b""
        self.rewound = False  # Set to True when rewind detected
        self.rewind_old_offset = 0
        self.rewind_new_offset = 0

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

    def read_new_lines(self, max_lines: int | None = None) -> list[dict]:
        """Read any new complete lines since last call. Returns parsed JSON dicts."""
        # Check for rewind before reading
        self.check_rewind()

        try:
            with open(self.filepath, "rb") as f:
                f.seek(self.offset + len(self._buffer))
                new_data = f.read()
        except OSError:
            return []

        if not new_data:
            return []

        self._buffer += new_data
        lines = []

        while b"\n" in self._buffer:
            if max_lines is not None and len(lines) >= max_lines:
                break
            nl_idx = self._buffer.index(b"\n")
            line_data = self._buffer[:nl_idx]
            self._buffer = self._buffer[nl_idx + 1 :]

            if not line_data.strip():
                self.offset += nl_idx + 1
                continue

            try:
                parsed = json.loads(line_data)
                if isinstance(parsed, dict):
                    lines.append(parsed)
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.debug("Skipping corrupt JSONL line at offset %d", self.offset)

            self.offset += nl_idx + 1

        return lines

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
        on_flush: Callable[[list[dict]], int | None],
        batch_size: int = 10,
        flush_interval_ms: int = 100,
    ):
        self.on_flush = on_flush
        self.batch_size = batch_size
        self.flush_interval_ms = flush_interval_ms
        self._buffer: list[dict] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()
        self.total_flushed = 0
        self.total_outputs = 0

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

    def _do_flush(self):
        """Internal flush — must be called with _lock held."""
        batch = self._buffer
        self._last_flush = time.monotonic()
        count = len(batch)
        try:
            result = self.on_flush(batch)
            self._buffer = []  # Clear only after successful flush
            self.total_flushed += count
            self.total_outputs += result if isinstance(result, int) else count
        except Exception as e:
            logger.error("Batch flush failed (%d items), retaining in buffer: %s", count, e)


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
        on_flush: Callable[[list[dict]], None] | None = None,
        on_rewind: Callable[[str, str, int, int], None] | None = None,
        watch_roots: list[WatchRoot] | None = None,
        db_path: str | Path | None = None,
        poll_interval_s: float = 1.0,
        batch_size: int = 10,
        flush_interval_ms: int = 100,
        registry_flush_interval_s: float = 5.0,
        health_path: str | Path | None = None,
        coverage_watchdog: CoverageWatchdog | None = None,
        max_lines_per_file: int = 100,
    ):
        if watch_roots is not None:
            self.watch_roots = [WatchRoot(root.provider, root.path) for root in watch_roots]
        elif watch_dir is not None:
            self.watch_roots = [WatchRoot("claude", Path(watch_dir).expanduser())]
        else:
            self.watch_roots = default_watch_roots()
        self.watch_dir = self.watch_roots[0].resolved_path if self.watch_roots else Path(".")
        self.on_rewind = on_rewind
        registry = (
            Path(registry_path).expanduser() if registry_path else Path.home() / ".local/share/brainlayer/offsets.json"
        )
        self.registry = OffsetRegistry(registry)
        self.indexer = BatchIndexer(
            on_flush=on_flush or (lambda _items: None),
            batch_size=batch_size,
            flush_interval_ms=flush_interval_ms,
        )
        self.poll_interval_s = poll_interval_s
        self.registry_flush_interval_s = registry_flush_interval_s
        self.max_lines_per_file = max(1, max_lines_per_file)
        self._tailers: dict[str, JSONLTailer] = {}
        self._file_providers: dict[str, str] = {}
        self._stop = threading.Event()
        self._last_registry_flush = time.monotonic()
        self.health_path = Path(health_path).expanduser() if health_path else None
        self.db_path = Path(db_path).expanduser() if db_path else None
        self.coverage_watchdog = coverage_watchdog or CoverageWatchdog()
        self._health_window_started = time.monotonic()
        self._health_window_started_epoch = time.time()
        self._health_entries_seen = 0
        self._health_output_at_start = 0

    def provider_for_file(self, filepath: str) -> str:
        return self._file_providers.get(filepath, "unknown")

    def _discover_jsonl_files(self) -> list[str]:
        """Find all .jsonl files under each watched project, including nested session artifacts."""
        discovered: list[tuple[float, str, str]] = []
        self._file_providers = {}
        for root in self.watch_roots:
            root_path = root.resolved_path
            if not root_path.exists():
                continue
            try:
                bases = [root_path]
                if root.provider == "claude":
                    bases = [path for path in root_path.iterdir() if path.is_dir()]
                for base in bases:
                    for f in base.rglob("*.jsonl"):
                        if f.is_file():
                            path = str(f)
                            try:
                                mtime = f.stat().st_mtime
                            except OSError as e:
                                logger.debug("Skipping JSONL file during discovery after stat failure: %s: %s", path, e)
                                continue
                            discovered.append((mtime, path, root.provider))
            except OSError:
                continue
        discovered.sort(key=lambda item: item[0], reverse=True)
        files = [path for _mtime, path, provider in discovered]
        self._file_providers = {path: provider for _mtime, path, provider in discovered}
        return files

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
            normalized.append(entry)
        return normalized

    def _max_offset_lag_bytes(self, files: list[str]) -> int:
        max_lag = 0
        for filepath in files:
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
                row = conn.execute(
                    """
                    SELECT COUNT(*) FROM chunks
                    WHERE source = 'realtime_watcher'
                      AND COALESCE(ingested_at, strftime('%s', created_at)) >= ?
                    """,
                    (int(self._health_window_started_epoch),),
                ).fetchone()
                return int(row[0]) if row else 0
            finally:
                conn.close()
        except sqlite3.Error:
            return None

    def _write_health_snapshot(self, files: list[str]):
        if not self.health_path:
            return

        now = time.monotonic()
        elapsed = max(now - self._health_window_started, 1.0)
        entries_per_min = self._health_entries_seen / elapsed * 60.0
        outputs_per_min = (self.indexer.total_outputs - self._health_output_at_start) / elapsed * 60.0
        db_inserts = self._db_realtime_inserts_since_window_start()
        inserts_per_min = (db_inserts / elapsed * 60.0) if db_inserts is not None else outputs_per_min
        max_lag = self._max_offset_lag_bytes(files)
        watchdog = self.coverage_watchdog.evaluate(
            active_entries_per_minute=entries_per_min,
            realtime_inserts_per_minute=inserts_per_min,
            max_offset_lag_bytes=max_lag,
        )
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "providers": sorted({root.provider for root in self.watch_roots}),
            "files_tracked": len(files),
            "active_jsonl_entries_per_minute": entries_per_min,
            "db_realtime_inserts_per_minute": inserts_per_min,
            "watcher_chunks_output_per_minute": outputs_per_min,
            "max_offset_lag_bytes": max_lag,
            **watchdog,
        }
        try:
            self.health_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.health_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            tmp_path.replace(self.health_path)
        except OSError:
            logger.debug("Failed to write watcher health snapshot", exc_info=True)

        if elapsed >= 60.0:
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
            if stored_inode != 0 and current_inode != stored_inode:
                # File was replaced — reset offset
                tailer = JSONLTailer(filepath, offset=0)
                self._tailers[filepath] = tailer
            return tailer

        stored_offset, stored_inode = self.registry.get(filepath)
        tailer = JSONLTailer(filepath, offset=stored_offset)

        # Verify inode matches
        current_inode = tailer.get_inode()
        if stored_inode != 0 and current_inode != stored_inode:
            tailer = JSONLTailer(filepath, offset=0)

        self._tailers[filepath] = tailer
        return tailer

    def poll_once(self) -> int:
        """Run one poll cycle. Returns number of new lines found."""
        total_new = 0
        files = self._discover_jsonl_files()

        for filepath in files:
            tailer = self._ensure_tailer(filepath)
            new_lines = tailer.read_new_lines(max_lines=self.max_lines_per_file)

            # Handle rewind detection (checkpoint restore)
            if tailer.rewound:
                session_id = Path(filepath).stem
                logger.warning(
                    "Checkpoint restore: %s (offset %d → %d)",
                    session_id,
                    tailer.rewind_old_offset,
                    tailer.rewind_new_offset,
                )
                try:
                    from .telemetry import emit

                    emit(
                        "brainlayer-watcher",
                        {
                            "_type": "rewind_detected",
                            "session_id": session_id,
                            "file_path": filepath,
                            "old_offset": tailer.rewind_old_offset,
                            "new_offset": tailer.rewind_new_offset,
                        },
                    )
                except Exception:
                    pass

                # Call rewind callback if set
                if self.on_rewind:
                    try:
                        self.on_rewind(
                            filepath,
                            session_id,
                            tailer.rewind_old_offset,
                            tailer.rewind_new_offset,
                        )
                    except Exception as e:
                        logger.error("Rewind callback failed: %s", e)

                tailer.rewound = False  # Reset flag

            if new_lines:
                normalized_lines = self._normalize_lines(filepath, new_lines)
                self.indexer.add(normalized_lines)
                self.registry.set(filepath, tailer.offset, tailer.get_inode())
                self._health_entries_seen += len(normalized_lines)
                total_new += len(normalized_lines)

        self.indexer.tick()

        # Periodic registry flush
        now = time.monotonic()
        if now - self._last_registry_flush >= self.registry_flush_interval_s:
            self.registry.flush()
            self._last_registry_flush = now

        self._write_health_snapshot(files)

        return total_new

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

        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as e:
                logger.error("Poll cycle error: %s", e)
                try:
                    emit_watcher_error("poll_cycle", str(e))
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
