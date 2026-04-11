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
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

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

    def read_new_lines(self) -> list[dict]:
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
        on_flush: Callable[[list[dict]], None],
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
            self.on_flush(batch)
            self._buffer = []  # Clear only after successful flush
            self.total_flushed += count
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
        watch_dir: str | Path,
        registry_path: str | Path,
        on_flush: Callable[[list[dict]], None],
        on_rewind: Callable[[str, str, int, int], None] | None = None,
        poll_interval_s: float = 1.0,
        batch_size: int = 10,
        flush_interval_ms: int = 100,
        registry_flush_interval_s: float = 5.0,
    ):
        self.watch_dir = Path(watch_dir).expanduser()
        self.on_rewind = on_rewind
        self.registry = OffsetRegistry(Path(registry_path).expanduser())
        self.indexer = BatchIndexer(
            on_flush=on_flush,
            batch_size=batch_size,
            flush_interval_ms=flush_interval_ms,
        )
        self.poll_interval_s = poll_interval_s
        self.registry_flush_interval_s = registry_flush_interval_s
        self._tailers: dict[str, JSONLTailer] = {}
        self._stop = threading.Event()
        self._last_registry_flush = time.monotonic()

    def _discover_jsonl_files(self) -> list[str]:
        """Find all .jsonl files under each watched project, including nested session artifacts."""
        files = []
        try:
            dirs = list(self.watch_dir.iterdir())
        except OSError:
            return files
        for project_dir in dirs:
            if not project_dir.is_dir():
                continue
            try:
                for f in project_dir.rglob("*.jsonl"):
                    if f.is_file():
                        files.append(str(f))
            except OSError:
                continue
        return files

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
            new_lines = tailer.read_new_lines()

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
                # Tag each line with source metadata
                for line in new_lines:
                    line.setdefault("_source_file", filepath)
                self.indexer.add(new_lines)
                self.registry.set(filepath, tailer.offset, tailer.get_inode())
                total_new += len(new_lines)

        self.indexer.tick()

        # Periodic registry flush
        now = time.monotonic()
        if now - self._last_registry_flush >= self.registry_flush_interval_s:
            self.registry.flush()
            self._last_registry_flush = now

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
