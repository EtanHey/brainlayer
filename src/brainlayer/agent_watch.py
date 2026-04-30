"""Polling watcher for Codex, Cursor, and Gemini session artifacts."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class AgentSessionRegistry:
    """Persists mtime/size state for agent session files."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, dict[str, int]] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path) as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                self._data = data
        except (OSError, json.JSONDecodeError, ValueError):
            self._data = {}

    def get(self, filepath: str) -> dict[str, int] | None:
        return self._data.get(filepath)

    def set(self, filepath: str, *, mtime_ns: int, size: int) -> None:
        self._data[filepath] = {"mtime_ns": mtime_ns, "size": size}
        self._dirty = True

    def flush(self) -> bool:
        if not self._dirty:
            return True

        tmp_path = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
            with os.fdopen(fd, "w") as fh:
                json.dump(self._data, fh)
            os.rename(tmp_path, self.path)
            self._dirty = False
            return True
        except OSError as exc:
            logger.warning("Failed to flush agent registry: %s", exc)
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            return False


@dataclass(frozen=True)
class AgentSessionSource:
    name: str
    patterns: list[str]
    ingest: Callable[[Path], int]
    root: Path


class AgentSessionWatcher:
    """Poll source roots and ingest files whose size/mtime changed."""

    def __init__(
        self,
        registry_path: str | Path,
        sources: list[AgentSessionSource],
        poll_interval_s: float = 30.0,
        registry_flush_interval_s: float = 5.0,
    ):
        self.registry = AgentSessionRegistry(registry_path)
        self.sources = sources
        self.poll_interval_s = poll_interval_s
        self.registry_flush_interval_s = registry_flush_interval_s
        self._stop = threading.Event()
        self._last_registry_flush = time.monotonic()

    def _discover_files(self, source: AgentSessionSource) -> list[Path]:
        files: list[Path] = []
        if not source.root.exists():
            return files
        for pattern in source.patterns:
            try:
                files.extend(path for path in source.root.glob(pattern) if path.is_file())
            except OSError:
                continue
        return sorted(set(files))

    def poll_once(self) -> int:
        processed = 0
        for source in self.sources:
            for file_path in self._discover_files(source):
                try:
                    stat = file_path.stat()
                except OSError:
                    continue

                state = {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
                previous = self.registry.get(str(file_path))
                if previous == state:
                    continue

                indexed = source.ingest(file_path)
                logger.info("Agent ingest %s %s -> %d chunks", source.name, file_path.name, indexed)
                self.registry.set(str(file_path), **state)
                processed += 1

        now = time.monotonic()
        if now - self._last_registry_flush >= self.registry_flush_interval_s:
            self.registry.flush()
            self._last_registry_flush = now
        return processed

    def start(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as exc:
                logger.error("Agent watcher poll failed: %s", exc)
            self._stop.wait(self.poll_interval_s)

        self.registry.flush()

    def stop(self) -> None:
        self._stop.set()

