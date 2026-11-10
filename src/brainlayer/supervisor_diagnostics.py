"""Bounded best-effort human diagnostics and optional JSONL, without fsync.

The timeout path only enqueues into a bounded memory queue. A daemon writer owns
formatting, logging and journal writes; a full queue or stalled writer cannot hold
up kill or reap.
Enqueue ordering is guaranteed, crash durability is deliberately not.
"""

import json
import logging
import os
import queue
import stat
import threading
import time
import traceback
from contextlib import suppress


class DiagnosticSink:
    def __init__(self, fd, *, capacity=64 * 1024):
        self.fd, self.capacity = fd, capacity
        self.pending = queue.Queue(maxsize=32)
        self.closed = threading.Event()
        self.writer = threading.Thread(target=self._write, daemon=True, name="guard-diagnostics")
        self.writer.start()

    @classmethod
    def open(cls, path, *, journal_optional=False, **kwargs):
        try:
            fd = cls.open_fd(path)
        except OSError:
            if not journal_optional:
                raise
            fd = None
        try:
            sink = cls(fd, **kwargs)
        except BaseException:
            if fd is not None:
                os.close(fd)
            raise
        if fd is None:
            sink.emit(event="diagnostic-journal-unavailable")
        return sink

    @staticmethod
    def open_fd(path):
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1:
            os.close(fd)
            raise PermissionError("unsafe diagnostic sink")
        return fd

    def emit(self, **event):
        if self.closed.is_set():
            return
        row = dict(schema_version=1, ts=time.time(), **event)
        try:
            self.pending.put_nowait(row)
        except queue.Full:
            pass

    def _write(self):
        try:
            while not self.closed.is_set() or not self.pending.empty():
                try:
                    event = self.pending.get(timeout=1)
                except queue.Empty:
                    continue
                if event is None:
                    break
                exc_info = event.pop("exc_info", None)
                if exc_info is not None:
                    event["traceback"] = "".join(traceback.format_exception(*exc_info))[-2048:]
                logger = logging.getLogger("brainlayer.service_supervisor")
                if event.get("event") == "interval-delayed":
                    logger.warning(
                        "periodic component %s delayed %.1fs: %s",
                        event["component"],
                        event["delay_s"],
                        event["reason"],
                    )
                elif event.get("event") == "diagnostic-journal-unavailable":
                    logger.warning("operation diagnostic journal unavailable; human logging remains active")
                elif event.get("event") == "component-startup-failed":
                    logger.error("supervisor component startup failed\n%s", event.get("traceback") or event["error"])
                if self.fd is None:
                    continue
                try:
                    self._append(event)
                except OSError:
                    fd, self.fd = self.fd, None
                    with suppress(OSError):
                        os.close(fd)
                    logger.warning("diagnostic journal failed; human logging and timeout enforcement remain active")
        finally:
            if self.fd is not None:
                os.close(self.fd)

    def _append(self, event):
        row = (json.dumps(event, sort_keys=True) + "\n").encode()
        if len(row) > min(4096, self.capacity):
            return
        end = os.lseek(self.fd, 0, os.SEEK_END)
        if end + len(row) > self.capacity:
            start = max(0, end - self.capacity)
            history = os.pread(self.fd, min(end, self.capacity), start)
            if start:
                history = history.partition(b"\n")[2]
            lines = [line for line in history.splitlines(keepends=True) if line.endswith(b"\n")]
            retained, size = [], 0
            room = min(self.capacity // 2, self.capacity - len(row))
            for line in reversed(lines):
                if size + len(line) > room:
                    break
                retained.insert(0, line)
                size += len(line)
            # Keep the newest timeout even through a subsequent startup storm.
            if event.get("event") != "guard-timeout":
                for line in reversed(lines):
                    try:
                        timeout = json.loads(line).get("event") == "guard-timeout"
                    except (ValueError, AttributeError):
                        continue
                    if timeout:
                        if line not in retained and len(line) + len(row) <= self.capacity:
                            while retained and size + len(line) + len(row) > self.capacity:
                                size -= len(retained.pop(0))
                            retained.insert(0, line)
                        break
            row = b"".join(retained) + row
            os.ftruncate(self.fd, 0)
            os.lseek(self.fd, 0, os.SEEK_SET)
        view = memoryview(row)
        while view:
            written = os.write(self.fd, view)
            if written <= 0:
                raise OSError("diagnostic short write")
            view = view[written:]

    def close(self, *, drain=0.02):
        self.closed.set()
        try:
            self.pending.put_nowait(None)
        except queue.Full:
            pass  # A full queue already wakes the writer; it exits after draining.
        self.writer.join(timeout=drain)  # Never wait for disk recovery, including after a kill.


def emit(sink, **event):
    """Diagnostics never decide whether a timeout is enforced."""
    if sink is not None:
        try:
            sink.emit(**event)
        except Exception:
            pass
