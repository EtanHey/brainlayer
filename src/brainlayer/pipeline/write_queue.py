from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable

_STOP = object()


class WriteQueueFullError(RuntimeError):
    """Raised when the single-writer queue cannot accept more work."""


@dataclass
class WriteIntent:
    name: str
    callback: Callable[[], Any]
    crash_on_error: bool = False
    future: Future[Any] = field(default_factory=Future)

    def execute(self) -> Any:
        return self.callback()


class WriteQueue:
    def __init__(self, maxsize: int = 1000):
        self._queue: queue.Queue[WriteIntent | object] = queue.Queue(maxsize=maxsize)
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._run, name="brainlayer-write-worker", daemon=True)
            self._worker.start()

    def submit(
        self,
        name: str,
        callback: Callable[[], Any],
        *,
        crash_on_error: bool = False,
        timeout: float | None = 30.0,
    ) -> Future[Any]:
        self.start()
        intent = WriteIntent(name=name, callback=callback, crash_on_error=crash_on_error)
        try:
            self._queue.put(intent, timeout=timeout)
        except queue.Full as exc:
            intent.future.set_exception(WriteQueueFullError(f"write queue is full: {name}"))
            raise WriteQueueFullError(f"write queue is full: {name}") from exc
        return intent.future

    def stop(self, timeout: float = 5.0) -> None:
        with self._lock:
            worker = self._worker
            if worker is None:
                return
        deadline = time.monotonic() + max(timeout, 0.0)
        while worker.is_alive():
            try:
                self._queue.put(_STOP, timeout=0.05)
                break
            except queue.Full:
                if time.monotonic() >= deadline:
                    break
        worker.join(timeout=timeout)
        if not worker.is_alive():
            with self._lock:
                self._worker = None

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                self._queue.task_done()
                break

            assert isinstance(item, WriteIntent)
            try:
                result = item.execute()
            except Exception as exc:
                if not item.future.done():
                    item.future.set_exception(exc)
                self._queue.task_done()
                if item.crash_on_error:
                    break
                continue

            if not item.future.done():
                item.future.set_result(result)
            self._queue.task_done()

        with self._lock:
            self._worker = None
