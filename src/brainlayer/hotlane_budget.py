"""Daemon-owned rolling CPU evidence; no database scans or sampling subprocess."""

import json
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone


class HotlaneCpuSampler:
    def __init__(self, db_path, *, clock=time.monotonic, cpu_clock=time.process_time):
        self.path = db_path.parent / "hotlane-cpu.json"
        self.db = str(db_path.resolve())
        self.clock, self.cpu_clock = clock, cpu_clock
        self.samples = deque()
        self.encoded = 0
        self.last_publish = float("-inf")
        self.state = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.worker = None
        self.observe("starting")

    def start(self):
        self.worker = threading.Thread(target=self._sample, name="hotlane-cpu", daemon=True)
        self.worker.start()

    def close(self):
        self.stop.set()
        if self.worker is not None:
            self.worker.join()

    def _sample(self):
        while not self.stop.wait(10):
            self.observe()

    def observe(self, state=None, backlog_encoded=0):
        with self.lock:
            self._observe(state or self.state or "starting", backlog_encoded)

    def _observe(self, state, backlog_encoded):
        now = self.clock()
        self.encoded += backlog_encoded
        self.samples.append((now, self.cpu_clock(), self.encoded))
        while len(self.samples) > 2 and self.samples[1][0] <= now - 600:
            self.samples.popleft()
        if now - self.last_publish < 10 and state == self.state:
            return
        self.last_publish, self.state = now, state
        first, last = self.samples[0], self.samples[-1]
        elapsed = last[0] - first[0]
        payload = dict(
            source="daemon",
            status="measured",
            platform=sys.platform,
            pid=os.getpid(),
            db=self.db,
            measured_at=datetime.now(timezone.utc).isoformat(),
            window_seconds=elapsed,
            cpu_percent=100 * (last[1] - first[1]) / elapsed if elapsed else 0,
            backlog_encoded=last[2] - first[2],
            backlog_state=state,
        )
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
            temporary.replace(self.path)
        except OSError:
            # Evidence loss must be visible; it must not stop the memory writer.
            import logging

            logging.getLogger(__name__).exception("cannot publish hotlane CPU evidence to %s", self.path)
