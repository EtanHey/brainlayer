from __future__ import annotations

import threading
import time


class TokenBucket:
    def __init__(self, rate_per_sec: float, burst: int = 10):
        self.rate_per_sec = rate_per_sec
        self.burst = max(1, burst)
        self._tokens = float(self.burst)
        self._updated_at = time.monotonic()
        self._condition = threading.Condition()

    def acquire(self, n: int = 1) -> None:
        if n <= 0:
            raise ValueError("requested tokens must be positive")
        if n > self.burst:
            raise ValueError("requested tokens exceed burst capacity")
        if self.rate_per_sec <= 0:
            return

        with self._condition:
            while True:
                now = time.monotonic()
                elapsed = max(0.0, now - self._updated_at)
                if elapsed:
                    self._tokens = min(self.burst, self._tokens + (elapsed * self.rate_per_sec))
                    self._updated_at = now

                if self._tokens >= n:
                    self._tokens -= n
                    return

                missing = n - self._tokens
                wait_time = missing / self.rate_per_sec
                self._condition.wait(timeout=wait_time)
