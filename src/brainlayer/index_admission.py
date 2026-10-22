"""Calendar index admission: wait for a writer-idle window inside the pass budget.

Resident writers (drain, hotlane) commit continuously, so a calendar pass that hits
a busy rollback has usually lost one race, not the night. The pass backs off and
retries the same batch from its checkpoint, never as a concurrent second writer:
the lock and the bounded busy_timeout stay exactly as they are. Only when no
admission is possible before the deadline does the pass defer (rc75).
"""

from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")
# The admission wait's own binding: a test can fake this pause without also
# speeding up every other thread that sleeps (writer-telemetry heartbeats).
_sleep = time.sleep

ADMISSION_BASE_S = 1.0
ADMISSION_CAP_S = 30.0
# Time one more write attempt needs after a wait: upsert's own bounded busy retries
# plus a batch commit. Never sleep into it, so a lost race ends as an admission
# deferral rather than a busy error at the cap.
ADMISSION_ATTEMPT_RESERVE_S = 15.0


class IndexAdmissionExhausted(Exception):
    """No writer-idle window opened before the pass deadline.

    Deliberately not a RuntimeError: the per-file handler treats those as input
    faults and would count lost admission toward quarantining a healthy file.
    """

    def __init__(self, cause: BaseException, *, retries: int, waited_s: float):
        super().__init__(f"no writer admission before the deadline after {retries} retries: {cause}")
        self.cause = cause
        self.retries = retries
        self.waited_s = waited_s


class AdmissionBackoff:
    """Exponential backoff with jitter and a cap, bounded by one shared deadline."""

    def __init__(
        self,
        deadline_monotonic: float,
        *,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        rng: Callable[[], float] = random.random,
        base_s: float = ADMISSION_BASE_S,
        cap_s: float = ADMISSION_CAP_S,
        attempt_reserve_s: float = ADMISSION_ATTEMPT_RESERVE_S,
        on_wait: Callable[[int, float, BaseException], None] | None = None,
    ):
        self.deadline_monotonic = deadline_monotonic
        # Resolve the clock and sleep at call time so a patched module clock is honoured.
        self._monotonic = monotonic or (lambda: time.monotonic())
        self._sleep = sleep or (lambda seconds: _sleep(seconds))
        self._rng = rng
        self.base_s, self.cap_s, self.attempt_reserve_s = base_s, cap_s, attempt_reserve_s
        self._on_wait = on_wait
        self._streak = 0
        self.retries = 0
        self.waited_s = 0.0

    def wait(self, cause: BaseException) -> None:
        """Sleep before the next attempt, or raise when the budget cannot admit one."""
        now = self._monotonic()
        if now >= self.deadline_monotonic:
            # The cap already passed: the caller's deadline attribution owns this error.
            raise cause
        step = min(self.cap_s, self.base_s * (2**self._streak))
        delay = step * (0.5 + 0.5 * self._rng())
        if now + delay + self.attempt_reserve_s > self.deadline_monotonic:
            raise IndexAdmissionExhausted(cause, retries=self.retries, waited_s=self.waited_s) from cause
        self._streak += 1
        self.retries += 1
        self.waited_s += delay
        if self._on_wait is not None:
            self._on_wait(self.retries, delay, cause)
        self._sleep(delay)

    def admitted(self) -> None:
        """A write committed: the next contention starts a fresh backoff streak."""
        self._streak = 0

    def run(self, attempt: Callable[[], T]) -> T:
        """Call ``attempt`` until it is admitted past busy/writer-in-use contention."""
        import apsw

        from .vector_store import WriterInUseError

        while True:
            try:
                result = attempt()
            except (apsw.BusyError, WriterInUseError) as exc:
                self.wait(exc)
                continue
            self.admitted()
            return result
