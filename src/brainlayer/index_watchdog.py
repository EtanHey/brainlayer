"""Wall-clock enforcement for a `brainlayer index` run, owned by a side thread.

Observed on the M1, 2026-09-05: `com.brainlayer.index` reached 12h13m against a 4h cap with
no alarm, ~100% CPU, WAL unchanged, no JSONL open, and a stdout log silent for 12h.

The plan's hypothesis was that the apsw progress handler in `VectorStore.upsert_chunks`
cannot fire inside a boundary-less phase. Measured on this apsw (3.51.2.0 / SQLite 3.51.2),
that is NOT true -- the handler ticks freely:

* FTS5 `rebuild` over 400k rows: 7,287 ticks in 0.68s
* vec0 bulk `DELETE`: 1,008 ticks in 0.042s
* vec0 KNN over 1.5M rows: 8 ticks in 0.167s (~48/s)

and the `id=`-keyed registration `upsert_chunks` uses does register (34,180 ticks). So a
long statement *inside `upsert_chunks`* would have been stopped.

The actual hole is narrower and simpler: enforcement exists ONLY inside `upsert_chunks` and
at `cli.index_fast`'s per-file boundaries. Every other phase of a run has no cap of any
kind -- notably opening the writer store (schema validation and migrations, which is
read-heavy and matches "100% CPU, WAL unchanged, only DB files open") and the first
embedding-model load. Any of those running long explains a 12h run with no alarm without
needing an opcode blackout at all.

So the cap cannot live at the work sites; it has to live beside them. This watchdog runs its
own thread, which a blocked main thread cannot delay:

1. the moment the deadline passes it emits ``INDEX_RUNTIME_EXCEEDED`` naming the phase, so
   the run is on the record even while a stuck phase is still unwinding. This is
   unconditional -- it holds for a phase issuing no SQL at all, such as a wedged model load;
2. it calls ``sqlite3_interrupt`` on the writer connection (apsw documents this as safe from
   another thread). Measured: it aborts a long FTS5 statement mid-flight -- 0.22s into a
   0.66s rebuild -- leaving `integrity_check` ok, the transaction rolled back, and every row
   readable;
3. it leaves ``expired`` set, which the main thread turns into ``IndexDeadlineExceeded`` at
   its next safe point;
4. between those it prints a heartbeat carrying the phase and the age of the last progress,
   so a 12h-silent log cannot recur -- and so the next real stall names its own phase.

Not covered: a phase that neither issues SQL nor returns cannot be *aborted* from here, only
alarmed and heartbeaten. Killing such a run is a policy call and its own change.

The deadline comes from this module's import-time binding of ``time.monotonic``. That is
deliberate: the CLI's clock is monkeypatched by several tests, and a watchdog reading a
patched clock would both mis-fire and steal values from those tests' iterators.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
from time import monotonic as _wall_monotonic
from typing import Any, Callable

from .vector_store import IndexDeadlineExceeded

logger = logging.getLogger(__name__)

ALARM_CODE = "INDEX_RUNTIME_EXCEEDED"
HEARTBEAT_PREFIX = "BRAINLAYER_INDEX_HEARTBEAT"

_DEFAULT_HEARTBEAT_INTERVAL_S = 300.0
_DEFAULT_POLL_INTERVAL_S = 0.25
# One interrupt aborts the statement running when it lands; SQLite documents it as a no-op
# for statements started afterwards. A small bounded retry covers a stall that restarts
# under upsert_chunks' busy-retry loop, without machine-gunning the ROLLBACK that follows.
_MAX_INTERRUPT_ATTEMPTS = 3
_JOIN_TIMEOUT_S = 2.0


def _heartbeat_interval_s_env() -> float:
    """Seconds between index heartbeats; ``BRAINLAYER_INDEX_HEARTBEAT_S`` overrides."""
    raw = os.environ.get("BRAINLAYER_INDEX_HEARTBEAT_S")
    if raw is None:
        return _DEFAULT_HEARTBEAT_INTERVAL_S
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_HEARTBEAT_INTERVAL_S
    if not math.isfinite(value) or value <= 0:
        return _DEFAULT_HEARTBEAT_INTERVAL_S
    return value


class IndexWatchdog:
    """Enforce a wall-clock cap on an index run from a side thread.

    ``counters`` is consulted only when the alarm fires, so the alarm can carry whatever
    the main thread had committed at that moment without the watchdog tracking it.
    """

    def __init__(
        self,
        *,
        max_runtime_s: float,
        interrupt: Callable[[], None] | None = None,
        emit: Callable[[Any], bool] | None = None,
        counters: Callable[[], dict[str, Any]] | None = None,
        monotonic: Callable[[], float] = _wall_monotonic,
        heartbeat_interval_s: float | None = None,
        poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
        heartbeat_sink: Callable[[str], None] | None = None,
        phase: str = "starting",
    ) -> None:
        self._max_runtime_s = float(max_runtime_s)
        self._monotonic = monotonic
        self._emit = emit
        self._counters = counters
        self._heartbeat_interval_s = (
            heartbeat_interval_s if heartbeat_interval_s is not None else _heartbeat_interval_s_env()
        )
        self._poll_interval_s = max(0.01, float(poll_interval_s))
        self._heartbeat_sink = heartbeat_sink

        self._lock = threading.Lock()
        self._interrupt = interrupt
        self._phase = phase
        self._last_progress = self._monotonic()
        self._started_at = self._last_progress
        self._deadline = self._started_at + self._max_runtime_s

        self._expired = threading.Event()
        self._alarm_emitted = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.interrupt_attempts = 0

    # ── wiring the main thread updates ──────────────────────────────────

    def set_interrupt(self, interrupt: Callable[[], None] | None) -> None:
        """Attach the writer connection's ``interrupt`` once the store is open.

        The watchdog starts before the store exists so that opening it -- which can run
        migrations, another boundary-less phase -- is still covered by the alarm and the
        heartbeat.
        """
        with self._lock:
            self._interrupt = interrupt

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase
            self._last_progress = self._monotonic()

    def note_progress(self) -> None:
        with self._lock:
            self._last_progress = self._monotonic()

    # ── state the main thread reads ─────────────────────────────────────

    @property
    def expired(self) -> bool:
        return self._expired.is_set()

    @property
    def alarm_emitted(self) -> bool:
        return self._alarm_emitted.is_set()

    @property
    def phase(self) -> str:
        with self._lock:
            return self._phase

    def elapsed_s(self) -> float:
        return max(0.0, self._monotonic() - self._started_at)

    def context(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Cap-alarm context: always carries the phase and the age of the last progress."""
        with self._lock:
            phase = self._phase
            last_progress = self._last_progress
        now = self._monotonic()
        context: dict[str, Any] = {
            "max_runtime_s": self._max_runtime_s,
            "elapsed_s": round(max(0.0, now - self._started_at), 3),
            "phase": phase,
            "last_progress_age_s": round(max(0.0, now - last_progress), 3),
        }
        if extra:
            context.update(extra)
        return context

    def raise_if_expired(self, processed_count: int = 0) -> None:
        if self._expired.is_set():
            raise IndexDeadlineExceeded(processed_count)

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(self) -> "IndexWatchdog":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, name="brainlayer-index-watchdog", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=_JOIN_TIMEOUT_S)

    def __enter__(self) -> "IndexWatchdog":
        return self.start()

    def __exit__(self, *_args: object) -> bool:
        self.stop()
        return False

    # ── the thread ──────────────────────────────────────────────────────

    def _run(self) -> None:
        next_heartbeat = self._started_at + self._heartbeat_interval_s
        while not self._stop.wait(self._poll_interval_s):
            now = self._monotonic()
            if now >= self._deadline:
                self._on_deadline()
                # Keep polling: a bounded interrupt retry still has attempts left, and the
                # heartbeat must keep reporting while the main thread unwinds.
            elif now >= next_heartbeat:
                self._heartbeat()
                next_heartbeat = now + self._heartbeat_interval_s

    def _on_deadline(self) -> None:
        # The alarm goes first, before the interrupt: if interrupting wedges or the stall
        # ignores it, the run is still on the record as over its cap.
        if not self._alarm_emitted.is_set():
            self._alarm_emitted.set()
            self._emit_cap_alarm()
        self._expired.set()
        self._pull_interrupt()

    def _emit_cap_alarm(self) -> None:
        from .alarm import build_alarm, emit_alarm

        extra: dict[str, Any] = {"stopped_by": "watchdog"}
        if self._counters is not None:
            try:
                extra.update(self._counters())
            except Exception as exc:  # a counter must never suppress the alarm
                logger.debug("Index watchdog counters failed: %s", exc)
        alarm = build_alarm(
            ALARM_CODE,
            "brainlayer index exceeded its maximum runtime inside a phase with no transaction boundary",
            self.context(extra),
        )
        emit = self._emit if self._emit is not None else emit_alarm
        try:
            emit(alarm)
        except Exception as exc:  # never let the watchdog thread die on its own alarm
            logger.debug("Index watchdog alarm emit failed: %s", exc)

    def _pull_interrupt(self) -> None:
        with self._lock:
            interrupt = self._interrupt
        if interrupt is None or self.interrupt_attempts >= _MAX_INTERRUPT_ATTEMPTS:
            return
        self.interrupt_attempts += 1
        try:
            interrupt()
        except Exception as exc:
            logger.debug("Index watchdog interrupt failed: %s", exc)

    def _heartbeat(self) -> None:
        context = self.context()
        line = (
            f"{HEARTBEAT_PREFIX} phase={context['phase']} "
            f"elapsed_s={context['elapsed_s']} "
            f"last_progress_age_s={context['last_progress_age_s']} "
            f"max_runtime_s={context['max_runtime_s']}"
        )
        # debug, not warning: stderr below is the visible path, and a stderr log handler
        # at default level would otherwise print every heartbeat twice (seen live).
        logger.debug(line)
        sink = self._heartbeat_sink
        try:
            if sink is not None:
                sink(line)
            else:
                print(line, file=sys.stderr, flush=True)
        except Exception as exc:
            logger.debug("Index heartbeat emit failed: %s", exc)
