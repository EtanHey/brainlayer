"""Observe suspend time without using adjustable wall time as a detector.

Darwin clock_gettime(3): MONOTONIC_RAW advances through sleep, without clock
adjustments; UPTIME_RAW does not. Wall timestamps only label inferred bounds.
"""

from __future__ import annotations

import datetime as dt
import time

NOTE_JOIN_GAP_SECONDS = 20 * 60


def _clock_ids() -> tuple[int, int]:
    if callable(getattr(time, "clock_gettime", None)):
        for pair in (("CLOCK_MONOTONIC_RAW", "CLOCK_UPTIME_RAW"), ("CLOCK_BOOTTIME", "CLOCK_MONOTONIC")):
            if all(hasattr(time, name) for name in pair):
                return tuple(getattr(time, name) for name in pair)
    raise RuntimeError(
        "Backup sleep observation requires Darwin MONOTONIC_RAW/UPTIME_RAW or Linux BOOTTIME/MONOTONIC clocks; "
        "this platform cannot measure suspend time safely"
    )


def _read_clock(clock: int) -> float:
    try:
        return time.clock_gettime(clock)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot measure backup sleep: clock_gettime({clock}) failed: {exc}") from exc


def _inclusive() -> float:
    return _read_clock(_clock_ids()[0])


def _awake() -> float:
    return _read_clock(_clock_ids()[1])


def sleep_note(intervals: list[dict], *, action: str) -> str | None:
    if not intervals:
        return None
    minutes = int(sum(item["seconds"] for item in intervals) // 60)
    hours, minutes = divmod(minutes, 60)
    duration = f"{hours} h {minutes} min" if hours else f"{minutes} min"
    spans: list[tuple[dt.datetime, dt.datetime]] = []
    for start, end in sorted(
        (dt.datetime.fromisoformat(item["from"]), dt.datetime.fromisoformat(item["to"])) for item in intervals
    ):
        if spans and (start - spans[-1][1]).total_seconds() <= NOTE_JOIN_GAP_SECONDS:
            spans[-1] = (spans[-1][0], max(end, spans[-1][1]))
        else:
            spans.append((start, end))
    bounds = ", ".join(
        f"{start:{'%Y-%m-%d %H:%M' if start.date() != end.date() else '%H:%M'}}-"
        f"{end:{'%Y-%m-%d %H:%M' if start.date() != end.date() else '%H:%M'}}"
        for start, end in spans
    )
    return f"paused {duration} while the Mac slept ({bounds}); {action}"


class SleepTracker:
    def __init__(self, *, inclusive=_inclusive, awake=_awake, wall=None):
        self.inclusive = inclusive
        self.awake = awake
        self.wall = wall or (lambda: dt.datetime.now().astimezone())
        self.started = self.previous_inclusive = inclusive()
        self.previous_awake = awake()
        self.intervals: list[dict] = []

    def tick(self, *, prefix: str, action: str) -> dict:
        inclusive, awake = self.inclusive(), self.awake()
        slept = (inclusive - self.previous_inclusive) - (awake - self.previous_awake)
        if slept >= 60:
            end = self.wall()
            self.intervals.append(
                {
                    "from": (end - dt.timedelta(seconds=slept)).isoformat(),
                    "to": end.isoformat(),
                    "seconds": round(slept, 3),
                    "bounds": "inferred",
                }
            )
        self.previous_inclusive, self.previous_awake = inclusive, awake
        result = {
            f"{prefix}_wall_seconds": round(inclusive - self.started, 3),
            f"{prefix}_slept_seconds": round(sum(item["seconds"] for item in self.intervals), 3),
            "sleep_intervals": [dict(item) for item in self.intervals],
        }
        note = sleep_note(self.intervals, action=action)
        if note:
            result["sleep_note"] = note
        return result
