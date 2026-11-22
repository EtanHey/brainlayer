"""Observe suspend time without using adjustable wall time as a detector.

Darwin clock_gettime(3): MONOTONIC_RAW advances through sleep, without clock
adjustments; UPTIME_RAW does not. Wall timestamps only label inferred bounds.
"""

from __future__ import annotations

import datetime as dt
import time


def _inclusive() -> float:
    clock = time.CLOCK_BOOTTIME if hasattr(time, "CLOCK_BOOTTIME") else time.CLOCK_MONOTONIC_RAW
    return time.clock_gettime(clock)


def _awake() -> float:
    clock = time.CLOCK_UPTIME_RAW if hasattr(time, "CLOCK_UPTIME_RAW") else time.CLOCK_MONOTONIC
    return time.clock_gettime(clock)


def sleep_note(intervals: list[dict], *, action: str) -> str | None:
    if not intervals:
        return None
    minutes = int(sum(item["seconds"] for item in intervals) // 60)
    hours, minutes = divmod(minutes, 60)
    duration = f"{hours} h {minutes} min" if hours else f"{minutes} min"
    bounds = ", ".join(
        f"{dt.datetime.fromisoformat(item['from']):%H:%M}-{dt.datetime.fromisoformat(item['to']):%H:%M}"
        for item in intervals
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
