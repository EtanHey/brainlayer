"""Bounded migration lifecycle gates; no installer activation on import."""

import math
import os
import re
import time

from .consolidation_plan import ALIASES, ARCHIVED, NEW, probe_service


def deadline_after(clock, grace):
    if type(grace) not in {int, float} or not math.isfinite(grace) or grace < 0:
        raise ValueError("migration deadline must be finite and nonnegative")
    return clock.monotonic() + grace


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class LaunchdRuntime:
    def __init__(self, domain, run, *, alive=pid_alive, clock=time):
        if not re.fullmatch(r"gui/[1-9][0-9]*", domain):
            raise ValueError("migration requires an explicit GUI domain")
        self.domain, self.run, self.alive, self.clock = domain, run, alive, clock
        self.observed_pids = {}

    def _checked(self, action, target, *args):
        result = self.run(["launchctl", action, target, *map(str, args)])
        if result.returncode:
            raise RuntimeError(f"launchctl {action} failed for {target}: {result.returncode}")
        return result

    def observe(self, label):
        if label not in set(ALIASES) | ARCHIVED | set(NEW):
            raise ValueError("refusing non-owned launchd service")
        return probe_service(self.domain, label, self.run)

    def stop(self, label, *, grace=30, expected=None):
        deadline = deadline_after(self.clock, grace) + 2
        state = self.observe(label)
        pids = self.observed_pids.setdefault(label, set())
        if state["pid"] is not None:
            pids.add(state["pid"])
        if expected is not None and any(state[key] != expected[key] for key in state):
            raise RuntimeError(f"service changed since snapshot: {label}")
        if state["loaded"]:
            self._checked("bootout", f"{self.domain}/{label}")
        while True:
            current = self.observe(label)
            if current["pid"] is not None:
                pids.add(current["pid"])
            if not current["loaded"] and not any(self.alive(pid) for pid in pids):
                return
            if self.clock.monotonic() >= deadline:
                raise RuntimeError(f"service did not exit: {label}")
            self.clock.sleep(0.1)


def wait_supervisor(query, expected, *, grace=10, clock=time, alive=pid_alive):
    """Wait for matching configuration, unpaused jobs and live enabled residents."""
    if not expected:
        raise ValueError("readiness requires a nonempty component manifest")
    deadline = deadline_after(clock, grace)
    while True:
        try:
            states = query()
            if (
                isinstance(states, dict)
                and set(states) == set(expected)
                and all(
                    isinstance(states[name], dict)
                    and states[name].get("configured") is child.enabled
                    and states[name].get("paused") is False
                    and (
                        states[name].get("pid") is None
                        if not child.enabled
                        else child.interval
                        and states[name].get("pid") is None
                        or type(states[name].get("pid")) is int
                        and states[name]["pid"] > 1
                        and alive(states[name]["pid"])
                    )
                    for name, child in expected.items()
                )
            ):
                return states
        except (OSError, ValueError, KeyError, RuntimeError):
            pass
        if clock.monotonic() >= deadline:
            raise RuntimeError("replacement supervisor not ready")
        clock.sleep(0.1)
