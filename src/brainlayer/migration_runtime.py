"""Bounded migration lifecycle gates; no installer activation on import."""

import math
import os
import re
import sys
import time

from .consolidation_plan import ALIASES, ARCHIVED, NEW, probe_service


def supervisor_ready_grace():
    """Shared consolidation readiness/exit budget, validated before handoff."""
    name = "BRAINLAYER_SUPERVISOR_READY_GRACE_S"
    value = os.environ.get(name, "90")
    try:
        if not re.fullmatch(r"[0-9]+", value):
            raise ValueError
        grace = int(value)
        if not 10 <= grace <= 900:
            raise ValueError
    except ValueError:
        raise ValueError(f"{name} must be an integer from 10 to 900") from None
    return grace


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

    def stop(self, label, *, grace=None, expected=None):
        deadline = deadline_after(self.clock, supervisor_ready_grace() if grace is None else grace) + 2
        next_report = self.clock.monotonic() + 10
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
            if self.clock.monotonic() >= next_report:
                print(f"Waiting for service to exit: {label}", file=sys.stderr, flush=True)
                next_report = self.clock.monotonic() + 10
            self.clock.sleep(0.1)


def wait_supervisor(query, expected, *, grace=None, clock=time, alive=pid_alive):
    """Wait for matching configuration, unpaused jobs and live enabled residents."""
    if not expected:
        raise ValueError("readiness requires a nonempty component manifest")
    deadline = deadline_after(clock, supervisor_ready_grace() if grace is None else grace)
    next_report = clock.monotonic() + 10
    while True:
        pending = list(expected)
        try:
            states = query()
            if isinstance(states, dict):
                pending = [
                    name
                    for name, child in expected.items()
                    if not (
                        isinstance(states.get(name), dict)
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
                    )
                ]
                pending.extend(f"unexpected {name}" for name in states.keys() - expected.keys())
                if not pending:
                    return states
        except (OSError, ValueError, KeyError, RuntimeError):
            pass
        if clock.monotonic() >= deadline:
            raise RuntimeError("replacement supervisor not ready")
        if clock.monotonic() >= next_report:
            print(f"Replacement supervisor not ready; waiting for: {', '.join(pending)}", file=sys.stderr, flush=True)
            next_report = clock.monotonic() + 10
        clock.sleep(0.1)
