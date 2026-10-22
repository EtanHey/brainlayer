"""Opt-in acknowledged component quiescence with an inherited writer-exclusion lease."""

import ctypes
import datetime as dt
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .drive_credentials import _atomic_write
from .supervisor_guard import component_gate, start_time

HOLD_MAX_ENV = "BRAINLAYER_MAINTENANCE_HOLD_MAX_S"
HOLD_GRACE_ENV = "BRAINLAYER_MAINTENANCE_HOLD_GRACE_S"
HOLD_KILL_AFTER_ENV = "BRAINLAYER_MAINTENANCE_HOLD_KILL_AFTER_S"
DEFAULT_HOLD_MAX_S = 3600.0
# The hold ends this long before the quiet window closes, leaving room to resume and catch up.
HOLD_WINDOW_RESERVE_S = 600.0
DEFAULT_WINDOW = (4, 120)  # 04:00 for 120 minutes, matching MaintenanceConfig's defaults.
# Boot-scoped, system-wide elapsed clock that counts suspend and ignores wall-clock steps:
# Darwin CLOCK_MONOTONIC keeps running while asleep; Linux needs CLOCK_BOOTTIME for that.
_HOLD_CLOCK_ID = time.CLOCK_MONOTONIC if sys.platform == "darwin" else time.CLOCK_BOOTTIME


class HoldWindowClosed(RuntimeError):
    """The quiet window leaves no time for a bounded hold."""

    message = "quiet window leaves no time for a bounded maintenance hold"


def hold_clock():
    """Seconds on the hold clock. Suspend counts toward a hold, so an overdue hold ends on wake."""
    return time.clock_gettime(_HOLD_CLOCK_ID)


def boot_id():
    """Stable per-boot identity; unlike kern.boottime it does not move when the wall clock is set."""
    if sys.platform == "darwin":
        value, size = ctypes.create_string_buffer(64), ctypes.c_size_t(64)
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.sysctlbyname(b"kern.bootsessionuuid", value, ctypes.byref(size), None, 0):
            raise OSError(ctypes.get_errno(), "kern.bootsessionuuid unreadable")
        return value.value.decode()
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def _seconds(name, default):
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        return default
    return value if math.isfinite(value) and value > 0 else default


def hold_budget(now=None, *, window=DEFAULT_WINDOW):
    """Seconds a hold may last.

    With a window clock the hold is window-bound: it starts only inside the quiet window and ends
    ``HOLD_WINDOW_RESERVE_S`` before it closes. Without one (manual FTS repair) only the cap applies.
    """
    budget = _seconds(HOLD_MAX_ENV, DEFAULT_HOLD_MAX_S)
    if now is None:
        return budget
    start_hour, minutes = window
    start = now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    if now < start:
        start -= dt.timedelta(days=1)
    end = start + dt.timedelta(minutes=minutes)
    if now >= end:
        return 0.0
    return min(budget, (end - now).total_seconds() - HOLD_WINDOW_RESERVE_S)


def _generation(directory):
    """The journaled hold generation, or None when it is absent or cannot be trusted."""
    try:
        entry = json.loads((directory / "maintenance.json").read_text())
        generation = [str(directory), entry["boot"], entry["owner"], entry["pid"], entry["pid_start"]]
        deadline = entry["deadline"]
    except (OSError, ValueError, KeyError, TypeError):
        return None, None  # No journal, or one written before holds carried an owner and a cap.
    _, boot, owner, pid, birth = generation
    if (
        not all(isinstance(v, str) and v for v in (boot, owner, birth))
        or type(pid) is not int
        or pid <= 1
        or type(deadline) not in (int, float)
        or not math.isfinite(deadline)
    ):
        return None, None
    return generation, deadline


def _lease_held(directory):
    try:
        with component_gate(directory):
            return False  # A shared lock succeeded: no exclusive hold remains.
    except BlockingIOError:
        return True


def enforce_hold_expiry(path, now=None):
    """Supervisor-side backstop: TERM, then KILL, the exact live owner of the current overdue hold.

    Escalation is keyed by the whole hold generation (directory, boot, owner token, pid, birth),
    persisted beside the journal so a supervisor restart keeps it, and recorded only once a TERM
    was delivered. The generation is revalidated after the identity read, immediately before each
    signal. What remains is the kernel gap between that last check and ``os.kill``; pidfd-style
    signalling is unavailable on Darwin.
    """
    directory = Path(path).parent
    generation, deadline = _generation(directory)
    if generation is None or generation[1] != boot_id() or generation[3] == os.getpid():
        return None  # A previous boot's owner cannot be alive; it is never a signal target.
    now = hold_clock() if now is None else now
    if now - deadline - _seconds(HOLD_GRACE_ENV, 60) < 0:
        return None
    escalation_path = directory / "maintenance-escalation.json"
    try:
        escalation = json.loads(escalation_path.read_text())
        term_at = escalation["term_at"] if escalation["generation"] == generation else None
    except (OSError, ValueError, KeyError, TypeError):
        term_at = None  # Missing, unreadable, or a completed/replaced generation's record.
    if term_at is not None and now - term_at < _seconds(HOLD_KILL_AFTER_ENV, 30):
        return None  # TERM already delivered; give the owner its unwind time.
    sig = signal.SIGTERM if term_at is None else signal.SIGKILL
    pid, birth = generation[3], generation[4]
    if start_time(pid) != birth:
        return None  # Dead owner or reused PID: never signal a stranger.
    if not _lease_held(directory) or _generation(directory)[0] != generation:
        return None  # The hold ended or was replaced while we looked.
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return None
    if sig == signal.SIGTERM:
        _atomic_write(escalation_path, dict(generation=generation, term_at=now))
    return sig


def recover_hold(path):
    """Called off the event loop; the free exclusive lease proves no inherited writer remains."""
    from .pause import DEFAULT_PAUSE_SENTINEL_PATH
    from .supervisor_adapters import SupervisorCommands

    client = SupervisorCommands(
        path,
        lambda args: subprocess.run(args, capture_output=True, text=True, timeout=1, check=False),
        pause_path=Path(os.environ.get("BRAINLAYER_PAUSE_SENTINEL", str(DEFAULT_PAUSE_SENTINEL_PATH))),
    )
    deadline, request = time.monotonic() + 3, client.request

    def bounded_request(*args, **kwargs):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("maintenance recovery deadline")
        kwargs["timeout"] = min(kwargs.get("timeout", 2), remaining)
        return request(*args, **kwargs)

    client.request = bounded_request
    hold = MaintenanceHold(client)
    try:
        with component_gate(hold.directory, exclusive=True):
            hold._restore()
    except BlockingIOError:
        enforce_hold_expiry(path)  # A live owner past its cap is ended; the next tick restores.
        raise


def control_client():
    from .supervisor_adapters import SupervisorCommands, adapt_commands

    client = adapt_commands(lambda args: subprocess.run(args, capture_output=True, text=True, timeout=20, check=False))
    if not isinstance(client, SupervisorCommands):
        raise RuntimeError("supervisor profile required for component quiescence")
    return client


def service_loaded(service):
    client = control_client()
    label = "com.etanhey.brainlayer-fleet-watchdog" if service == "fleet-watchdog" else f"com.brainlayer.{service}"
    if service == "index":
        return False  # Calendar step, never an archived resident plist.
    state = client.request()[client.aliases[label]]
    return state["configured"] and not state["paused"]


class MaintenanceHold:
    """Recover owned pauses; callers inherit fd/token and retain the lease until children are reaped."""

    def __init__(self, client=None, *, now_fn=None, window=DEFAULT_WINDOW):
        self.client = client if client is not None else control_client()
        self.directory, self.journal = self.client.path.parent, self.client.path.with_name("maintenance.json")
        self.fd, self.token, self.gate, self.nested = None, "", None, False
        self.now_fn, self.window = now_fn, window  # A window clock makes the hold window-bound.
        self.deadline = None  # On hold_clock(); every bound (VACUUM, calendar, watchdog) reads it.

    def _states(self):
        states = self.client.request()
        if any(not isinstance(v.get("pause_owner"), str) for v in states.values()):
            raise RuntimeError("supervisor lacks pause ownership")
        return states

    def _restore(self):
        if not self.journal.exists():
            return
        entry = json.loads(self.journal.read_text())
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("owner"), str)
            or not entry["owner"]
            or not isinstance(entry.get("components"), dict)
            or any(type(v) is not bool for v in entry["components"].values())
        ):
            raise ValueError("unknown maintenance hold journal")
        states = self._states()
        disabled = self.client._disabled(states, include_paused=False)
        labels = {name: label for label, name in self.client.aliases.items()}
        keep_down = {
            s.strip().removeprefix("com.brainlayer.")
            for s in os.environ.get("BRAINLAYER_MAINTENANCE_KEEP_DOWN", "").split(",")
        }
        for name, restore in entry["components"].items():
            if name not in states:
                continue
            state = states[name]
            if not restore or not state["configured"] or state.get("pause_owner") != entry["owner"]:
                continue
            label = labels.get(name, name)
            if (
                {name, label, label.removeprefix("com.brainlayer.")} & keep_down
                or label in disabled
                or name in disabled
                or "com.brainlayer.daemon" in disabled
            ):
                self.client.request("pause", name, owner="", timeout=state["exit_timeout"] + 5)
                continue
            self.client.request("resume", name, owner=entry["owner"], timeout=state["exit_timeout"] + 5)
        self.journal.unlink()
        (self.directory / "maintenance-escalation.json").unlink(missing_ok=True)
        self._sync_directory()

    def _sync_directory(self):
        fd = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def __enter__(self):
        inherited = os.environ.get("BRAINLAYER_MAINTENANCE_FD")
        # Calendar children must inherit this open lease and its journal token; never reopen the file.
        if inherited is not None:
            self.fd = int(inherited)
            self.token = os.environ.get("BRAINLAYER_MAINTENANCE_TOKEN", "")
            actual, expected = os.fstat(self.fd), (self.directory / "maintenance.lock").lstat()
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            entry = json.loads(self.journal.read_text())
            states = self._states()
            if (
                (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino)
                or not self.token
                or entry["owner"] != self.token
                or any(v["pid"] or v["configured"] and not v["paused"] for v in states.values())
            ):
                raise RuntimeError("unverified inherited maintenance hold")
            if entry.get("boot") != boot_id():
                raise RuntimeError("inherited maintenance hold belongs to another boot")
            self.nested = True
            self.deadline = entry.get("deadline")
            return self
        gate = component_gate(self.directory, exclusive=True)
        fd = gate.__enter__()
        self.gate, self.fd = gate, fd
        try:
            self._restore()
            budget = hold_budget(self.now_fn() if self.now_fn else None, window=self.window)
            if budget <= 0:
                raise HoldWindowClosed(HoldWindowClosed.message)
            birth, boot = start_time(os.getpid()), boot_id()
            if not birth or not boot:
                raise RuntimeError("maintenance hold owner identity unavailable")  # No killable owner, no hold.
            states = self._states()
            self.token = uuid.uuid4().hex
            self.deadline = hold_clock() + budget
            _atomic_write(
                self.journal,
                dict(
                    owner=self.token,
                    components={n: v["configured"] and not v["paused"] for n, v in states.items()},
                    pid=os.getpid(),
                    pid_start=birth,
                    boot=boot,
                    deadline=self.deadline,
                    expires_at=time.time() + budget,  # Wall time, for reporting only.
                ),
            )
            self._sync_directory()
            stopped = self.client.request(
                "pause", "*", owner=self.token, timeout=max((v["exit_timeout"] for v in states.values()), default=0) + 5
            )
            if any(v["pid"] or v["configured"] and not v["paused"] for v in stopped.values()):
                raise RuntimeError("component quiescence not acknowledged")
            return self
        except BaseException as exc:
            try:
                self.close()
            except Exception as cleanup:
                exc.add_note(f"hold restore failed: {cleanup}")
            raise

    def close(self):
        if self.nested or self.gate is None:
            return
        try:
            self._restore()
        finally:
            self.gate.__exit__(None, None, None)
            self.gate = None

    def __exit__(self, *_):
        self.close()
