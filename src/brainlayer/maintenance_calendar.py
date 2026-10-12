"""Inactive ordered calendar runner; each step owns a bounded, guarded process group."""

import asyncio
import json
import math
import os
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from .job_alerts import report
from .maintenance_control import MaintenanceHold
from .supervisor_guard import Custody, gated_spawn, launch_guard, start_time, stop_group

NAMES = tuple(
    "index,backup-daily,maintenance-nightly,jsonl-backup,p0-counter,wal-checkpoint,"
    "maintenance-weekly,decay,repair-fts".split(",")
)


@dataclass(frozen=True)
class Step:
    name: str
    argv: tuple[str, ...]
    timeout: float
    env: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.name not in NAMES or not self.argv or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("invalid calendar step")


def calendar_steps(python, cli, lib_dir, weekday):
    commands = (
        ((cli, "index"), 21600),
        ((str(lib_dir / "backup-daily.sh"),), 28800),
        ((python, "-m", "brainlayer.maintenance", "--light"), 1800),
        ((str(lib_dir / "jsonl-backup.sh"),), 7200),
        ((cli, "p0-counter"), 600),
        ((cli, "wal-checkpoint", "--mode", "TRUNCATE", "--retry-busy"), 1200),
        ((python, "-m", "brainlayer.maintenance", "--full"), 7200),
        ((cli, "decay", "--json"), 600),
        ((cli, "repair-fts"), 3600),
    )
    steps = [
        Step(name, (str(lib_dir / "brainlayer-env-run.sh"), *args), timeout, {"BRAINLAYER_PYTHON": str(python)})
        for name, (args, timeout) in zip(NAMES, commands)
        if weekday == 6 or name in NAMES[:6]
    ]
    steps[3].env["BRAINLAYER_JSONL_BACKUP_DRIVE_FOLDER"] = "Brain Drive/06_ARCHIVE/backups/claude-jsonl"
    return steps


def record(directory, **event):
    with (directory / "runs.jsonl").open("a") as handle:
        handle.write(json.dumps(dict(ts=time.time(), **event), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


async def pulse(guard, stop, interval):
    while not stop.is_set() and guard.poll() is None:
        os.write(guard.stdin.fileno(), b"H\n")
        await asyncio.sleep(interval)


@contextmanager
def calendar_custody(directory, timeout):
    with Custody(directory) as custody:
        custody.recover(min(1, timeout / 2))  # Reap dead-runner groups before their inherited lease can block us.
        yield custody


async def run_calendar(steps, directory, logs, *, guard_timeout=30, shutdown_timeout=15, install_signals=False):
    if not all(math.isfinite(v) and v > 0 for v in (guard_timeout, shutdown_timeout)):
        raise ValueError("invalid calendar timing")
    steps = tuple(steps)
    if len({s.name for s in steps}) != len(steps):
        raise ValueError("duplicate calendar step")
    stop, loop, handlers = asyncio.Event(), asyncio.get_running_loop(), {}
    aggregate, guard = 0, None
    with calendar_custody(directory, shutdown_timeout) as custody, MaintenanceHold() as hold:
        logs.mkdir(parents=True, exist_ok=True)
        guard = launch_guard(custody, guard_timeout, 0.2, shutdown_timeout, extra_fds=(hold.fd,))
        heartbeat = None
        try:
            record(directory, event="runner", guard=guard.pid)
            heartbeat = asyncio.create_task(pulse(guard, stop, min(0.1, guard_timeout / 4)))
            if install_signals:
                for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                    handlers[sig] = signal.getsignal(sig)
                    loop.add_signal_handler(sig, stop.set)
            for step in steps:
                if stop.is_set():
                    return 1
                record(directory, event="started", step=step.name)
                child, gate, code = None, None, 1
                env = (
                    os.environ
                    | step.env
                    | dict(
                        BRAINLAYER_MAINTENANCE_FD=str(hold.fd),
                        BRAINLAYER_MAINTENANCE_TOKEN=hold.token,
                        BRAINLAYER_SUPERVISOR_SOCKET=str(hold.client.path),
                        BRAINLAYER_CALENDAR_STEP="1",
                        BRAINLAYER_LAUNCHD_SERVICE=step.name,
                        XPC_SERVICE_NAME=f"com.brainlayer.{step.name}",
                    )
                )
                if step.name == "backup-daily":
                    env["BRAINLAYER_BACKUP_SUPERVISED_CHILD"] = "1"  # This runner owns its timeout/group.
                try:
                    disabled = hold.client._disabled(hold._states(), include_paused=False)
                    if f"com.brainlayer.{step.name}" in disabled or "com.brainlayer.maintenance" in disabled:
                        record(directory, event="finished", step=step.name, status="deferred", returncode=75)
                        aggregate = 1
                        continue
                    with (
                        (logs / f"{step.name}.out.log").open("ab") as out,
                        (logs / f"{step.name}.err.log").open("ab") as err,
                    ):
                        child, gate = gated_spawn(step.argv, env, out, err, extra_fds=(hold.fd,))
                    birth = start_time(child.pid)
                    if birth is None:
                        raise RuntimeError("calendar child birth unknown")
                    custody.record(child.pid, birth)
                    os.write(guard.stdin.fileno(), f"+{child.pid}\n".encode())
                    os.write(gate, b"R")
                    deadline = time.monotonic() + step.timeout
                    while child.poll() is None and not stop.is_set() and time.monotonic() < deadline:
                        if guard.poll() is not None:
                            raise RuntimeError("calendar guard exited")
                        await asyncio.sleep(min(0.1, guard_timeout / 4))
                    code = child.returncode if child.returncode is not None else (143 if stop.is_set() else 124)
                except (OSError, RuntimeError) as exc:
                    record(directory, event="step-error", step=step.name, error=type(exc).__name__)
                finally:
                    if gate is not None:
                        os.close(gate)
                    if child is not None:
                        deadline = time.monotonic() + shutdown_timeout
                        try:
                            os.write(guard.stdin.fileno(), b"Q\n" if stop.is_set() else b"H\n")
                        except BrokenPipeError:
                            stop.set()  # Still reap our child before releasing any lease.
                        await stop_group(child, 0.2, lambda: (deadline, shutdown_timeout / 2))
                        custody.forget(guard, child.pid)  # Only after group ESRCH, including descendants.
                record(
                    directory,
                    event="finished",
                    step=step.name,
                    status="ok" if code == 0 else "deferred" if code == 75 else "failed",
                    returncode=code,
                )
                aggregate |= code != 0
                key = {"maintenance-nightly": "maintenance-light", "maintenance-weekly": "maintenance-full"}.get(
                    step.name, step.name
                )
                try:
                    if code != 75:  # A deferral never erases the previous failure episode.
                        report(key, None if code == 0 else f"BrainLayer {step.name} exited {code}; see its log")
                except Exception as exc:
                    aggregate = 1
                    record(directory, event="alert-error", step=step.name, error=type(exc).__name__)
            return int(aggregate)
        finally:
            stop.set()
            if heartbeat:
                await asyncio.gather(heartbeat, return_exceptions=True)
            for sig, previous in handlers.items():
                loop.remove_signal_handler(sig)
                signal.signal(sig, previous)
            if guard.poll() is None:
                os.write(guard.stdin.fileno(), b"Q\n")
                guard.stdin.close()  # EOF retains orphan cleanup and its inherited writer lease.
                deadline = time.monotonic() + shutdown_timeout
                while guard.poll() is None and time.monotonic() < deadline:
                    await asyncio.sleep(0.02)
                if guard.poll() is None:
                    raise RuntimeError("calendar guard retains unresolved writer custody")
