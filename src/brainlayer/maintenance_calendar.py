"""Inactive scheduled calendar runner with isolated groups and scoped writer holds."""

import asyncio
import datetime as dt
import json
import math
import os
import signal
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import maintenance_control
from .config import configured_brainlayer_env_value, get_user_env_path
from .drive_credentials import _atomic_write
from .index_budget import CALENDAR_INDEX_UNWIND_S, calendar_index_budget_s
from .job_alerts import report
from .maintenance_control import MaintenanceHold
from .supervisor_guard import Custody, gated_spawn, guard_lease, launch_guard, start_time, stop_group

NAMES = tuple(
    "index,backup-daily,maintenance-nightly,jsonl-backup,p0-counter,wal-checkpoint,maintenance-weekly,decay,repair-fts".split(
        ","
    )
)
KEYS = {
    "backup-daily": "backup-daily",
    "jsonl-backup": "jsonl-backup",
    "maintenance-nightly": "maintenance-light",
    "maintenance-weekly": "maintenance-full",
}


@dataclass(frozen=True)
class Step:
    name: str
    argv: tuple[str, ...]
    timeout: float
    env: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.name not in NAMES or not self.argv or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("invalid calendar step")


@dataclass(frozen=True)
class CalendarPlan:
    scheduled_date: dt.date
    candidates: tuple[Step, ...]

    def select(self, last_weekly=None):
        sunday = self.scheduled_date - dt.timedelta(days=(self.scheduled_date.weekday() + 1) % 7)
        weekly = last_weekly is None or last_weekly < sunday
        names = list(NAMES[:6])
        if weekly:
            names[2] = "maintenance-weekly"
            names += ["decay", "repair-fts"]
        by_name = {s.name: s for s in self.candidates}
        return [by_name[n] for n in names]


def calendar_steps(python, cli, lib_dir, scheduled_date):
    commands = (
        ((cli, "index"), 1800 + CALENDAR_INDEX_UNWIND_S),
        ((str(lib_dir / "backup-daily.sh"),), 28890),
        ((python, "-m", "brainlayer.maintenance", "--light"), 1800),
        ((str(lib_dir / "jsonl-backup.sh"),), 7200),
        ((cli, "p0-counter"), 600),
        ((cli, "wal-checkpoint", "--mode", "TRUNCATE", "--retry-busy"), 1200),
        ((python, "-m", "brainlayer.maintenance", "--full"), 7300),
        ((cli, "decay", "--json"), 600),
        ((cli, "repair-fts"), 3600),
    )
    steps = tuple(
        Step(n, (str(lib_dir / "brainlayer-env-run.sh"), *args), timeout, {"BRAINLAYER_PYTHON": str(python)})
        for n, (args, timeout) in zip(NAMES, commands)
    )
    steps[3].env["BRAINLAYER_JSONL_BACKUP_DRIVE_FOLDER"] = "Brain Drive/06_ARCHIVE/backups/claude-jsonl"
    return CalendarPlan(scheduled_date, steps)


def record(directory, **event):
    with (directory / "runs.jsonl").open("a") as handle:
        handle.write(json.dumps(dict(ts=time.time(), **event), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def backup_deadline(step):
    if step.name != "backup-daily":
        return step.timeout
    path = Path(os.environ.get("BRAINLAYER_ENV_FILE", get_user_env_path()))
    configured = configured_brainlayer_env_value("BRAINLAYER_BACKUP_TIMEOUT_SECONDS", path)
    seconds = int(
        configured if configured is not None else os.environ.get("BRAINLAYER_BACKUP_TIMEOUT_SECONDS", "28800")
    )
    if seconds < 1:
        raise ValueError("invalid backup timeout")
    return max(step.timeout, seconds + 90)


def _bounded_index_step(step, now, shutdown_timeout):
    """Use existing group custody, reserving cleanup before the quiet window."""
    path = Path(os.environ.get("BRAINLAYER_ENV_FILE", get_user_env_path()))
    configured = configured_brainlayer_env_value("BRAINLAYER_INDEX_MAX_RUNTIME_S", path)
    try:
        requested = float(
            configured if configured is not None else os.environ.get("BRAINLAYER_INDEX_MAX_RUNTIME_S", 1800)
        )
    except (TypeError, ValueError):
        requested = 1800
    if not math.isfinite(requested) or requested <= 0:
        requested = 1800
    budget = calendar_index_budget_s(
        min(requested, step.timeout), now, reserve_s=max(60, shutdown_timeout + CALENDAR_INDEX_UNWIND_S + 15)
    )
    if budget <= 0:
        return None
    return replace(
        step,
        timeout=min(step.timeout, budget + CALENDAR_INDEX_UNWIND_S),
        env=step.env | {"BRAINLAYER_INDEX_CALENDAR_BUDGET_S": str(budget)},
    )


async def pulse(guard, stop, interval):
    while not stop.is_set() and guard.poll() is None:
        os.write(guard.stdin.fileno(), b"H\n")
        await asyncio.sleep(interval)


async def _execute_step(step, env, *, directory, logs, custody, guard, stop, guard_timeout, shutdown_timeout):
    child, gate = None, None
    try:
        with (logs / f"{step.name}.out.log").open("ab") as out, (logs / f"{step.name}.err.log").open("ab") as err:
            fds = tuple(int(env[k]) for k in ("BRAINLAYER_CALENDAR_GUARD_FD", "BRAINLAYER_MAINTENANCE_FD") if k in env)
            child, gate = gated_spawn(step.argv, env, out, err, extra_fds=fds)
        birth = start_time(child.pid)
        if birth is None:
            raise RuntimeError("calendar child birth unknown")
        custody.record(child.pid, birth)
        os.write(guard.stdin.fileno(), f"+{child.pid}\n".encode())
        record(directory, event="spawn", step=step.name, pid=child.pid)
        os.write(gate, b"R")
        deadline = time.monotonic() + backup_deadline(step)
        while child.poll() is None and not stop.is_set() and time.monotonic() < deadline:
            if guard.poll() is not None:
                raise RuntimeError("calendar guard exited")
            await asyncio.sleep(min(0.1, guard_timeout / 4))
        return child.returncode if child.returncode is not None else (143 if stop.is_set() else 124)
    finally:
        if gate is not None:
            os.close(gate)
        if child is not None:
            deadline = time.monotonic() + shutdown_timeout
            try:
                os.write(guard.stdin.fileno(), b"Q\n" if stop.is_set() else b"H\n")
            except BrokenPipeError:
                stop.set()
            await stop_group(child, 0.2, lambda: (deadline, shutdown_timeout / 2))
            custody.forget(guard, child.pid)


async def _window(directory, step, now_fn, sleep_fn, stop):
    if step.name not in {"maintenance-nightly", "maintenance-weekly"}:
        return True
    if now_fn().hour < 4:
        record(directory, event="wait-start", step=step.name)
        while now_fn().hour < 4 and not stop.is_set():
            await sleep_fn(0.1)  # Responsive to TERM; tests advance their private clock.
        record(directory, event="wait-finished", step=step.name)
    return not stop.is_set() and 4 <= now_fn().hour < 6


def finish(directory, state, step, code, date, defer_after, missed=False, stopped=False, *, reason=None, replaces=None):
    status = "stopped" if stopped else "deferred" if code == 75 else "ok" if code == 0 else "failed"
    detail = {key: value for key, value in dict(reason=reason, replaces=replaces).items() if value is not None}
    record(directory, event="finished", step=step.name, status=status, returncode=code, **detail)
    key = KEYS.get(step.name)
    if key and not stopped:
        if code == 75:
            prior = state["deferrals"].get(key, {})
            gap = (date - dt.date.fromisoformat(prior.get("date", date.isoformat()))).days
            count = prior.get("count", 0) + 1 if gap == 1 else max(1, prior.get("count", 0)) if gap == 0 else 1
            state["deferrals"][key] = dict(date=date.isoformat(), count=count)
            if missed or count >= defer_after:
                report(key, f"BrainLayer {step.name}: quiet window missed" if missed else f"Deferred {count} nights")
        else:
            state["deferrals"].pop(key, None)
            report(key, None if code == 0 else f"BrainLayer {step.name} exited {code}; see its log")
    _atomic_write(directory / "state.json", state)
    return status


@asynccontextmanager
async def step_hold(client, guard, env, step):
    if step.name not in {"maintenance-nightly", "maintenance-weekly", "repair-fts"}:
        yield env
        return
    hold = MaintenanceHold(client)
    await asyncio.to_thread(hold.__enter__)  # Keep watchdog pulses alive during acknowledged pause.
    try:
        guard_lease(guard, hold.fd)
        yield env | dict(BRAINLAYER_MAINTENANCE_FD=str(hold.fd), BRAINLAYER_MAINTENANCE_TOKEN=hold.token)
        guard_lease(guard, None)  # Only after the child group is confirmed gone.
    finally:
        await asyncio.to_thread(hold.__exit__, None, None, None)


async def run_calendar(
    steps,
    directory,
    logs,
    *,
    guard_timeout=30,
    shutdown_timeout=15,
    install_signals=False,
    scheduled_date=None,
    now_fn=lambda: dt.datetime.now().astimezone(),
    sleep_fn=asyncio.sleep,
    defer_after=3,
):
    if not all(math.isfinite(v) and v > 0 for v in (guard_timeout, shutdown_timeout)) or defer_after < 1:
        raise ValueError("invalid calendar timing")
    date = steps.scheduled_date if isinstance(steps, CalendarPlan) else scheduled_date or now_fn().date()
    custody = Custody(directory)
    try:
        custody.__enter__()
    except BlockingIOError:
        record(directory, event="already-running", scheduled_date=date.isoformat())
        return 0
    stop, handlers, results = asyncio.Event(), {}, {}
    guard, heartbeat = None, None
    loop = asyncio.get_running_loop()
    try:
        custody.recover(min(1, shutdown_timeout / 2))  # Reap stale writers before reacquiring their lease.
        state_path = directory / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else dict(deferrals={})
        if not isinstance(state, dict) or not isinstance(state.get("deferrals"), dict):
            raise ValueError("unknown calendar state")
        planned = isinstance(steps, CalendarPlan)
        if planned:
            marker = dt.date.fromisoformat(state["last_weekly"]) if state.get("last_weekly") else None
            steps = steps.select(marker)
        steps = tuple(steps)
        weekly_substitution = planned and any(step.name == "maintenance-weekly" for step in steps)
        index_overran_window = False
        client = maintenance_control.control_client()
        logs.mkdir(parents=True, exist_ok=True)
        guard = launch_guard(custody, guard_timeout, 0.2, shutdown_timeout, lease_control=True)
        heartbeat = asyncio.create_task(pulse(guard, stop, min(0.1, guard_timeout / 4)))
        record(directory, event="runner", guard=guard.pid, scheduled_date=date.isoformat())
        if install_signals:
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                handlers[sig] = signal.getsignal(sig)
                loop.add_signal_handler(sig, stop.set)
        env = os.environ | dict(
            BRAINLAYER_CALENDAR_STEP="1",
            BRAINLAYER_CALENDAR_GUARD_FD=str(guard.stdin.fileno()),
            BRAINLAYER_SUPERVISOR_SOCKET=str(client.path),
        )
        env.pop("BRAINLAYER_BACKUP_SUPERVISED_CHILD", None)
        for step in steps:
            if stop.is_set():
                record(directory, event="finished", step=step.name, status="skipped", returncode=None)
                continue
            code, missed, reason = 1, False, None
            try:
                if step.name == "index":
                    bounded = _bounded_index_step(step, now_fn(), shutdown_timeout)
                    if bounded is None:
                        results[step.name] = finish(directory, state, step, 75, date, defer_after)
                        continue
                    step = bounded
                disabled = client._disabled(client.request(), include_paused=False)
                if f"com.brainlayer.{step.name}" in disabled or "com.brainlayer.maintenance" in disabled:
                    code, reason = 75, "disabled"
                elif not await _window(directory, step, now_fn, sleep_fn, stop):
                    code, missed = (143, False) if stop.is_set() else (75, True)
                    if missed:
                        reason = "index-overran-window" if index_overran_window else "quiet-window-missed"
                else:
                    child_env = (
                        env
                        | step.env
                        | dict(BRAINLAYER_LAUNCHD_SERVICE=step.name, XPC_SERVICE_NAME=f"com.brainlayer.{step.name}")
                    )
                    async with step_hold(client, guard, child_env, step) as held_env:
                        if stop.is_set():
                            code = 143
                        else:
                            started = now_fn()
                            code = await _execute_step(
                                step,
                                held_env,
                                directory=directory,
                                logs=logs,
                                custody=custody,
                                guard=guard,
                                stop=stop,
                                guard_timeout=guard_timeout,
                                shutdown_timeout=shutdown_timeout,
                            )
                            ended = now_fn()
                            if step.name == "index":
                                # Only attribute the miss to indexing when its measured interval
                                # crossed this day's window end. Backup delays get a generic reason.
                                window_end = started.replace(hour=6, minute=0, second=0, microsecond=0)
                                index_overran_window = started < window_end <= ended
            except (OSError, RuntimeError, ValueError) as exc:
                record(directory, event="step-error", step=step.name, error=type(exc).__name__)
                stop.set()  # Custody/protocol failures abort; ordinary child failures remain isolated.
            results[step.name] = finish(
                directory,
                state,
                step,
                code,
                date,
                defer_after,
                missed,
                stop.is_set(),
                reason=reason,
                replaces="maintenance-nightly" if weekly_substitution and step.name == "maintenance-weekly" else None,
            )
        if all(results.get(n) == "ok" for n in ("maintenance-weekly", "decay", "repair-fts")):
            state["last_weekly"] = (date - dt.timedelta(days=(date.weekday() + 1) % 7)).isoformat()
            _atomic_write(directory / "state.json", state)
        failed = sum(v == "failed" for v in results.values())
        record(directory, event="summary", failed=failed, deferred=sum(v == "deferred" for v in results.values()))
        return int(failed > 0 or stop.is_set())
    finally:
        stop.set()
        if heartbeat:
            await asyncio.gather(heartbeat, return_exceptions=True)
        try:
            if guard:
                guard.lease_channel.close()  # Parent EOF is independent of pipes inherited by children.
            if guard and guard.poll() is None:
                os.write(guard.stdin.fileno(), b"Q\n")
                guard.stdin.close()
                deadline = time.monotonic() + shutdown_timeout
                while guard.poll() is None and time.monotonic() < deadline:
                    await asyncio.sleep(0.02)
                if guard.poll() is None:
                    raise RuntimeError("calendar guard retains unresolved custody")
        finally:
            for sig, previous in handlers.items():
                loop.remove_signal_handler(sig)
                signal.signal(sig, previous)
            custody.__exit__(None, None, None)
