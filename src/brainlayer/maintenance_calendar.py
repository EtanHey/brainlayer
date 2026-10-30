"""Inactive scheduled calendar runner with isolated groups and scoped writer holds."""

import asyncio
import datetime as dt
import fcntl
import json
import logging
import math
import os
import plistlib
import signal
import stat
import time
import uuid
from contextlib import ExitStack, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from xml.parsers.expat import ExpatError

from . import maintenance_control
from .backup_daily import (
    BACKUP_INVOCATION_ENV,
    DEFAULT_SNAPSHOT_TIMEOUT_SECONDS,
    SNAPSHOT_RECEIPT_RESERVE_SECONDS,
    default_snapshot_timeout_seconds,
    local_backup_date,
)
from .config import configured_brainlayer_env_value, get_user_env_path
from .drive_credentials import _atomic_write
from .index_budget import CALENDAR_INDEX_UNWIND_S, calendar_index_budget_s
from .job_alerts import active_alerts, report
from .maintenance_control import MaintenanceHold
from .supervisor_guard import (
    MAX_STALL_GRACE_S,
    Custody,
    configured_guard_timeout,
    gated_spawn,
    guard_lease,
    launch_guard,
    start_time,
    stop_group,
)
from .supervisor_jobs import TRANSCRIPT_RECOVERY_EXIT_TIMEOUT, TRANSCRIPT_RECOVERY_RUN_TIMEOUT

LOG = logging.getLogger(__name__)

NAMES = tuple(
    "index,backup-daily,maintenance-nightly,jsonl-backup,p0-counter,wal-checkpoint,maintenance-weekly,decay,repair-fts,backup-snapshot".split(
        ","
    )
)
KEYS = {
    "index": "index",
    "backup-daily": "backup-daily",
    "backup-snapshot": "backup-snapshot",
    "jsonl-backup": "jsonl-backup",
    "maintenance-nightly": "maintenance-light",
    "maintenance-weekly": "maintenance-full",
}


# Existing hourly recovery runs this step without a hold or any service pause.
# Other candidates remain scheduled until they have an explicit daytime contract.
DAYTIME_RECOVERY_STEPS = frozenset({"jsonl-backup"})
RECOVERY_DEFER_REASONS = {
    "index": "Indexing has a calendar runtime bound before the maintenance window",
    "backup-snapshot": "The verified snapshot has a calendar runtime bound before 06:00",
    "backup-daily": "Database backup may create a snapshot; daytime recovery is not admitted",
    **dict.fromkeys(
        ("maintenance-nightly", "maintenance-weekly", "repair-fts"),
        "It pauses database writes, so it only runs between 04:00 and 06:00",
    ),
    **dict.fromkeys(
        ("p0-counter", "wal-checkpoint", "decay"), "This database step has not been enabled for daytime retry"
    ),
}


def _recovery_failure_reason(row):
    reasons = {
        "quiet-window-missed": "The 04:00–06:00 maintenance window closed before this step could run",
        "maintenance-window": "Not enough time remained in the 04:00–06:00 maintenance window",
        "index-overran-window": "Indexing ran past the end of the maintenance window",
        "resident-writer-busy": "Another database writer was busy",
        "disabled": "This step was disabled or paused",
        "runner-vanished": "The scheduled runner stopped before finishing",
    }
    return reasons.get(row.get("reason")) or {
        "failed": "it stopped with an error",
        "stopped": "it stopped before finishing",
        "skipped": "it did not run",
        "deferred": "it was postponed",
    }.get(row.get("status"), "it did not finish in the scheduled night")


class RecoveryHistoryError(ValueError):
    pass


def recovery_refusal_reason(exc):
    if isinstance(exc, RecoveryHistoryError):
        return "Maintenance history is damaged or unreadable, so retry is off; tonight’s run will still happen."
    if isinstance(exc, ConnectionError):
        return "The maintenance supervisor is unavailable. Retry when it is running again."
    return "Retry availability could not be checked. Check the maintenance log before trying again."


def failed_recovery_selection(directory):
    try:
        return _failed_recovery_selection(directory)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RecoveryHistoryError() from exc


def _failed_recovery_selection(directory):
    """Replay only the newest scheduled night, then its same-date recovery records.

    A missing finish is incomplete. Manual runs and other nights cannot erase failures.
    Only damaged records in the latest night can affect its selection.
    The actual retry must re-check this snapshot under calendar custody.
    """
    date, names, outcomes, current = None, [], {}, False
    path = directory / "runs.jsonl"
    if path.exists():
        rows, newest = [], None
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                row = None
            if isinstance(row, dict) and row.get("event") == "runner" and row.get("kind", "nightly") == "nightly":
                # Older damage cannot change the newest nightly attempt. Only a header for the same
                # or a newer date opens a new window: a later --date rerun of an older night does not.
                stamp = row.get("scheduled_date")
                if not isinstance(stamp, str) or newest is None or stamp >= newest:
                    rows, newest = [], stamp if isinstance(stamp, str) else newest
            rows.append(row)
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("event"), str):
                raise RecoveryHistoryError()
            event = row["event"]
            if event == "runner" and row.get("kind", "nightly") == "nightly":
                candidate = dt.date.fromisoformat(row["scheduled_date"]).isoformat()
                if date is None or candidate >= date:
                    date, names, outcomes = candidate, row["steps"], {}
                    if not isinstance(names, list) or any(name not in NAMES for name in names):
                        raise ValueError("Unknown scheduled steps")
                    current = True
                else:
                    current = False
            elif event in {"runner", "transcript-recovery"}:
                current = row.get("kind") == "recovery" and row.get("scheduled_date") == date
            elif event == "finished" and current and row.get("step") in names:
                outcomes[row["step"]] = row
            elif event == "summary":
                current = False
    failures = [
        dict(
            step=name,
            reason=_recovery_failure_reason(outcomes.get(name, {})),
        )
        for name in names
        if not (outcomes.get(name, {}).get("status") == "ok" and outcomes[name].get("returncode") == 0)
    ]
    return dict(
        scheduled_date=date,
        failures=failures,
        steps=[item["step"] for item in failures if item["step"] in DAYTIME_RECOVERY_STEPS],
        deferred=[
            dict(step=item["step"], reason=RECOVERY_DEFER_REASONS[item["step"]], next_run="will run tonight at 04:00")
            for item in failures
            if item["step"] not in DAYTIME_RECOVERY_STEPS
        ],
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


@dataclass(frozen=True)
class CalendarPlan:
    scheduled_date: dt.date
    candidates: tuple[Step, ...]
    nightly_start: dt.time | None = None

    def select(self, last_weekly=None):
        sunday = self.scheduled_date - dt.timedelta(days=(self.scheduled_date.weekday() + 1) % 7)
        weekly = last_weekly is None or last_weekly < sunday
        # Weekly full follows the local proof; both nightly and weekly put upload last.
        names = ["index", "maintenance-nightly", "jsonl-backup", "p0-counter", "wal-checkpoint"]
        if weekly:
            names[1] = "maintenance-weekly"
            names.insert(1, "backup-snapshot")
            names += ["decay", "repair-fts", "backup-daily"]
        else:
            names.append("backup-daily")
        by_name = {s.name: s for s in self.candidates}
        selected = [by_name[n] for n in names]
        if weekly:
            selected[-1] = replace(selected[-1], env=selected[-1].env | {"BRAINLAYER_BACKUP_PHASE": "upload"})
        return selected


def _configured_calendar_start(lib_dir=None):
    # The installed package ships the launchd templates; source checkouts use the
    # same template. An explicitly supplied lib-dir schedule takes precedence.
    candidates = ([lib_dir / "com.brainlayer.maintenance.plist"] if lib_dir else []) + [
        Path(__file__).parent / "launchd/com.brainlayer.maintenance.plist",
        Path(__file__).parents[2] / "scripts/launchd/com.brainlayer.maintenance.plist",
    ]
    path = next((path for path in candidates if path.exists()), None)
    if path is None:
        raise ValueError("Nightly calendar start is unavailable")
    with path.open("rb") as stream:
        try:
            interval = plistlib.load(stream)["StartCalendarInterval"]
        except ExpatError as exc:  # Malformed XML is not a ValueError; callers handle ValueError.
            raise ValueError("Nightly calendar start is unreadable") from exc
    return dt.time(interval["Hour"], interval.get("Minute", 0))


def _retry_window(plan, steps, now, guard_timeout, shutdown_timeout):
    start = plan.nightly_start if isinstance(plan, CalendarPlan) else _configured_calendar_start()
    if start is None:
        raise ValueError("Nightly calendar start is unavailable")
    duration = sum(step.timeout for step in steps) + guard_timeout + MAX_STALL_GRACE_S + 2 * shutdown_timeout
    # Include tomorrow's reservation when a long configured bound crosses midnight.
    for offset in (0, 1):
        day = now + dt.timedelta(days=offset)
        nightly = day.replace(hour=start.hour, minute=start.minute, second=0, microsecond=0)
        pause_start = nightly - dt.timedelta(seconds=duration)
        pause_end = nightly.replace(hour=6, minute=0)
        if now < pause_end:
            break
    blocked = pause_start <= now < pause_end
    reason = f"Retries pause from {pause_start:%H:%M} until nightly maintenance finishes at {pause_end:%H:%M}."
    return (
        blocked,
        dict(
            retry_pause_start=pause_start.isoformat(timespec="seconds"),
            retry_pause_end=pause_end.isoformat(timespec="seconds"),
        ),
        reason,
    )


def _probe_calendar_custody(directory):
    """Try existing locks only: no mkdir, lock creation, group replay or writes."""
    # Never probe full-calendar: the nightly takes it outside its wait loop, so even a
    # microsecond hold there makes the 03:15 run record already-running and skip the night.
    # A running calendar also holds both locks below, which is what the retry admission checks.
    with ExitStack() as locks:
        for path in (directory, directory / "recovery-admission"):
            if not path.exists():
                continue
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise PermissionError("unsafe calendar directory")
            try:
                fd = os.open(path / "control.lock", os.O_RDONLY | os.O_NOFOLLOW)
            except FileNotFoundError:
                continue
            locks.callback(os.close, fd)
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
                or info.st_nlink != 1
            ):
                raise PermissionError("unsafe calendar lock")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
    return True


async def _failed_recovery_admission(plan, directory, now, guard_timeout, shutdown_timeout, reply):
    selection = await _blocking(failed_recovery_selection, directory)
    reply.update(selection)
    candidates = plan.candidates if isinstance(plan, CalendarPlan) else plan
    steps = tuple(step for step in candidates if step.name in selection["steps"])
    if {step.name for step in steps} != set(selection["steps"]):
        raise ValueError("Recovery plan is missing a selected step")
    blocked, window, reason = await _blocking(_retry_window, plan, steps, now, guard_timeout, shutdown_timeout)
    reply.update(window)
    if blocked:
        reply.update(status="refused", reason=reason)
    elif not selection["steps"]:
        reply.update(status="nothing-to-retry", reason="Everything that can run during the day has already finished.")
    else:
        client = maintenance_control.control_client()
        disabled = client._disabled(await _blocking(client.request), include_paused=True)
        if any(f"com.brainlayer.{name}" in disabled for name in (*selection["steps"], "maintenance")):
            reply.update(status="refused", reason="Maintenance or a retry step is disabled or paused")
        else:
            reply.update(status="ready", reason="Only failed daytime-safe steps will retry; no services pause")
    return steps


async def _failed_recovery_probe(plan, directory, now, guard_timeout, shutdown_timeout, reply):
    try:
        free = await _blocking(_probe_calendar_custody, directory)
        # All try-locks are released before reading history or contacting the supervisor.
        if not free:
            try:
                # Display-only while busy: the live run may be mid-write, so a torn line is not damage.
                reply.update(await _blocking(failed_recovery_selection, directory))
            except (OSError, ValueError, KeyError, TypeError):
                pass
            reply.update(
                status="refused", reason="Maintenance is running right now. Try again when it finishes.", steps=[]
            )
        else:
            await _failed_recovery_admission(plan, directory, now, guard_timeout, shutdown_timeout, reply)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        reply.update(status="refused", reason=recovery_refusal_reason(exc), steps=[])
        return 1


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
        for n, (args, timeout) in zip(NAMES[: len(commands)], commands)
    )
    steps[1].env["BRAINLAYER_BACKUP_PHASE"] = "all"
    steps += (
        replace(
            steps[1],
            name="backup-snapshot",
            timeout=DEFAULT_SNAPSHOT_TIMEOUT_SECONDS,
            env=steps[1].env | {"BRAINLAYER_BACKUP_PHASE": "snapshot"},
        ),
    )
    steps[3].env["BRAINLAYER_JSONL_BACKUP_DRIVE_FOLDER"] = "Brain Drive/06_ARCHIVE/backups/claude-jsonl"
    try:
        start = _configured_calendar_start(lib_dir)
    except (OSError, ValueError, KeyError, TypeError):
        # Only retries need the start time, and they refuse without it; the nightly must still run.
        start = None
    return CalendarPlan(scheduled_date, steps, start)


def record(directory, **event):
    with (directory / "runs.jsonl").open("a+b") as handle:
        end = handle.seek(0, 2)
        if end:
            handle.seek(-1, 2)
            if handle.read(1) != b"\n":
                handle.write(b"\n")  # Preserve torn bytes, separating them from the next valid record.
        handle.write((json.dumps({"ts": time.time(), **event}, sort_keys=True) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())


def _scan_record(directory, **event):
    try:
        record(directory, **event)
    except OSError as exc:
        LOG.warning("abandonment %s write failed: %s", event["event"], type(exc).__name__)


def _abandonment_alerts(event):
    names = event["incomplete_steps"] or [event["step"] or "maintenance-weekly"]
    detail = (
        f"incomplete {', '.join(event['incomplete_steps'])}"
        if event["incomplete_steps"]
        else "steps finished; terminal summary missing"
    )
    return {
        KEYS[name]: f"BrainLayer calendar {event['scheduled_date']} abandoned during "
        f"{event['step'] or 'startup'}; {detail}; retries at next scheduled run"
        for name in names
        if name in KEYS
    }


def _abandonment_notice(directory, key, reason):
    try:
        report(key, reason)
    except OSError as exc:
        LOG.warning("abandonment alert %s failed: %s", key, type(exc).__name__)
        _scan_record(directory, event="alert-error", key=key, error=type(exc).__name__)


def _retire_incomplete(directory, state, steps):
    event = state.get("incomplete_run")
    if not event:
        return
    selected_keys = {KEYS.get(name) for name in steps}
    alerts = active_alerts()
    for key, reason in _abandonment_alerts(event).items():
        if key not in selected_keys and alerts.get(key) == reason:
            _abandonment_notice(directory, key, None)
    state.pop("incomplete_run", None)
    _atomic_write(directory / "state.json", state)


def _recover_abandoned(directory, state, plan, today=None):
    """Best effort, under custody AFTER writer reap and BEFORE launch_guard (blocking I/O)."""
    latest, current, skipped = None, None, 0
    try:
        path = directory / "runs.jsonl"
        if not path.exists():
            return
        with path.open(errors="replace") as handle:
            for number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict) or not isinstance(row.get("event"), str):
                        raise ValueError("not an event object")
                    event = row["event"]
                    if event == "runner":
                        dt.date.fromisoformat(row["scheduled_date"])
                        if not isinstance(row["ts"], (float, int)) or not math.isfinite(row["ts"]):
                            raise ValueError("invalid timestamp")
                        latest = current = dict(runner=row, spawn={}, finished=set(), abandoned=None, terminal=None)
                    elif event in {"transcript-recovery", "transcript-recovery-skip"}:
                        current = None  # Hourly work is not a nightly maintenance outcome.
                    elif event == "abandoned" and latest and row["runner_started_at"] == latest["runner"]["ts"]:
                        latest["abandoned"] = row
                    elif event == "summary" and latest:
                        if row.get("runner_started_at", current and current["runner"]["ts"]) == latest["runner"]["ts"]:
                            latest["terminal"] = row
                        current = None
                    elif current and event == "spawn":
                        if row["step"] not in NAMES or not isinstance(row["pid"], int):
                            raise ValueError("invalid spawn")
                        current["spawn"] = row
                    elif current and event == "finished":
                        current["finished"].add(row["step"])
                except (ValueError, KeyError, TypeError) as exc:
                    skipped += 1
                    LOG.warning(
                        "abandonment scan skipped line=%d count=%d error=%s", number, skipped, type(exc).__name__
                    )
        if skipped:
            LOG.warning("abandonment scan skipped=%d", skipped)
            _scan_record(directory, event="abandon-scan-skipped", count=skipped)
        if not latest:
            return
        runner, spawn = latest["runner"], latest["spawn"]
        kind = runner.get("kind", "nightly")
        terminal = latest["terminal"]
        if terminal:
            if terminal.get("complete") and state.get("incomplete_run"):
                _retire_incomplete(directory, state, runner["steps"])
            elif state.get("incomplete_run"):
                for key, reason in _abandonment_alerts(state["incomplete_run"]).items():
                    _abandonment_notice(directory, key, reason)
            return
        date = dt.date.fromisoformat(runner["scheduled_date"])
        today = today or dt.datetime.now().astimezone().date()
        # Ad-hoc custody cleanup is separate from scheduled nightly outcome history.
        if kind != "manual" and (today - date).days not in (0, 1):
            return
        event = latest["abandoned"]
        if event is None:
            identities = [
                (runner.get("guard"), runner.get("guard_start_time")),
                (spawn.get("pid"), spawn.get("start_time")),
            ]
            dead = [
                pid
                for pid, birth in identities
                if isinstance(pid, int)
                and pid > 1
                and ((actual := start_time(pid)) is None or (birth is not None and actual != birth))
            ]
            if not dead:
                return
            names = runner.get("steps")
            if names is None:
                marker = dt.date.fromisoformat(state["last_weekly"]) if state.get("last_weekly") else None
                names = (
                    [s.name for s in CalendarPlan(date, plan.candidates).select(marker)]
                    if isinstance(plan, CalendarPlan)
                    else [s.name for s in plan]
                )
            detected = time.time()
            event = dict(
                event="abandoned",
                kind="nightly",
                scheduled_date=date.isoformat(),
                step=spawn.get("step"),
                pid=spawn.get("pid", runner.get("guard")),
                runner_started_at=runner["ts"],
                detected_at=detected,
                ts=detected,
                # Under custody a new guard is gone; orphan covers legacy PID reuse.
                reason="runner-vanished" if runner.get("guard") in dead else "step-orphaned",
                incomplete_steps=[name for name in names if name not in latest["finished"]],
                retry="next-scheduled-run",
            )
            record(directory, **event)
        state["incomplete_run"] = event | dict(failed=1)
        _atomic_write(directory / "state.json", state)
        for key, reason in _abandonment_alerts(event).items():
            _abandonment_notice(directory, key, reason)
        record(
            directory,
            event="summary",
            kind="nightly",
            scheduled_date=event["scheduled_date"],
            runner_started_at=runner["ts"],
            failed=1,
            deferred=0,
            abandoned=True,
            complete=False,
        )
    except Exception as exc:
        LOG.warning("abandonment scan failed: %s", type(exc).__name__)
        _scan_record(directory, event="abandon-scan-error", error=type(exc).__name__, skipped=skipped)


def backup_deadline(step):
    if step.name == "backup-snapshot":
        return step.timeout
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


def _service_name(step):
    return "backup-daily" if step.name == "backup-snapshot" else step.name


def _bounded_snapshot_step(step, now, shutdown_timeout):
    path = Path(os.environ.get("BRAINLAYER_ENV_FILE", get_user_env_path()))
    configured = configured_brainlayer_env_value("BRAINLAYER_BACKUP_SNAPSHOT_TIMEOUT_SECONDS", path)
    explicit = configured if configured is not None else os.environ.get("BRAINLAYER_BACKUP_SNAPSHOT_TIMEOUT_SECONDS")
    # An explicit override wins outright; otherwise the budget is sized from today's database.
    requested = float(explicit if explicit is not None else min(step.timeout, default_snapshot_timeout_seconds()))
    if not math.isfinite(requested) or requested <= 0:
        raise ValueError("invalid snapshot timeout")
    end = now.replace(hour=6, minute=0, second=0, microsecond=0)
    seconds = int(min(requested, (end - now).total_seconds() - shutdown_timeout - 1))
    # The backup's own deadline fires first, so its timeout receipt is written before this kill.
    internal = seconds - min(SNAPSHOT_RECEIPT_RESERVE_SECONDS, max(1, seconds // 4))
    if internal < 1:
        return None
    return replace(
        step,
        timeout=seconds,
        env=step.env
        | {
            "BRAINLAYER_BACKUP_TIMEOUT_SECONDS": str(internal),
            "BRAINLAYER_BACKUP_CALENDAR_SNAPSHOT_BUDGET_SECONDS": str(internal),
        },
    )


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


_calendar_operations = ContextVar("calendar_operations", default=None)


@dataclass
class CalendarOperation:
    label: str
    started: float
    reported: bool = False


def _open_operation_diagnostics(directory):
    try:
        from .supervisor_diagnostics import DiagnosticSink
    except ImportError:
        return None  # Optional until the independently reviewed diagnostics PR lands.
    try:
        return DiagnosticSink.open(directory / "operation-events.jsonl")
    except (OSError, RuntimeError):
        logging.warning("calendar operation diagnostic sink unavailable; enforcement remains active")
        return None


async def pulse(guard, stop, interval, operations=None, operation_timeout=120, diagnostics=None):
    operations = operations if operations is not None else {}
    while not stop.is_set() and guard.poll() is None:
        now = time.monotonic()
        overdue = [op for op in operations.values() if now - op.started >= operation_timeout]
        for op in overdue:
            if not op.reported:
                event = dict(
                    event="heartbeat-suppressed",
                    reason="operation-overdue",
                    operation=op.label,
                    started_monotonic=op.started,
                    deadline_monotonic=op.started + operation_timeout,
                    age_s=now - op.started,
                    budget_s=operation_timeout,
                    parent_pid=os.getpid(),
                    parent_start_time=start_time(os.getpid()) or "unknown",
                    guard_pid=guard.pid,
                )
                if diagnostics:
                    try:
                        diagnostics.emit(**event)
                    except Exception:
                        pass
                else:
                    logging.error("calendar heartbeat-suppressed: %s", event)
                op.reported = True
        if not overdue:
            os.write(guard.stdin.fileno(), b"H\n")
        await asyncio.sleep(interval)


def _index_deferral_reason(path, offset):
    """Only this invocation's bounded stdout tail can explain an index exit 75."""
    try:
        with path.open("rb") as handle:
            end = handle.seek(0, 2)
            start = max(offset, end - 64 * 1024)
            handle.seek(start)
            data = handle.read(64 * 1024)
        if start > offset:
            data = data.partition(b"\n")[2]  # Never accept a truncated line.
        lines = data.decode("utf-8", errors="replace").splitlines()
        last = next((line for line in reversed(lines) if line.startswith("INDEX_DEFERRED ")), "")
        fields = last.split()
        token = fields[1] if len(fields) > 1 else ""
        return {
            "resident_writer_busy": "resident-writer-busy",
            "maintenance_window=04:00-06:00": "maintenance-window",
        }.get(token)
    except OSError:
        return None


async def _execute_step(
    step, env, *, directory, logs, custody, guard, stop, guard_timeout, shutdown_timeout, deadline=None
):
    child, gate = None, None
    try:
        budget, held, clock = await _blocking(backup_deadline, step), False, time.monotonic

        def prepare():
            nonlocal child, gate
            with (logs / f"{step.name}.out.log").open("ab") as out, (logs / f"{step.name}.err.log").open("ab") as err:
                fds = tuple(
                    int(env[k]) for k in ("BRAINLAYER_CALENDAR_GUARD_FD", "BRAINLAYER_MAINTENANCE_FD") if k in env
                )
                child, gate = gated_spawn(step.argv, env, out, err, extra_fds=fds)
            birth = start_time(child.pid)
            if birth is None:
                raise RuntimeError("calendar child birth unknown")
            custody.record(child.pid, birth)
            identity = (
                {"backup_invocation_id": env[BACKUP_INVOCATION_ENV]}
                if step.name == "backup-daily" and env.get(BACKUP_INVOCATION_ENV)
                else {}
            )
            record(directory, event="spawn", step=step.name, pid=child.pid, start_time=birth, **identity)

        await _blocking(prepare)
        os.write(guard.stdin.fileno(), f"+{child.pid}\n".encode())
        os.write(gate, b"R")
        if deadline is not None:
            # A held step runs on the hold clock, the same one VACUUM and the watchdog read, so a
            # wall-clock step can neither drop nor reorder the bound.
            clock = maintenance_control.hold_clock
            remaining = max(0.0, deadline - clock())
            held = remaining <= budget
            budget = min(budget, remaining)
        end = clock() + budget
        while child.poll() is None and not stop.is_set() and clock() < end:
            if guard.poll() is not None:
                raise RuntimeError("calendar guard exited")
            await asyncio.sleep(min(0.1, guard_timeout / 4))
        if held and child.poll() is None and not stop.is_set():
            await _blocking(record, directory, event="hold-expired", step=step.name, pid=child.pid)
        return child.returncode if child.returncode is not None else (143 if stop.is_set() else 124)
    finally:
        if gate is not None:
            os.close(gate)
        if child is not None:
            deadline = time.monotonic() + shutdown_timeout
            try:
                os.write(guard.stdin.fileno(), b"H\n")  # Guard shutdown follows the finish worker.
            except BrokenPipeError:
                stop.set()
            await _drain(asyncio.create_task(stop_group(child, 0.2, lambda: (deadline, shutdown_timeout / 2))))
            await _blocking(custody.forget, guard, child.pid)


async def _window(directory, step, now_fn, sleep_fn, stop):
    if step.name not in {"maintenance-nightly", "maintenance-weekly"}:
        return True
    if now_fn().hour < 4:
        await _blocking(record, directory, event="wait-start", step=step.name)
        while now_fn().hour < 4 and not stop.is_set():
            await sleep_fn(0.1)  # Responsive to TERM; tests advance their private clock.
        await _blocking(record, directory, event="wait-finished", step=step.name)
    return not stop.is_set() and 4 <= now_fn().hour < 6


def finish(
    directory,
    state,
    step,
    code,
    date,
    defer_after,
    missed=False,
    stopped=False,
    *,
    reason=None,
    replaces=None,
    backup_invocation_id=None,
    preserve_abandonment_alerts=False,
):
    status = "stopped" if stopped else "deferred" if code == 75 else "ok" if code == 0 else "failed"
    detail = {
        key: value
        for key, value in dict(reason=reason, replaces=replaces, backup_invocation_id=backup_invocation_id).items()
        if value is not None
    }
    record(directory, event="finished", step=step.name, status=status, returncode=code, **detail)
    key = KEYS.get(step.name)
    notice = None
    if key and not stopped:
        if code == 75:
            prior = state["deferrals"].get(key, {})
            gap = (date - dt.date.fromisoformat(prior.get("date", date.isoformat()))).days
            count = prior.get("count", 0) + 1 if gap == 1 else max(1, prior.get("count", 0)) if gap == 0 else 1
            state["deferrals"][key] = dict(date=date.isoformat(), count=count)
            if missed or count >= defer_after:
                notice = (key, f"BrainLayer {step.name}: quiet window missed" if missed else f"Deferred {count} nights")
        else:
            state["deferrals"].pop(key, None)
            notice = (key, None if code == 0 else f"BrainLayer {step.name} exited {code}; see its log")
    _atomic_write(directory / "state.json", state)
    protected = state.get("incomplete_run")
    if notice is not None and not (preserve_abandonment_alerts and protected and key in _abandonment_alerts(protected)):
        try:
            report(*notice)
        except OSError as exc:
            # An alert-file failure does not undo a completed step or imply lost custody.
            record(directory, event="alert-error", step=step.name, error=type(exc).__name__)
    return status


async def _drain(worker):
    """Retain custody until executor work or group cleanup survives repeated cancellation."""
    cancelled = None
    while True:
        try:
            result = await asyncio.shield(worker)
            break
        except asyncio.CancelledError as exc:
            cancelled = exc
    if cancelled is not None:
        raise cancelled
    return result


async def _blocking(operation, *args, **kwargs):
    label = getattr(operation, "__qualname__", type(operation).__name__)
    began = time.monotonic()
    worker = asyncio.get_running_loop().run_in_executor(None, partial(operation, *args, **kwargs))
    operations = _calendar_operations.get()
    if operations is not None:
        operations[worker] = CalendarOperation(label, began)
    try:
        return await _drain(worker)
    finally:
        if operations is not None:
            operations.pop(worker, None)


async def _finish_step(*args, **kwargs):
    return await _blocking(finish, *args, **kwargs)


HELD_STEPS = frozenset({"maintenance-nightly", "maintenance-weekly", "repair-fts"})


@asynccontextmanager
async def step_hold(client, guard, env, step, now_fn=None):
    if step.name not in HELD_STEPS:
        yield env
        return
    hold = MaintenanceHold(client, now_fn=now_fn)  # Calendar holds are always window-bound.
    try:
        await _blocking(hold.__enter__)
        await _blocking(guard_lease, guard, hold.fd)
        yield env | dict(
            BRAINLAYER_MAINTENANCE_FD=str(hold.fd),
            BRAINLAYER_MAINTENANCE_TOKEN=hold.token,
            BRAINLAYER_MAINTENANCE_HOLD_DEADLINE=repr(hold.deadline),
        )
        await _blocking(guard_lease, guard, None)  # Only after the child group is confirmed gone.
    finally:
        await _blocking(hold.__exit__, None, None, None)


def _transcripts_fresh(now):
    """Only a recent verified upload or verified idle receipt suppresses recovery."""
    env_file = Path(os.environ.get("BRAINLAYER_ENV_FILE", get_user_env_path()))
    key = "BRAINLAYER_JSONL_BACKUP_LOG_PATH"
    configured = configured_brainlayer_env_value(key, env_file)
    path = Path(configured or os.environ.get(key) or Path.home() / ".local/share/brainlayer/logs/jsonl-backup.log")
    try:
        with path.expanduser().open("rb") as stream:
            end = stream.seek(0, 2)
            stream.seek(max(0, end - 64 * 1024))
            row = json.loads(stream.read().splitlines()[-1])
        if not isinstance(row, dict) or row.get("verified") is not True:
            return False
        if row.get("status") != "no-op" and not (row.get("status") == "uploaded" and row.get("uploaded") is True):
            return False
        attempted = dt.datetime.fromisoformat(row["attempted_at"])
        if attempted.tzinfo is None:
            return False
        age = (now.astimezone(dt.UTC) - attempted).total_seconds()
        return 0 <= age < 24 * 60 * 60
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return False


async def _admit_calendar(steps, directory, date, transcript_recovery, guard_timeout, shutdown_timeout, sleep_fn):
    """Reserve a full invocation before retrying custody, never retry already-started work."""
    leases = ExitStack()
    full = isinstance(steps, CalendarPlan) and not transcript_recovery
    try:
        if full or transcript_recovery:
            await _blocking(directory.mkdir, mode=0o700, parents=True, exist_ok=True)
        if full:
            # Only full invocations touch this singleton: a recovery's short admission
            # probe must never make a scheduled full invocation look like a duplicate.
            await _blocking(leases.enter_context, Custody(directory / "full-calendar", reset_previous_boot=False))
        deadline = maintenance_control.hold_clock()
        if full:
            recovery = [step for step in steps.candidates if step.name == "jsonl-backup"]
            if len(recovery) != 1:
                raise ValueError("full calendar admission requires one bounded transcript step")
            # Cover the actual step bound, the supervisor backstop and its exit grace,
            # plus guard startup/expiry (its host-stall grace ends at an absolute
            # missed-deadline + MAX_STALL_GRACE_S) and group/guard shutdown. The
            # suspend-aware elapsed clock cannot be extended by a wall-clock change or sleep.
            deadline += (
                max(recovery[0].timeout, TRANSCRIPT_RECOVERY_RUN_TIMEOUT)
                + TRANSCRIPT_RECOVERY_EXIT_TIMEOUT
                + guard_timeout
                + MAX_STALL_GRACE_S
                + 2 * shutdown_timeout
                + 30
            )
        waiting, admission = False, None
        while True:
            if full and maintenance_control.hold_clock() >= deadline:
                await _blocking(record, directory, event="calendar-admission-timeout", scheduled_date=date.isoformat())
                leases.close()
                return None, 1
            try:
                if (full or transcript_recovery) and admission is None:
                    admission = await _blocking(
                        leases.enter_context, Custody(directory / "recovery-admission", reset_previous_boot=False)
                    )
                custody = await _blocking(leases.enter_context, Custody(directory))
                if transcript_recovery:
                    await _blocking(admission.__exit__, None, None, None)
                return leases, custody
            except BlockingIOError:
                if not full:
                    raise
                if not waiting:
                    await _blocking(record, directory, event="calendar-admission-wait", scheduled_date=date.isoformat())
                    waiting = True
                await sleep_fn(min(1, max(0, deadline - maintenance_control.hold_clock())))
    except BlockingIOError:
        leases.close()
        await _blocking(record, directory, event="already-running", scheduled_date=date.isoformat())
        return None, 0
    except BaseException:
        leases.close()
        raise


async def run_calendar(
    steps,
    directory,
    logs,
    *,
    guard_timeout=None,
    shutdown_timeout=15,
    operation_timeout=120,
    install_signals=False,
    scheduled_date=None,
    now_fn=lambda: dt.datetime.now().astimezone(),
    sleep_fn=asyncio.sleep,
    defer_after=3,
    transcript_recovery=False,
    failed_recovery=False,
    recovery_status=False,
    recovery_result=None,
):
    if failed_recovery and transcript_recovery:
        raise ValueError("Choose only one recovery mode")
    reply = recovery_result if recovery_result is not None else {}
    if guard_timeout is None:
        guard_timeout = await _blocking(configured_guard_timeout)
    if (
        not all(math.isfinite(v) and v > 0 for v in (guard_timeout, shutdown_timeout, operation_timeout))
        or defer_after < 1
    ):
        raise ValueError("invalid calendar timing")
    date = steps.scheduled_date if isinstance(steps, CalendarPlan) else scheduled_date or local_backup_date(now_fn())
    if failed_recovery and recovery_status:
        return await _failed_recovery_probe(steps, directory, now_fn(), guard_timeout, shutdown_timeout, reply)
    operations = {}
    token = _calendar_operations.set(operations)
    try:
        leases, custody = await _admit_calendar(
            steps, directory, date, transcript_recovery or failed_recovery, guard_timeout, shutdown_timeout, sleep_fn
        )
    except BaseException:
        _calendar_operations.reset(token)
        raise
    if leases is None:
        if failed_recovery:
            try:
                # Display-only snapshot while busy; never use it to admit or launch work.
                reply.update(await _blocking(failed_recovery_selection, directory))
            except (OSError, ValueError, KeyError, TypeError):
                pass
            reply.update(
                status="refused", reason="Maintenance is running right now. Try again when it finishes.", steps=[]
            )
        _calendar_operations.reset(token)
        return custody
    stop, heartbeat_stop, handlers, results = asyncio.Event(), asyncio.Event(), {}, {}
    guard, heartbeat, diagnostics = None, None, None
    active_step, runner_failed = None, 0
    planned, plan_complete = isinstance(steps, CalendarPlan), False
    kind = "recovery" if transcript_recovery or failed_recovery else "nightly" if planned else "manual"
    manual_failed = 0
    runner_started_at = None
    loop = asyncio.get_running_loop()
    try:
        if failed_recovery:
            plan = steps
            steps = await _failed_recovery_admission(plan, directory, now_fn(), guard_timeout, shutdown_timeout, reply)
            if reply["status"] != "ready":
                return 0
            date = dt.date.fromisoformat(reply["scheduled_date"])
        await _blocking(custody.recover, min(1, shutdown_timeout / 2))  # Reap stale writers before any journal scan.
        state_path = directory / "state.json"
        state = await _blocking(
            lambda: json.loads(state_path.read_text()) if state_path.exists() else dict(deferrals={})
        )
        if not isinstance(state, dict) or not isinstance(state.get("deferrals"), dict):
            raise ValueError("unknown calendar state")
        await _blocking(_recover_abandoned, directory, state, steps, today=now_fn().date())
        manual_failed = int(kind == "manual" and bool(state.get("incomplete_run")))
        if transcript_recovery:
            # The freshness decision shares custody with the entire nightly plan. A competing
            # runner cannot begin an upload between this check and our transcript step.
            if await _blocking(_transcripts_fresh, now_fn()):
                await _blocking(record, directory, event="transcript-recovery-skip", reason="fresh-verified-receipt")
                return 0
            candidates = steps.candidates if planned else steps
            steps = tuple(step for step in candidates if step.name == "jsonl-backup")
            if len(steps) != 1:
                raise ValueError("transcript recovery requires one transcript step")
        elif planned and not failed_recovery:
            marker = dt.date.fromisoformat(state["last_weekly"]) if state.get("last_weekly") else None
            steps = steps.select(marker)
        steps = tuple(steps)
        weekly_substitution = planned and any(step.name == "maintenance-weekly" for step in steps)
        index_overran_window = False
        client = maintenance_control.control_client()
        await _blocking(logs.mkdir, parents=True, exist_ok=True)

        def start_guard():
            nonlocal guard
            guard = launch_guard(custody, guard_timeout, 0.2, shutdown_timeout, lease_control=True)

        diagnostics = await _blocking(_open_operation_diagnostics, directory)
        await _blocking(start_guard)
        heartbeat = asyncio.create_task(
            pulse(guard, heartbeat_stop, min(0.1, guard_timeout / 4), operations, operation_timeout, diagnostics)
        )
        runner_started_at = time.time()
        await _blocking(
            record,
            directory,
            event="transcript-recovery" if transcript_recovery or failed_recovery else "runner",
            kind=kind,
            guard=guard.pid,
            guard_start_time=start_time(guard.pid),
            steps=[step.name for step in steps],
            ts=runner_started_at,
            scheduled_date=date.isoformat(),
        )
        if failed_recovery:
            reply.update(status="accepted", reason="Transcript backup retried successfully")
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
        env.pop(BACKUP_INVOCATION_ENV, None)
        for step in steps:
            active_step = step.name
            if stop.is_set():
                await _blocking(record, directory, event="finished", step=step.name, status="skipped", returncode=None)
                continue
            code, missed, reason, step_failed = 1, False, None, False
            invocation = uuid.uuid4().hex if step.name == "backup-daily" else None
            try:
                if failed_recovery:
                    blocked, window, window_reason = await _blocking(
                        _retry_window, plan, steps, now_fn(), guard_timeout, shutdown_timeout
                    )
                    reply.update(window)
                    if blocked:
                        reply.update(status="refused", reason=window_reason)
                        break
                bounded = step
                if step.name == "index":
                    bounded = await _blocking(_bounded_index_step, step, now_fn(), shutdown_timeout)
                elif step.name == "backup-snapshot":
                    bounded = await _blocking(_bounded_snapshot_step, step, now_fn(), shutdown_timeout)
                if bounded is None:
                    code, reason = 75, "maintenance-window"
                else:
                    step = bounded
                    disabled = client._disabled(await _blocking(client.request), include_paused=failed_recovery)
                    if f"com.brainlayer.{_service_name(step)}" in disabled or "com.brainlayer.maintenance" in disabled:
                        code, reason = 75, "disabled"
                    elif not await _window(directory, step, now_fn, sleep_fn, stop):
                        code, missed = (143, False) if stop.is_set() else (75, True)
                        if missed:
                            reason = "index-overran-window" if index_overran_window else "quiet-window-missed"
                    elif step.name in HELD_STEPS and maintenance_control.hold_budget(now_fn()) <= 0:
                        code, reason = 75, "maintenance-window"  # Outside the window, or no bounded hold fits.
                    else:
                        child_env = (
                            env
                            | step.env
                            | dict(
                                BRAINLAYER_LAUNCHD_SERVICE=_service_name(step),
                                XPC_SERVICE_NAME=f"com.brainlayer.{_service_name(step)}",
                            )
                            | ({BACKUP_INVOCATION_ENV: invocation} if invocation else {})
                        )
                        try:
                            async with step_hold(client, guard, child_env, step, now_fn) as held_env:
                                if stop.is_set():
                                    code = 143
                                else:
                                    started = now_fn()
                                    index_log = logs / "index.out.log"
                                    index_offset = await _blocking(
                                        lambda: (
                                            index_log.stat().st_size
                                            if step.name == "index" and index_log.exists()
                                            else 0
                                        )
                                    )
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
                                        deadline=float(held_env["BRAINLAYER_MAINTENANCE_HOLD_DEADLINE"])
                                        if "BRAINLAYER_MAINTENANCE_HOLD_DEADLINE" in held_env
                                        else None,
                                    )
                                    ended = now_fn()
                                    if step.name == "index":
                                        if code == 75:
                                            reason = await _blocking(_index_deferral_reason, index_log, index_offset)
                                        # Only attribute the miss to indexing when its measured interval
                                        # crossed this day's window end. Backup delays get a generic reason.
                                        window_end = started.replace(hour=6, minute=0, second=0, microsecond=0)
                                        index_overran_window = started < window_end <= ended
                        except maintenance_control.HoldWindowClosed:
                            # The window closed between the check and the grant: defer, keep the plan going.
                            code, reason = 75, "maintenance-window"
            except (OSError, RuntimeError, ValueError) as exc:
                await _blocking(record, directory, event="step-error", step=step.name, error=type(exc).__name__)
                code = 1
                step_failed = True
                stop.set()  # Custody/protocol failures abort; ordinary child failures remain isolated.
            # Receipt fsync, alert locking and osascript must not starve the guard heartbeat.
            stopped = stop.is_set() and not step_failed
            if code not in (0, 75) and not stopped:
                results[step.name] = "failed"  # Retain the child failure even if completion is cancelled.
            results[step.name] = await _finish_step(
                directory,
                state,
                step,
                code,
                date,
                defer_after,
                missed,
                stopped,
                reason=reason,
                replaces="maintenance-nightly" if weekly_substitution and step.name == "maintenance-weekly" else None,
                backup_invocation_id=invocation,
                preserve_abandonment_alerts=transcript_recovery or failed_recovery,
            )
        active_step = None
        plan_complete = bool(steps) and not stop.is_set() and len(results) == len(steps)
        if all(results.get(n) == "ok" for n in ("maintenance-weekly", "decay", "repair-fts")):
            state["last_weekly"] = (date - dt.timedelta(days=(date.weekday() + 1) % 7)).isoformat()
            await _blocking(_atomic_write, directory / "state.json", state)
        failed = sum(v == "failed" for v in results.values())
        if failed_recovery:
            reply.update(results=results)
            if reply.get("status") != "refused" and (
                failed or stop.is_set() or any(value != "ok" for value in results.values())
            ):
                reply.update(status="failed", reason="Retry did not finish successfully; see the step results and logs")
        return int(
            failed > 0 or stop.is_set() or manual_failed or (failed_recovery and reply.get("status") == "failed")
        )
    except Exception as exc:
        if active_step is not None:
            results[active_step] = "failed"
        else:
            runner_failed = 1
        await _blocking(record, directory, event="runner-error", step=active_step, error=type(exc).__name__)
        if failed_recovery:
            reply.update(status="failed" if runner_started_at else "refused", reason=recovery_refusal_reason(exc))
            return 1
        raise
    finally:
        try:
            stop.set()
            heartbeat_stop.set()  # All finish workers have exited; bounded guard shutdown can begin.
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
        except Exception as exc:
            runner_failed += 1
            await _blocking(record, directory, event="runner-error", phase="cleanup", error=type(exc).__name__)
            raise
        finally:
            try:
                if not failed_recovery or runner_started_at is not None:
                    await _blocking(
                        record,
                        directory,
                        event="summary",
                        kind=kind,
                        complete=plan_complete and not runner_failed,
                        scheduled_date=date.isoformat(),
                        runner_started_at=runner_started_at,
                        failed=max(
                            int(kind == "nightly" and not plan_complete),
                            sum(v == "failed" for v in results.values()) + runner_failed + manual_failed,
                        ),
                        deferred=sum(v == "deferred" for v in results.values()),
                    )
                if kind == "nightly" and plan_complete and not runner_failed:
                    try:
                        await _blocking(_retire_incomplete, directory, state, [step.name for step in steps])
                    except OSError as exc:
                        LOG.warning("abandonment state retirement failed: %s", type(exc).__name__)
                        await _blocking(_scan_record, directory, event="abandon-scan-error", error=type(exc).__name__)
            finally:
                try:
                    if diagnostics:
                        await _blocking(diagnostics.close)
                finally:
                    try:
                        leases.close()
                    finally:
                        _calendar_operations.reset(token)
