from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .jobs import _job_plists, _mapped_kegs, _pid, _runs_keg
from .jobs import installed_opt_path as _installed_opt_path

CommandRunner = Callable[[list[str]], Any]
EXIT_RE = re.compile(r"(?m)^\s*last exit code\s*=\s*(-?\d+)\s*$")
RUNS_RE = re.compile(r"(?m)^\s*runs\s*=\s*(\d+)\s*$")
MAX_HEAL_ATTEMPTS = 3
MIN_FAILURES = 2
SKIP_NAMES = {"enrichment", "brainbar", "brainbar-daemon", "backup-daily", "jsonl-backup", "health-check"}


@dataclass(frozen=True)
class JobTick:
    state: dict[str, dict[str, Any]]
    actions: list[str]
    escalations: list[str]
    scan_error: str | None = None


def installed_opt_path() -> Path | None:
    try:
        return _installed_opt_path()
    except ValueError:
        return None


def _code(result: Any) -> int:
    return int(getattr(result, "returncode", 127))


def _output(result: Any) -> str:
    return str(getattr(result, "stdout", "") or "")


def _failure(
    output: str, current_keg: Path, opt_path: Path, command_runner: CommandRunner
) -> tuple[str | None, int | None]:
    exit_match = EXIT_RE.search(output)
    last_exit = int(exit_match.group(1)) if exit_match else None
    if (pid := _pid(output)) is not None:
        mapped, error = _mapped_kegs(pid, command_runner)
        if mapped == {current_keg}:
            return None, last_exit
        if not mapped and error is None:
            command = command_runner(["ps", "-p", str(pid), "-o", "command="])
            if _code(command) == 0 and any(
                arg.startswith((f"{opt_path}/", f"{current_keg}/")) for arg in _output(command).split()
            ):
                return None, last_exit
        if error or not mapped:
            return f"keg mapping unavailable for pid {pid}", last_exit
        return f"stale keg for pid {pid}: {', '.join(sorted(map(str, mapped)))}", last_exit
    if last_exit is not None and last_exit != 0:
        return f"crashloop, last exit code {last_exit}", last_exit
    return None, last_exit


def _escalations(state: dict[str, dict[str, Any]]) -> list[str]:
    return [
        f"{label}: {episode.get('reason', 'job unavailable')}; last exit code "
        f"{episode.get('last_exit_code', 'unknown')}; {MAX_HEAL_ATTEMPTS} "
        f"{'failed runs' if episode.get('interval') else 'heal attempts failed to restore a healthy job'}"
        for label, episode in state.items()
        if isinstance(episode, dict) and episode.get("failed_heals")
    ]


def scan_job_lifecycle(
    plist_dir: Path,
    opt_path: Path,
    previous_state: dict[str, dict[str, Any]],
    *,
    now_epoch: int,
    command_runner: CommandRunner,
    uid: int,
    paused_labels: set[str] | frozenset[str] = frozenset(),
    heal: bool = True,
) -> JobTick:
    def fail(message: str) -> JobTick:
        return JobTick(previous_state, [], _escalations(previous_state), message)

    try:
        current_keg = opt_path.resolve(strict=True)
    except OSError as exc:
        return fail(f"current BrainLayer keg unavailable: {exc}")
    if not plist_dir.is_dir():
        return fail(f"BrainLayer LaunchAgents directory unavailable: {plist_dir}")
    for label, episode in previous_state.items():
        if not isinstance(episode, dict) or any(
            key in episode and (type(episode[key]) is not int or episode[key] < 0)
            for key in ("consecutive", "attempts", "next_retry_epoch", "runs")
        ):
            return fail(f"invalid persisted job state for {label}")
    try:
        jobs = _job_plists(plist_dir)
    except ValueError as exc:
        return fail(str(exc))
    disabled = command_runner(["launchctl", "print-disabled", f"gui/{uid}"])
    if _code(disabled) != 0:
        return fail("launchd disabled state unavailable; refusing to heal jobs")
    disabled_listing = _output(disabled)
    state = {label: episode for label, episode in previous_state.items() if isinstance(episode, dict)}
    actions: list[str] = []
    seen = set()
    for label, plist in jobs:
        seen.add(label)
        if (
            label.removeprefix("com.brainlayer.") in SKIP_NAMES
            or label in paused_labels
            or not _runs_keg(plist, opt_path, current_keg)
            or re.search(rf'["\']?{re.escape(label)}["\']?\s*=>\s*(?:true|disabled)', disabled_listing, re.I)
        ):
            state.pop(label, None)
            continue
        target = f"gui/{uid}/{label}"
        printed = command_runner(["launchctl", "print", target])
        if _code(printed) in {3, 36, 113}:  # The existing fleet watchdog owns unloaded jobs.
            if state.pop(label, None) is not None:
                actions.append(f"pruned:{label}")
            continue
        if _code(printed) != 0:
            return fail(f"cannot inspect loaded job {label}: launchctl exit {_code(printed)}")
        if (runs_match := RUNS_RE.search(_output(printed))) is None:
            return fail(f"launchd runs unavailable for {label}")
        runs = int(runs_match.group(1))
        interval = "StartInterval" in plist or "StartCalendarInterval" in plist
        prior = previous_state.get(label, {})
        prior_runs = prior.get("runs")
        advanced = prior_runs is not None and runs > prior_runs
        reason, last_exit = _failure(_output(printed), current_keg, opt_path, command_runner)
        if reason and reason.startswith("keg mapping unavailable"):
            if interval:
                continue  # A running interval script may not have loaded a keg library yet.
            return fail(f"{label}: {reason}")
        if interval and reason and reason.startswith("stale keg"):
            # Defer counting until launchd reports a completed, advanced run.
            continue
        if reason is None and advanced and last_exit not in (None, 0):
            reason = f"crashloop, last exit code {last_exit}"
        if reason is None and (last_exit in (None, 0) or (not interval and not advanced)):
            state[label] = {"runs": runs, "consecutive": 0, "attempts": 0, "failed_heals": False}
            continue
        if reason is None:
            reason = prior.get("reason", f"last exit code {last_exit}")
        stale = reason.startswith("stale keg")
        consecutive = prior.get("consecutive", 0) + (1 if advanced or stale else 0)
        attempts = prior.get("attempts", 0)
        failed_heals = attempts >= MAX_HEAL_ATTEMPTS
        next_retry = prior.get("next_retry_epoch", 0)
        if (
            not interval
            and heal
            and (stale or consecutive >= MIN_FAILURES)
            and attempts < MAX_HEAL_ATTEMPTS
            and now_epoch >= next_retry
        ):
            kicked = command_runner(["launchctl", "kickstart", "-k", target])
            attempts += 1
            next_retry = now_epoch + 300 * (2 ** (attempts - 1))
            actions.append(f"{'kickstart' if _code(kicked) == 0 else 'kickstart_failed'}:{label}")
        if interval and consecutive >= MAX_HEAL_ATTEMPTS:
            failed_heals = True
        state[label] = {
            "runs": runs,
            "interval": interval,
            "consecutive": consecutive,
            "attempts": attempts,
            "failed_heals": failed_heals,
            "next_retry_epoch": next_retry,
            "reason": reason,
            "last_exit_code": last_exit,
        }
    for label in previous_state.keys() - seen:
        state.pop(label, None)
        actions.append(f"pruned:{label}")
    return JobTick(state, actions, _escalations(state))
