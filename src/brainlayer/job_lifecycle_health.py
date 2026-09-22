"""Bounded recovery for loaded BrainLayer jobs with persistent badge escalation."""

from __future__ import annotations

import plistlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

CommandRunner = Callable[[list[str]], Any]
PID_RE = re.compile(r"(?m)^\s*pid\s*=\s*(\d+)\s*$")
EXIT_RE = re.compile(r"(?m)^\s*last exit code\s*=\s*(-?\d+)\s*$")
MAX_HEAL_ATTEMPTS = 3
MIN_FAILURES = 2
SKIP_LABELS = {
    "com.brainlayer.enrichment",
    "com.brainlayer.brainbar",
    "com.brainlayer.brainbar-daemon",
    "com.brainlayer.backup-daily",
    "com.brainlayer.jsonl-backup",
}


@dataclass(frozen=True)
class JobTick:
    state: dict[str, dict[str, Any]]
    actions: list[str]
    escalations: list[str]
    scan_error: str | None = None


def installed_opt_path() -> Path | None:
    prefix = Path(sys.prefix)
    for parent in (prefix, *prefix.parents):
        if parent.parent.name == "brainlayer" and parent.parent.parent.name == "Cellar":
            return parent.parent.parent.parent / "opt" / "brainlayer"
    return None


def _code(result: Any) -> int:
    return int(getattr(result, "returncode", 127))


def _output(result: Any) -> str:
    return str(getattr(result, "stdout", "") or "")


def _mapped_kegs(pid: int, command_runner: CommandRunner) -> set[Path] | None:
    output = command_runner(["lsof", "-p", str(pid), "-Fn"])
    if _code(output) != 0:
        return None
    kegs = set()
    for line in _output(output).splitlines():
        if not line.startswith("n"):
            continue
        parts = Path(line[1:]).parts
        for index in range(len(parts) - 2):
            if parts[index : index + 2] == ("Cellar", "brainlayer"):
                kegs.add(Path(*parts[: index + 3]))
                break
    return kegs


def _failure(
    output: str, current_keg: Path, opt_path: Path, command_runner: CommandRunner
) -> tuple[str | None, int | None]:
    exit_match = EXIT_RE.search(output)
    last_exit = int(exit_match.group(1)) if exit_match else None
    pid_match = PID_RE.search(output)
    if pid_match:
        pid = int(pid_match.group(1))
        mapped = _mapped_kegs(pid, command_runner)
        if mapped == {current_keg}:
            return None, last_exit
        if mapped == set():
            command = command_runner(["ps", "-p", str(pid), "-o", "command="])
            if _code(command) == 0 and _output(command).strip().startswith((f"{opt_path}/", f"{current_keg}/")):
                return None, last_exit
        if mapped is None or not mapped:
            return f"keg mapping unavailable for pid {pid}", last_exit
        return f"stale keg for pid {pid}: {', '.join(sorted(map(str, mapped)))}", last_exit
    if last_exit is not None and last_exit != 0:
        return f"crashloop, last exit code {last_exit}", last_exit
    return None, last_exit


def _escalations(state: dict[str, dict[str, Any]]) -> list[str]:
    messages = []
    for label, episode in state.items():
        if not isinstance(episode, dict) or not episode.get("failed_heals"):
            continue
        last_exit = episode.get("last_exit_code")
        exit_detail = f"last exit code {last_exit}" if last_exit is not None else "last exit code unknown"
        messages.append(
            f"{label}: {episode.get('reason', 'job unavailable')}; {exit_detail}; "
            f"{MAX_HEAL_ATTEMPTS} heal attempts failed to restore a healthy job"
        )
    return messages


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
    try:
        current_keg = opt_path.resolve(strict=True)
    except OSError as exc:
        return JobTick(previous_state, [], _escalations(previous_state), f"current BrainLayer keg unavailable: {exc}")
    if not plist_dir.is_dir():
        return JobTick(
            previous_state,
            [],
            _escalations(previous_state),
            f"BrainLayer LaunchAgents directory unavailable: {plist_dir}",
        )
    disabled = command_runner(["launchctl", "print-disabled", f"gui/{uid}"])
    if _code(disabled) != 0:
        return JobTick(
            previous_state,
            [],
            _escalations(previous_state),
            "launchd disabled state unavailable; refusing to heal jobs",
        )
    disabled_listing = _output(disabled)
    state = {label: episode for label, episode in previous_state.items() if isinstance(episode, dict)}
    actions: list[str] = []
    for path in sorted(plist_dir.glob("com.brainlayer.*.plist")):
        try:
            with path.open("rb") as handle:
                plist = plistlib.load(handle)
        except (OSError, ValueError, TypeError, plistlib.InvalidFileException) as exc:
            return JobTick(previous_state, [], _escalations(previous_state), f"cannot inspect {path}: {exc}")
        label = plist.get("Label") if isinstance(plist, dict) else None
        if label != path.stem or not isinstance(label, str):
            return JobTick(
                previous_state, [], _escalations(previous_state), f"invalid BrainLayer LaunchAgent label in {path}"
            )
        if label in SKIP_LABELS or label in paused_labels:
            state.pop(label, None)
            continue
        arguments = plist.get("ProgramArguments")
        if not isinstance(arguments, list) or not any(
            isinstance(argument, str) and argument.startswith((f"{opt_path}/", f"{current_keg}/"))
            for argument in arguments
        ):
            state.pop(label, None)
            continue
        if re.search(rf'["\']?{re.escape(label)}["\']?\s*=>\s*(?:true|disabled)', disabled_listing, re.I):
            state.pop(label, None)
            continue
        target = f"gui/{uid}/{label}"
        printed = command_runner(["launchctl", "print", target])
        if _code(printed) in {3, 36, 113}:  # The existing fleet watchdog owns unloaded jobs.
            continue
        if _code(printed) != 0:
            return JobTick(
                previous_state,
                [],
                _escalations(previous_state),
                f"cannot inspect loaded job {label}: launchctl exit {_code(printed)}",
            )
        reason, last_exit = _failure(_output(printed), current_keg, opt_path, command_runner)
        if reason is None:
            state.pop(label, None)
            continue  # Healthy measurement clears the previous episode.
        prior = previous_state.get(label, {})
        prior = prior if isinstance(prior, dict) else {}
        consecutive = max(0, int(prior.get("consecutive", 0))) + 1
        attempts = max(0, int(prior.get("attempts", 0)))
        failed_heals = attempts >= MAX_HEAL_ATTEMPTS
        next_retry = max(0, int(prior.get("next_retry_epoch", 0)))
        threshold = 1 if reason.startswith("stale keg") else MIN_FAILURES
        if heal and consecutive >= threshold and attempts < MAX_HEAL_ATTEMPTS and now_epoch >= next_retry:
            kicked = command_runner(["launchctl", "kickstart", "-k", target])
            attempts += 1
            backoff = 300 * (2 ** (attempts - 1))
            next_retry = now_epoch + backoff
            action = "kickstart" if _code(kicked) == 0 else "kickstart_failed"
            actions.append(f"{action}:{label}")
        state[label] = {
            "consecutive": consecutive,
            "attempts": attempts,
            "failed_heals": failed_heals,
            "next_retry_epoch": next_retry,
            "reason": reason,
            "last_exit_code": last_exit,
        }
    return JobTick(state, actions, _escalations(state))
