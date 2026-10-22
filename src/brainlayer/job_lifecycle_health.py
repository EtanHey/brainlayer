from __future__ import annotations

import hashlib
import json
import plistlib
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .consolidation_plan import read_owned
from .jobs import _job_plists, _mapped_kegs, _pid, _runs_keg
from .jobs import installed_opt_path as _installed_opt_path

CommandRunner = Callable[[list[str]], Any]
EXIT_RE = re.compile(r"(?m)^\s*last exit code\s*=\s*(-?\d+)\s*$")
RUNS_RE = re.compile(r"(?m)^\s*runs\s*=\s*(\d+)\s*$")
MAX_HEAL_ATTEMPTS = 3
MIN_FAILURES = 2
SKIP_NAMES = {"enrich", "enrichment", "brainbar", "brainbar-daemon", "backup-daily", "jsonl-backup", "health-check"}


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


def _supervisor_keg(command: str, plist: dict, plist_dir: Path, current_keg: Path, opt_path: Path) -> Path | None:
    """Verify the migrated daemon's recorded venv, including macOS framework re-exec.

    The transaction publishes the receipt only after the replacement is ready.
    An opt symlink or a matching framework alone does not establish keg identity.
    """
    try:
        control = plist_dir.parent.parent / ".local/share/brainlayer/supervisor"
        manifest = control / "jobs.json"
        journal = json.loads(read_owned(control / "migration.json"))
        if journal["phase"] != "migrated":
            return None
        keg = Path(journal["keg"])
        # install.sh journals readlink's target verbatim, relative to opt's parent.
        keg = (keg if keg.is_absolute() else opt_path.parent / keg).resolve(strict=True)
        if keg.parent != current_keg.parent:
            return None
        raw = read_owned(plist_dir / "com.brainlayer.daemon.plist")
        if (
            plistlib.loads(raw) != plist
            or hashlib.sha256(raw).hexdigest() != journal["published_sha256"]["com.brainlayer.daemon.plist"]
            or hashlib.sha256(read_owned(manifest)).hexdigest() != journal["manifest_sha256"]
        ):
            return None
        pinned = plistlib.loads(bytes.fromhex(journal["daemon_pinned"]))
        args, pinned_args = plist["ProgramArguments"], pinned["ProgramArguments"]
        tail = ["-m", "brainlayer", "daemon", "--manifest", str(manifest)]
        if (
            pinned["Label"] != "com.brainlayer.daemon"
            or len(args) != 7
            or len(pinned_args) != 7
            or args[2:] != tail
            or pinned_args[2:] != tail
        ):
            return None
        python = Path(pinned_args[1])
        prefix = keg / "libexec/venv"
        if python != prefix / "bin/python" or not (prefix / "pyvenv.cfg").is_file():
            return None
        launcher = python.resolve(strict=True)
        if Path(args[1]).resolve(strict=True) != launcher:
            return None
        framework = launcher
        if (
            launcher.parent.name == "bin"
            and launcher.parents[2].name == "Versions"
            and launcher.parents[3].name == "Python.framework"
        ):
            # macOS's framework launcher execs Python.app; resolving its symlink
            # alone still names bin/python3.x rather than the running executable.
            framework = (launcher.parents[1] / "Resources/Python.app/Contents/MacOS/Python").resolve(strict=True)
        if shlex.split(command) != [str(framework), *tail]:
            return None
        if keg != current_keg and Path(args[1]) != python:
            # An opt flip leaves the old receipt behind. Both venvs can share
            # this executable, so the receipt cannot identify the running keg.
            # A false stale verdict lets a RunAtLoad healer restart its parent
            # before persisting backoff, repeating at every supervisor launch.
            return None
        return keg
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None


def _failure(
    output: str,
    current_keg: Path,
    opt_path: Path,
    command_runner: CommandRunner,
    *,
    supervisor_keg: Callable[[str], Path | None] | None = None,
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
            if _code(command) == 0 and supervisor_keg is not None:
                if (recorded := supervisor_keg(_output(command))) is not None:
                    if recorded == current_keg:
                        return None, last_exit
                    return f"stale keg for pid {pid}: {recorded}", last_exit
        if error or not mapped:
            return f"keg mapping unavailable for pid {pid}", last_exit
        return f"stale keg for pid {pid}: {', '.join(sorted(map(str, mapped)))}", last_exit
    if last_exit is not None and last_exit != 0:
        return f"crashloop, last exit code {last_exit}", last_exit
    return None, last_exit


NIGHTLY_LABELS = frozenset({"com.brainlayer.maintenance"})


def _escalation(label: str, episode: dict[str, Any]) -> str:
    exit_code = episode.get("last_exit_code", "unknown")
    count = episode.get("consecutive")
    if episode.get("interval") and type(count) is int:
        if label in NIGHTLY_LABELS:
            # Once a night is not a crashloop; the daemon-oriented reason would repeat the exit.
            return f"{label}: failed {count} nights in a row; last exit {exit_code}"
        failures = f"{count} failed runs"
    else:
        # Pre-fix persisted interval episodes carry no count; failed_heals implies the threshold.
        failures = f"{MAX_HEAL_ATTEMPTS} " + (
            "failed runs" if episode.get("interval") else "heal attempts failed to restore a healthy job"
        )
    return f"{label}: {episode.get('reason', 'job unavailable')}; last exit code {exit_code}; {failures}"


def _escalations(state: dict[str, dict[str, Any]]) -> list[str]:
    return [
        _escalation(label, episode)
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
    scan_errors: list[str] = []
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
        reason, last_exit = _failure(
            _output(printed),
            current_keg,
            opt_path,
            command_runner,
            supervisor_keg=(lambda command: _supervisor_keg(command, plist, plist_dir, current_keg, opt_path))
            if label == "com.brainlayer.daemon"
            else None,
        )
        if interval and _pid(_output(printed)) is not None:
            # The live invocation has no exit yet; last_exit belongs to runs - 1.
            # Consume that completion only once, independently of the live PID's keg.
            if prior_runs is None:
                state[label] = {**prior, "runs": max(0, runs - 1)}
                continue
            if runs <= prior_runs + 1:
                continue
            runs -= 1
            reason = None
        advanced = prior_runs is not None and runs > prior_runs
        if reason and reason.startswith("keg mapping unavailable"):
            if interval:
                continue  # A running interval script may not have loaded a keg library yet.
            # Preserve this uncertain episode without healing it; known jobs still progress.
            scan_errors.append(f"{label}: {reason}")
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
    return JobTick(state, actions, _escalations(state), "; ".join(scan_errors) or None)
