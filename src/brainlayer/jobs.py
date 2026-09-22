"""Restart loaded BrainLayer LaunchAgents after a Homebrew keg change."""

from __future__ import annotations

import plistlib
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]
PID_RE = re.compile(r"(?m)^\s*pid\s*=\s*(\d+)\s*$")
ENRICHMENT_LABEL = "com.brainlayer.enrichment"
BACKUP_LABELS = {"com.brainlayer.backup-daily", "com.brainlayer.jsonl-backup"}


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(args, 127, "", str(exc))


def installed_opt_path() -> Path:
    """Find opt/brainlayer from the interpreter's venv, without trusting PATH."""
    prefix = Path(sys.prefix)
    for parent in (prefix, *prefix.parents):
        if parent.parent.name == "brainlayer" and parent.parent.parent.name == "Cellar":
            return parent.parent.parent.parent / "opt" / "brainlayer"
    raise ValueError("jobs restart requires an installed Homebrew BrainLayer interpreter")


def _pid(output: str) -> int | None:
    match = PID_RE.search(output)
    return int(match.group(1)) if match else None


def _mapped_kegs(pid: int, command_runner: CommandRunner) -> tuple[set[Path], str | None]:
    response = command_runner(["lsof", "-p", str(pid), "-Fn"])
    if response.returncode != 0:
        return set(), f"lsof failed for pid {pid}: {response.stderr.strip() or response.returncode}"
    kegs: set[Path] = set()
    for line in response.stdout.splitlines():
        if not line.startswith("n"):
            continue
        path = Path(line[1:])
        parts = path.parts
        for index in range(len(parts) - 2):
            if parts[index : index + 2] == ("Cellar", "brainlayer"):
                kegs.add(Path(*parts[: index + 3]))
                break
    if not kegs:
        return set(), f"pid {pid} has no mapped BrainLayer keg in lsof output"
    return kegs, None


def _job_plists(directory: Path) -> list[tuple[str, dict[str, Any]]]:
    jobs = []
    for path in sorted(directory.glob("com.brainlayer.*.plist")):
        try:
            with path.open("rb") as handle:
                payload = plistlib.load(handle)
        except (OSError, ValueError, TypeError, plistlib.InvalidFileException) as exc:
            raise ValueError(f"cannot inspect BrainLayer plist {path}: {exc}") from exc
        label = payload.get("Label") if isinstance(payload, dict) else None
        if label != path.stem or not isinstance(label, str) or not label.startswith("com.brainlayer."):
            raise ValueError(f"invalid BrainLayer LaunchAgent label in {path}")
        jobs.append((label, payload))
    return jobs


def _runs_keg(plist: dict[str, Any], opt_path: Path, current_keg: Path) -> bool:
    arguments = plist.get("ProgramArguments")
    if not isinstance(arguments, list):
        return False
    prefixes = (f"{opt_path}/", f"{current_keg}/")
    return any(isinstance(argument, str) and argument.startswith(prefixes) for argument in arguments)


def restart_loaded_jobs(
    plist_dir: Path,
    opt_path: Path,
    *,
    command_runner: CommandRunner = _run,
    uid: int,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Kickstart loaded daemons, then verify every running job maps the current keg."""
    current_keg = opt_path.resolve(strict=True)
    if current_keg.parent.name != "brainlayer" or current_keg.parent.parent.name != "Cellar":
        raise ValueError(f"opt path does not resolve to a BrainLayer Cellar keg: {opt_path}")
    report: dict[str, Any] = {
        "current_keg": str(current_keg),
        "loaded": [],
        "restarted": [],
        "skipped": {},
        "stale": {},
        "errors": {},
    }
    try:
        if not plist_dir.is_dir():
            raise ValueError(f"BrainLayer LaunchAgents directory unavailable: {plist_dir}")
        jobs = _job_plists(plist_dir)
        if not jobs:
            raise ValueError(f"no BrainLayer LaunchAgent plists found in {plist_dir}")
    except ValueError as exc:
        report["errors"]["discovery"] = str(exc)
        report["ok"] = False
        return report
    measured_jobs = 0
    for label, plist in jobs:
        target = f"gui/{uid}/{label}"
        initial = command_runner(["launchctl", "print", target])
        if initial.returncode != 0:
            if initial.returncode == 113:
                report["skipped"][label] = "not loaded"
            else:
                report["errors"][label] = f"launchctl print failed: {initial.stderr.strip() or initial.returncode}"
            continue
        report["loaded"].append(label)
        if label == ENRICHMENT_LABEL:
            report["skipped"][label] = "enrichment excluded"
            continue
        if label in {"com.brainlayer.brainbar", "com.brainlayer.brainbar-daemon"}:
            report["skipped"][label] = "cask-owned BrainBar job"
            continue
        if label in BACKUP_LABELS:
            report["skipped"][label] = "backup job waits for next run"
            continue
        if not _runs_keg(plist, opt_path, current_keg):
            report["skipped"][label] = "not a BrainLayer keg command"
            continue
        measured_jobs += 1
        pid = _pid(initial.stdout)
        interval = "StartInterval" in plist or "StartCalendarInterval" in plist
        daemon = bool(plist.get("KeepAlive") or plist.get("RunAtLoad")) and not interval
        stale_inflight = False
        if interval and pid is not None:
            mapped, error = _mapped_kegs(pid, command_runner)
            stale_inflight = error is not None or mapped != {current_keg}
        if daemon or stale_inflight:
            kicked = command_runner(["launchctl", "kickstart", "-k", target])
            if kicked.returncode != 0:
                report["errors"][label] = f"kickstart failed: {kicked.stderr.strip() or kicked.returncode}"
            else:
                report["restarted"].append(label)
        elif interval:
            report["skipped"][label] = "interval job waits for next run"
        else:
            report["skipped"][label] = "not a resident daemon"
        for attempt in range(17):
            observed = command_runner(["launchctl", "print", target])
            if observed.returncode != 0:
                report["errors"][label] = (
                    f"verification launchctl print failed: {observed.stderr.strip() or observed.returncode}"
                )
                break
            if observed.returncode == 0 and (running_pid := _pid(observed.stdout)) is not None:
                mapped, error = _mapped_kegs(running_pid, command_runner)
                if error is None and mapped == {current_keg}:
                    break
                if attempt == 16:
                    report["stale"][label] = error or f"pid {running_pid} maps {sorted(map(str, mapped))}"
            else:
                if not daemon and not stale_inflight:
                    break  # an idle interval job will execute the new keg next time
                if attempt == 16:
                    report["errors"][label] = "job did not reach a running pid before verification deadline"
            if attempt < 16:
                sleep_fn(0.5)
    if measured_jobs == 0:
        report["errors"]["selection"] = "no loaded BrainLayer keg jobs were measured"
    report["ok"] = not report["errors"] and not report["stale"]
    return report
