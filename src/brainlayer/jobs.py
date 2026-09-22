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
        except (OSError, ValueError, TypeError, plistlib.InvalidFileException):
            continue
        label = payload.get("Label") if isinstance(payload, dict) else None
        if label == path.stem and isinstance(label, str) and label.startswith("com.brainlayer."):
            jobs.append((label, payload))
    return jobs


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
    for label, plist in _job_plists(plist_dir):
        target = f"gui/{uid}/{label}"
        initial = command_runner(["launchctl", "print", target])
        if initial.returncode != 0:
            report["skipped"][label] = "not loaded"
            continue
        report["loaded"].append(label)
        if label == ENRICHMENT_LABEL:
            report["skipped"][label] = "enrichment excluded"
            continue
        bundle_ids = plist.get("AssociatedBundleIdentifiers", [])
        if isinstance(bundle_ids, str):
            bundle_ids = [bundle_ids]
        if "com.brainlayer.brainbar" in bundle_ids:
            report["skipped"][label] = "cask-owned BrainBar job"
            continue
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
        for attempt in range(5):
            observed = command_runner(["launchctl", "print", target])
            if observed.returncode == 0 and (running_pid := _pid(observed.stdout)) is not None:
                mapped, error = _mapped_kegs(running_pid, command_runner)
                if error is None and mapped == {current_keg}:
                    break
                if attempt == 4:
                    report["stale"][label] = error or f"pid {running_pid} maps {sorted(map(str, mapped))}"
            else:
                if daemon and label in report["restarted"]:
                    report["errors"][label] = "resident job did not reach a running pid after kickstart"
                break  # an interval job may exit; its next run executes the new keg
            sleep_fn(0.2)
    report["ok"] = not report["errors"] and not report["stale"]
    return report
