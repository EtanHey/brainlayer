"""One bounded supervisor pass; durable per-build suppression survives daemon restarts."""

import argparse
import fcntl
import json
import os
import plistlib
import subprocess
import sys
import time
from pathlib import Path

from .brainbar_identity import JOBS, app_identity, loaded, running_bundle_matches
from .deploy_drift import _atomic_write_json
from .job_alerts import report
from .pipeline.secret_scrub import scrub_secrets


def process_table():
    result = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,comm=,args="], capture_output=True, text=True, check=True, timeout=5
    )
    return result.stdout


def brew_processes():
    # Include brew's Ruby entrypoint and Homebrew-owned child executables, even
    # after reparenting. Unknown/unreadable process tables defer activation.
    rows = process_table().splitlines()
    matches = []
    for row in rows:
        fields = row.split(None, 3)
        if len(fields) != 4 or not fields[0].isascii() or not fields[0].isdecimal():
            raise ValueError("unreadable process table")
        tokens = fields[3].split()
        if Path(fields[2]).name == "brew" or any(
            Path(token).name in {"brew", "brew.rb"} or "/Homebrew/" in token or "/homebrew/Library/Homebrew/" in token
            for token in tokens
        ):
            # argv is detection-only: Homebrew curl headers can carry tokens.
            executable = scrub_secrets(Path(fields[2]).name).text[:128]
            matches.append(f"{fields[0]} {executable}")
    return matches


def brew_active():
    return bool(brew_processes())


class AutoActivator:
    def __init__(self, agents: Path, state: Path):
        self.agents, self.state = agents, state

    def tick(self):
        self.state.parent.mkdir(parents=True, exist_ok=True)
        with self.state.with_suffix(".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            self._tick()

    def _defer_for_brew(self):
        # Separate from the per-build attempt: brew may have removed both plists.
        path = self.state.with_suffix(".brew.json")
        prior = json.loads(path.read_text()) if path.exists() else {}
        processes = brew_processes()
        absent = bool(processes) and all(loaded(label).returncode == 113 for label, _ in JOBS)
        if not absent:
            if prior:
                path.unlink()
                if prior.get("alerted"):
                    report("brainbar-activation-brew", None)
        else:
            now = time.time()
            since = prior.get("since", now)
            if since > now:
                since, prior = now, {}
            alerted = prior.get("alerted", False)
            if now - since > 600 and not alerted:
                names = processes[:5]
                if len(processes) > 5:
                    names.append(f"+{len(processes) - 5} more")
                report(
                    "brainbar-activation-brew",
                    "BrainBar activation deferred by Homebrew for >10 minutes: " + "; ".join(names),
                )
                alerted = True
            _atomic_write_json(path, {"since": since, "alerted": alerted})
        return bool(processes)

    def _tick(self):
        if self._defer_for_brew():
            return
        apps = set()
        for label, _ in JOBS:
            path = self.agents / f"{label}.plist"
            if not path.is_file():
                return
            with path.open("rb") as stream:
                plist = plistlib.load(stream)
            apps.add(Path(plist["ProgramArguments"][0]).parents[2])
        if len(apps) != 1:
            raise ValueError("cask plists use different bundles")
        app = apps.pop()
        if not (app / "Contents/Info.plist").is_file():
            return
        identity = {"app": str(app), **app_identity(app)}
        prior = json.loads(self.state.read_text()) if self.state.exists() else {}
        if prior.get("identity") == identity and prior.get("stage") == "ready":
            return
        states = [loaded(label).returncode for label, _ in JOBS]
        if all(code == 0 for code in states) and running_bundle_matches(app):
            _atomic_write_json(self.state, {"identity": identity, "stage": "ready"})
            report("brainbar-activation", None)
            return
        if prior.get("identity") == identity:
            if prior.get("stage") == "attempted":
                reason = "BrainBar activation was interrupted; run brainlayer brainbar activate"
                report("brainbar-activation", reason)
                _atomic_write_json(self.state, {"identity": identity, "stage": "failed", "reason": reason})
            return
        if any(code == 0 for code in states) or self._defer_for_brew():
            return
        # Write before launching: crash/restart is never permission to repeat.
        _atomic_write_json(self.state, {"identity": identity, "stage": "attempted"})
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "brainlayer",
                    "brainbar",
                    "activate",
                    "--verify",
                    "--automatic",
                    "--timeout",
                    "90",
                ],
                capture_output=True,
                text=True,
                timeout=110,
                check=False,
            )
            receipt = json.loads(result.stdout)
            if receipt.get("stage") == "deferred":
                self.state.unlink()
                return
            ok = result.returncode == 0 and receipt.get("ok") is True
            reason = (
                None
                if ok
                else f"BrainBar activation failed: {receipt.get('reason', 'no successful receipt')}; run brainlayer brainbar activate"
            )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            ok, reason = False, f"BrainBar activation failed: {exc}; run brainlayer brainbar activate"
        _atomic_write_json(self.state, {"identity": identity, "stage": "ready" if ok else "failed", "reason": reason})
        report("brainbar-activation", reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("BRAINLAYER_FORBID_BRAINBAR_SOCKET"):
        raise SystemExit("test guard forbids automatic installed BrainBar activation")
    try:
        AutoActivator(Path.home() / "Library/LaunchAgents", args.state).tick()
    except Exception as exc:
        report("brainbar-activation", f"BrainBar activation monitor refused: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
