"""One bounded supervisor pass; durable per-build suppression survives daemon restarts."""

import argparse
import fcntl
import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path

from .brainbar_identity import JOBS, app_identity, loaded
from .deploy_drift import _atomic_write_json
from .job_alerts import report


def process_table():
    result = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,comm=,args="], capture_output=True, text=True, check=True, timeout=5
    )
    return result.stdout


def brew_active():
    # Include brew's Ruby entrypoint and Homebrew-owned child executables, even
    # after reparenting. Unknown/unreadable process tables defer activation.
    rows = process_table().splitlines()
    for row in rows:
        fields = row.split(None, 3)
        if len(fields) != 4:
            raise ValueError("unreadable process table")
        tokens = fields[3].split()
        if Path(fields[2]).name == "brew" or any(
            Path(token).name in {"brew", "brew.rb"} or "/Homebrew/" in token or "/homebrew/Library/Homebrew/" in token
            for token in tokens
        ):
            return True
    return False


class AutoActivator:
    def __init__(self, agents: Path, state: Path):
        self.agents, self.state = agents, state

    def tick(self):
        if brew_active():
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
        self.state.parent.mkdir(parents=True, exist_ok=True)
        with self.state.with_suffix(".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            prior = json.loads(self.state.read_text()) if self.state.exists() else {}
            if prior.get("identity") == identity:
                if prior.get("stage") == "attempted":
                    reason = "BrainBar activation was interrupted; run brainlayer brainbar activate"
                    report("brainbar-activation", reason)
                    _atomic_write_json(self.state, {"identity": identity, "stage": "failed", "reason": reason})
                return
            if any(loaded(label).returncode == 0 for label, _ in JOBS) or brew_active():
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
            _atomic_write_json(
                self.state, {"identity": identity, "stage": "ready" if ok else "failed", "reason": reason}
            )
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
