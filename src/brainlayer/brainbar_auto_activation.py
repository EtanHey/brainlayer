"""One bounded supervisor pass; durable per-build suppression survives daemon restarts."""

import argparse
import fcntl
import json
import os
import plistlib
import re
import subprocess
import sys
import time
from pathlib import Path

from .brainbar_identity import JOBS, app_identity, loaded, running_bundle_matches
from .deploy_drift import _atomic_write_json
from .job_alerts import active_alerts, report
from .pipeline.secret_scrub import scrub_secrets


class ProcessScanUnavailable(Exception):
    """Detection is unknown; only the safe exception class name may escape."""


def inspect_processes(function, *args):
    try:
        return function(*args)
    except (subprocess.TimeoutExpired, TimeoutError) as exc:
        error = ProcessScanUnavailable(type(exc).__name__)
        error.inspection_timeout = True
        raise error from None


def process_table():
    result = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,comm=,args="], capture_output=True, text=True, check=True, timeout=5
    )
    return result.stdout


def brew_processes():
    # Include brew's Ruby entrypoint and Homebrew-owned child executables, even
    # after reparenting. Unknown/unreadable process tables defer activation.
    rows = process_table().splitlines()
    matches, readable = [], 0
    for row in rows:
        fields = row.split(None, 3)
        # A process exiting or exec-ing mid-scan can tear one row: empty args
        # are still checked by comm, a row without a pid and comm is skipped.
        if len(fields) < 3 or not fields[0].isascii() or not fields[0].isdecimal():
            continue
        readable += 1
        tokens = fields[3].split() if len(fields) == 4 else []
        if Path(fields[2]).name == "brew" or any(
            Path(token).name in {"brew", "brew.rb"} or "/Homebrew/" in token or "/homebrew/Library/Homebrew/" in token
            for token in tokens
        ):
            # argv is detection-only: Homebrew curl headers can carry tokens.
            executable = scrub_secrets(Path(fields[2]).name).text[:128]
            matches.append(f"{fields[0]} {executable}")
    if not readable:
        raise ValueError("unreadable process table")
    return matches


def brew_active():
    return bool(brew_processes())


# The pid in update.lock is the installer's bash child, which always runs as
# `<copy or keg>/(brainbar-update|update-<op>).sh --custody-held ...`.
UPDATER_COMMAND = re.compile(r"(?:brainbar-update|/update-[A-Za-z0-9-]+)\.sh --custody-held ")


def process_command(pid: int) -> str:
    # Popen, not run: one bounded ps of a single pid, timeout counted like the other scans.
    with subprocess.Popen(
        ["/bin/ps", "-o", "command=", "-p", str(pid)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
    ) as proc:
        try:
            out, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise
    return out.strip() if proc.returncode == 0 else ""


def update_in_progress(root: Path) -> bool:
    """True only while update.lock names a pid that is still the updater (a reused pid never counts)."""
    try:
        pid = int((root / "update.lock" / "pid").read_text().strip())
    except (OSError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        return bool(UPDATER_COMMAND.search(inspect_processes(process_command, pid)))
    except OSError as exc:
        raise ProcessScanUnavailable(type(exc).__name__) from None


class AutoActivator:
    def __init__(self, agents: Path, state: Path, update_root: Path | None = None):
        self.agents, self.state = agents, state
        default = Path.home() / "Library/Application Support/BrainBar"
        self.update_root = update_root or Path(os.environ.get("BRAINBAR_UPDATE_ROOT") or default)

    def tick(self):
        self.state.parent.mkdir(parents=True, exist_ok=True)
        with self.state.with_suffix(".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            failures = self.state.with_suffix(".scan.json")
            prior = json.loads(failures.read_text()) if failures.exists() else {}
            try:
                self._tick()
            except ProcessScanUnavailable as exc:
                if not getattr(exc, "inspection_timeout", False):
                    prior = json.loads(failures.read_text()) if failures.exists() else {}
                count = min(prior.get("failures", 0) + 1, 12)
                _atomic_write_json(failures, {"failures": count})
                if count >= 12:
                    try:
                        absent = all(loaded(label).returncode == 113 for label, _ in JOBS)
                    except (OSError, subprocess.SubprocessError, RuntimeError):
                        return  # Unknown launchd state is not evidence BrainBar is down.
                    if absent:
                        report("brainbar-activation", f"BrainBar activation process scan failed for >=12 ticks: {exc}")

    def _defer_for_brew(self):
        # Separate from the per-build attempt: brew may have removed both plists.
        path = self.state.with_suffix(".brew.json")
        prior = json.loads(path.read_text()) if path.exists() else {}
        try:
            processes = brew_processes()
        except (OSError, subprocess.SubprocessError) as exc:
            raise ProcessScanUnavailable(type(exc).__name__) from None
        except ValueError:
            # Counted and alerted only after 12 ticks, never an immediate refusal.
            raise ProcessScanUnavailable("unreadable process table") from None
        # Recover before activation can report a new failure on the same alert key.
        self.state.with_suffix(".scan.json").unlink(missing_ok=True)
        reason = active_alerts().get("brainbar-activation", "")
        migration = self.state.with_suffix(".scan-migration.json")
        migrate_legacy = not migration.exists()
        if migrate_legacy:
            # Consume even without a legacy alert, before later work can raise an OSError.
            _atomic_write_json(migration, {"completed": True})
        if (migrate_legacy and reason.startswith("BrainBar activation monitor refused: [Errno")) or reason.startswith(
            (
                "BrainBar activation process scan failed for >=12 ticks:",
                "BrainBar activation monitor refused: Command '['/bin/ps',",
                "BrainBar activation monitor refused: Command '['/usr/sbin/lsof',",
                "BrainBar activation monitor refused: Command '['/bin/launchctl',",
                "BrainBar activation monitor refused: running-bundle inspection deadline exceeded",
                "BrainBar activation monitor refused: unreadable process table",
            )
        ):
            report("brainbar-activation", None)
        absent = bool(processes) and all(inspect_processes(loaded, label).returncode == 113 for label, _ in JOBS)
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

    def _defer_for_update(self):
        # Between the cask step and its own activation the updater restores plists and carries
        # approval without the activation lock; a launch here would race it on a new build.
        path = self.state.with_suffix(".update.json")
        prior = json.loads(path.read_text()) if path.exists() else {}
        if not update_in_progress(self.update_root):
            if prior:
                path.unlink()
                if prior.get("alerted"):
                    report("brainbar-activation-update", None)
            return False
        now = time.time()
        since = min(prior.get("since", now), now)
        alerted = prior.get("alerted", False)
        if now - since > 900 and not alerted:
            report(
                "brainbar-activation-update",
                "BrainBar activation deferred by an in-app update for >15 minutes; "
                "inspect update-state and update.log in ~/Library/Application Support/BrainBar",
            )
            alerted = True
        _atomic_write_json(path, {"since": since, "alerted": alerted})
        return True

    def _tick(self):
        if self._defer_for_update() or self._defer_for_brew():
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
        states = [inspect_processes(loaded, label).returncode for label, _ in JOBS]
        if all(code == 0 for code in states) and inspect_processes(running_bundle_matches, app):
            _atomic_write_json(self.state, {"identity": identity, "stage": "ready"})
            report("brainbar-activation", None)
            return
        if prior.get("identity") == identity:
            if prior.get("stage") == "attempted":
                reason = "BrainBar activation was interrupted; run brainlayer brainbar activate"
                report("brainbar-activation", reason)
                _atomic_write_json(self.state, {"identity": identity, "stage": "failed", "reason": reason})
            return
        if any(code == 0 for code in states) or self._defer_for_brew() or update_in_progress(self.update_root):
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
