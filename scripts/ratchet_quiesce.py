"""Real launchd replay on GitHub macOS only; local tests must use fake commands."""

from __future__ import annotations

import argparse
import json
import os
import platform
import plistlib
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

INTERVAL = 3
WINDOW = 2 * INTERVAL + 2


def launchctl(*args, check=False):
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, timeout=15, check=check)


def require_runner():
    if (
        os.environ.get("GITHUB_ACTIONS") != "true"
        or os.environ.get("RUNNER_OS") != "macOS"
        or platform.system() != "Darwin"
    ):
        raise RuntimeError("real launchd replay is restricted to a GitHub macOS runner")
    if not shutil.which("launchctl"):
        raise RuntimeError("launchctl missing")
    result = launchctl("print", f"gui/{os.getuid()}")
    if result.returncode:
        raise RuntimeError(f"no usable gui domain: exit {result.returncode}: {result.stderr}")


def labels(namespace):
    if not re.fullmatch(r"com\.brainlayer\.ratchettest\.[a-zA-Z0-9-]+", namespace):
        raise ValueError("only an isolated ratchettest namespace is allowed")
    return {"brainbar-daemon": f"{namespace}.subject", "fleet-watchdog": f"{namespace}.watchdog"}


def cleanup(home, mapping, *, command=launchctl):
    failures = []
    for label in reversed(tuple(mapping.values())):  # supervisor first
        target = f"gui/{os.getuid()}/{label}"
        command("disable", target)
        command("bootout", target)
        state = command("print", target)
        if state.returncode not in {3, 36, 113} and "could not find service" not in state.stderr.lower():
            failures.append(f"cleanup cannot prove {label} unloaded: {state.stdout} {state.stderr}")
        enabled = command("enable", target)
        if enabled.returncode:
            failures.append(f"cleanup enable failed for {label}: {enabled.stderr}")
        (home / "Library/LaunchAgents" / f"{label}.plist").unlink(missing_ok=True)
    if failures:
        raise RuntimeError("; ".join(failures))


def wait_until(predicate, description):
    deadline = time.monotonic() + 25
    while not predicate():
        if time.monotonic() >= deadline:
            raise RuntimeError(f"timeout: {description}")
        time.sleep(0.2)


def replay(source, home, mapping):
    sys.path.insert(0, str(source / "src"))
    from brainlayer import maintenance
    from brainlayer.launchd_primitive import is_launchd_label_disabled

    if Path(maintenance.__file__).resolve() != source / "src/brainlayer/maintenance.py":
        raise RuntimeError("maintenance imported from a different checkout")
    # This is configuration injection, never a replacement quiesce/state algorithm.
    with (
        patch.dict(os.environ, {"HOME": str(home), "BRAINLAYER_MAINTENANCE_KEEP_DOWN": ""}),
        patch.object(maintenance, "_launchd_label", side_effect=lambda service: mapping[service]),
        patch.object(maintenance, "PAUSE_SENTINEL_PATH", home / "data/pause.sentinel"),
    ):
        agents = home / "Library/LaunchAgents"
        agents.mkdir(parents=True)
        watchdog = home / "fleet-watchdog.sh"
        shutil.copyfile(source / "scripts/launchd/fleet-watchdog.sh", watchdog)
        for service, label in mapping.items():
            args = ["/bin/sleep", "600"] if service == "brainbar-daemon" else ["/bin/sh", str(watchdog)]
            payload = {
                "Label": label,
                "ProgramArguments": args,
                "RunAtLoad": True,
                "EnvironmentVariables": {"HOME": str(home), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
            }
            if service == "fleet-watchdog":
                payload["StartInterval"] = INTERVAL
            else:
                payload["KeepAlive"] = True
            plist = agents / f"{label}.plist"
            plist.write_bytes(plistlib.dumps(payload))
            launchctl("enable", f"gui/{os.getuid()}/{label}", check=True)
            launchctl("bootstrap", f"gui/{os.getuid()}", str(plist), check=True)

        def loaded():
            return maintenance._service_is_loaded("brainbar-daemon")

        def disabled():
            return is_launchd_label_disabled(mapping["fleet-watchdog"])

        wait_until(loaded, "initial subject bootstrap")
        if disabled() is not False:
            raise RuntimeError("watchdog was not enabled initially")
        # Positive control: prove this exact watchdog revives this exact subject.
        launchctl("bootout", f"gui/{os.getuid()}/{mapping['brainbar-daemon']}", check=True)
        wait_until(lambda: not loaded(), "positive-control subject bootout")
        wait_until(loaded, "positive-control watchdog revival")
        if "re-bootstrapped" not in (home / "Library/Logs/brainlayer/fleet-watchdog.log").read_text():
            raise RuntimeError("subject revival has no real-watchdog log receipt")

        stopped = {}
        revived = False
        held = True
        try:
            maintenance._quiesce_services(("brainbar-daemon",), stopped)
            deadline = time.monotonic() + WINDOW
            while time.monotonic() < deadline:
                revived |= loaded()
                held &= disabled() is True and not maintenance._service_is_loaded("fleet-watchdog")
                time.sleep(0.2)
        finally:
            failures = maintenance._resume_services(source, ("brainbar-daemon",), stopped)
            if failures:
                raise RuntimeError(f"real resume failed: {failures}")
        wait_until(lambda: loaded() and maintenance._service_is_loaded("fleet-watchdog"), "both jobs resumed")
        if disabled() is not False:
            raise RuntimeError("watchdog still disabled after resume")
        marker_removed = not (home / "data/fleet-watchdog-hold.json").exists()
        if not marker_removed:
            raise RuntimeError("watchdog hold marker left behind")
        return {
            "revived": revived,
            "held": held,
            "resumed": True,
            "marker_removed": marker_removed,
            "positive_control": True,
            "window_seconds": WINDOW,
            "interval_seconds": INTERVAL,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expect-red", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    if not args.cleanup and args.source is None:
        parser.error("--source is required unless --cleanup is set")
    require_runner()  # before any import, plist creation, or mutating command
    mapping = labels(args.namespace)
    if not args.home.resolve().is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve()):
        raise ValueError("fixture HOME must be inside RUNNER_TEMP")
    if args.cleanup:
        cleanup(args.home, mapping)
        return 0
    sha = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    report = {"sha": sha, "status": "failed"}
    try:
        report.update(replay(args.source.resolve(), args.home.resolve(), mapping))
        report["status"] = "RED" if report["revived"] else "GREEN" if report["held"] else "failed"
        passed = report["status"] == ("RED" if args.expect_red else "GREEN")
        return 0 if passed else 1
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if args.out:
            args.out.write_text(json.dumps(report) + "\n")
        print(json.dumps(report), flush=True)
        cleanup(args.home, mapping)


if __name__ == "__main__":
    raise SystemExit(main())
