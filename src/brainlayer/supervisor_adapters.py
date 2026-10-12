"""Opt-in local control for existing health/watchdog launchd command seams."""

from __future__ import annotations

import json
import os
import re
import socket
import stat
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from .pause import DEFAULT_PAUSE_SENTINEL_PATH, _parse_iso_datetime
from .supervisor_guard import peer_uid

ALIASES = {
    f"com.brainlayer.{name}": name
    for name in (
        "watch",
        "drain",
        "health-check",
        "observability",
        "throughput-watchdog",
        "tier0-watchdog",
        "log-cap",
        "t3-ingest",
    )
}
ALIASES.update(
    {"com.brainlayer.hotlane-brainbar": "hotlane", "com.etanhey.brainlayer-fleet-watchdog": "fleet-watchdog"}
)
ARCHIVED = {
    f"com.brainlayer.{name}"
    for name in (
        "index",
        "backup-daily",
        "jsonl-backup",
        "maintenance-nightly",
        "maintenance-weekly",
        "decay",
        "repair-fts",
        "p0-counter",
        "wal-checkpoint",
        "enrich",
        "enrichment",
    )
}


class SupervisorCommands:
    """Adapt legacy *policy*, never launch archived component plists.

    Only unmanaged services (e.g. unchanged BrainBar) reach the real command runner.
    OFF/paused/operator disable/unknown/corrupt holds never authorize revival.
    """

    def __init__(self, path: Path, fallback, *, pause_path: Path = DEFAULT_PAUSE_SENTINEL_PATH):
        self.path, self.fallback, self.pause_path = path, fallback, pause_path
        self.aliases = ALIASES.copy()
        self.domain = f"gui/{os.getuid()}"

    def request(self, action="status", component="*"):
        directory, endpoint = self.path.parent.lstat(), self.path.lstat()
        if (
            directory.st_uid != os.getuid()
            or directory.st_mode & 0o077
            or not stat.S_ISDIR(directory.st_mode)
            or endpoint.st_uid != os.getuid()
            or endpoint.st_mode & 0o077
            or not stat.S_ISSOCK(endpoint.st_mode)
        ):
            raise PermissionError("supervisor endpoint must be owner-only")
        with socket.socket(socket.AF_UNIX) as client:
            client.settimeout(2 if action == "status" else 20)
            client.connect(str(self.path))
            if peer_uid(client) != os.getuid():
                raise PermissionError("supervisor peer uid differs")
            client.sendall(json.dumps(dict(action=action, component=component)).encode() + b"\n")
            with client.makefile("rb") as reader:
                response = json.loads(reader.readline(65537))
        if "error" in response:
            raise RuntimeError(response["error"])
        states = response["components"]
        for value in states.values():
            if (
                type(value["configured"]) is not bool
                or type(value["paused"]) is not bool
                or value["pid"] is not None
                and (type(value["pid"]) is not int or value["pid"] <= 1)
            ):
                raise ValueError("unknown supervisor component state")
        return states

    def _disabled(self, states, *, include_paused=True):
        observed = self.fallback(["launchctl", "print-disabled", self.domain])
        if observed.returncode != 0:
            raise RuntimeError("operator disabled state unavailable")
        disabled = {
            label
            for label, value in re.findall(r'["\']([^"\']+)["\']\s*=>\s*(true|false|disabled|enabled)', observed.stdout)
            if value in {"true", "disabled"}
        }
        try:
            payload = json.loads(self.pause_path.read_text())
        except FileNotFoundError:
            if os.path.lexists(self.pause_path):
                raise OSError("pause sentinel unreadable")
            payload = {"labels": []}
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("labels"), list)
            or not all(isinstance(v, str) for v in payload["labels"])
        ):
            raise ValueError("pause sentinel is corrupt")
        expiry = _parse_iso_datetime(payload.get("expires_at"))
        if expiry is None or datetime.now(UTC) <= expiry:
            disabled.update(payload["labels"])
        return disabled | {
            label
            for label, name in self.aliases.items()
            if name in states and (not states[name]["configured"] or include_paused and states[name]["paused"])
        }

    def __call__(self, args):
        def result(code=0, out="", error=""):
            return subprocess.CompletedProcess(args, code, out, error)

        command = Path(args[0]).name
        if command not in {"launchctl", "kill"}:
            return self.fallback(args)
        label = Path(args[-1]).name.removesuffix(".plist")
        if label in ARCHIVED:
            return result(1, error=f"archived/retired label: {label}")
        if args[:2] != ["launchctl", "print-disabled"] and label not in self.aliases and command != "kill":
            return self.fallback(args)
        try:
            states = self.request()
            if args[:2] == ["launchctl", "print-disabled"]:
                disabled = self._disabled(states)
                if args[2] != self.domain:
                    raise PermissionError("wrong launchd domain")
                return result(out="\n".join(f'"{label}" => disabled' for label in sorted(disabled)))
            if command == "kill":
                if args[1:2] != ["-9"]:
                    raise ValueError("unsupported signal command")
                matches = [name for name, value in states.items() if value["pid"] == int(args[-1])]
                if len(matches) != 1:
                    raise ValueError("pid is not an owned component")
                name = matches[0]
                label = next(label for label, candidate in self.aliases.items() if candidate == name)
                action = "restart"
            else:
                name, verb = self.aliases[label], args[1]
                if name not in states:
                    raise ValueError(f"unknown component: {name}")
                if verb != "bootstrap" and args[-1] != f"{self.domain}/{label}":
                    raise PermissionError("wrong launchd target")
                if verb == "print":
                    state = states[name]
                    return result(
                        out=f"state = {'running' if state['pid'] else 'waiting'}\nruns = {state['starts']}\n"
                        + (f"pid = {state['pid']}\n" if state["pid"] else "")
                    )
                if verb == "bootstrap":
                    if args[2] != self.domain:
                        raise PermissionError("wrong launchd domain")
                    action = "resume"
                elif verb == "bootout":
                    action = "pause"
                elif verb == "kickstart":
                    action = "restart"
                else:
                    raise ValueError("unsupported managed launchd action")
            if action != "pause":
                disabled = self._disabled(states, include_paused=action != "resume")
                if label in disabled or "com.brainlayer.daemon" in disabled:
                    raise PermissionError("component is OFF, paused or disabled")
            self.request(action, name)
            return result()
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, StopIteration) as exc:
            return result(127, error=f"supervisor control failed: {exc}")


def adapt_commands(runner, pause_path=None, *, required=()):
    path = os.environ.get("BRAINLAYER_SUPERVISOR_SOCKET")
    if not path or isinstance(runner, SupervisorCommands):
        return runner
    backend = SupervisorCommands(
        Path(path),
        runner,
        pause_path=pause_path or Path(os.environ.get("BRAINLAYER_PAUSE_SENTINEL", str(DEFAULT_PAUSE_SENTINEL_PATH))),
    )
    states = backend.request()  # Unknown transport must fail the worker.
    if any(backend.aliases[label] not in states for label in required if label in backend.aliases):
        raise RuntimeError("required supervisor component is unavailable")
    return backend


def main():
    def fallback(args):
        return subprocess.run(args, capture_output=True, text=True, timeout=20, check=False)

    runner = adapt_commands(fallback)
    if runner is fallback:
        raise RuntimeError("BRAINLAYER_SUPERVISOR_SOCKET is required")
    result = runner(["launchctl", *sys.argv[1:]])
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
