"""Opt-in local control for existing health/watchdog launchd command seams."""

from __future__ import annotations

import json
import math
import os
import re
import socket
import stat
import subprocess
import sys
from pathlib import Path

from .paths import supervisor_socket_path
from .pause import DEFAULT_PAUSE_SENTINEL_PATH, pause_applies_to_label, pause_sentinel_state
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
        "wal-checkpoint-fallback",  # Supervisor-only; never resurrect the calendar's archived label.
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


def supervisor_owns_jobs() -> bool:
    """Read the canonical migration owner with BrainBarJobOwnership's custody rules."""
    path = Path.home() / ".local/share/brainlayer/supervisor/migration.json"
    limit = 4 * 1024 * 1024
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        parts = path.parts[1:]
        for index, part in enumerate(parts):
            if part in {".", ".."}:
                raise ValueError("unsafe migration journal path")
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
            if index < len(parts) - 1:
                flags |= os.O_DIRECTORY
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or info.st_mode & 0o077
            or info.st_size > limit
        ):
            raise PermissionError("migration journal is untrusted")
        with os.fdopen(os.dup(fd), "rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("migration journal exceeds read limit")
        row = json.loads(raw)
        phase = row.get("phase") if isinstance(row, dict) else None
        if isinstance(phase, str) and phase in {"migrating", "migrated", "reverting"}:
            return True
        if isinstance(phase, str) and phase in {"reverted", "rolled_back"}:
            return False
        raise ValueError("migration journal has unknown phase")
    except FileNotFoundError:
        return False
    finally:
        os.close(fd)


class SupervisorCommands:
    """Adapt legacy *policy*, never launch archived component plists.

    Only unmanaged services (e.g. unchanged BrainBar) reach the real command runner.
    OFF/paused/operator disable/unknown/corrupt holds never authorize revival.
    """

    def __init__(self, path: Path, fallback, *, pause_path: Path = DEFAULT_PAUSE_SENTINEL_PATH):
        self.path, self.fallback, self.pause_path = path, fallback, pause_path
        self.aliases = ALIASES.copy()
        self.domain = f"gui/{os.getuid()}"
        self.control_error = ""

    def request(self, action="status", component="*", *, timeout=2, owner=""):
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
            client.settimeout(timeout)
            client.connect(str(self.path))
            if peer_uid(client) != os.getuid():
                raise PermissionError("supervisor peer uid differs")
            client.sendall(json.dumps(dict(action=action, component=component, owner=owner)).encode() + b"\n")
            with client.makefile("rb") as reader:
                response = json.loads(reader.readline(65537))
        if not isinstance(response, dict):
            raise ValueError("supervisor response must be an object")
        if "error" in response:
            raise RuntimeError(response["error"])
        states = response["components"]
        if not isinstance(states, dict):
            raise ValueError("supervisor components must be an object")
        for value in states.values():
            if (
                not isinstance(value, dict)
                or type(value["configured"]) is not bool
                or type(value["paused"]) is not bool
                or type(value["starts"]) is not int
                or value["starts"] < 0
                or type(value["exit_timeout"]) not in {int, float}
                or not math.isfinite(value["exit_timeout"])
                or value["exit_timeout"] < 0
                or type(value.get("interval", 0)) not in {int, float}
                or not math.isfinite(value.get("interval", 0))
                or value.get("interval", 0) < 0
                or value.get("next_run_at") is not None
                and (
                    type(value["next_run_at"]) not in {int, float}
                    or not math.isfinite(value["next_run_at"])
                    or value["next_run_at"] < 0
                )
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
        payload, active, _ = pause_sentinel_state(self.pause_path, strict=True)
        if active:
            disabled.update(
                label
                for label in (*self.aliases, *states, *ARCHIVED, "com.brainlayer.daemon", "com.brainlayer.maintenance")
                if pause_applies_to_label(payload, label)
            )
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
        if command == "launchctl" and args[1:2] == ["kickstart"] and label == "com.brainlayer.daemon":
            # Managed healers must not kill their parent before saving retry state.
            return result(127, error="parent supervisor restart refused; use install.sh consolidate")
        if label in ARCHIVED:
            return result(1, error=f"archived/retired label: {label}")
        if args[:2] != ["launchctl", "print-disabled"] and label not in self.aliases and command != "kill":
            return self.fallback(args)
        try:
            if self.control_error:
                raise RuntimeError(self.control_error)
            states = self.request()
            if args[:2] == ["launchctl", "print-disabled"]:
                if args[2:3] != [self.domain]:
                    raise PermissionError("wrong launchd domain")
                disabled = self._disabled(states)
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
                    # An interval job remains loaded between runs, including a health
                    # check that exits 1 to report issues. A dead worker with no next
                    # scheduled run must remain unhealthy.
                    down = (
                        state["configured"]
                        and not state["paused"]
                        and not state["pid"]
                        and not (state.get("interval") and state.get("next_run_at") is not None)
                    )
                    return result(
                        code=113 if down else 0,
                        out=f"state = {'running' if state['pid'] else 'waiting'}\nruns = {state['starts']}\n"
                        + (f"pid = {state['pid']}\n" if state["pid"] else ""),
                    )
                if verb == "bootstrap":
                    if args[2] != self.domain:
                        raise PermissionError("wrong launchd domain")
                    state = states[name]
                    # Resume only selects paused children. Heal an already-unpaused
                    # dead worker directly, without creating an owner-less pause.
                    action = "restart" if state["configured"] and not state["paused"] and not state["pid"] else "resume"
                elif verb == "bootout":
                    if not states[name]["pid"]:
                        return result()
                    action = "pause"
                elif verb == "kickstart":
                    action = "restart"
                else:
                    raise ValueError("unsupported managed launchd action")
            if action != "pause":
                disabled = self._disabled(states, include_paused=action != "resume")
                if label in disabled or "com.brainlayer.daemon" in disabled:
                    raise PermissionError("component is OFF, paused or disabled")
            self.request(action, name, timeout=states[name]["exit_timeout"] + 5)
            return result()
        except (
            OSError,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
            OverflowError,
            RuntimeError,
            StopIteration,
        ) as exc:
            return result(127, error=f"supervisor control failed: {exc}")


def adapt_commands(runner, pause_path=None, *, required=(), tolerate_unavailable=False, discover_ownership=False):
    path = os.environ.get("BRAINLAYER_SUPERVISOR_SOCKET")
    if isinstance(runner, SupervisorCommands):
        return runner
    ownership_error = ""
    if discover_ownership:
        try:
            if not supervisor_owns_jobs():
                return runner
        except (OSError, ValueError) as exc:
            ownership_error = f"can't determine job owner: {exc}"
    elif not path:
        return runner
    backend = SupervisorCommands(
        supervisor_socket_path(),
        runner,
        pause_path=pause_path or Path(os.environ.get("BRAINLAYER_PAUSE_SENTINEL", str(DEFAULT_PAUSE_SENTINEL_PATH))),
    )
    try:
        if ownership_error:
            raise RuntimeError(ownership_error)
        states = backend.request()
        if any(backend.aliases[label] not in states for label in required if label in backend.aliases):
            raise RuntimeError("required supervisor component is unavailable")
    except (OSError, ValueError, KeyError, TypeError, OverflowError, RuntimeError) as exc:
        if not tolerate_unavailable:
            raise
        backend.control_error = str(exc) or type(exc).__name__

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
