"""Explicit, single-attempt cask activation after Homebrew returns."""

import fcntl
import math
import os
import plistlib
import re
import subprocess
import sys
import time
from pathlib import Path
from xml.parsers.expat import ExpatError

from .brainbar_identity import JOBS, running_bundle_matches
from .setup import get_current_mcp_bridge_bin, verify_mcp_transport


def _activate(agents: Path, *, timeout_seconds: float = 90.0, verify: bool = True, automatic: bool = False) -> dict:
    """Replace only the cask's two labels; unload attempts on failure, never relaunch."""
    attempted = []
    mutated = False
    stage = "preflight"
    report = {"ok": False, "service_ready": False, "rollback_attempted": False, "cleanup_errors": []}

    def launchctl(*args, timeout=10.0):
        return subprocess.run(["/bin/launchctl", *args], capture_output=True, text=True, timeout=timeout, check=False)

    try:
        deadline = time.monotonic() + timeout_seconds
        if sys.platform != "darwin":
            raise ValueError("activation requires macOS")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("activation timeout must be finite and positive")
        socket = Path("/tmp/brainbar.sock")
        for key in ("BRAINLAYER_MCP_SOCKET", "MCPLAYER_BRAINLAYER_FRONT"):
            if os.environ.get(key) and os.environ[key] != str(socket):
                raise ValueError(f"cask activation requires {key} to be unset or {socket}")
        bridge = get_current_mcp_bridge_bin()
        if verify and not bridge:
            raise ValueError("installed brainlayer-mcp-stdio-bridge is missing")
        bundles = set()
        for label, binary in JOBS:
            with (agents / f"{label}.plist").open("rb") as stream:
                plist = plistlib.load(stream)
            if not isinstance(plist, dict):
                raise ValueError(f"invalid installed cask plist: {label}")
            arguments = plist.get("ProgramArguments", [])
            executable = (
                Path(arguments[0])
                if isinstance(arguments, list) and len(arguments) == 1 and isinstance(arguments[0], str)
                else Path()
            )
            if (
                plist.get("Label") != label
                or not executable.is_absolute()
                or executable.name != binary
                or executable.parent.name != "MacOS"
                or executable.parent.parent.name != "Contents"
                or executable.parents[2].name != "BrainBar.app"
                or not os.access(executable, os.X_OK)
            ):
                raise ValueError(f"invalid installed cask plist: {label}")
            bundles.add(executable.parents[2])
        if len(bundles) != 1:
            raise ValueError("BrainBar labels must use the same installed app")
        domain = f"gui/{os.getuid()}"

        def remaining():
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise TimeoutError("activation deadline exceeded")
            return seconds

        stage = "disabled"
        disabled = launchctl("print-disabled", domain, timeout=min(10, remaining()))
        if disabled.returncode:
            raise RuntimeError("could not inspect disabled launchd labels")
        for label, _ in JOBS:
            if re.search(r'"' + re.escape(label) + r'"\s*=>\s*true', disabled.stdout):
                raise RuntimeError(f"launchd label explicitly disabled: {label}")
        stage = "running_probe"
        if running_bundle_matches(next(iter(bundles)), timeout=min(10, remaining())):
            # A failed probe of this already-running bundle never authorizes stopping it.
            tool_count = verify_mcp_transport(bridge_command=bridge, timeout_seconds=remaining()) if verify else None
            return {
                **report,
                "ok": True,
                "service_ready": verify,
                "stage": "already_running",
                "reason": "",
                "tool_count": tool_count,
            }

        def brew_guard():
            if automatic:
                from .brainbar_auto_activation import brew_active

                if brew_active():
                    raise RuntimeError("active Homebrew run; automatic activation deferred")

        brew_guard()
        for label, _ in JOBS:
            brew_guard()
            stage = f"launch:{label}"
            target = f"{domain}/{label}"
            current = launchctl("print", target, timeout=min(10, remaining()))
            if current.returncode not in (0, 113):
                raise RuntimeError(f"could not inspect {label}: {current.returncode}")
            if current.returncode == 0:
                mutated = True
                if launchctl("bootout", target, timeout=min(10, remaining())).returncode != 0:
                    raise RuntimeError(f"could not unload {label}")
                while True:
                    state = launchctl("print", target, timeout=min(10, remaining()))
                    if state.returncode == 113:
                        break
                    if state.returncode != 0:
                        raise RuntimeError(f"could not observe teardown: {label}")
                    time.sleep(min(0.1, remaining()))
            brew_guard()
            attempted.append(target)
            mutated = True
            result = launchctl("bootstrap", domain, str(agents / f"{label}.plist"), timeout=min(10, remaining()))
            if result.returncode:
                if launchctl("print", target, timeout=min(10, remaining())).returncode != 0:
                    raise RuntimeError(f"could not bootstrap {label}: {result.returncode}")
        stage = "socket"
        if verify:
            while not socket.is_socket():
                time.sleep(min(0.1, remaining()))
        stage = "mcp"
        tool_count = verify_mcp_transport(bridge_command=bridge, timeout_seconds=remaining()) if verify else None
        if verify:
            stage = "running_identity"
            if not running_bundle_matches(next(iter(bundles)), timeout=min(10, remaining())):
                raise RuntimeError("MCP responded but running executables do not match the installed bundle")
        return {
            **report,
            "ok": True,
            "service_ready": verify,
            "stage": "verified" if verify else "started_unverified",
            "reason": "",
            "tool_count": tool_count,
        }
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
        plistlib.InvalidFileException,
        ExpatError,
    ) as exc:
        if automatic and not mutated and "active Homebrew run" in str(exc):
            return {**report, "stage": "deferred", "reason": str(exc)}
        # KeepAlive jobs must not keep retrying after failed activation. No automatic
        # old-bundle rollback, config rewrite, or second bootstrap is attempted.
        for target in reversed(attempted):
            try:
                if launchctl("bootout", target).returncode != 0:
                    report["cleanup_errors"].append(target)
            except (OSError, subprocess.SubprocessError):
                report["cleanup_errors"].append(target)
        return {**report, "stage": stage, "reason": str(exc)}


def activate(agents: Path, *, timeout_seconds: float = 90.0, verify: bool = True, automatic: bool = False) -> dict:
    try:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("activation timeout must be finite and positive")
        deadline = time.monotonic() + timeout_seconds
        if os.environ.get("BRAINLAYER_FORBID_BRAINBAR_SOCKET") and agents == Path.home() / "Library/LaunchAgents":
            raise ValueError("test guard forbids installed BrainBar activation")
        # Shared by manual commands and the supervisor child; no concurrent restart.
        with (agents / ".brainbar-activation.lock").open("a") as lock:
            while True:
                seconds = deadline - time.monotonic()
                if seconds <= 0:
                    raise TimeoutError("activation deadline exceeded waiting for activation lock")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if automatic:
                        return {
                            "ok": False,
                            "stage": "deferred",
                            "reason": "activation already in progress",
                            "service_ready": False,
                            "rollback_attempted": False,
                            "cleanup_errors": [],
                        }
                    time.sleep(min(0.1, seconds))
            return _activate(agents, timeout_seconds=deadline - time.monotonic(), verify=verify, automatic=automatic)
    except (OSError, ValueError) as exc:
        return {
            "ok": False,
            "stage": "activation_lock",
            "reason": str(exc),
            "service_ready": False,
            "rollback_attempted": False,
            "cleanup_errors": [],
        }
