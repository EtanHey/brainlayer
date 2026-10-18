"""Explicit, single-attempt cask activation after Homebrew returns."""

import math
import os
import plistlib
import subprocess
import sys
import time
from pathlib import Path
from xml.parsers.expat import ExpatError

from .setup import get_current_mcp_bridge_bin, verify_mcp_transport

JOBS = (("com.brainlayer.brainbar-daemon", "BrainBarDaemon"), ("com.brainlayer.brainbar", "BrainBar"))


def activate(agents: Path, *, timeout_seconds: float = 90.0) -> dict:
    """Replace only the cask's two labels; unload attempts on failure, never relaunch."""
    attempted = []
    stage = "preflight"
    report = {"ok": False, "service_ready": False, "rollback_attempted": False, "cleanup_errors": []}

    def launchctl(*args, timeout=10.0):
        return subprocess.run(["/bin/launchctl", *args], capture_output=True, timeout=timeout, check=False).returncode

    try:
        if sys.platform != "darwin":
            raise ValueError("activation requires macOS")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("activation timeout must be finite and positive")
        socket = Path("/tmp/brainbar.sock")
        for key in ("BRAINLAYER_MCP_SOCKET", "MCPLAYER_BRAINLAYER_FRONT"):
            if os.environ.get(key) and os.environ[key] != str(socket):
                raise ValueError(f"cask activation requires {key} to be unset or {socket}")
        bridge = get_current_mcp_bridge_bin()
        if not bridge:
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
        deadline = time.monotonic() + timeout_seconds

        def remaining():
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise TimeoutError("activation deadline exceeded")
            return seconds

        for label, _ in JOBS:
            stage = f"launch:{label}"
            target = f"{domain}/{label}"
            if launchctl("print", target, timeout=min(10, remaining())) == 0:
                if launchctl("bootout", target, timeout=min(10, remaining())) != 0:
                    raise RuntimeError(f"could not unload {label}")
            attempted.append(target)
            if launchctl("bootstrap", domain, str(agents / f"{label}.plist"), timeout=min(10, remaining())) != 0:
                raise RuntimeError(f"could not bootstrap {label}")
        stage = "socket"
        while not socket.is_socket():
            time.sleep(min(0.1, remaining()))
        stage = "mcp"
        tool_count = verify_mcp_transport(bridge_command=bridge, timeout_seconds=remaining())
        return {
            **report,
            "ok": True,
            "service_ready": True,
            "stage": "verified",
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
        # KeepAlive jobs must not keep retrying after failed activation. No automatic
        # old-bundle rollback, config rewrite, or second bootstrap is attempted.
        for target in reversed(attempted):
            try:
                if launchctl("bootout", target) != 0:
                    report["cleanup_errors"].append(target)
            except (OSError, subprocess.SubprocessError):
                report["cleanup_errors"].append(target)
        return {**report, "stage": stage, "reason": str(exc)}
