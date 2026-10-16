"""Run one probe job in the existing macOS GUI domain; never unlock or sign in."""

import json
import os
import plistlib
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path


def run(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError("Aqua job command failed")
    return result.stdout.strip()


def require_aqua():
    if os.environ.get("BRAINLAYER_HARNESS_AQUA_REQUIRED") == "1" and run(["/bin/launchctl", "managername"]) != "Aqua":
        raise RuntimeError("Probe requires verified Aqua context; no harness launched")


def launch(script, argv, output, timeout, *, harnesses=()):
    """Bootstrap a unique one-shot job; tear down only that exact job on exit."""
    label = "com.brainlayer.harness-probe-" + uuid.uuid4().hex
    target = f"gui/{os.getuid()}/{label}"
    output = output.resolve()
    if output.exists():
        raise RuntimeError("Probe output already exists")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    for parent in (output.parent, *output.parent.parents):
        info = parent.lstat()
        unsafe = not stat.S_ISDIR(info.st_mode) or info.st_uid not in (0, os.getuid())
        shared = info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX
        if unsafe or shared:
            raise RuntimeError("Unsafe Aqua receipt parent")
        if parent == output.parent and (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700):
            raise RuntimeError("Unsafe Aqua receipt parent")
    with tempfile.TemporaryDirectory(prefix=".harness-aqua-", dir=output.parent) as work:
        work = Path(work)
        spec = {
            "Label": label,
            "ProgramArguments": [sys.executable, str(script.resolve()), *argv],
            "WorkingDirectory": str(Path.cwd()),
            "RunAtLoad": True,
            "KeepAlive": False,
            "LimitLoadToSessionType": "Aqua",
            "EnvironmentVariables": {
                "PATH": str(Path.home() / ".local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
                "BRAINLAYER_HARNESS_AQUA_REQUIRED": "1",
            },
            "StandardOutPath": str(work / "stdout"),
            "StandardErrorPath": str(work / "stderr"),
        }
        for name in ("T3_ACP_MCP_ENDPOINT", "T3_ACP_MCP_AUTHORIZATION"):
            if "t3" in harnesses and os.environ.get(name):
                spec["EnvironmentVariables"][name] = os.environ[name]
        plist = work / "probe.plist"
        fd = os.open(plist, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(plistlib.dumps(spec))
        bootstrapped = False
        try:
            run(["/bin/launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist)])
            bootstrapped = True
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                receipt = output / "aqua-exit.json"
                if receipt.exists():
                    code = json.loads(receipt.read_text())["exit_code"]
                    if type(code) is not int or code not in (0, 1):
                        raise RuntimeError("Invalid Aqua completion receipt")
                    return code
                state = run(["/bin/launchctl", "print", target])
                if "state = not running" in state and "last exit code = (never exited)" not in state:
                    raise RuntimeError("Aqua job exited without a probe receipt")
                time.sleep(1)
            raise TimeoutError("Own Aqua probe exceeded its deadline")
        finally:
            if bootstrapped:
                try:
                    run(["/bin/launchctl", "bootout", target])
                finally:
                    output.mkdir(parents=True, exist_ok=True, mode=0o700)
                    for name in ("stdout", "stderr"):
                        path = work / name
                        if path.exists():
                            (output / ("aqua-" + name + ".log")).write_bytes(path.read_bytes())
                    (output / "aqua-launch.json").write_text(json.dumps({"label": label, "domain": target}))
                    if (work / "stdout").exists():
                        print((work / "stdout").read_text(), end="")
