"""Bind loaded cask PIDs to the executable vnodes in the installed bundle."""

import os
import plistlib
import re
import subprocess
import time
from pathlib import Path

JOBS = (("com.brainlayer.brainbar-daemon", "BrainBarDaemon"), ("com.brainlayer.brainbar", "BrainBar"))


def app_identity(app: Path) -> dict:
    with (app / "Contents/Info.plist").open("rb") as stream:
        info = plistlib.load(stream)
    if not isinstance(info, dict):
        raise ValueError("installed BrainBar identity is not a dictionary")
    version = info.get("CFBundleShortVersionString")
    commit = info.get("GitCommit", "")
    if not isinstance(version, str) or not version or not isinstance(commit, str):
        raise ValueError("installed BrainBar identity is missing")
    return {"version": version, "git_commit": commit}


def run(args, timeout=10):
    return subprocess.run(args, capture_output=True, text=True, check=False, timeout=timeout)


def loaded(label, timeout=10):
    result = run(["/bin/launchctl", "print", f"gui/{os.getuid()}/{label}"], timeout)
    if result.returncode not in (0, 113):
        raise RuntimeError(f"launchctl print failed for {label}: {result.returncode}")
    return result


def running_bundle_matches(app: Path, *, timeout=10) -> bool:
    deadline = time.monotonic() + timeout

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("running-bundle inspection deadline exceeded")
        return value

    for label, binary in JOBS:
        state = loaded(label, remaining())
        pid = re.search(r"(?m)^\s*pid\s*=\s*(\d+)\s*$", state.stdout)
        executable = app / "Contents/MacOS" / binary
        program = re.search(r"(?m)^\s*program\s*=\s*(.*?)\s*$", state.stdout)
        if state.returncode or not pid or not program or program[1].strip('"') != str(executable):
            return False
        result = run(["/usr/sbin/lsof", "-a", "-p", pid[1], "-d", "txt", "-F", "Din"], remaining())
        if result.returncode:
            return False
        device = inode = None
        matched = False
        info = executable.stat()
        for line in result.stdout.splitlines():
            if line.startswith("f"):
                device = inode = None
            elif line.startswith("D"):
                device = int(line[1:], 16)
            elif line.startswith("i"):
                inode = int(line[1:])
            elif line == f"n{executable}":
                matched = device == info.st_dev and inode == info.st_ino
                if matched:
                    break
        if not matched:
            return False
        # Refuse PID/label replacement while the vnode snapshot was inspected.
        final = loaded(label, remaining())
        final_pid = re.search(r"(?m)^\s*pid\s*=\s*(\d+)\s*$", final.stdout)
        if final.returncode or not final_pid or final_pid[1] != pid[1] or executable.stat() != info:
            return False
    return True
