#!/opt/homebrew/opt/brainlayer/libexec/venv/bin/python
"""BrainLayer SessionStart cleanup hook — runs on every new Claude Code session.

Performs two maintenance tasks:
  1. Stops verified, explicitly recorded MCP children after they become orphaned
  2. Checkpoints the WAL if it exceeds WAL_THRESHOLD_MB

Output: brief status line to stdout (injected as Claude context).
Target: <1s total.
"""

import ctypes
import json
import os
import signal
import sqlite3
import stat
import struct
import subprocess
import sys
import time
from pathlib import Path

# --- Config ---
WAL_THRESHOLD_MB = 100  # only checkpoint if WAL > this size
OWNERSHIP_DIR = Path.home() / ".local/state/brainlayer/session-cleanup-owned"
_CANONICAL_DB = os.path.expanduser("~/.local/share/brainlayer/brainlayer.db")


def get_db_path():
    env = os.environ.get("BRAINLAYER_DB")
    if env and os.path.exists(env):
        return env
    if os.path.exists(_CANONICAL_DB):
        return _CANONICAL_DB
    return None


def parse_etime(etime_str):
    """Parse ps etime (DD-HH:MM:SS / HH:MM:SS / MM:SS) to seconds."""
    days = 0
    if "-" in etime_str:
        day_part, etime_str = etime_str.split("-", 1)
        days = int(day_part)
    parts = etime_str.split(":")
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    elif len(parts) == 2:
        h, m, s = 0, int(parts[0]), int(parts[1])
    else:
        return 0
    return days * 86400 + h * 3600 + m * 60 + s


def _process_identity(pid):
    """Read executable, parent and microsecond birth from the kernel, never argv."""
    if type(pid) is not int or pid <= 1:
        return None
    try:
        if sys.platform == "darwin":
            lib = ctypes.CDLL("/usr/lib/libproc.dylib")
            info = ctypes.create_string_buffer(136)  # proc_bsdinfo (sys/proc_info.h)
            path = ctypes.create_string_buffer(4096)
            if lib.proc_pidinfo(pid, 3, 0, info, len(info)) != len(info):
                return None
            if struct.unpack_from("I", info.raw, 4)[0] == 5:  # SZOMB
                return None
            if lib.proc_pidpath(pid, path, len(path)) <= 0:
                return None
            ppid, uid = struct.unpack_from("II", info.raw, 16)
            start = ":".join(map(str, struct.unpack_from("QQ", info.raw, 120)))
            executable = os.fsdecode(path.value)
        elif sys.platform == "linux":
            root = Path(f"/proc/{pid}")
            fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] == "Z":
                return None
            ppid, uid = int(fields[1]), root.stat().st_uid
            # Include boot identity so a record cannot survive PID reuse on reboot.
            start = Path("/proc/sys/kernel/random/boot_id").read_text().strip() + ":" + fields[19]
            executable = os.readlink(root / "exe")
        else:
            return None
        return dict(pid=pid, ppid=ppid, uid=uid, start=start, executable=executable)
    except (OSError, ValueError, IndexError):
        return None


def _ownership_directory(create=False):
    if create:
        OWNERSHIP_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(OWNERSHIP_DIR, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    info = os.fstat(fd)
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        os.close(fd)
        raise PermissionError("ownership directory must be private")
    return fd


def record_owned_child(pid):
    """Record only a direct child of this hook; never adopt an existing seat."""
    child = _process_identity(pid)
    if not child or child["ppid"] != os.getpid() or child["uid"] != os.getuid():
        return False
    try:
        directory = _ownership_directory(create=True)
        try:
            fd = os.open(f"{pid}.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
            with os.fdopen(fd, "w") as output:
                json.dump(child, output)
                output.flush()
                os.fsync(output.fileno())
        finally:
            os.close(directory)
        return True
    except OSError:
        return False  # Failed registration never authorizes cleanup.


def start_owned_mcp(argv):
    """Explicit launch/record path; SessionStart does not currently launch MCPs."""
    if not argv or not os.path.isabs(argv[0]):
        raise ValueError("owned MCP executable must be absolute")
    process = subprocess.Popen(
        argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
    )
    record_owned_child(process.pid)
    return process  # Caller retains its handle even if registration failed.


def cleanup_stale_mcp():
    """Stop only recorded, identity-matching orphans. No record means no scan."""
    try:
        directory = _ownership_directory()
    except OSError:
        return 0
    killed = 0
    try:
        for name in os.listdir(directory):
            if not name.endswith(".json"):
                continue
            try:
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
                with os.fdopen(fd) as source:
                    info = os.fstat(source.fileno())
                    if (
                        not stat.S_ISREG(info.st_mode)
                        or info.st_uid != os.getuid()
                        or info.st_mode & 0o077
                        or info.st_nlink != 1
                        or info.st_size > 16384
                    ):
                        continue
                    saved = json.load(source)
                if not isinstance(saved, dict):
                    continue
                pid = saved.get("pid")
                if type(pid) is not int or pid <= 1 or name != f"{pid}.json":
                    continue
                current = _process_identity(pid)
                if not current or current["ppid"] != 1 or current["uid"] != os.getuid():
                    continue  # Attached Codex/T3/Cursor/test children stay alive at any age.
                if any(current[key] != saved.get(key) for key in ("pid", "uid", "start", "executable")):
                    continue
                if _process_identity(pid) != current:
                    continue  # Revalidate birth, executable and orphan state before TERM.
                os.kill(pid, signal.SIGTERM)
                killed += 1
                os.unlink(name, dir_fd=directory)
            except (OSError, ValueError, TypeError):
                continue
    finally:
        os.close(directory)
    return killed


def wal_checkpoint():
    """Checkpoint WAL if it exceeds threshold. Returns (before_mb, after_mb) or None."""
    db_path = get_db_path()
    if not db_path:
        return None

    wal_path = db_path + "-wal"
    try:
        wal_size = os.path.getsize(wal_path)
    except OSError:
        return None

    wal_mb = wal_size / (1024 * 1024)
    if wal_mb < WAL_THRESHOLD_MB:
        return None  # not worth checkpointing

    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_checkpoint(PASSIVE)")  # non-blocking
        conn.close()
    except sqlite3.Error:
        return None

    try:
        after_size = os.path.getsize(wal_path)
    except OSError:
        after_size = 0

    return (wal_mb, after_size / (1024 * 1024))


def main():
    start = time.monotonic()
    messages = []

    # 1. Cleanup stale MCP processes
    killed = cleanup_stale_mcp()
    if killed:
        messages.append(f"Cleaned {killed} stale MCP process{'es' if killed != 1 else ''}")

    # 2. WAL checkpoint (if needed)
    if (time.monotonic() - start) < 0.8:  # time budget
        result = wal_checkpoint()
        if result:
            before, after = result
            messages.append(f"WAL checkpoint: {before:.0f}MB → {after:.0f}MB")

    if messages:
        print("[BrainLayer cleanup] " + "; ".join(messages))

    sys.exit(0)


if __name__ == "__main__":
    main()
