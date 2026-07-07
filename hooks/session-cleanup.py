#!/usr/bin/env python3
"""BrainLayer SessionStart cleanup hook — runs on every new Claude Code session.

Performs two maintenance tasks:
  1. Kills orphaned MCP processes (ppid=1 or stale >6h with idle parent)
  2. Checkpoints the WAL if it exceeds WAL_THRESHOLD_MB

Output: brief status line to stdout (injected as Claude context).
Target: <1s total.
"""

import os
import signal
import sqlite3
import subprocess
import sys
import time

# --- Config ---
MAX_AGE_HOURS = 6
WAL_THRESHOLD_MB = 100  # only checkpoint if WAL > this size
MCP_PATTERNS = ("brainlayer-mcp", "voicelayer-mcp")
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


def cleanup_stale_mcp():
    """Kill orphaned/stale MCP processes. Returns count killed."""
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,ppid,etime,command"], text=True, timeout=3)
    except (subprocess.SubprocessError, FileNotFoundError):
        return 0

    killed = 0
    my_pid = os.getpid()
    age_threshold = MAX_AGE_HOURS * 3600

    for line in out.strip().split("\n")[1:]:
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, ppid, etime, cmd = int(parts[0]), int(parts[1]), parts[2], parts[3]

        if pid == my_pid:
            continue
        if not any(pat in cmd for pat in MCP_PATTERNS):
            continue

        age = parse_etime(etime)
        should_kill = False

        # Rule 1: orphaned (reparented to init/launchd)
        if ppid == 1:
            should_kill = True

        # Rule 2: old process with idle parent
        elif age > age_threshold:
            # Check if parent claude has active node/bun children (excluding voicelayer)
            try:
                children = subprocess.check_output(["pgrep", "-P", str(ppid)], text=True, timeout=2).strip().split("\n")
                # If parent only has MCP children, it's idle
                active_children = 0
                for child_pid in children:
                    if not child_pid.strip():
                        continue
                    try:
                        child_cmd = subprocess.check_output(
                            ["ps", "-p", child_pid.strip(), "-o", "command="], text=True, timeout=1
                        ).strip()
                        if not any(pat in child_cmd for pat in MCP_PATTERNS):
                            active_children += 1
                    except subprocess.SubprocessError:
                        pass
                if active_children == 0:
                    should_kill = True
            except subprocess.SubprocessError:
                # Parent has no children at all — stale
                should_kill = True

        if should_kill:
            try:
                os.kill(pid, signal.SIGTERM)
                killed += 1
            except (ProcessLookupError, PermissionError):
                pass

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
