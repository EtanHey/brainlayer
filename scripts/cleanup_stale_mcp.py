#!/usr/bin/env python3
"""Kill orphaned brainlayer-mcp and voicelayer-mcp processes.

An MCP process is considered stale if:
  1. Its parent PID is 1 (reparented to init/launchd — parent died)
  2. It has been running longer than MAX_AGE_HOURS and its parent
     claude process has no child node process (session ended)

Safe: never kills processes whose parent claude is actively running
a node/bun child (sign of an active Claude Code session).

Usage:
    python3 scripts/cleanup_stale_mcp.py          # dry-run (default)
    python3 scripts/cleanup_stale_mcp.py --kill    # actually kill
    python3 scripts/cleanup_stale_mcp.py --json    # JSON output for hooks
"""

import argparse
import json
import os
import signal
import subprocess
import sys

MCP_PATTERNS = ("brainlayer-mcp", "voicelayer-mcp")
MAX_AGE_HOURS = 6  # processes older than this are candidates even with live parents


def _set_max_age(hours):
    global MAX_AGE_HOURS
    MAX_AGE_HOURS = hours


def get_mcp_processes():
    """Return list of dicts with pid, ppid, etime, command for MCP processes."""
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid,ppid,etime,command"],
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return []

    results = []
    for line in out.strip().split("\n")[1:]:  # skip header
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, ppid, etime, cmd = int(parts[0]), int(parts[1]), parts[2], parts[3]
        if any(pat in cmd for pat in MCP_PATTERNS):
            # Don't include ourselves
            if pid == os.getpid():
                continue
            results.append(
                {
                    "pid": pid,
                    "ppid": ppid,
                    "etime": etime,
                    "command": cmd,
                    "age_seconds": parse_etime(etime),
                }
            )
    return results


def parse_etime(etime_str):
    """Parse ps etime format (DD-HH:MM:SS or HH:MM:SS or MM:SS) to seconds."""
    days = 0
    if "-" in etime_str:
        day_part, etime_str = etime_str.split("-", 1)
        days = int(day_part)

    parts = etime_str.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
    elif len(parts) == 2:
        hours, minutes, seconds = 0, int(parts[0]), int(parts[1])
    else:
        return 0

    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def parent_has_active_session(ppid):
    """Check if ppid (a claude process) has active node/bun children (active session)."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-P", str(ppid), "-lf", "node|bun"],
            text=True,
            timeout=3,
        )
        # Filter out voicelayer-mcp (bun) — those don't count as active session indicators
        for line in out.strip().split("\n"):
            if line and "voicelayer-mcp" not in line:
                return True
        return False
    except subprocess.SubprocessError:
        return False


def classify_stale(procs):
    """Classify processes as stale or active. Returns (stale, active) lists."""
    stale = []
    active = []
    age_threshold = MAX_AGE_HOURS * 3600

    for p in procs:
        reason = None

        # Rule 1: orphaned (ppid=1)
        if p["ppid"] == 1:
            reason = "orphaned (ppid=1)"

        # Rule 2: very old and parent has no active session
        elif p["age_seconds"] > age_threshold:
            if not parent_has_active_session(p["ppid"]):
                reason = f"stale (>{MAX_AGE_HOURS}h, parent idle)"

        if reason:
            p["reason"] = reason
            stale.append(p)
        else:
            active.append(p)

    return stale, active


def kill_processes(stale, dry_run=True):
    """Kill stale processes. Returns count killed."""
    killed = 0
    for p in stale:
        if dry_run:
            print(f"  [DRY-RUN] Would kill PID {p['pid']}: {p['reason']}")
        else:
            try:
                os.kill(p["pid"], signal.SIGTERM)
                killed += 1
                print(f"  Killed PID {p['pid']}: {p['reason']}")
            except ProcessLookupError:
                pass  # already dead
            except PermissionError:
                print(f"  Permission denied for PID {p['pid']}", file=sys.stderr)
    return killed


def main():
    parser = argparse.ArgumentParser(description="Clean up stale MCP processes")
    parser.add_argument("--kill", action="store_true", help="Actually kill (default: dry-run)")
    parser.add_argument("--json", action="store_true", help="JSON output (for hooks)")
    parser.add_argument(
        "--max-age", type=int, default=MAX_AGE_HOURS, help=f"Max age in hours (default: {MAX_AGE_HOURS})"
    )
    args = parser.parse_args()

    _set_max_age(args.max_age)

    procs = get_mcp_processes()
    stale, active = classify_stale(procs)

    if args.json:
        print(
            json.dumps(
                {
                    "stale_count": len(stale),
                    "active_count": len(active),
                    "killed": len(stale) if args.kill else 0,
                    "stale": [{"pid": p["pid"], "reason": p["reason"], "age_s": p["age_seconds"]} for p in stale],
                }
            )
        )
        if args.kill:
            kill_processes(stale, dry_run=False)
        return

    print(f"MCP processes: {len(procs)} total, {len(stale)} stale, {len(active)} active")
    if stale:
        kill_processes(stale, dry_run=not args.kill)
        if not args.kill:
            print(f"\nRun with --kill to actually terminate {len(stale)} processes.")
    else:
        print("No stale processes found.")


if __name__ == "__main__":
    main()
