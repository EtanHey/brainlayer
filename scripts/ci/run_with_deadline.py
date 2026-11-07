#!/usr/bin/env python3
"""Bound a gate command and report the last visible pytest node on timeout.

The command runs in its own session, so a signal aimed at this wrapper (Ctrl-C,
or the heavy-suite helper's INT/TERM/HUP) never reaches it by itself. The
wrapper forwards those signals to the command's process group, waits a bounded
grace, then SIGKILLs the group, and only returns once the leader is reaped:
releasing the heavy-suite lock with a live suite behind it is the bug (#1075).

The group is only ever signalled while its leader is unreaped. A zombie leader
pins the process-group id, so a recycled PID can never receive these signals.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time

PYTEST_NODE = re.compile(rb"(tests/[^\s]+::[^\s]+)")
FORWARDED_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
POLL_SECONDS = 0.05


def _exited(child, may_reap: bool) -> bool:
    """True once the leader has exited; reaps only when ``may_reap``."""
    if child.returncode is not None:
        return True
    if hasattr(os, "waitid"):
        try:
            flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
            return os.waitid(os.P_PID, child.pid, flags) is not None
        except ChildProcessError:
            return child.poll() is not None
    # Without waitid the leader cannot be observed without reaping it.
    return may_reap and child.poll() is not None


def _wait_leader(child, seconds: float, interrupted, may_reap: bool) -> bool:
    deadline = time.monotonic() + seconds
    while not _exited(child, may_reap):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or interrupted():
            return False
        time.sleep(min(POLL_SECONDS, remaining))
    return True


def _signal_group(child, signum: int) -> None:
    if child.returncode is not None:
        return  # reaped: the group id is no longer provably ours
    try:
        os.killpg(child.pid, signum)
    except (ProcessLookupError, PermissionError):
        pass  # only a zombie leader is left (macOS answers EPERM for that)


def _stop_group(child, signum: int, grace: float, escalate) -> None:
    """Signal the group, wait boundedly for the leader, SIGKILL the rest, reap."""
    _signal_group(child, signum)
    _wait_leader(child, grace, escalate, may_reap=False)
    _signal_group(child, signal.SIGKILL)  # leader still unreaped: pgid pinned
    child.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--grace", type=float, default=5.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.seconds <= 0 or args.grace < 0 or not command:
        parser.error("a positive --seconds, a non-negative --grace and a command are required")

    # Installed before spawning, so a signal racing the spawn is recorded, not lost.
    pending: list[int] = []
    previous = {signum: signal.signal(signum, lambda s, _f: pending.append(s)) for signum in FORWARDED_SIGNALS}
    try:
        if pending:
            return 128 + pending[0]
        return _run(args, command, pending)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _run(args: argparse.Namespace, command: list[str], pending: list[int]) -> int:
    child = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    assert child.stdout is not None
    last_node = b"unknown"
    tail = b""
    output_lock = threading.Lock()

    def forward() -> None:
        nonlocal last_node, tail
        while chunk := os.read(child.stdout.fileno(), 4096):
            with output_lock:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                tail = (tail + chunk)[-8192:]
                nodes = PYTEST_NODE.findall(tail)
                if nodes:
                    last_node = nodes[-1]

    reader = threading.Thread(target=forward, name="gate-output", daemon=True)
    reader.start()
    try:
        if _wait_leader(child, args.seconds, lambda: bool(pending), may_reap=True) and not pending:
            return child.wait()
        if pending:
            signum = pending[0]
            seen = len(pending)
            # A repeated signal during the grace escalates straight to SIGKILL.
            _stop_group(child, signum, args.grace, lambda: len(pending) > seen)
            print(
                f"ERROR: {args.label} interrupted by {signal.Signals(signum).name}; "
                "forwarded to the command's process group and reaped it",
                file=sys.stderr,
                flush=True,
            )
            return 128 + signum
        _stop_group(child, signal.SIGTERM, args.grace, lambda: bool(pending))
        reader.join(timeout=2)
        with output_lock:
            node = last_node.decode(errors="replace")
        print(
            f"ERROR: {args.label} exceeded {args.seconds:g}s; last visible pytest node: {node}; push blocked",
            file=sys.stderr,
            flush=True,
        )
        return 124
    finally:
        reader.join(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())
