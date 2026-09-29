#!/usr/bin/env python3
"""Bound a gate command and report the last visible pytest node on timeout."""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import threading

PYTEST_NODE = re.compile(rb"(tests/[^\s]+::[^\s]+)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.seconds <= 0 or not command:
        parser.error("a positive --seconds and a command are required")

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
        return child.wait(timeout=args.seconds)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
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
