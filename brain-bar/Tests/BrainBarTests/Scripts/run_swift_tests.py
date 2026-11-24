#!/usr/bin/env python3
"""Run Swift tests in an owned TMPDIR and reject leaked brainbar-* fixtures.

Arguments pass through unchanged to `swift test`, including --filter and --skip.
With no arguments, the full suite runs.

Output is relayed line by line from an unbuffered xctest, so a hang shows the test
it hung in. After STALL_SECONDS without any output, the runner samples every live
xctest process once per stall, so the log carries each thread's stack.
"""

import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# Single UI tests legitimately run up to ~51 s on a slow runner (CI, 24 Nov), so a
# shorter threshold spends the samples before a real hang. A false sample only prints stacks.
STALL_SECONDS = 75
MAX_STALL_SAMPLES = 3


def sample_xctest(directory, number):
    pids = subprocess.run(["pgrep", "-x", "xctest"], capture_output=True, text=True).stdout.split()
    if not pids:
        print("BrainBar stall watchdog: no xctest process to sample", flush=True)
    for pid in pids:
        report = Path(directory, f"stall-sample-{number}-{pid}.txt")
        print(f"BrainBar stall watchdog: no output for {STALL_SECONDS}s; sampling xctest pid {pid}", flush=True)
        subprocess.run(["sample", pid, "3", "-file", str(report)], capture_output=True, check=False)
        if report.exists():
            print(report.read_text(errors="replace"), flush=True)
            report.unlink()
        print(f"BrainBar stall watchdog: end of sample for pid {pid}", flush=True)


def run_relayed(command, env, directory):
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    last_output = [time.monotonic()]

    def relay():
        for line in iter(process.stdout.readline, b""):
            last_output[0] = time.monotonic()
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()

    reader = threading.Thread(target=relay, daemon=True)
    reader.start()
    samples = 0
    while process.poll() is None:
        time.sleep(5)
        if samples < MAX_STALL_SAMPLES and time.monotonic() - last_output[0] >= STALL_SECONDS:
            samples += 1
            sample_xctest(directory, samples)
            last_output[0] = time.monotonic()
    reader.join()
    return process.returncode


def main():
    # macOS Unix socket paths have a small fixed limit; nesting under its long default
    # /var/folders TMPDIR makes the existing private daemon fixtures exceed it.
    with tempfile.TemporaryDirectory(prefix="brainbar-suite-", suffix=".noindex", dir="/tmp") as directory:
        root = Path(directory)
        before = set(root.glob("brainbar-*"))
        returncode = run_relayed(
            [
                "swift",
                "test",
                "--scratch-path",
                str(Path(__file__).resolve().parents[4] / ".build.noindex/swift"),
                *sys.argv[1:],
            ],
            {**os.environ, "TMPDIR": directory + "/", "BRAINBAR_TEST_TMP_ROOT": directory, "NSUnbufferedIO": "YES"},
            directory,
        )
        leaked = sorted(set(root.glob("brainbar-*")) - before)
        print(f"BrainBar suite-owned temp gate: before={len(before)} new={len(leaked)}", flush=True)
        if leaked:
            print("BrainBar fixture leaks in suite-owned TMPDIR:", file=sys.stderr)
            for path in leaked:
                print(path.name, file=sys.stderr)
        return returncode or bool(leaked)


if __name__ == "__main__":
    sys.exit(main())
