#!/usr/bin/env python3
"""Run Swift tests in an owned TMPDIR and reject leaked brainbar-* fixtures."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    # macOS Unix socket paths have a small fixed limit; nesting under its long default
    # /var/folders TMPDIR makes the existing private daemon fixtures exceed it.
    with tempfile.TemporaryDirectory(prefix="brainbar-suite-", suffix=".noindex", dir="/tmp") as directory:
        root = Path(directory)
        before = set(root.glob("brainbar-*"))
        result = subprocess.run(
            ["swift", "test", *sys.argv[1:]],
            env={**os.environ, "TMPDIR": directory + "/", "BRAINBAR_TEST_TMP_ROOT": directory},
        )
        leaked = sorted(set(root.glob("brainbar-*")) - before)
        print(f"BrainBar suite-owned temp gate: before={len(before)} new={len(leaked)}", flush=True)
        if leaked:
            print("BrainBar fixture leaks in suite-owned TMPDIR:", file=sys.stderr)
            for path in leaked:
                print(path.name, file=sys.stderr)
        return result.returncode or bool(leaked)


if __name__ == "__main__":
    sys.exit(main())
