#!/usr/bin/env python3
"""Reject tracked KG exports and raw artifact JSONL files by path only."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable


def matching_tracked_exports(paths: Iterable[str]) -> list[str]:
    """Return paths matching the private-export boundaries, without opening files."""
    matches = (
        path
        for path in paths
        if (path.startswith("eval_results/") and Path(path).name.startswith("kg-") and path.endswith(".json"))
        or (path.startswith("artifacts/") and path.endswith(".jsonl"))
    )
    return sorted(set(matches))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo_root)
    paths = tracked.decode("utf-8", errors="surrogateescape").split("\0")
    offending = matching_tracked_exports(path for path in paths if path)
    if not offending:
        print("No tracked privacy export paths.")
        return 0

    print("Tracked privacy export paths are not allowed:", file=sys.stderr)
    for path in offending:
        print(path, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
