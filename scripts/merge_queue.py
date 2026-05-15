#!/usr/bin/env python3
"""Union an AirDrop'd BrainLayer queue directory into the live queue.

Invocation:
  python3 scripts/merge_queue.py ~/Downloads/queue
  python3 scripts/merge_queue.py ~/Downloads/queue --dry-run
  python3 scripts/merge_queue.py ~/Downloads/queue --dest ~/.brainlayer/queue

After a real merge, kick the drain LaunchAgent:
  launchctl kickstart -k gui/$(id -u)/com.brainlayer.drain

This script is append-only. It never mirrors, deletes, or overwrites queue files.
Byte-identical JSONL files are skipped so re-runs are no-ops.
Same-name/different-content files are copied under a deterministic -merge-<hash>
name so both queue events survive for the drain daemon.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brainlayer.queue_io import get_queue_dir  # noqa: E402
from brainlayer.queue_merge import merge_queue_dirs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Union an AirDrop'd BrainLayer queue into the live queue.")
    parser.add_argument("source", type=Path, help="AirDrop'd queue directory, for example ~/Downloads/queue")
    parser.add_argument("--dest", type=Path, default=None, help="Destination queue dir (default: ~/.brainlayer/queue)")
    parser.add_argument("--dry-run", action="store_true", help="List planned actions without writing files")
    args = parser.parse_args()

    dest = args.dest.expanduser() if args.dest else get_queue_dir()
    result = merge_queue_dirs(args.source, dest, dry_run=args.dry_run)
    mode = "DRY-RUN" if args.dry_run else "MERGED"
    print(f"{mode}: source={args.source.expanduser()} dest={dest}")
    for name in result.copied:
        print(f"copy {name}")
    for name in result.collisions:
        print(f"collision-renamed {name}")
    for name in result.skipped_exact:
        print(f"skip-exact {name}")
    for name in result.skipped_non_jsonl:
        print(f"skip-non-jsonl {name}")
    print(
        "summary "
        f"copied={len(result.copied)} "
        f"collisions={len(result.collisions)} "
        f"skipped_exact={len(result.skipped_exact)} "
        f"skipped_non_jsonl={len(result.skipped_non_jsonl)}"
    )
    if not args.dry_run and result.copied:
        print("next: launchctl kickstart -k gui/$(id -u)/com.brainlayer.drain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
