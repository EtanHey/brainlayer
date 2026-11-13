"""Per-PR review scratch must not be committed at the repository root.

22 such files (19 BUGBOT_*.md, bugbot-review-notes.md, SLICE-B-BRIEF.md, progress.txt)
accumulated there between 2026-03 and 2026-06; nothing referenced them.
"""

from __future__ import annotations

import fnmatch
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRATCH_PATTERNS = ("BUGBOT_*.md", "bugbot-review-notes.md", "SLICE-*-BRIEF.md", "progress.txt")


def _clean_git_env() -> dict[str, str]:
    # An inherited GIT_DIR/GIT_WORK_TREE (a git hook runs the suite) would list the wrong repo.
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def test_no_review_scratch_tracked_at_root():
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, check=True, capture_output=True, text=True, env=_clean_git_env()
    ).stdout.splitlines()
    root_files = [path for path in tracked if "/" not in path]
    offenders = sorted(
        path for path in root_files if any(fnmatch.fnmatchcase(path, pattern) for pattern in SCRATCH_PATTERNS)
    )
    assert offenders == [], f"review scratch committed at repo root: {offenders}"
