#!/usr/bin/env python3
"""Auto-store git commits into BrainLayer.

Install: brainlayer hooks install
Or manually: ln -sf $(pwd)/hooks/post-commit.py .git/hooks/post-commit

This hook runs after each `git commit` and stores the commit metadata
into BrainLayer for searchable history. Requires `brainlayer` CLI on PATH.
"""

import os
import subprocess
import sys


def main():
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        commit_msg = (
            subprocess.check_output(["git", "log", "-1", "--pretty=%B"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        files_changed = (
            subprocess.check_output(
                ["git", "diff-tree", "--no-commit-id", "-r", "--name-only", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
            .split("\n")
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not in a git repo or git not available — skip silently
        return

    # Detect project from repo root directory name
    try:
        repo_root = (
            subprocess.check_output(["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        project = os.path.basename(repo_root)
    except (subprocess.CalledProcessError, FileNotFoundError):
        project = None

    # Truncate file list for large commits
    files_display = files_changed[:10]
    if len(files_changed) > 10:
        files_display.append(f"... and {len(files_changed) - 10} more")

    content = f"Commit {commit_hash[:8]}: {commit_msg}\nFiles: {', '.join(files_display)}"

    # Build brainlayer store command
    cmd = ["brainlayer-store", "--type", "journal", "--content", content, "--tags", "commit,git"]
    if project:
        cmd.extend(["--project", project])

    try:
        subprocess.run(cmd, capture_output=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # brainlayer not installed or hanging — don't block the commit
        pass


if __name__ == "__main__":
    main()
