#!/usr/bin/env python3
"""Run a foreground release step in a disposable detached worktree, never a clone.

Usage: python3 scripts/release_worktree.py <tag-or-SHA> -- <command> [args...]
The command runs in the worktree; an exact {worktree} argument expands to its path.
No fetch, tag creation, push, hook configuration change, or force cleanup is implicit.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import uuid
from pathlib import Path


def git_env() -> dict[str, str]:
    # Hooks can export repository locators. Keep config/auth policy inherited,
    # but let -C and the child's cwd select the intended repository.
    locators = {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
        "GIT_PREFIX",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    }
    return {key: value for key, value in os.environ.items() if key not in locators}


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], env=git_env(), text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def run(repo: Path, ref: str, command: list[str]) -> int:
    common = Path(git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir")).resolve()
    canonical = common.parent
    if common.name != ".git" or Path(git(canonical, "rev-parse", "--show-toplevel")).resolve() != canonical:
        raise RuntimeError("Expected the canonical non-bare checkout with a .git directory")
    sha = git(repo, "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}")
    container = canonical / ".worktrees"
    if container.is_symlink():
        raise RuntimeError("Refusing a symlinked .worktrees directory")
    container.mkdir(exist_ok=True)
    worktree = container / f"release-{sha[:12]}-{uuid.uuid4().hex[:12]}"
    git(canonical, "worktree", "add", "--detach", str(worktree), sha)
    print(f"release-worktree: {worktree} at {sha}", file=sys.stderr, flush=True)
    code = 1
    try:
        argv = [str(worktree) if item == "{worktree}" else item for item in command]
        env = {**git_env(), "BRAINLAYER_RELEASE_WORKTREE": str(worktree), "BRAINLAYER_RELEASE_SHA": sha}
        try:
            code = subprocess.run(argv, cwd=worktree, env=env).returncode
            if code < 0:
                code = 128 - code
        except OSError as exc:
            print(f"release-worktree: command could not start: {exc}", file=sys.stderr)
            code = 127
        except KeyboardInterrupt:
            code = 130
    finally:
        # Never force: a tracked edit or untracked failure artifact belongs to
        # the operator. A clean worktree's ignored build output goes with it.
        try:
            git(canonical, "worktree", "remove", str(worktree))
        except RuntimeError as exc:
            print(f"release-worktree: preserved {worktree}; cleanup refused: {exc}", file=sys.stderr)
            if code == 0:
                code = 1
        else:
            print(f"release-worktree: removed {worktree}", file=sys.stderr)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("ref")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a foreground command after -- is required")
    try:
        return run(args.repo, args.ref, command)
    except (OSError, RuntimeError) as exc:
        print(f"release-worktree: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
