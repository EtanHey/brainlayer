#!/usr/bin/env python3
"""Run a foreground release step in a disposable detached worktree, never a clone.

Usage: python3 scripts/release_worktree.py <tag-or-SHA> -- <command> [args...]
The command runs in the worktree; an exact {worktree} argument expands to its path.
No fetch, tag creation, push, hook configuration change, or force cleanup is implicit.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

SIGNAL_GRACE_SECONDS = 15  # Inner pre-push supervisors need 10 seconds plus kill/reap time.


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


def signal_group(pgid: int, signum: int) -> None:
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        pass
    except PermissionError:
        if group_running(pgid):
            raise  # A live or unmeasurable group must not be silently ignored.


def group_running(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # Darwin can return EPERM for a group containing only unreaped zombies.
    # Orphaned zombies can outlive the command until launchd/init reaps them.
    # They cannot execute a push or access the checkout. Unknown state is live.
    try:
        processes = subprocess.run(["ps", "-axo", "pgid=,stat="], text=True, capture_output=True, timeout=1)
    except (OSError, subprocess.TimeoutExpired):
        return True
    if processes.returncode:
        return True
    for row in processes.stdout.splitlines():
        parts = row.split()
        if len(parts) == 2 and parts[0] == str(pgid) and not parts[1].startswith("Z"):
            return True
    return False


def wait_for_group(child: subprocess.Popen, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while True:
        child.poll()  # Reap the direct child as well as checking its descendants.
        if not group_running(child.pid):
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                return False
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


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
    child = None
    received = 0

    def forward(signum, frame):
        nonlocal received
        received = received or signum
        if child is not None:
            signal_group(child.pid, signum)

    previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        argv = [str(worktree) if item == "{worktree}" else item for item in command]
        env = {**git_env(), "BRAINLAYER_RELEASE_WORKTREE": str(worktree), "BRAINLAYER_RELEASE_SHA": sha}
        try:
            child = subprocess.Popen(argv, cwd=worktree, env=env, start_new_session=True)
            if received:  # A signal may arrive between installing handlers and Popen returning.
                signal_group(child.pid, received)
            while child.poll() is None and not received:
                time.sleep(0.05)
            if received:
                if not wait_for_group(child, SIGNAL_GRACE_SECONDS):
                    print("release-worktree: signal grace expired; killing command group", file=sys.stderr)
                    signal_group(child.pid, signal.SIGKILL)
                    wait_for_group(child, 2)
                code = 128 + received
            else:
                code = child.returncode if child.returncode >= 0 else 128 - child.returncode
        except OSError as exc:
            print(f"release-worktree: command could not start: {exc}", file=sys.stderr)
            code = 127
    finally:
        # Never force: a tracked edit or untracked failure artifact belongs to
        # the operator. A clean worktree's ignored build output goes with it.
        try:
            if child is not None and group_running(child.pid):
                print(
                    f"release-worktree: preserved {worktree}; command group still running or unmeasurable",
                    file=sys.stderr,
                )
                if code == 0:
                    code = 3
            else:
                try:
                    git(canonical, "worktree", "remove", str(worktree))
                except RuntimeError as exc:
                    print(f"release-worktree: preserved {worktree}; cleanup refused: {exc}", file=sys.stderr)
                    if code == 0:
                        code = 3
                else:
                    print(f"release-worktree: removed {worktree}", file=sys.stderr)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return 128 + received if received else code


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
