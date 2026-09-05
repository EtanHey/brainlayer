#!/usr/bin/env python3
"""Prove a release tag actually CONTAINS the fixes the release claims.

The incident this exists to prevent: ``v1.5.15`` was tagged at ``51a72a06`` (2026-09-05
20:12:29). #778 -- the index runtime watchdog -- merged at ``78d92bcb`` 20:34:55, **22 minutes
later**. A deploy brief, a collab post, and a report to Etan all said "1.5.15 carries #778", and a
whole M1 deploy was cut for a fix the release did not contain. Nobody lied: the claim was a MEMORY
of a merge that happened the same evening. The fix is to make the claim CHECKABLE, not to try
harder to remember.

Usage:
    scripts/release_tag_contains.py v1.5.16 778 '#779' <sha> [--assert-module brainlayer.index_watchdog]

Exit status:
    0  every claimed item is in the tag, and every --assert-module imported
    1  the gate failed -- something claimed is NOT IN, or an assert-module raised
    2  usage/environment error -- unknown tag, not a git repo, bad arguments

Environment overrides (tests and non-default installs):
    BRAINLAYER_GH_BIN      gh executable used to resolve a PR number (default: gh)
    BRAINLAYER_KEG_PYTHON  installed keg python used by --assert-module
                           (default: /opt/homebrew/opt/brainlayer/libexec/venv/bin/python)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass

EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_USAGE = 2

DEFAULT_KEG_PYTHON = "/opt/homebrew/opt/brainlayer/libexec/venv/bin/python"
# A bare number is a PR; a SHA is hex and long. `#` always wins, and the 6-digit ceiling keeps
# an all-digit abbreviated SHA from being sent to `gh pr view` (PR numbers here are 3 digits).
PR_CLAIM = re.compile(r"^(?:#(\d+)|(\d{1,6}))$")
NOT_IN_REPO = "(commit not in this repo -- fetch it before claiming it)"


def _scrubbed_git_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def _run(args: list[str], cwd: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """A missing executable is a FAILED check, never a traceback that reads as a crash."""
    try:
        return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False, env=env)
    except OSError as error:
        return subprocess.CompletedProcess(args, 127, "", f"{args[0]}: {error.strerror}")


def _git(args: list[str], cwd: str) -> subprocess.CompletedProcess[str]:
    """Always answer for ``cwd``, never for an inherited repo.

    Git exports ``GIT_DIR``/``GIT_INDEX_FILE`` into hooks and into ``git rebase --exec``, and they
    win over the working directory. A release gate that silently answered for the WRONG repository
    would be the same class of unfalsifiable claim this script exists to close, so scrub them.
    """
    return _run(["git", *args], cwd, env=_scrubbed_git_env())


def _escape(cell: str) -> str:
    """A commit subject may contain ``|``; the table must survive it."""
    return cell.replace("|", r"\|")


@dataclass
class Claim:
    label: str  # what the operator claimed, verbatim (#778, or a sha)
    sha: str | None  # resolved commit, if there is one
    subject: str
    contained: bool

    @property
    def status(self) -> str:
        return "IN" if self.contained else "NOT IN"


def resolve_tag(tag: str, cwd: str) -> str:
    result = _git(["rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}"], cwd)
    if result.returncode != 0 or not result.stdout.strip():
        raise SystemExit_(f"ERROR: no such tag or revision: {tag}")
    return result.stdout.strip()


class SystemExit_(Exception):
    """Usage error -- exits 2, never 1, so a broken invocation is never read as a clean gate."""


def pr_merge_commit(number: str, cwd: str) -> tuple[str | None, str]:
    """Resolve a PR number to its merge commit. Returns ``(sha_or_None, subject)``."""
    gh_bin = os.environ.get("BRAINLAYER_GH_BIN", "gh")
    result = _run([gh_bin, "pr", "view", number, "--json", "mergeCommit,title"], cwd)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        reason = detail[0] if detail else "no output"
        return None, f"(gh pr view #{number} failed: {reason})"
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None, f"(gh pr view #{number} returned unparseable JSON)"
    title = str(payload.get("title") or "").strip()
    merge_commit = payload.get("mergeCommit") or {}
    oid = merge_commit.get("oid") if isinstance(merge_commit, dict) else None
    if not oid:
        return None, f"{title} (PR not merged)"
    return str(oid), title


def evaluate(claim: str, tag_sha: str, cwd: str) -> Claim:
    match = PR_CLAIM.match(claim)
    if match:
        number = match.group(1) or match.group(2)
        label = f"#{number}"
        sha, subject = pr_merge_commit(number, cwd)
        if sha is None:
            return Claim(label, None, subject, contained=False)
    else:
        label = claim
        resolved = _git(["rev-parse", "--verify", "--quiet", f"{claim}^{{commit}}"], cwd)
        if resolved.returncode != 0 or not resolved.stdout.strip():
            return Claim(label, None, NOT_IN_REPO, contained=False)
        sha = resolved.stdout.strip()
        subject = ""

    if _git(["cat-file", "-e", f"{sha}^{{commit}}"], cwd).returncode != 0:
        return Claim(label, sha, NOT_IN_REPO, contained=False)

    if not subject:
        subject = _git(["log", "-1", "--format=%s", sha], cwd).stdout.strip()
    contained = _git(["merge-base", "--is-ancestor", sha, tag_sha], cwd).returncode == 0
    return Claim(label, sha, subject, contained)


def previous_release_tag(tag: str, cwd: str) -> str | None:
    """The predecessor must be a FULL release -- same policy as ``.githooks/pre-push``.

    ``--match 'v*'`` keeps a nightly/``archive/*`` tag from starting the range, and
    ``--exclude '*-*'`` keeps a pre-release (``v1.2.3-rc1``) from doing the same one name up.
    Over-scoping from the last full release is safe; under-scoping a release listing is not.
    """
    result = _git(
        ["describe", "--tags", "--abbrev=0", "--match", "v*", "--exclude", "*-*", f"{tag}^"],
        cwd,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout.strip()


def print_carries(tag: str, tag_sha: str, cwd: str) -> None:
    """Unprompted: what the tag ACTUALLY carries, so a release lists that instead of a memory."""
    previous = previous_release_tag(tag, cwd)
    print()
    if previous is None:
        print(f"WARNING: {tag} has no previous full-release tag (v*, non-pre-release) to diff from.")
        print("WARNING: cannot list what it carries — say so in the receipt rather than guessing.")
        return
    log = _git(["log", "--format=%h %s", f"{previous}..{tag_sha}"], cwd)
    commits = [line for line in log.stdout.splitlines() if line.strip()]
    print(f"## {tag} carries ({previous}..{tag}, {len(commits)} commits)")
    for line in commits:
        print(f"- {line}")


def assert_modules(modules: list[str], cwd: str) -> bool:
    """The other end of the check: does the INSTALLED keg actually have the module?

    ``import brainlayer.index_watchdog`` -> ``ModuleNotFoundError`` is exactly what a keg built
    from a tag that predates the fix answers.
    """
    keg_python = os.environ.get("BRAINLAYER_KEG_PYTHON", DEFAULT_KEG_PYTHON)
    print()
    print(f"## installed-keg module asserts ({keg_python})")
    all_ok = True
    for module in modules:
        result = _run([keg_python, "-c", f"import {module}"], cwd)
        if result.returncode == 0:
            print(f"import {module} | OK")
            continue
        all_ok = False
        detail = (result.stderr or result.stdout).strip().splitlines()
        reason = detail[-1] if detail else f"exit {result.returncode}"
        print(f"import {module} | FAILED — {reason}")
    return all_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="release_tag_contains.py",
        description="Prove a release tag contains every fix the release claims.",
    )
    parser.add_argument("tag", help="the release tag, e.g. v1.5.16")
    parser.add_argument("claims", nargs="*", help="PR numbers (778, #778) or commit SHAs")
    parser.add_argument(
        "--assert-module",
        dest="assert_module",
        action="append",
        default=[],
        metavar="MODULE",
        help="importable name the INSTALLED keg python must import (repeatable)",
    )
    parser.add_argument("--repo", default=".", help="repository to check (default: cwd)")
    args = parser.parse_args(argv)

    cwd = args.repo
    if _git(["rev-parse", "--git-dir"], cwd).returncode != 0:
        print(f"ERROR: not a git repository: {cwd}", file=sys.stderr)
        return EXIT_USAGE
    try:
        tag_sha = resolve_tag(args.tag, cwd)
    except SystemExit_ as error:
        print(str(error), file=sys.stderr)
        return EXIT_USAGE

    print(f"# tag-contains gate — {args.tag} ({tag_sha[:12]})")
    print()
    claims = [evaluate(claim, tag_sha, cwd) for claim in args.claims]
    if not claims:
        print("(no claims given — reporting what the tag carries only)")
    else:
        print("| claim | sha | subject | status |")
        print("| --- | --- | --- | --- |")
        for claim in claims:
            sha_cell = claim.sha[:12] if claim.sha else "—"
            print(f"| {_escape(claim.label)} | {sha_cell} | {_escape(claim.subject)} | {claim.status} |")

    missing = [claim for claim in claims if not claim.contained]
    print_carries(args.tag, tag_sha, cwd)

    modules_ok = assert_modules(args.assert_module, cwd) if args.assert_module else True

    print()
    if missing:
        print(f"FAIL: {args.tag} does NOT contain {len(missing)} of {len(claims)} claimed items.")
        print("FAIL: do not name them in a release receipt or a deploy brief.")
    if not modules_ok:
        print("FAIL: the installed keg could not import a module this release claims to ship.")
    if missing or not modules_ok:
        return EXIT_GATE_FAILED
    if claims:
        print(f"OK: {args.tag} contains all {len(claims)} claimed items.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
