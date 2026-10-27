#!/usr/bin/env python3
"""Select PR sweeps and require a full exact-SHA batch receipt before release."""

import argparse
import fnmatch
import json
import os
import re
import subprocess
from pathlib import Path

SENSITIVE = (
    "src/brainlayer/maintenance*.py",
    "src/brainlayer/*supervisor*.py",
    "src/brainlayer/watcher*.py",
    "src/brainlayer/vector_store.py",
    "src/brainlayer/pipeline/**",
    "src/brainlayer/backup_daily.py",
    "src/brainlayer/*migration*.py",
    "src/brainlayer/launchd*.py",
    "scripts/launchd/**",
    "tests/test_launchd*.py",
    "tests/test_migration*.py",
    "scripts/ci/**",
    ".github/workflows/ci.yml",
    "tests/test_ci_crash_sweeps.py",
    "tests/test_crash_sweep_gate.py",
)


def required(event, files, labels):
    return (
        event != "pull_request"
        or "crash-sweep" in labels
        or any(fnmatch.fnmatchcase(path, pattern) for path in files for pattern in SENSITIVE)
    )


def api(endpoint):
    result = subprocess.run(
        ["gh", "api", "--paginate", "--slurp", endpoint],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return json.loads(result.stdout)


def verify_release(repo, sha):
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise RuntimeError("release requires a full commit SHA")
    pages = api(f"repos/{repo}/actions/workflows/ci.yml/runs?head_sha={sha}&status=success&per_page=100")
    for page in pages:
        for run in page["workflow_runs"]:
            if not (
                run["head_sha"] == sha
                and run["conclusion"] == "success"
                and (run["event"] == "workflow_dispatch" or (run["event"] == "push" and run["head_branch"] == "main"))
            ):
                continue
            jobs = api(f"repos/{repo}/actions/runs/{run['id']}/attempts/{run['run_attempt']}/jobs?per_page=100")
            for job_page in jobs:
                for job in job_page["jobs"]:
                    if job["name"] == "crash sweeps" and job["head_sha"] == sha and job["conclusion"] == "success":
                        print(f"Full crash sweep passed at {sha}: {job['html_url']}")
                        return
    raise RuntimeError(f"No successful full crash sweep for exact batch SHA {sha}; run CI on that ref before release")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("required")
    release = commands.add_parser("verify-release")
    release.add_argument("repo")
    release.add_argument("sha")
    release.add_argument("--tag", action="store_true", help="peel a local tag object to its commit")
    args = parser.parse_args()
    if args.command == "verify-release":
        sha = (
            subprocess.check_output(["git", "rev-parse", f"{args.sha}^{{commit}}"], text=True, timeout=60).strip()
            if args.tag
            else args.sha
        )
        verify_release(args.repo, sha)
        return
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    files, labels = [], []
    if os.environ["GITHUB_EVENT_NAME"] == "pull_request":
        pr = event["pull_request"]
        files = subprocess.check_output(
            ["git", "diff", "--name-only", "--no-renames", f"{pr['base']['sha']}...{pr['head']['sha']}", "--"],
            text=True,
            timeout=60,
        ).splitlines()
        labels = [label["name"] for label in pr["labels"]]
    value = str(required(os.environ["GITHUB_EVENT_NAME"], files, labels)).lower()
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"required={value}\n")
    print(f"Full crash sweep required={value}")


if __name__ == "__main__":
    main()
