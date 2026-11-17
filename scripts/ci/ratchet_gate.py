#!/usr/bin/env python3
"""Require a successful main-push ratchet attestation on the exact release SHA."""

import argparse
import json
import re
import subprocess


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
    pages = api(
        f"repos/{repo}/actions/workflows/ratchet-attest.yml/runs?head_sha={sha}&event=push&branch=main&per_page=100"
    )
    matching = []
    for page in pages:
        for run in page["workflow_runs"]:
            if run["head_sha"] != sha or run["event"] != "push" or run["head_branch"] != "main":
                continue
            matching.append(run)
            if run["status"] == "completed" and run["conclusion"] == "success":
                print(f"Ratchet attestation passed at {sha}: {run['html_url']}")
                return
    if not matching:
        reason = "missing main-push attestation"
    elif any(run["status"] != "completed" for run in matching):
        reason = "pending main-push attestation"
    else:
        reason = "failure: " + ", ".join(sorted({str(run["conclusion"]) for run in matching}))
    raise RuntimeError(f"Ratchet release refused for exact SHA {sha}: {reason}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    release = commands.add_parser("verify-release")
    release.add_argument("repo")
    release.add_argument("sha")
    args = parser.parse_args()
    verify_release(args.repo, args.sha)


if __name__ == "__main__":
    main()
