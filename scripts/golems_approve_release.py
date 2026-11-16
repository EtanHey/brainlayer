#!/usr/bin/env python3
"""Public-input gates for the dispatch-only approval helper release. No credentials."""

import hashlib
import json
import os
import plistlib
import re
import subprocess
import sys
from pathlib import Path

PROFILE_SHA256 = "7287dac8a8a9b3308da5b96428471db1afb100ee6bf26250f2d5ff79082eef4f"
TEAM = "PPN23G925Y"
IDENTIFIER = "com.golems.approve"
ENTITLEMENTS = {
    "com.apple.application-identifier": f"{TEAM}.{IDENTIFIER}",
    "com.apple.developer.team-identifier": TEAM,
    "keychain-access-groups": [f"{TEAM}.{IDENTIFIER}"],
}
SECRETS = (
    "BRAINBAR_DEVELOPER_ID_APPLICATION",
    "BRAINBAR_DEVELOPER_ID_CERTIFICATE_BASE64",
    "BRAINBAR_DEVELOPER_ID_CERTIFICATE_PASSWORD",
    "BRAINBAR_NOTARY_APPLE_ID",
    "BRAINBAR_NOTARY_PASSWORD",
    "BRAINBAR_NOTARY_TEAM_ID",
)


def refuse(message):
    raise SystemExit(f"Release refused: {message}")


def sha(value):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        refuse("golems_sha must be a full 40-hex commit SHA")
    return value.lower()


def merged_sha(root, value):
    value = sha(value)
    # Fetch the full master history even from checkout's default shallow SHA checkout.
    shallow = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--is-shallow-repository"], capture_output=True, text=True, timeout=30
    )
    if shallow.returncode:
        refuse("cannot inspect golems checkout")
    command = ["git", "-C", str(root), "fetch", "--no-tags"]
    if shallow.stdout.strip() == "true":
        command.append("--unshallow")
    command += ["origin", "+refs/heads/master:refs/remotes/origin/master"]
    fetched = subprocess.run(command, capture_output=True, timeout=120)
    if fetched.returncode:
        refuse("cannot freshly fetch EtanHey/golems master")
    result = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", value, "refs/remotes/origin/master"],
        capture_output=True,
        timeout=30,
    )
    if result.returncode:
        refuse("golems_sha is not an ancestor of freshly fetched EtanHey/golems master")


def environment_policy(metadata, policies):
    if metadata.get("name") != "developer-id":
        refuse("developer-id environment missing")
    required = [r for r in metadata.get("protection_rules", []) if r.get("type") == "required_reviewers"]
    expected_reviewers = [("User", "etanhey")]
    if len(required) != 1 or required[0].get("prevent_self_review") is not True:
        refuse("developer-id requires owner review with self-review prevented")
    reviewers = [
        (r.get("type"), r.get("reviewer", {}).get("login", "").lower()) for r in required[0].get("reviewers", [])
    ]
    if reviewers != expected_reviewers:
        refuse("developer-id requires EtanHey as the sole reviewer")
    if metadata.get("deployment_branch_policy") != {"protected_branches": False, "custom_branch_policies": True}:
        refuse("developer-id requires a main-only deployment branch policy")
    if [(p.get("name"), p.get("type")) for p in policies] != [("main", "branch")]:
        refuse("developer-id deployment branches must be main only (no tags)")


def environment_ready():
    # GET only. Never auto-create an environment or read secret values.
    endpoint = "repos/EtanHey/brainlayer/environments/developer-id"

    def get(path):
        response = subprocess.run(["gh", "api", "--paginate", "--slurp", path], capture_output=True, timeout=60)
        if response.returncode:
            refuse("developer-id protected environment unavailable; owner setup or metadata access required")
        try:
            return json.loads(response.stdout)
        except (ValueError, TypeError):
            refuse("developer-id environment metadata unreadable")

    try:
        metadata_pages = get(endpoint)
        policy_pages = get(endpoint + "/deployment-branch-policies?per_page=100")
        if len(metadata_pages) != 1:
            refuse("developer-id environment metadata ambiguous")
        policies = [p for page in policy_pages for p in page["branch_policies"]]
        environment_policy(metadata_pages[0], policies)
    except (KeyError, TypeError, AttributeError):
        refuse("developer-id environment metadata incomplete")


def profile(path):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != PROFILE_SHA256:
        refuse("provisioning profile SHA256 mismatch")


def committed_files(root):
    root = Path(root)
    for name in ("GolemsApprove.entitlements", "Info.plist"):
        path = root / name
        if not path.is_file() or path.is_symlink():
            refuse(f"missing committed native/golems-approve/{name}")
    try:
        entitlements = plistlib.loads((root / "GolemsApprove.entitlements").read_bytes())
        info = plistlib.loads((root / "Info.plist").read_bytes())
    except Exception:
        refuse("invalid committed signing plist")
    if entitlements != ENTITLEMENTS:
        refuse("committed entitlements must exactly match the provisioned identity; no runtime relaxations")
    for key, value in {
        "CFBundleIdentifier": IDENTIFIER,
        "CFBundleExecutable": "golems-approve",
        "CFBundlePackageType": "APPL",
    }.items():
        if info.get(key) != value:
            refuse(f"committed Info.plist has invalid {key}")


def secrets():
    # Report names only, never values or subprocess errors carrying arguments.
    missing = [name for name in SECRETS if not os.environ.get(name)]
    if missing:
        refuse("developer-id missing required secrets: " + ", ".join(missing))
    if os.environ["BRAINBAR_NOTARY_TEAM_ID"] != TEAM:
        refuse("notary team does not match pinned profile team")


def verify(app):
    app = Path(app)
    profile(app / "Contents/embedded.provisionprofile")
    committed_files(app / "Contents/Resources")
    if (app / "Contents/Info.plist").read_bytes() != (app / "Contents/Resources/Info.plist").read_bytes():
        refuse("bundle Info.plist differs from committed Info.plist")
    result = subprocess.run(["codesign", "-d", "--entitlements", ":-", str(app)], capture_output=True)
    try:
        actual = plistlib.loads(result.stdout)
    except Exception:
        refuse("cannot decode signed entitlements")
    if result.returncode or actual != ENTITLEMENTS:
        refuse("signed entitlements differ from committed entitlements")
    requirement = f'anchor apple generic and certificate leaf[subject.OU] = "{TEAM}" and identifier "{IDENTIFIER}"'
    subprocess.run(["codesign", "--verify", "--deep", "--strict", "-R", "=" + requirement, str(app)], check=True)
    # Unknown argument refuses after signature validation, before stdin parsing, LAContext or key access.
    result = subprocess.run(
        [str(app / "Contents/MacOS/golems-approve"), "--ci-no-keychain-probe"],
        input=b"",
        capture_output=True,
        timeout=15,
    )
    if result.returncode != 2 or result.stdout or not result.stderr.startswith(b"golems-approve refused:"):
        refuse("signed no-keychain launch probe failed (output withheld)")
    print("Signed helper launched and refused the no-keychain probe as expected")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "sha":
        print(sha(sys.argv[2]))
    elif mode == "merged-sha":
        merged_sha(sys.argv[2], sys.argv[3])
    elif mode == "environment":
        environment_ready()
    elif mode == "profile":
        profile(sys.argv[2])
    elif mode == "source":
        committed_files(sys.argv[2])
    elif mode == "secrets":
        secrets()
    elif mode == "verify":
        verify(sys.argv[2])
    else:
        refuse("unknown mode")
