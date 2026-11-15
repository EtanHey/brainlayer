#!/usr/bin/env python3
"""Public-input gates for the dispatch-only approval helper release. No credentials."""

import hashlib
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
        refuse("missing required secrets: " + ", ".join(missing))
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
