"""Resource-only manifest transitions, pinned inside the keg restart journal."""

import hashlib
import json
import plistlib
import sys

from .consolidation_plan import read_owned
from .supervisor_jobs import RESOURCE_POLICY_KEYS, resource_policy


def transition(old, new):
    before, after = json.loads(old), json.loads(new)
    expected, diff = json.loads(old), {}
    if before.keys() != after.keys():
        raise ValueError("policy reconcile cannot change component inventory")
    for label, record in before.items():
        prior, requested = resource_policy(record["plist"]), resource_policy(after[label]["plist"])
        changes = {
            key: {"old": prior.get(key), "new": requested.get(key)}
            for key in RESOURCE_POLICY_KEYS
            if prior.get(key) != requested.get(key)
        }
        if changes:
            diff[label] = changes
        target = expected[label]["plist"]
        for key in RESOURCE_POLICY_KEYS:
            target.pop(key, None)
        target.update(requested)
    if expected != after:
        raise ValueError("policy reconcile changed non-policy configuration")
    return dict(
        old_bytes=old.hex(),
        new_bytes=new.hex(),
        old_sha256=hashlib.sha256(old).hexdigest(),
        new_sha256=hashlib.sha256(new).hexdigest(),
        diff=diff,
    )


def plan(previous, templates):
    old = bytes.fromhex(previous["manifest_bytes"])
    if hashlib.sha256(old).hexdigest() != previous["manifest_sha256"]:
        raise ValueError("invalid manifest custody hash")
    document = json.loads(old)
    for label, record in document.items():
        try:
            raw = read_owned(templates / f"{label}.plist")
        except FileNotFoundError:
            print(f"{label}: policy unchanged: no keg template", file=sys.stderr)
            continue
        template = plistlib.loads(raw)
        if template.get("Label") != label:
            raise ValueError(f"policy template identity changed: {label}")
        if sys.platform != "darwin":
            # These launchd templates carry Darwin-only resource contracts.
            # Preserve a runnable POSIX preimage instead of importing them.
            print(f"{label}: policy unchanged: non-Darwin platform {sys.platform}", file=sys.stderr)
            continue
        requested = resource_policy(template)
        for key in RESOURCE_POLICY_KEYS:
            record["plist"].pop(key, None)
        record["plist"].update(requested)
    result = transition(old, json.dumps(document).encode())
    return result if result["diff"] else None


def versions(restarting):
    """Validate journal intent independently of today's templates during recovery."""
    previous = restarting["restart_preimage"]
    old = bytes.fromhex(previous["manifest_bytes"])
    if hashlib.sha256(old).hexdigest() != previous["manifest_sha256"]:
        raise ValueError("invalid manifest custody hash")
    pending = restarting.get("policy_reconcile")
    new = bytes.fromhex(pending["new_bytes"]) if pending else old
    if pending and transition(old, new) != pending:
        raise ValueError("invalid journaled policy transition")
    return old, new


def log_diff(diff):
    for label, changes in diff.items():
        for key, values in changes.items():
            print(f"{label}: {key} {json.dumps(values['old'])} → {json.dumps(values['new'])}", file=sys.stderr)
