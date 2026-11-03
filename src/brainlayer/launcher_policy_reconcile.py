"""Journal launcher plist transitions that change policy keys only."""

import hashlib
import plistlib

POLICY_KEYS = frozenset({"ProcessType", "Nice", "LowPriorityIO", "SoftResourceLimits"})


def _transition(previous, prepared):
    """Return a verifiable policy-only transition, or raise for any other edit."""
    files = {}
    diff = {}
    for label, new in prepared.items():
        name = f"{label}.plist"
        old = bytes.fromhex(previous["published_files"][name])
        before, after = plistlib.loads(old), plistlib.loads(new)
        if before.get("Label") != label or after.get("Label") != label:
            raise ValueError(f"launcher identity changed: {label}")
        old_nonpolicy = {key: value for key, value in before.items() if key not in POLICY_KEYS}
        new_nonpolicy = {key: value for key, value in after.items() if key not in POLICY_KEYS}
        if old_nonpolicy != new_nonpolicy:
            raise ValueError("launcher configuration changed; explicit replan required")
        changes = {
            key: {"old": before.get(key), "new": after.get(key)}
            for key in sorted(POLICY_KEYS)
            if before.get(key) != after.get(key)
        }
        if changes:
            files[name] = {
                "old_bytes": old.hex(),
                "new_bytes": new.hex(),
                "old_sha256": hashlib.sha256(old).hexdigest(),
                "new_sha256": hashlib.sha256(new).hexdigest(),
            }
            diff[label] = changes
    if not diff:
        return None
    return {"files": files, "diff": diff}


def transition(previous, prepared):
    """Collect policy-only changes per launcher.

    A launcher whose non-policy keys differ from its published bytes is left out, not refused:
    the existing journal checks own that case (a restart fallback pins a different daemon than
    the requested configuration, and a real configuration change must still refuse there).
    """
    files, diff = {}, {}
    for label, new in prepared.items():
        try:
            one = _transition(previous, {label: new})
        except ValueError as failure:
            if "explicit replan required" in str(failure):
                continue
            raise
        if one:
            files.update(one["files"])
            diff.update(one["diff"])
    return {"files": files, "diff": diff} if diff else None


def validate(pending):
    """Validate journal custody without consulting mutable templates."""
    if set(pending) != {"files", "diff"} or set(pending["files"]) != {f"{label}.plist" for label in pending["diff"]}:
        raise ValueError("invalid launcher policy journal")
    for name, item in pending["files"].items():
        old, new = bytes.fromhex(item["old_bytes"]), bytes.fromhex(item["new_bytes"])
        if hashlib.sha256(old).hexdigest() != item["old_sha256"]:
            raise ValueError("invalid old launcher policy hash")
        if hashlib.sha256(new).hexdigest() != item["new_sha256"]:
            raise ValueError("invalid new launcher policy hash")
        label = name.removesuffix(".plist")
        candidate = _transition({"published_files": {name: old.hex()}}, {label: new})
        if candidate != {"files": {name: item}, "diff": {label: pending["diff"][label]}}:
            raise ValueError("invalid launcher policy transition")
