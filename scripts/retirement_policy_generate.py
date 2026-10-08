"""Deterministic closed-policy generator; new sites require explicit purposes."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.retirement_inventory import inventory  # noqa: E402


def render(root, additions=None):
    policy = json.loads((root / "scripts/retirement_policy.json").read_text())
    if policy.get("schema_version") != 1 or not isinstance(policy.get("sites"), dict):
        raise ValueError("Unknown/missing policy schema")
    additions = additions or {}
    sites, missing = {}, []
    for key, site in inventory(root).items():
        previous = policy["sites"].get(key, {})
        if previous and previous.get("site") != site:
            raise ValueError("Policy site identity does not match its key: " + key)
        purpose = additions.get(key, previous.get("non_cloud_purpose"))
        if not isinstance(purpose, str) or not purpose.strip():
            missing.append({"key": key, "site": site})
        else:
            sites[key] = {"site": site, "non_cloud_purpose": purpose}
    if missing:
        raise ValueError("New sites require reviewed non-cloud purposes: " + json.dumps(missing, sort_keys=True))
    result = {**policy, "sites": sites}
    return json.dumps(result, indent=2, sort_keys=True) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--purposes", type=Path, help="Explicit site-key to reviewed non-cloud purpose JSON")
    args = parser.parse_args(argv)
    try:
        additions = json.loads(args.purposes.read_text()) if args.purposes else None
        if additions is not None and not isinstance(additions, dict):
            raise ValueError("Purposes must be a site-key mapping")
        generated = render(args.root, additions)
        path = args.root / "scripts/retirement_policy.json"
        if args.check:
            if path.read_text() != generated:
                raise ValueError("Policy is stale; review explicit purposes and run --write")
        else:
            path.write_text(generated)
    except (OSError, ValueError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
