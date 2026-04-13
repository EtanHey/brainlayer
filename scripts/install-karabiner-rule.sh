#!/usr/bin/env bash
# Merge BrainBar Karabiner complex modification into ~/.config/karabiner/karabiner.json
# without overwriting existing rules. Requires Python 3.
#
# Usage: bash scripts/install-karabiner-rule.sh
# From repo root, or any cwd (script resolves paths relative to repo).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RULE_JSON="$REPO_ROOT/brain-bar/karabiner/brainbar-f4.json"
KARABINER_JSON="${HOME}/.config/karabiner/karabiner.json"

if [[ ! -f "$RULE_JSON" ]]; then
  echo "install-karabiner-rule: missing rule file: $RULE_JSON" >&2
  exit 1
fi

RULE_JSON="$RULE_JSON" KARABINER_JSON="$KARABINER_JSON" python3 <<'PY'
import json
import os

rule_path = os.environ["RULE_JSON"]
kb_path = os.path.expanduser(os.environ["KARABINER_JSON"])

with open(rule_path, encoding="utf-8") as f:
    new_rule = json.load(f)

if os.path.isfile(kb_path):
    with open(kb_path, encoding="utf-8") as f:
        data = json.load(f)
else:
    data = {"global": {}, "profiles": []}

if "profiles" not in data or not data["profiles"]:
    data["profiles"] = [
        {
            "name": "Default profile",
            "complex_modifications": {"rules": []},
        }
    ]

prof = data["profiles"][0]
if "complex_modifications" not in prof:
    prof["complex_modifications"] = {}
if "rules" not in prof["complex_modifications"]:
    prof["complex_modifications"]["rules"] = []

rules = prof["complex_modifications"]["rules"]
desc = new_rule.get("description", "")
if any(r.get("description") == desc for r in rules):
    print(f"Already installed (skip): {desc}")
else:
    rules.append(new_rule)
    print(f"Added: {desc}")

os.makedirs(os.path.dirname(kb_path), exist_ok=True)
with open(kb_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
    f.write("\n")

print(f"Wrote: {kb_path}")
print("Open Karabiner-Elements and enable the rule under Complex modifications.")
PY
