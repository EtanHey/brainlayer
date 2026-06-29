#!/usr/bin/env bash
# Release metadata consistency guard for BrainLayer/BrainBar.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="${BRAINLAYER_VERSION_CHECK_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
FAILED=0

default_tap_root() {
    local sibling
    sibling="$(cd "$PACKAGE_ROOT/.." && pwd)/homebrew-layers"
    if [[ -d "$sibling" ]]; then
        printf '%s\n' "$sibling"
    fi
}

TAP_ROOT="${BRAINLAYER_VERSION_CHECK_TAP_ROOT:-${BRAINLAYER_HOMEBREW_TAP_ROOT:-$(default_tap_root)}}"

err() {
    printf '[brainlayer-version-check] ERROR: %s\n' "$*" >&2
}

require_file() {
    local label="$1"
    local path="$2"
    if [[ ! -f "$path" ]]; then
        err "$label not found: $path"
        FAILED=1
    fi
}

require_equal() {
    local label="$1"
    local actual="$2"
    local expected="$3"
    if [[ "$actual" != "$expected" ]]; then
        err "$label is '$actual', expected '$expected'"
        FAILED=1
    fi
}

read_pyproject_version() {
    python3 - "$1" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as handle:
    print(tomllib.load(handle)["project"]["version"])
PY
}

read_init_version() {
    python3 - "$1" <<'PY'
import ast
import sys

module = ast.parse(open(sys.argv[1], encoding="utf-8").read())
for node in module.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__version__":
                print(ast.literal_eval(node.value))
                raise SystemExit(0)
raise SystemExit("__version__ not found")
PY
}

read_server_value() {
    python3 - "$1" "$2" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
field = sys.argv[2]
if field == "version":
    print(manifest.get("version", ""))
elif field == "packages[0].version":
    print((manifest.get("packages") or [{}])[0].get("version", ""))
else:
    raise SystemExit(f"unknown field: {field}")
PY
}

read_plist_string() {
    python3 - "$1" "$2" <<'PY'
import plistlib
import sys

with open(sys.argv[1], "rb") as handle:
    print(plistlib.load(handle).get(sys.argv[2], ""))
PY
}

extract_cask_version() {
    awk '
        /^[[:space:]]*version "/ {
            gsub(/"/, "", $2)
            print $2
            exit
        }
    ' "$1"
}

latest_git_tag() {
    if [[ -n "${BRAINLAYER_VERSION_CHECK_GIT_TAG:-}" ]]; then
        printf '%s\n' "$BRAINLAYER_VERSION_CHECK_GIT_TAG"
        return
    fi
    if ! git -C "$PACKAGE_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
        printf '\n'
        return
    fi
    git -C "$PACKAGE_ROOT" tag --list 'v[0-9]*.[0-9]*.[0-9]*' --sort=-v:refname 2>/dev/null | head -n 1
}

if [[ -z "$TAP_ROOT" ]]; then
    err "Homebrew tap root is required. Set BRAINLAYER_VERSION_CHECK_TAP_ROOT or BRAINLAYER_HOMEBREW_TAP_ROOT."
    exit 2
fi

PYPROJECT="$PACKAGE_ROOT/pyproject.toml"
INIT_PY="$PACKAGE_ROOT/src/brainlayer/__init__.py"
SERVER_JSON="$PACKAGE_ROOT/server.json"
INFO_PLIST="$PACKAGE_ROOT/brain-bar/bundle/Info.plist"
CASK_PATH="$TAP_ROOT/Casks/brainbar.rb"

require_file "pyproject.toml" "$PYPROJECT"
require_file "src/brainlayer/__init__.py" "$INIT_PY"
require_file "server.json" "$SERVER_JSON"
require_file "brain-bar/bundle/Info.plist" "$INFO_PLIST"
require_file "Homebrew cask" "$CASK_PATH"

if [[ "$FAILED" -ne 0 ]]; then
    exit 1
fi

canonical_version="$(read_pyproject_version "$PYPROJECT")"
init_version="$(read_init_version "$INIT_PY")"
server_version="$(read_server_value "$SERVER_JSON" "version")"
server_package_version="$(read_server_value "$SERVER_JSON" "packages[0].version")"
plist_short_version="$(read_plist_string "$INFO_PLIST" "CFBundleShortVersionString")"
cask_version="$(extract_cask_version "$CASK_PATH")"
git_tag="$(latest_git_tag)"
expected_git_tag="v$canonical_version"

require_equal "src/brainlayer/__init__.py __version__" "$init_version" "$canonical_version"
require_equal "server.json version" "$server_version" "$canonical_version"
require_equal "server.json packages[0].version" "$server_package_version" "$canonical_version"
require_equal "Info.plist CFBundleShortVersionString" "$plist_short_version" "$canonical_version"
require_equal "Casks/brainbar.rb version" "$cask_version" "$canonical_version"
if [[ -z "$git_tag" ]]; then
    err "latest git tag could not be determined under $PACKAGE_ROOT"
    FAILED=1
else
    require_equal "latest git tag" "$git_tag" "$expected_git_tag"
fi

if [[ "$FAILED" -ne 0 ]]; then
    exit 1
fi

printf '[brainlayer-version-check] PASS: BrainLayer/BrainBar %s release metadata is consistent\n' "$canonical_version"
