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

# Prints -1 when $1 < $2, 0 when equal, 1 when $1 > $2. Exits non-zero when
# either side is not a dotted numeric version (callers must treat that as a
# hard mismatch rather than an allowed lag).
compare_versions() {
    python3 - "$1" "$2" <<'PY'
import sys


def parse(value):
    parts = value.split(".")
    if not value or not all(part.isdigit() for part in parts):
        raise SystemExit(2)
    return tuple(int(part) for part in parts)


left = parse(sys.argv[1])
right = parse(sys.argv[2])
print(-1 if left < right else (1 if left > right else 0))
PY
}

warn() {
    printf '[brainlayer-version-check] WARN: %s\n' "$*"
}

GIT_LOCAL_ENV_VARS=(
    GIT_ALTERNATE_OBJECT_DIRECTORIES
    GIT_CONFIG
    GIT_CONFIG_PARAMETERS
    GIT_CONFIG_COUNT
    GIT_OBJECT_DIRECTORY
    GIT_DIR
    GIT_WORK_TREE
    GIT_IMPLICIT_WORK_TREE
    GIT_GRAFT_FILE
    GIT_INDEX_FILE
    GIT_NO_REPLACE_OBJECTS
    GIT_REPLACE_REF_BASE
    GIT_PREFIX
    GIT_SHALLOW_FILE
    GIT_COMMON_DIR
)

git_package_root() {
    local -a env_args=()
    local name

    for name in "${GIT_LOCAL_ENV_VARS[@]}"; do
        env_args+=(-u "$name")
    done

    env "${env_args[@]}" git -C "$PACKAGE_ROOT" "$@"
}

latest_git_tag() {
    if [[ -n "${BRAINLAYER_VERSION_CHECK_GIT_TAG:-}" ]]; then
        printf '%s\n' "$BRAINLAYER_VERSION_CHECK_GIT_TAG"
        return
    fi
    if ! git_package_root rev-parse --git-dir >/dev/null 2>&1; then
        printf '\n'
        return
    fi
    git_package_root tag --list 'v[0-9]*.[0-9]*.[0-9]*' --sort=-v:refname 2>/dev/null | head -n 1
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
plist_bundle_version="$(read_plist_string "$INFO_PLIST" "CFBundleVersion")"
plist_release_version="$(read_plist_string "$INFO_PLIST" "BrainLayerReleaseVersion")"
cask_version="$(extract_cask_version "$CASK_PATH")"
git_tag="$(latest_git_tag)"
expected_git_tag="v$canonical_version"

if [[ "$canonical_version" =~ ^([0-9]+\.[0-9]+\.[0-9]+)(\.[0-9]+)?$ ]]; then
    expected_plist_short_version="${BASH_REMATCH[1]}"
else
    err "pyproject.toml version '$canonical_version' must match X.Y.Z or interim X.Y.Z.N"
    FAILED=1
    expected_plist_short_version="$canonical_version"
fi

require_equal "src/brainlayer/__init__.py __version__" "$init_version" "$canonical_version"
require_equal "server.json version" "$server_version" "$canonical_version"
require_equal "server.json packages[0].version" "$server_package_version" "$canonical_version"
require_equal "Info.plist CFBundleShortVersionString" "$plist_short_version" "$expected_plist_short_version"
require_equal "Info.plist BrainLayerReleaseVersion" "$plist_release_version" "$canonical_version"
if [[ ! "$plist_bundle_version" =~ ^[0-9]+(\.[0-9]+){0,2}$ ]]; then
    err "Info.plist CFBundleVersion '$plist_bundle_version' must contain one to three numeric components"
    FAILED=1
fi
cask_lag_reason="${BRAINLAYER_VERSION_CHECK_CASK_LAG_REASON:-}"
cask_lag_allowed=0
if [[ "$cask_version" == "$canonical_version" ]]; then
    :
elif [[ -n "$cask_lag_reason" ]] && cask_order="$(compare_versions "$cask_version" "$canonical_version" 2>/dev/null)" \
    && [[ "$cask_order" == "-1" ]]; then
    # BrainBar.app ships as its own GitHub-release artifact, so the cask may
    # legitimately trail the Python package when no Swift change was released.
    cask_lag_allowed=1
    warn "Casks/brainbar.rb is $cask_version (package $canonical_version) — allowed: $cask_lag_reason"
else
    require_equal "Casks/brainbar.rb version" "$cask_version" "$canonical_version"
fi
if [[ -z "$git_tag" ]]; then
    err "latest git tag could not be determined under $PACKAGE_ROOT"
    FAILED=1
else
    require_equal "latest git tag" "$git_tag" "$expected_git_tag"
fi

if [[ "$FAILED" -ne 0 ]]; then
    exit 1
fi

if [[ "$cask_lag_allowed" -eq 1 ]]; then
    printf '[brainlayer-version-check] PASS: BrainLayer/BrainBar %s release metadata is consistent (cask %s lag allowed: %s)\n' \
        "$canonical_version" "$cask_version" "$cask_lag_reason"
else
    printf '[brainlayer-version-check] PASS: BrainLayer/BrainBar %s release metadata is consistent\n' "$canonical_version"
fi
