#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: brain-bar/Scripts/dev-preview.sh <worktree-or-branch>
       brain-bar/Scripts/dev-preview.sh --all
       brain-bar/Scripts/dev-preview.sh --clean

Build and open UI-only BrainBar DEV previews. The installed BrainBar is never replaced.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PREVIEW_ROOT="${BRAINBAR_DEV_PREVIEW_ROOT:-$HOME/Applications/BrainBar DEV}"
PLIST_BUDDY="${BRAINBAR_PLIST_BUDDY:-/usr/libexec/PlistBuddy}"
OPEN_BIN="${BRAINBAR_DEV_OPEN_BIN:-/usr/bin/open}"

safe_name() {
    printf '%s' "$1" | sed 's#[[:space:]/]#-#g; s/[^A-Za-z0-9._-]/-/g'
}

find_worktree_for_branch() {
    local wanted="$1" path="" branch=""
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            worktree\ *) path="${line#worktree }" ;;
            branch\ refs/heads/*) branch="${line#branch refs/heads/}" ;;
            "")
                if [ "$branch" = "$wanted" ]; then
                    printf '%s\n' "$path"
                    return 0
                fi
                path=""
                branch=""
                ;;
        esac
    done < <(git -C "$REPO_ROOT" worktree list --porcelain)
    if [ "$branch" = "$wanted" ]; then
        printf '%s\n' "$path"
        return 0
    fi
    return 1
}

resolve_worktree() {
    local target="$1" resolved common expected_common
    if [ -d "$target" ]; then
        resolved="$(git -C "$target" rev-parse --show-toplevel 2>/dev/null || true)"
    else
        resolved="$(find_worktree_for_branch "$target" || true)"
    fi
    if [ -z "$resolved" ]; then
        echo "[dev-preview] ERROR: no checked-out worktree for '$target'" >&2
        return 1
    fi
    common="$(cd "$resolved" && git rev-parse --path-format=absolute --git-common-dir)"
    expected_common="$(cd "$REPO_ROOT" && git rev-parse --path-format=absolute --git-common-dir)"
    if [ "$common" != "$expected_common" ]; then
        echo "[dev-preview] ERROR: $resolved is not a BrainLayer worktree" >&2
        return 1
    fi
    printf '%s\n' "$resolved"
}

verify_preview_bundle() {
    local app="$1" branch="$2" sha="$3"
    local plist="$app/Contents/Info.plist"
    local preview stamped_branch stamped_sha bundle_id
    preview="$($PLIST_BUDDY -c 'Print :BrainBarDevPreview' "$plist" 2>/dev/null || true)"
    stamped_branch="$($PLIST_BUDDY -c 'Print :BrainBarDevBranch' "$plist" 2>/dev/null || true)"
    stamped_sha="$($PLIST_BUDDY -c 'Print :GitCommit' "$plist" 2>/dev/null || true)"
    bundle_id="$($PLIST_BUDDY -c 'Print :CFBundleIdentifier' "$plist" 2>/dev/null || true)"
    if [ "$preview" != "true" ] || [ "$stamped_branch" != "$branch" ] || [ "$stamped_sha" != "$sha" ] ||
       [[ "$bundle_id" != com.brainlayer.brainbar.dev.* ]]; then
        echo "[dev-preview] ERROR: unsafe or stale DEV bundle stamp at $app" >&2
        return 1
    fi
    if [ -e "$app/Contents/MacOS/BrainBarDaemon" ] || [ -d "$app/Contents/Resources/LaunchAgents" ]; then
        echo "[dev-preview] ERROR: DEV bundle contains daemon or LaunchAgent payloads: $app" >&2
        return 1
    fi
}

build_one() {
    local worktree branch sha short safe branch_hash app
    worktree="$(resolve_worktree "$1")"
    branch="$(git -C "$worktree" branch --show-current)"
    if [ -z "$branch" ]; then
        branch="detached-$(git -C "$worktree" rev-parse --short HEAD)"
    fi
    sha="$(git -C "$worktree" rev-parse HEAD)"
    short="${sha:0:8}"
    safe="$(safe_name "$branch")"
    branch_hash="$(printf '%s' "$branch" | shasum -a 256 | cut -c1-8)"
    app="$PREVIEW_ROOT/BrainBar DEV · $safe-$branch_hash.app"
    mkdir -p "$PREVIEW_ROOT"

    BRAINBAR_DEV_APP_DIR="$app" bash "$worktree/brain-bar/build-app.sh" \
        --force-worktree-build --force-dirty
    verify_preview_bundle "$app" "$branch" "$sha"
    "$OPEN_BIN" -n "$app"
    printf 'PREVIEW\t%s\t%s\t%s\n' "$branch" "$short" "$app"
}

build_all() {
    local repo number branch has_brainbar worktree built=0
    repo="$(gh repo view --json nameWithOwner --jq .nameWithOwner)"
    while IFS=$'\t' read -r number branch; do
        [ -n "$number" ] || continue
        has_brainbar="$(gh pr view "$number" --repo "$repo" --json files --jq \
            '[.files[].path | startswith("brain-bar/")] | any')"
        [ "$has_brainbar" = "true" ] || continue
        worktree="$(find_worktree_for_branch "$branch" || true)"
        if [ -z "$worktree" ]; then
            echo "[dev-preview] WARNING: PR #$number branch '$branch' has no checked-out worktree; skipped" >&2
            continue
        fi
        build_one "$worktree"
        built=$((built + 1))
    done < <(gh pr list --repo "$repo" --state open --limit 100 --json number,headRefName \
        --jq '.[] | [.number, .headRefName] | @tsv')
    if [ "$built" -eq 0 ]; then
        echo "[dev-preview] ERROR: no checked-out open PRs touching brain-bar/" >&2
        return 1
    fi
}

clean_all() {
    local app removed=0
    for app in "$PREVIEW_ROOT"/BrainBar\ DEV\ ·\ *.app; do
        [ -d "$app" ] || continue
        rm -rf "$app"
        printf 'REMOVED\t%s\n' "$app"
        removed=$((removed + 1))
    done
    for app in "$HOME"/Applications/BrainBar-DEV-*.app; do
        [ -d "$app" ] || continue
        rm -rf "$app"
        printf 'REMOVED\t%s\n' "$app"
        removed=$((removed + 1))
    done
    [ "$removed" -gt 0 ] || echo "[dev-preview] No DEV previews found under $PREVIEW_ROOT"
}

case "${1:-}" in
    --all) [ "$#" -eq 1 ] || { usage >&2; exit 2; }; build_all ;;
    --clean) [ "$#" -eq 1 ] || { usage >&2; exit 2; }; clean_all ;;
    -h|--help) usage ;;
    "") usage >&2; exit 2 ;;
    -*) echo "[dev-preview] ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) [ "$#" -eq 1 ] || { usage >&2; exit 2; }; build_one "$1" ;;
esac
