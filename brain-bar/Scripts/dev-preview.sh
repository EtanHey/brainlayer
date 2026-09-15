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
CANONICAL_REPO_ROOT="${BRAINBAR_CANONICAL_REPO_ROOT:-$HOME/Gits/brainlayer}"
PREVIEW_ROOT="${BRAINBAR_DEV_PREVIEW_ROOT:-$HOME/Applications/BrainBar DEV}"
PLIST_BUDDY="${BRAINBAR_PLIST_BUDDY:-/usr/libexec/PlistBuddy}"
OPEN_BIN="${BRAINBAR_DEV_OPEN_BIN:-/usr/bin/open}"
OVERLAY_WORKTREE=""
OVERLAY_TEMP_ROOT=""

cleanup_overlay() {
    if [ -n "$OVERLAY_WORKTREE" ]; then
        git -C "$REPO_ROOT" worktree remove --force "$OVERLAY_WORKTREE" >/dev/null 2>&1 || true
        OVERLAY_WORKTREE=""
    fi
    if [ -n "$OVERLAY_TEMP_ROOT" ] && [ -d "$OVERLAY_TEMP_ROOT" ]; then
        rmdir "$OVERLAY_TEMP_ROOT" >/dev/null 2>&1 || true
        OVERLAY_TEMP_ROOT=""
    fi
}

trap cleanup_overlay EXIT

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
    local target="$1" resolved common expected_common canonical
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
    resolved="$(cd "$resolved" && pwd -P)"
    if [ -d "$CANONICAL_REPO_ROOT" ]; then
        canonical="$(cd "$CANONICAL_REPO_ROOT" && pwd -P)"
        if [ "$resolved" = "$canonical" ]; then
            echo "[dev-preview] ERROR: refusing canonical repo root as a preview source: $resolved" >&2
            return 1
        fi
    fi
    printf '%s\n' "$resolved"
}

verify_preview_bundle() {
    local app="$1" branch="$2" sha="$3" harness_sha="$4"
    local plist="$app/Contents/Info.plist"
    local preview stamped_branch stamped_sha stamped_harness bundle_id
    preview="$($PLIST_BUDDY -c 'Print :BrainBarDevPreview' "$plist" 2>/dev/null || true)"
    stamped_branch="$($PLIST_BUDDY -c 'Print :BrainBarDevBranch' "$plist" 2>/dev/null || true)"
    stamped_sha="$($PLIST_BUDDY -c 'Print :GitCommit' "$plist" 2>/dev/null || true)"
    stamped_harness="$($PLIST_BUDDY -c 'Print :BrainBarDevHarnessCommit' "$plist" 2>/dev/null || true)"
    bundle_id="$($PLIST_BUDDY -c 'Print :CFBundleIdentifier' "$plist" 2>/dev/null || true)"
    if [ "$preview" != "true" ] || [ "$stamped_branch" != "$branch" ] || [ "$stamped_sha" != "$sha" ] ||
       [ "$stamped_harness" != "$harness_sha" ] ||
       [[ "$bundle_id" != com.brainlayer.brainbar.dev.* ]]; then
        echo "[dev-preview] ERROR: unsafe or stale DEV bundle stamp at $app" >&2
        return 1
    fi
    if [ -e "$app/Contents/MacOS/BrainBarDaemon" ] || [ -d "$app/Contents/Resources/LaunchAgents" ]; then
        echo "[dev-preview] ERROR: DEV bundle contains daemon or LaunchAgent payloads: $app" >&2
        return 1
    fi
}

prepare_overlay_worktree() {
    local target_worktree="$1" target_sha="$2"
    local harness_commit harness_base
    if [ -n "$(git -C "$target_worktree" status --porcelain --untracked-files=all)" ]; then
        echo "[dev-preview] ERROR: target lacks the preview harness and is dirty; refusing to omit uncommitted source" >&2
        return 1
    fi
    if [ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]; then
        echo "[dev-preview] ERROR: preview harness worktree is dirty; commit and review it before overlaying feature source" >&2
        return 1
    fi
    harness_commit="$(git -C "$REPO_ROOT" log --diff-filter=A --format=%H -1 -- brain-bar/Scripts/dev-preview.sh)"
    if [ -z "$harness_commit" ]; then
        echo "[dev-preview] ERROR: cannot locate the preview harness introduction commit" >&2
        return 1
    fi
    harness_base="$(git -C "$REPO_ROOT" rev-parse "$harness_commit^")"
    OVERLAY_TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/brainbar-dev-preview.XXXXXX")"
    OVERLAY_WORKTREE="$OVERLAY_TEMP_ROOT/checkout"
    git -C "$REPO_ROOT" worktree add --detach "$OVERLAY_WORKTREE" "$target_sha" >/dev/null
    if ! git -C "$REPO_ROOT" diff --binary "$harness_base..HEAD" -- brain-bar tests/test_brainbar_build_app_guards.py |
        git -C "$OVERLAY_WORKTREE" apply --index --3way -; then
        echo "[dev-preview] ERROR: preview harness does not apply cleanly to $target_sha" >&2
        return 1
    fi
}

build_one() {
    local worktree build_source branch sha describe short safe branch_hash app harness_sha
    worktree="$(resolve_worktree "$1")"
    branch="$(git -C "$worktree" branch --show-current)"
    if [ -z "$branch" ]; then
        branch="detached-$(git -C "$worktree" rev-parse --short HEAD)"
    fi
    sha="$(git -C "$worktree" rev-parse HEAD)"
    describe="$(git -C "$worktree" describe --always --dirty)"
    harness_sha="$(git -C "$REPO_ROOT" rev-parse HEAD)"
    short="${sha:0:8}"
    safe="$(safe_name "$branch")"
    branch_hash="$(printf '%s' "$branch" | shasum -a 256 | cut -c1-8)"
    app="$PREVIEW_ROOT/BrainBar DEV · $safe-$branch_hash.app"
    mkdir -p "$PREVIEW_ROOT"

    build_source="$worktree"
    if ! grep -q 'BrainBarDevHarnessCommit' "$worktree/brain-bar/build-app.sh"; then
        prepare_overlay_worktree "$worktree" "$sha"
        build_source="$OVERLAY_WORKTREE"
    fi
    BRAINBAR_DEV_APP_DIR="$app" \
        BRAINBAR_DEV_SOURCE_BRANCH="$branch" \
        BRAINBAR_DEV_SOURCE_COMMIT="$sha" \
        BRAINBAR_DEV_SOURCE_DESCRIBE="$describe" \
        BRAINBAR_DEV_HARNESS_COMMIT="$harness_sha" \
        bash "$build_source/brain-bar/build-app.sh" \
        --force-worktree-build --force-dirty
    verify_preview_bundle "$app" "$branch" "$sha" "$harness_sha"
    cleanup_overlay
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
