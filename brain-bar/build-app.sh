#!/usr/bin/env bash
# Build BrainBar as a proper macOS .app bundle.
#
# Usage: bash brain-bar/build-app.sh [--dry-run] [--force-worktree-build] [--force-dirty]
#
# Output: ~/Applications/BrainBar.app (override with BRAINBAR_APP_DIR)

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash brain-bar/build-app.sh [--dry-run] [--force-worktree-build] [--force-dirty]

Options:
  --dry-run               Validate guards and print the resolved app path without building
  --force-worktree-build  Allow non-canonical repo builds, but route them to a DEV app bundle
  --force-dirty           Allow builds from a dirty tree after explicit review
EOF
}

DRY_RUN=0
FORCE_WORKTREE_BUILD=0
FORCE_DIRTY=0
DEV_BUNDLE_BUILD=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            ;;
        --force-worktree-build)
            FORCE_WORKTREE_BUILD=1
            ;;
        --force-dirty)
            FORCE_DIRTY=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[build-app] ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGE_DIR="$SCRIPT_DIR"
BUNDLE_DIR="$SCRIPT_DIR/bundle"
SIGN_IDENTITY="${BRAINBAR_CODESIGN_IDENTITY:-Apple Development: Etan Heyman (DXHB5E7P2D)}"
UI_PLIST_LABEL="com.brainlayer.brainbar"
DAEMON_PLIST_LABEL="com.brainlayer.brainbar-daemon"
UI_PLIST_FILENAME="$UI_PLIST_LABEL.plist"
DAEMON_PLIST_FILENAME="$DAEMON_PLIST_LABEL.plist"
UI_PLIST_SRC="$BUNDLE_DIR/$UI_PLIST_FILENAME"
DAEMON_PLIST_SRC="$BUNDLE_DIR/$DAEMON_PLIST_FILENAME"
UI_PLIST_DST="$HOME/Library/LaunchAgents/$UI_PLIST_FILENAME"
DAEMON_PLIST_DST="$HOME/Library/LaunchAgents/$DAEMON_PLIST_FILENAME"
LAUNCH_DOMAIN="gui/$(id -u)"
SOCKET_PATH="${BRAINBAR_SOCKET_PATH:-/tmp/brainbar.sock}"
PLIST_BUDDY="${BRAINBAR_PLIST_BUDDY:-/usr/libexec/PlistBuddy}"
LSREGISTER="${BRAINBAR_LSREGISTER:-/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A/Support/lsregister}"
CANONICAL_REPO_ROOT="${BRAINBAR_CANONICAL_REPO_ROOT:-$HOME/Gits/brainlayer}"
BRAINLAYER_LOG_DIR="$HOME/Library/Logs/brainlayer"

if [ -d "$CANONICAL_REPO_ROOT" ]; then
    CANONICAL_REPO_ROOT="$(cd "$CANONICAL_REPO_ROOT" && pwd -P)"
fi

CURRENT_REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$CURRENT_REPO_ROOT" ]; then
    echo "[build-app] ERROR: build-app.sh must run from a git checkout" >&2
    exit 1
fi
CURRENT_REPO_ROOT="$(cd "$CURRENT_REPO_ROOT" && pwd -P)"

resolve_branch_name() {
    local branch
    branch="$(git -C "$CURRENT_REPO_ROOT" rev-parse --abbrev-ref HEAD)"
    if [ "$branch" = "HEAD" ]; then
        branch="detached-$(git -C "$CURRENT_REPO_ROOT" rev-parse --short HEAD)"
    fi
    printf '%s\n' "$branch"
}

sanitize_branch_name() {
    printf '%s' "$1" | sed 's#[[:space:]/]#-#g; s/[^A-Za-z0-9._-]/-/g'
}

dev_bundle_apps_dir() {
    printf '%s\n' "$HOME/Applications"
}

safe_branch_is_checked_out_anywhere() {
    local safe_branch="$1"
    local bundle_sha="${2:-}"
    local detached_sha=""
    case "$safe_branch" in
        detached-*)
            detached_sha="${safe_branch#detached-}"
            ;;
    esac

    local line
    local head_sha=""
    local has_branch=0
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            HEAD\ *)
                head_sha="${line#HEAD }"
                ;;
            branch\ refs/heads/*)
                has_branch=1
                local branch
                branch="${line#branch refs/heads/}"
                if [ "$(sanitize_branch_name "$branch")" = "$safe_branch" ]; then
                    return 0
                fi
                ;;
            "")
                if [ -n "$detached_sha" ] && [ "${head_sha#"$detached_sha"}" != "$head_sha" ]; then
                    return 0
                fi
                if [ "$has_branch" -eq 0 ] && [ -n "$bundle_sha" ] && [ "$head_sha" = "$bundle_sha" ]; then
                    return 0
                fi
                head_sha=""
                has_branch=0
                ;;
        esac
    done < <(git -C "$CURRENT_REPO_ROOT" worktree list --porcelain)

    if [ -n "$detached_sha" ] && [ "${head_sha#"$detached_sha"}" != "$head_sha" ]; then
        return 0
    fi
    if [ "$has_branch" -eq 0 ] && [ -n "$bundle_sha" ] && [ "$head_sha" = "$bundle_sha" ]; then
        return 0
    fi
    return 1
}

safe_branch_ref_exists() {
    local safe_branch="$1"
    local ref
    while IFS= read -r ref; do
        local branch
        case "$ref" in
            refs/heads/*)
                branch="${ref#refs/heads/}"
                ;;
            refs/remotes/origin/*)
                branch="${ref#refs/remotes/origin/}"
                ;;
            *)
                continue
                ;;
        esac
        if [ "$(sanitize_branch_name "$branch")" = "$safe_branch" ]; then
            return 0
        fi
    done < <(git -C "$CURRENT_REPO_ROOT" for-each-ref --format='%(refname)' refs/heads refs/remotes/origin 2>/dev/null)
    return 1
}

bundle_mtime_epoch() {
    local bundle="$1"
    stat -c %Y "$bundle" 2>/dev/null || stat -f %m "$bundle"
}

dev_bundle_age_days() {
    local bundle="$1"
    echo $(( ($(date +%s) - $(bundle_mtime_epoch "$bundle")) / 86400 ))
}

read_bundle_git_commit() {
    local bundle="$1"
    "$PLIST_BUDDY" -c "Print :GitCommit" "$bundle/Contents/Info.plist" 2>/dev/null || true
}

resolve_dev_bundle_branch() {
    local safe_branch="$1"
    local ref
    while IFS= read -r ref; do
        local branch
        case "$ref" in
            refs/heads/*)
                branch="${ref#refs/heads/}"
                ;;
            refs/remotes/origin/*)
                branch="${ref#refs/remotes/origin/}"
                ;;
            *)
                continue
                ;;
        esac
        if [ "$(sanitize_branch_name "$branch")" = "$safe_branch" ]; then
            printf '%s\n' "$branch"
            return 0
        fi
    done < <(git -C "$CURRENT_REPO_ROOT" for-each-ref --format='%(refname)' refs/heads refs/remotes/origin 2>/dev/null)

    printf '%s\n' "$safe_branch"
}

cleanup_stale_dev_bundles() {
    local apps_dir
    apps_dir="$(dev_bundle_apps_dir)"
    local stale_days="${BRAINBAR_DEV_STALE_DAYS:-14}"
    if ! [[ "$stale_days" =~ ^[0-9]+$ ]]; then
        echo "[build-app] WARNING: invalid BRAINBAR_DEV_STALE_DAYS='$stale_days', using 14" >&2
        stale_days=14
    fi
    local bundle
    local found=0

    for bundle in "$apps_dir"/BrainBar-DEV-*.app; do
        [ -d "$bundle" ] || continue
        found=1

        local name
        name="$(basename "$bundle")"
        local safe_branch
        safe_branch="${name#BrainBar-DEV-}"
        safe_branch="${safe_branch%.app}"
        local branch
        branch="$(resolve_dev_bundle_branch "$safe_branch")"
        local sha
        sha="$(read_bundle_git_commit "$bundle")"
        local is_stale=0
        local reason=""

        if safe_branch_is_checked_out_anywhere "$safe_branch" "$sha"; then
            reason="bundle branch token '$safe_branch' or SHA is checked out in a worktree"
        elif [ -n "$sha" ] && git -C "$CURRENT_REPO_ROOT" merge-base --is-ancestor "$sha" origin/main 2>/dev/null; then
            is_stale=1
            reason="bundle SHA $sha is in origin/main"
        elif ! safe_branch_ref_exists "$safe_branch"; then
            is_stale=1
            reason="branch '$branch' not found locally or upstream"
        else
            local age_days
            age_days="$(dev_bundle_age_days "$bundle")"
            if [ "$age_days" -gt "$stale_days" ]; then
                is_stale=1
                reason="bundle age $age_days days exceeds threshold $stale_days"
            else
                reason="bundle age $age_days days is within threshold $stale_days"
            fi
        fi

        if [ "$DRY_RUN" -eq 1 ]; then
            if [ "$is_stale" -eq 1 ]; then
                echo "[build-app] Dry run: would clean stale DEV bundle: $name ($reason)"
            else
                echo "[build-app] Keeping DEV bundle: $name ($reason)"
            fi
        elif [ "$is_stale" -eq 1 ]; then
            echo "[build-app] Cleaning stale DEV bundle: $name ($reason)"
            rm -rf "$bundle"
        fi
    done

    if [ "$DRY_RUN" -eq 1 ] && [ "$found" -eq 0 ]; then
        echo "[build-app] Dry run: no DEV bundles found under $apps_dir"
    fi
}

if [ "$CURRENT_REPO_ROOT" != "$CANONICAL_REPO_ROOT" ] && [ "$FORCE_WORKTREE_BUILD" -ne 1 ]; then
    echo "[build-app] ERROR: refusing non-canonical build from $CURRENT_REPO_ROOT" >&2
    echo "[build-app] Re-run with --force-worktree-build to install a DEV bundle instead of ~/Applications/BrainBar.app" >&2
    exit 1
fi

if [ "$CURRENT_REPO_ROOT" != "$CANONICAL_REPO_ROOT" ]; then
    DEV_BUNDLE_BUILD=1
    SAFE_BRANCH_NAME="$(sanitize_branch_name "$(resolve_branch_name)")"
    APP_DIR="$HOME/Applications/BrainBar-DEV-$SAFE_BRANCH_NAME.app"
else
    APP_DIR="${BRAINBAR_APP_DIR:-$HOME/Applications/BrainBar.app}"
fi

DIRTY_STATUS="$(git -C "$CURRENT_REPO_ROOT" status --porcelain --untracked-files=all)"
if [ -n "$DIRTY_STATUS" ] && [ "$FORCE_DIRTY" -ne 1 ]; then
    echo "[build-app] ERROR: refusing dirty build from $CURRENT_REPO_ROOT" >&2
    echo "[build-app] Re-run with --force-dirty once these changes are explicitly reviewed:" >&2
    printf '%s\n' "$DIRTY_STATUS" >&2
    exit 1
fi

if [ "$DEV_BUNDLE_BUILD" -eq 0 ] && [ "$DRY_RUN" -eq 1 ]; then
    cleanup_stale_dev_bundles
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[build-app] Dry run OK"
    echo "[build-app] Repo: $CURRENT_REPO_ROOT"
    echo "[build-app] App path: $APP_DIR"
    if [ "$DEV_BUNDLE_BUILD" -eq 1 ]; then
        echo "[build-app] LaunchAgents: skipped for DEV worktree build"
    else
        echo "[build-app] UI LaunchAgent: canonical install to $UI_PLIST_DST"
        echo "[build-app] Daemon LaunchAgent: canonical install to $DAEMON_PLIST_DST"
    fi
    exit 0
fi

git_commit() {
    git -C "$PACKAGE_DIR" rev-parse HEAD
}

git_describe() {
    git -C "$PACKAGE_DIR" describe --always --dirty
}

build_time_utc() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

plist_set_string() {
    local plist_path="$1"
    local key="$2"
    local value="$3"

    if "$PLIST_BUDDY" -c "Print :$key" "$plist_path" >/dev/null 2>&1; then
        "$PLIST_BUDDY" -c "Set :$key \"$value\"" "$plist_path"
    else
        "$PLIST_BUDDY" -c "Add :$key string \"$value\"" "$plist_path"
    fi
}

check_brainlayer_package_installed() {
    local repo_root="$1"
    local python_path="$repo_root/.venv/bin/python"

    local python_exec="python3"
    if [ -x "$python_path" ]; then
        python_exec="$python_path"
    fi

    if ! "$python_exec" -c "import brainlayer" 2>/dev/null; then
        echo "[build-app] ERROR: brainlayer package not installed" >&2
        echo "" >&2
        echo "This build requires the brainlayer package to be installed." >&2
        echo "BrainBar and launchd services no longer use PYTHONPATH for imports." >&2
        echo "" >&2
        echo "Install with:" >&2
        echo "  cd $repo_root" >&2
        if [ -x "$python_path" ]; then
            echo "  $python_path -m pip install -e ." >&2
        else
            echo "  python3 -m pip install -e ." >&2
        fi
        echo "" >&2
        echo "For temporary source-tree fallback:" >&2
        echo "  export BRAINLAYER_SOURCE_FALLBACK=1" >&2
        echo "" >&2
        return 1
    fi

    echo "[build-app] brainlayer package is installed"
    return 0
}

configure_launchagent_environment() {
    local plist_path="$1"
    local repo_root="$2"
    local python_path="$repo_root/.venv/bin/python"

    if ! check_brainlayer_package_installed "$repo_root"; then
        exit 1
    fi

    "$PLIST_BUDDY" -c "Delete :EnvironmentVariables" "$plist_path" >/dev/null 2>&1 || true
    "$PLIST_BUDDY" -c "Add :EnvironmentVariables dict" "$plist_path"
    "$PLIST_BUDDY" -c "Add :EnvironmentVariables:BRAINLAYER_REPO_ROOT string \"$repo_root\"" "$plist_path"
    if [ -x "$python_path" ]; then
        "$PLIST_BUDDY" -c "Add :EnvironmentVariables:BRAINBAR_PYTHON string \"$python_path\"" "$plist_path"
    fi
}

stamp_info_plist() {
    local plist_path="$1"
    local commit_sha="$2"
    local describe_ref="$3"
    local build_utc="$4"

    plist_set_string "$plist_path" "GitCommit" "$commit_sha"
    plist_set_string "$plist_path" "GitDescribe" "$describe_ref"
    plist_set_string "$plist_path" "BuildTimeUTC" "$build_utc"
}

bootout_launchagent() {
    launchctl bootout "$LAUNCH_DOMAIN/$UI_PLIST_LABEL" 2>/dev/null || true
    launchctl bootout "$LAUNCH_DOMAIN/$DAEMON_PLIST_LABEL" 2>/dev/null || true
    if [ -f "$UI_PLIST_DST" ]; then
        launchctl bootout "$LAUNCH_DOMAIN" "$UI_PLIST_DST" 2>/dev/null || true
    fi
    if [ -f "$DAEMON_PLIST_DST" ]; then
        launchctl bootout "$LAUNCH_DOMAIN" "$DAEMON_PLIST_DST" 2>/dev/null || true
    fi
}

wait_for_brainbar_exit() {
    for _ in $(seq 1 100); do
        if ! pgrep -x BrainBar > /dev/null 2>&1; then
            return 0
        fi
        sleep 0.2
    done
    return 1
}

wait_for_brainbar_daemon_exit() {
    for _ in $(seq 1 100); do
        if ! pgrep -x BrainBarDaemon > /dev/null 2>&1; then
            return 0
        fi
        sleep 0.2
    done
    return 1
}

wait_for_socket() {
    local path="$1"
    local attempts="${BRAINBAR_SOCKET_WAIT_ATTEMPTS:-300}"
    for _ in $(seq 1 "$attempts"); do
        if [ -S "$path" ]; then
            return 0
        fi
        sleep 0.2
    done
    return 1
}

if [ "$DEV_BUNDLE_BUILD" -eq 0 ]; then
    # Stop LaunchAgents first so KeepAlive cannot race the rebuild and unlink the
    # freshly rebound socket from an older instance that is still terminating.
    echo "[build-app] Stopping LaunchAgents..."
    bootout_launchagent

    # Kill any running BrainBar instances before installing.
    if pgrep -x BrainBar > /dev/null 2>&1; then
        echo "[build-app] Stopping running BrainBar instances..."
        killall BrainBar 2>/dev/null || true
        if ! wait_for_brainbar_exit; then
            echo "[build-app] ERROR: BrainBar did not exit cleanly"
            pgrep -fl BrainBar || true
            exit 1
        fi
    fi
    if pgrep -x BrainBarDaemon > /dev/null 2>&1; then
        echo "[build-app] Stopping running BrainBarDaemon instances..."
        killall BrainBarDaemon 2>/dev/null || true
        if ! wait_for_brainbar_daemon_exit; then
            echo "[build-app] ERROR: BrainBarDaemon did not exit cleanly"
            pgrep -fl BrainBarDaemon || true
            exit 1
        fi
    fi
    cleanup_stale_dev_bundles
    rm -f "$SOCKET_PATH"
else
    echo "[build-app] DEV worktree build: preserving canonical LaunchAgent and socket"
fi

echo "[build-app] Building BrainBar and BrainBarDaemon (release)..."
swift build -c release --package-path "$PACKAGE_DIR" --product BrainBar
swift build -c release --package-path "$PACKAGE_DIR" --product BrainBarDaemon

# Find the built binary
BIN_DIR="$(swift build -c release --package-path "$PACKAGE_DIR" --show-bin-path)"
BINARY="$BIN_DIR/BrainBar"
DAEMON_BINARY="$BIN_DIR/BrainBarDaemon"
if [ ! -f "$BINARY" ]; then
    echo "[build-app] ERROR: Binary not found at $BINARY"
    exit 1
fi
if [ ! -f "$DAEMON_BINARY" ]; then
    echo "[build-app] ERROR: Daemon binary not found at $DAEMON_BINARY"
    exit 1
fi

# Clean stale bundle
if [ -d "$APP_DIR" ]; then
    echo "[build-app] Removing old bundle..."
    rm -rf "$APP_DIR"
fi

echo "[build-app] Creating .app bundle at $APP_DIR..."
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"
mkdir -p "$BRAINLAYER_LOG_DIR"

cp "$BUNDLE_DIR/Info.plist" "$APP_DIR/Contents/"
cp "$BINARY" "$APP_DIR/Contents/MacOS/BrainBar"
cp "$DAEMON_BINARY" "$APP_DIR/Contents/MacOS/BrainBarDaemon"

COMMIT_SHA="$(git_commit)"
DESCRIBE_REF="$(git_describe)"
BUILD_UTC="$(build_time_utc)"
stamp_info_plist "$APP_DIR/Contents/Info.plist" "$COMMIT_SHA" "$DESCRIBE_REF" "$BUILD_UTC"
echo "[build-app] Stamped Info.plist:"
echo "  GitCommit=$COMMIT_SHA"
echo "  GitDescribe=$DESCRIBE_REF"
echo "  BuildTimeUTC=$BUILD_UTC"

# Developer signing keeps TCC permissions stable across rebuilds.
echo "[build-app] Signing..."
codesign --force --deep --sign "$SIGN_IDENTITY" --timestamp=none "$APP_DIR"

echo "[build-app] Verifying signature..."
if ! codesign -dv --verbose=4 "$APP_DIR" 2>&1 | grep -F "Authority=$SIGN_IDENTITY" >/dev/null; then
    echo "[build-app] ERROR: Installed app is not signed with $SIGN_IDENTITY"
    codesign -dv --verbose=4 "$APP_DIR" 2>&1
    exit 1
fi

# Register URL scheme with Launch Services (ensures brainbar:// works after rebuild)
"$LSREGISTER" -R "$APP_DIR"

install_launchagent() {
    local source_plist="$1"
    local target_plist="$2"
    local label="$3"

    echo "[build-app] Installing LaunchAgent to $target_plist..."
    TMP_PLIST="$(mktemp)"
    trap 'rm -f "$TMP_PLIST"' EXIT
    sed \
        -e "s|/Applications/BrainBar.app|$APP_DIR|g" \
        -e "s|__HOME__|$HOME|g" \
        "$source_plist" > "$TMP_PLIST"
    configure_launchagent_environment "$TMP_PLIST" "$CURRENT_REPO_ROOT"
    mv "$TMP_PLIST" "$target_plist"
    trap - EXIT
    launchctl bootstrap "$LAUNCH_DOMAIN" "$target_plist"
    launchctl kickstart -k "$LAUNCH_DOMAIN/$label"
}

# Install LaunchAgents (expands path to actual APP_DIR)
if [ "$DEV_BUNDLE_BUILD" -eq 0 ]; then
    if [ -f "$DAEMON_PLIST_SRC" ]; then
        install_launchagent "$DAEMON_PLIST_SRC" "$DAEMON_PLIST_DST" "$DAEMON_PLIST_LABEL"
    fi
    if [ -f "$UI_PLIST_SRC" ]; then
        install_launchagent "$UI_PLIST_SRC" "$UI_PLIST_DST" "$UI_PLIST_LABEL"
    fi
    echo "[build-app] LaunchAgents installed — daemon and UI restart independently"
fi

if [ "$DEV_BUNDLE_BUILD" -eq 0 ]; then
    if ! wait_for_socket "$SOCKET_PATH"; then
        echo "[build-app] ERROR: BrainBar did not recreate $SOCKET_PATH"
        pgrep -fl BrainBar || true
        exit 1
    fi

    python3 - <<'PY' "$SOCKET_PATH"
import os
import socket
import sys

path = sys.argv[1]
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
try:
    s.connect(path)
except OSError as exc:
    print(f"[build-app] ERROR: socket connect failed for {path}: {exc}", file=sys.stderr)
    raise SystemExit(1)
finally:
    s.close()
PY
fi

echo "[build-app] Done: $APP_DIR"
if [ "$DEV_BUNDLE_BUILD" -eq 0 ]; then
    echo "[build-app] Socket: $SOCKET_PATH"
else
    echo "[build-app] Socket: unchanged (canonical service preserved)"
fi
echo "[build-app] DB: ~/.local/share/brainlayer/brainlayer.db"
