#!/usr/bin/env bash
# Build BrainBar as a proper macOS .app bundle.
#
# Usage: bash brain-bar/build-app.sh
#
# Output: ~/Applications/BrainBar.app (override with BRAINBAR_APP_DIR)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGE_DIR="$SCRIPT_DIR"
BUNDLE_DIR="$SCRIPT_DIR/bundle"
APP_DIR="${BRAINBAR_APP_DIR:-$HOME/Applications/BrainBar.app}"
SIGN_IDENTITY="${BRAINBAR_CODESIGN_IDENTITY:-Apple Development: Etan Heyman (DXHB5E7P2D)}"
PLIST_LABEL="com.brainlayer.brainbar"
PLIST_FILENAME="$PLIST_LABEL.plist"
PLIST_SRC="$BUNDLE_DIR/$PLIST_FILENAME"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_FILENAME"
LAUNCH_DOMAIN="gui/$(id -u)"
SOCKET_PATH="${BRAINBAR_SOCKET_PATH:-/tmp/brainbar.sock}"

bootout_launchagent() {
    launchctl bootout "$LAUNCH_DOMAIN/$PLIST_LABEL" 2>/dev/null || true
    if [ -f "$PLIST_DST" ]; then
        launchctl bootout "$LAUNCH_DOMAIN" "$PLIST_DST" 2>/dev/null || true
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

wait_for_socket() {
    local path="$1"
    for _ in $(seq 1 100); do
        if [ -S "$path" ]; then
            return 0
        fi
        sleep 0.2
    done
    return 1
}

# Stop LaunchAgent first so KeepAlive cannot race the rebuild and unlink the
# freshly rebound socket from an older instance that is still terminating.
echo "[build-app] Stopping LaunchAgent..."
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
rm -f "$SOCKET_PATH"

echo "[build-app] Building BrainBar (release)..."
swift build -c release --package-path "$PACKAGE_DIR"

# Find the built binary
BIN_DIR="$(swift build -c release --package-path "$PACKAGE_DIR" --show-bin-path)"
BINARY="$BIN_DIR/BrainBar"
if [ ! -f "$BINARY" ]; then
    echo "[build-app] ERROR: Binary not found at $BINARY"
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

cp "$BUNDLE_DIR/Info.plist" "$APP_DIR/Contents/"
cp "$BINARY" "$APP_DIR/Contents/MacOS/BrainBar"

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
/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A/Support/lsregister -R "$APP_DIR"

# Install LaunchAgent (expands path to actual APP_DIR)
if [ -f "$PLIST_SRC" ]; then
    echo "[build-app] Installing LaunchAgent to $PLIST_DST..."
    bootout_launchagent
    sed "s|/Applications/BrainBar.app|$APP_DIR|g" "$PLIST_SRC" > "$PLIST_DST"
    launchctl bootstrap "$LAUNCH_DOMAIN" "$PLIST_DST"
    launchctl kickstart -k "$LAUNCH_DOMAIN/$PLIST_LABEL"
    echo "[build-app] LaunchAgent installed — BrainBar will auto-restart after quit"
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
else
    echo "[build-app] LaunchAgent plist missing at $PLIST_SRC — skipping auto-launch and socket verification"
fi

echo "[build-app] Done: $APP_DIR"
echo "[build-app] Socket: $SOCKET_PATH"
echo "[build-app] DB: ~/.local/share/brainlayer/brainlayer.db"
