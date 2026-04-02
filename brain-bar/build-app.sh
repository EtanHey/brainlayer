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

# Kill any running BrainBar instances before installing
if pgrep -x BrainBar > /dev/null 2>&1; then
    echo "[build-app] Stopping running BrainBar instances..."
    killall BrainBar 2>/dev/null || true
    sleep 1
    rm -f /tmp/brainbar.sock
fi

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
PLIST_NAME="com.brainlayer.brainbar.plist"
PLIST_SRC="$BUNDLE_DIR/$PLIST_NAME"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME"
if [ -f "$PLIST_SRC" ]; then
    echo "[build-app] Installing LaunchAgent to $PLIST_DST..."
    launchctl bootout "gui/$(id -u)/$PLIST_NAME" 2>/dev/null || true
    sed "s|/Applications/BrainBar.app|$APP_DIR|g" "$PLIST_SRC" > "$PLIST_DST"
    launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
    echo "[build-app] LaunchAgent installed — BrainBar will auto-restart after quit"
fi

echo "[build-app] Done: $APP_DIR"
echo "[build-app] Socket: /tmp/brainbar.sock"
echo "[build-app] DB: ~/.local/share/brainlayer/brainlayer.db"
