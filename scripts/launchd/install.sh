#!/usr/bin/env bash
# Install BrainLayer launchd plists for auto-indexing and enrichment.
#
# Usage:
#   ./scripts/launchd/install.sh          # Install both
#   ./scripts/launchd/install.sh index    # Install indexing only
#   ./scripts/launchd/install.sh enrich   # Install enrichment only
#   ./scripts/launchd/install.sh remove   # Unload and remove all
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAUNCH_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/.local/share/brainlayer/logs"
BRAINLAYER_BIN="${BRAINLAYER_BIN:-$(which brainlayer 2>/dev/null || echo "$HOME/.local/bin/brainlayer")}"

if [ ! -x "$BRAINLAYER_BIN" ]; then
    echo "ERROR: brainlayer binary not found at $BRAINLAYER_BIN"
    echo "Install with: pip install -e . (from brainlayer repo)"
    echo "Or set BRAINLAYER_BIN=/path/to/brainlayer"
    exit 1
fi

mkdir -p "$LAUNCH_DIR" "$LOG_DIR"

install_plist() {
    local name="$1"
    local src="$SCRIPT_DIR/com.brainlayer.${name}.plist"
    local dst="$LAUNCH_DIR/com.brainlayer.${name}.plist"

    if [ ! -f "$src" ]; then
        echo "ERROR: $src not found"
        return 1
    fi

    # Replace placeholders
    sed \
        -e "s|__HOME__|$HOME|g" \
        -e "s|__BRAINLAYER_BIN__|$BRAINLAYER_BIN|g" \
        "$src" > "$dst"

    echo "Installed: $dst"
    echo "  Binary: $BRAINLAYER_BIN"
    echo "  Logs: $LOG_DIR/"

    # Unload if already loaded, then load
    launchctl bootout "gui/$(id -u)/com.brainlayer.${name}" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$dst"
    echo "  Loaded: com.brainlayer.${name}"
}

remove_plist() {
    local name="$1"
    local dst="$LAUNCH_DIR/com.brainlayer.${name}.plist"
    launchctl bootout "gui/$(id -u)/com.brainlayer.${name}" 2>/dev/null || true
    rm -f "$dst"
    echo "Removed: com.brainlayer.${name}"
}

case "${1:-all}" in
    index)
        install_plist index
        ;;
    enrich)
        install_plist enrich
        ;;
    all)
        install_plist index
        install_plist enrich
        ;;
    remove)
        remove_plist index
        remove_plist enrich
        ;;
    *)
        echo "Usage: $0 [index|enrich|all|remove]"
        exit 1
        ;;
esac

echo ""
echo "Done. Check logs at: $LOG_DIR/"
echo "Status: launchctl list | grep brainlayer"
