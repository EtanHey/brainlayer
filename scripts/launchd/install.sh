#!/usr/bin/env bash
# Install BrainLayer launchd plists for auto-indexing, enrichment, and WAL checkpoint.
#
# Usage:
#   ./scripts/launchd/install.sh              # Install all
#   ./scripts/launchd/install.sh index        # Install indexing only
#   ./scripts/launchd/install.sh enrich       # Install enrichment only
#   ./scripts/launchd/install.sh checkpoint   # Install WAL checkpoint only
#   ./scripts/launchd/install.sh remove       # Unload and remove all
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAUNCH_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/.local/share/brainlayer/logs"
BRAINLAYER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BRAINLAYER_BIN="${BRAINLAYER_BIN:-$(which brainlayer 2>/dev/null || echo "$HOME/.local/bin/brainlayer")}"
PYTHON3="${PYTHON3:-$(which python3 2>/dev/null || echo "/usr/bin/python3")}"
GROQ_API_KEY="${GROQ_API_KEY:-$(op item get GROQ_API_KEY --reveal --fields credential 2>/dev/null || echo "")}"
GOOGLE_API_KEY="${GOOGLE_API_KEY:-$(op item get "Google AI API Key" --reveal --fields credential 2>/dev/null || echo "")}"

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
        -e "s|__BRAINLAYER_DIR__|$BRAINLAYER_DIR|g" \
        -e "s|__PYTHON3__|$PYTHON3|g" \
        -e "s|__GROQ_API_KEY__|$GROQ_API_KEY|g" \
        -e "s|__GOOGLE_API_KEY__|$GOOGLE_API_KEY|g" \
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
    checkpoint)
        install_plist wal-checkpoint
        ;;
    all)
        install_plist index
        install_plist enrich
        install_plist wal-checkpoint
        ;;
    remove)
        remove_plist index
        remove_plist enrich
        remove_plist wal-checkpoint
        ;;
    *)
        echo "Usage: $0 [index|enrich|checkpoint|all|remove]"
        exit 1
        ;;
esac

echo ""
echo "Done. Check logs at: $LOG_DIR/"
echo "Status: launchctl list | grep brainlayer"
