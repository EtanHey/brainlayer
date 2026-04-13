#!/usr/bin/env bash
# Install BrainLayer launchd plists for auto-indexing and maintenance jobs.
#
# Usage:
#   ./scripts/launchd/install.sh              # Install all
#   ./scripts/launchd/install.sh index        # Install indexing only
#   ./scripts/launchd/install.sh enrich       # Install enrichment only
#   ./scripts/launchd/install.sh decay        # Install decay only
#   ./scripts/launchd/install.sh load enrichment
#   ./scripts/launchd/install.sh unload enrichment
#   ./scripts/launchd/install.sh checkpoint   # Install WAL checkpoint only
#   ./scripts/launchd/install.sh remove       # Unload and remove all
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAUNCH_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/Library/Logs"
BRAINLAYER_LOG_DIR="$HOME/.local/share/brainlayer/logs"
BRAINLAYER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BRAINLAYER_BIN="${BRAINLAYER_BIN:-$(which brainlayer 2>/dev/null || echo "$HOME/.local/bin/brainlayer")}"
GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"

if [ ! -x "$BRAINLAYER_BIN" ]; then
    echo "ERROR: brainlayer binary not found at $BRAINLAYER_BIN"
    echo "Install with: pip install -e . (from brainlayer repo)"
    echo "Or set BRAINLAYER_BIN=/path/to/brainlayer"
    exit 1
fi

mkdir -p "$LAUNCH_DIR" "$LOG_DIR" "$BRAINLAYER_LOG_DIR"

resolve_google_api_key() {
    if [ -n "${GOOGLE_API_KEY:-}" ]; then
        printf '%s' "$GOOGLE_API_KEY"
        return 0
    fi

    if [ -f "$HOME/.zshrc" ] && command -v zsh >/dev/null 2>&1; then
        local sourced_key
        sourced_key="$(zsh -lc 'source ~/.zshrc >/dev/null 2>&1; printf %s "${GOOGLE_API_KEY:-${GOOGLE_GENERATIVE_AI_API_KEY:-}}"' 2>/dev/null || true)"
        if [ -n "$sourced_key" ]; then
            printf '%s' "$sourced_key"
            return 0
        fi
    fi

    printf '%s' "${GOOGLE_GENERATIVE_AI_API_KEY:-}"
}

load_plist() {
    local name="$1"
    local dst="$LAUNCH_DIR/com.brainlayer.${name}.plist"
    launchctl unload "$dst" 2>/dev/null || true
    launchctl load "$dst"
    echo "  Loaded: com.brainlayer.${name}"
}

unload_plist() {
    local name="$1"
    local dst="$LAUNCH_DIR/com.brainlayer.${name}.plist"
    launchctl unload "$dst" 2>/dev/null || true
    echo "  Unloaded: com.brainlayer.${name}"
}

install_plist() {
    local name="$1"
    local src="$SCRIPT_DIR/com.brainlayer.${name}.plist"
    local dst="$LAUNCH_DIR/com.brainlayer.${name}.plist"
    local google_api_key="${GOOGLE_API_KEY:-}"

    if [ ! -f "$src" ]; then
        echo "ERROR: $src not found"
        return 1
    fi

    if [ "$name" = "enrichment" ] || [ "$name" = "enrich" ]; then
        google_api_key="$(resolve_google_api_key)"
        if [ -z "$google_api_key" ]; then
            echo "ERROR: GOOGLE_API_KEY not found in environment or ~/.zshrc"
            return 1
        fi
    fi

    # Replace placeholders
    sed \
        -e "s|__HOME__|$HOME|g" \
        -e "s|__BRAINLAYER_BIN__|$BRAINLAYER_BIN|g" \
        -e "s|__BRAINLAYER_DIR__|$BRAINLAYER_DIR|g" \
        -e "s|__GOOGLE_API_KEY__|$google_api_key|g" \
        "$src" > "$dst"

    echo "Installed: $dst"
    echo "  Logs: $LOG_DIR/ and $BRAINLAYER_LOG_DIR/"

    load_plist "$name"
}

remove_plist() {
    local name="$1"
    local dst="$LAUNCH_DIR/com.brainlayer.${name}.plist"
    unload_plist "$name"
    rm -f "$dst"
    echo "Removed: com.brainlayer.${name}"
}

case "${1:-all}" in
    index)
        install_plist index
        ;;
    enrich)
        # Legacy alias: install the unified enrichment plist
        install_plist enrichment
        ;;
    enrichment)
        install_plist enrichment
        ;;
    decay)
        install_plist decay
        ;;
    load)
        load_plist "${2:-enrichment}"
        ;;
    unload)
        unload_plist "${2:-enrichment}"
        ;;
    checkpoint)
        install_plist wal-checkpoint
        ;;
    all)
        install_plist index
        install_plist enrichment
        install_plist decay
        install_plist wal-checkpoint
        # Remove old enrich plist if present
        remove_plist enrich 2>/dev/null || true
        ;;
    remove)
        remove_plist index
        remove_plist enrich 2>/dev/null || true
        remove_plist enrichment 2>/dev/null || true
        remove_plist decay 2>/dev/null || true
        remove_plist wal-checkpoint
        ;;
    *)
        echo "Usage: $0 [index|enrich|enrichment|decay|load [name]|unload [name]|checkpoint|all|remove]"
        exit 1
        ;;
esac

echo ""
echo "Done. Check logs at: $LOG_DIR/"
echo "Enrichment label: com.brainlayer.enrichment"
echo "Status: launchctl list | grep brainlayer"
