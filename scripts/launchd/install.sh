#!/usr/bin/env bash
# Install BrainLayer launchd plists for auto-indexing and maintenance jobs.
#
# Usage:
#   ./scripts/launchd/install.sh              # Install all
#   ./scripts/launchd/install.sh index        # Install indexing only
#   ./scripts/launchd/install.sh watch        # Install watcher only
#   ./scripts/launchd/install.sh enrich       # Install enrichment only
#   ./scripts/launchd/install.sh drain        # Install queue drain only
#   ./scripts/launchd/install.sh decay        # Install decay only
#   ./scripts/launchd/install.sh load enrichment
#   ./scripts/launchd/install.sh unload enrichment
#   ./scripts/launchd/install.sh checkpoint   # Install WAL checkpoint only
#   ./scripts/launchd/install.sh repair-fts   # Install weekly explicit FTS repair
#   ./scripts/launchd/install.sh backup       # Install daily DB backup only
#   ./scripts/launchd/install.sh jsonl-backup # Install daily JSONL backup only
#   ./scripts/launchd/install.sh maintenance  # Install recurring maintenance jobs
#   ./scripts/launchd/install.sh remove       # Unload and remove all
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAUNCH_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/Library/Logs"
BRAINLAYER_LOG_DIR="$HOME/.local/share/brainlayer/logs"
BRAINLAYER_LIB_DIR="$HOME/.local/lib/brainlayer"
BRAINLAYER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BRAINLAYER_LAUNCHD_DIR="${BRAINLAYER_LAUNCHD_DIR:-$SCRIPT_DIR}"
BRAINLAYER_BIN="${BRAINLAYER_BIN:-$(which brainlayer 2>/dev/null || echo "$HOME/.local/bin/brainlayer")}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
BRAINLAYER_PYTHON="${BRAINLAYER_PYTHON:-$PYTHON_BIN}"
BRAINLAYER_ENV_FILE="${BRAINLAYER_ENV_FILE:-$HOME/.config/brainlayer/brainlayer.env}"
BRAINLAYER_ENV_RUN="$BRAINLAYER_LIB_DIR/brainlayer-env-run.sh"

if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: python3 not found in PATH"
    echo "Install Python 3 or set PYTHON_BIN=/path/to/python3"
    exit 1
fi

if [ ! -x "$BRAINLAYER_BIN" ]; then
    echo "ERROR: brainlayer binary not found at $BRAINLAYER_BIN"
    echo "Install with: pip install -e . (from brainlayer repo)"
    echo "Or set BRAINLAYER_BIN=/path/to/brainlayer"
    exit 1
fi

mkdir -p "$LAUNCH_DIR" "$LOG_DIR" "$LOG_DIR/brainlayer" "$BRAINLAYER_LOG_DIR" "$BRAINLAYER_LIB_DIR"
mkdir -p "$HOME/.brainlayer/logs" "$HOME/.brainlayer/queue"

install_env_runner() {
    local src="$SCRIPT_DIR/brainlayer-env-run.sh"

    if [ ! -f "$src" ]; then
        echo "ERROR: $src not found"
        return 1
    fi

    install -m 0755 "$src" "$BRAINLAYER_ENV_RUN"
    echo "Installed: $BRAINLAYER_ENV_RUN"
}

verify_gemini_env_file() {
    if [ ! -f "$BRAINLAYER_ENV_FILE" ]; then
        echo "ERROR: BrainLayer Gemini env file not found at $BRAINLAYER_ENV_FILE"
        echo "Run 'brainlayer init' or create it from scripts/launchd/brainlayer.env.example"
        return 1
    fi

    if ! BRAINLAYER_ENV_FILE="$BRAINLAYER_ENV_FILE" /bin/sh -c '
        set -a
        . "$BRAINLAYER_ENV_FILE"
        set +a
        test -n "${GOOGLE_API_KEY:-${GOOGLE_GENERATIVE_AI_API_KEY:-}}"
    ' >/dev/null; then
        echo "ERROR: $BRAINLAYER_ENV_FILE did not provide GOOGLE_API_KEY"
        return 1
    fi

    if ! BRAINLAYER_ENV_FILE="$BRAINLAYER_ENV_FILE" /bin/sh -c '
        set -a
        . "$BRAINLAYER_ENV_FILE"
        set +a
        for key in BRAINLAYER_ENRICH_ENABLED BRAINLAYER_ENRICH_MODE BRAINLAYER_ENRICH_PROVIDER BRAINLAYER_ENRICH_BACKEND BRAINLAYER_ENRICH_RATE BRAINLAYER_ENRICH_CONCURRENCY BRAINLAYER_MAX_COMMIT_BATCH BRAINLAYER_GEMINI_SERVICE_TIER; do
            eval "value=\${$key:-}"
            if [ -z "$value" ]; then
                echo "missing $key" >&2
                exit 1
            fi
        done
    ' >/dev/null; then
        echo "ERROR: $BRAINLAYER_ENV_FILE is missing required enrichment config keys"
        echo "Run 'brainlayer init' or create it from scripts/launchd/brainlayer.env.example"
        return 1
    fi
}

verify_config_file() {
    if [ ! -f "$BRAINLAYER_ENV_FILE" ]; then
        echo "ERROR: BrainLayer config file not found at $BRAINLAYER_ENV_FILE"
        echo "Run 'brainlayer init' or create it from scripts/launchd/brainlayer.env.example"
        return 1
    fi
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

    if [ ! -f "$src" ]; then
        echo "ERROR: $src not found"
        return 1
    fi

    install_env_runner
    verify_config_file

    if [ "$name" = "enrichment" ] || [ "$name" = "enrich" ]; then
        verify_gemini_env_file
    fi

    # Replace placeholders
    sed \
        -e "s|__HOME__|$HOME|g" \
        -e "s|__BRAINLAYER_BIN__|$BRAINLAYER_BIN|g" \
        -e "s|__BRAINLAYER_DIR__|$BRAINLAYER_DIR|g" \
        -e "s|__BRAINLAYER_LAUNCHD_DIR__|$BRAINLAYER_LAUNCHD_DIR|g" \
        -e "s|__PYTHON_BIN__|$PYTHON_BIN|g" \
        -e "s|__BRAINLAYER_PYTHON__|$BRAINLAYER_PYTHON|g" \
        -e "s|__REPO_ROOT__|$BRAINLAYER_DIR|g" \
        -e "s|__BRAINLAYER_ENV_FILE__|$BRAINLAYER_ENV_FILE|g" \
        -e "s|__BRAINLAYER_ENV_RUN__|$BRAINLAYER_ENV_RUN|g" \
        "$src" > "$dst"

    echo "Installed: $dst"
    echo "  Logs: $LOG_DIR/ and $BRAINLAYER_LOG_DIR/"

    load_plist "$name"
}

install_backup_script() {
    local src="$SCRIPT_DIR/backup-daily.sh"
    local dst="$BRAINLAYER_LIB_DIR/backup-daily.sh"
    local escaped_brainlayer_dir

    if [ ! -f "$src" ]; then
        echo "ERROR: $src not found"
        return 1
    fi

    escaped_brainlayer_dir="$(printf '%s' "$BRAINLAYER_DIR" | sed 's/[\\&|]/\\&/g')"
    sed \
        -e "s|__BRAINLAYER_DIR_VALUE__|$escaped_brainlayer_dir|g" \
        "$src" > "$dst"
    chmod 755 "$dst"
    echo "Installed: $dst"
}

install_jsonl_backup_script() {
    local src="$SCRIPT_DIR/jsonl-backup.sh"
    local dst="$BRAINLAYER_LIB_DIR/jsonl-backup.sh"
    local escaped_brainlayer_dir

    if [ ! -f "$src" ]; then
        echo "ERROR: $src not found"
        return 1
    fi

    escaped_brainlayer_dir="$(printf '%s' "$BRAINLAYER_DIR" | sed 's/[\\&|]/\\&/g')"
    sed \
        -e "s|__BRAINLAYER_DIR_VALUE__|$escaped_brainlayer_dir|g" \
        "$src" > "$dst"
    chmod 755 "$dst"
    echo "Installed: $dst"
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
        remove_plist enrich 2>/dev/null || true
        ;;
    enrichment)
        install_plist enrichment
        ;;
    watch)
        install_plist watch
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
    drain)
        install_plist drain
        ;;
    repair-fts)
        install_plist repair-fts
        ;;
    backup)
        install_backup_script
        install_plist backup-daily
        ;;
    jsonl|jsonl-backup)
        install_jsonl_backup_script
        install_plist jsonl-backup
        ;;
    maintenance-nightly)
        install_plist maintenance-nightly
        ;;
    maintenance-weekly)
        install_plist maintenance-weekly
        ;;
    maintenance)
        install_plist maintenance-nightly
        install_plist maintenance-weekly
        ;;
    all)
        install_env_runner
        verify_config_file
        verify_gemini_env_file
        install_plist index
        install_plist drain
        install_plist watch
        install_plist enrichment
        install_plist decay
        install_plist wal-checkpoint
        install_plist repair-fts
        install_backup_script
        install_plist backup-daily
        install_jsonl_backup_script
        install_plist jsonl-backup
        install_plist maintenance-nightly
        install_plist maintenance-weekly
        # Remove old enrich plist if present
        remove_plist enrich 2>/dev/null || true
        ;;
    remove)
        remove_plist index
        remove_plist enrich 2>/dev/null || true
        remove_plist enrichment 2>/dev/null || true
        remove_plist watch 2>/dev/null || true
        remove_plist decay 2>/dev/null || true
        remove_plist drain 2>/dev/null || true
        remove_plist wal-checkpoint
        remove_plist repair-fts 2>/dev/null || true
        remove_plist backup-daily 2>/dev/null || true
        remove_plist jsonl-backup 2>/dev/null || true
        remove_plist maintenance-nightly 2>/dev/null || true
        remove_plist maintenance-weekly 2>/dev/null || true
        rm -f "$BRAINLAYER_LIB_DIR/backup-daily.sh"
        rm -f "$BRAINLAYER_LIB_DIR/jsonl-backup.sh"
        ;;
    *)
        echo "Usage: $0 [index|watch|enrich|enrichment|decay|drain|repair-fts|load [name]|unload [name]|checkpoint|backup|jsonl-backup|maintenance|maintenance-nightly|maintenance-weekly|all|remove]"
        exit 1
        ;;
esac

echo ""
echo "Done. Check logs at: $LOG_DIR/"
echo "Enrichment label: com.brainlayer.enrichment"
echo "Status: launchctl list | grep brainlayer"
