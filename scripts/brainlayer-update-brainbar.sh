#!/usr/bin/env bash
# Update BrainBar through the notarized Homebrew cask artifact.
set -euo pipefail

DRY_RUN=0
CASK_TOKEN="${BRAINLAYER_UPDATE_BRAINBAR_CASK_TOKEN:-etanhey/layers/brainbar}"
TEST_CASK_INSTALLED="${BRAINLAYER_UPDATE_TEST_BREW_CASK_INSTALLED:-}"
DRY_RUN_COMMANDS="${BRAINLAYER_UPDATE_DRY_RUN_COMMANDS:-0}"

usage() {
    cat <<EOF
Usage: brainlayer-update-brainbar.sh [--dry-run]

Routes BrainBar app updates through Homebrew:
  installed:     brew reinstall --cask $CASK_TOKEN
  not installed: brew install --cask $CASK_TOKEN

Default reinstall command:
  brew reinstall --cask etanhey/layers/brainbar

recovery-no-sudo:
  If brew fails mid-uninstall on root-owned leftovers, do not rebuild locally.
  The notarized .app and Homebrew receipt may still survive. Restore the user
  LaunchAgents from the app bundle and then rerun the cask command:

    app="/Applications/BrainBar.app"
    agents="\$app/Contents/Resources/LaunchAgents"
    domain="gui/\$(id -u)"
    cp "\$agents/com.brainlayer.brainbar.plist" "\$HOME/Library/LaunchAgents/"
    cp "\$agents/com.brainlayer.brainbar-daemon.plist" "\$HOME/Library/LaunchAgents/"
    launchctl bootstrap "\$domain" "\$HOME/Library/LaunchAgents/com.brainlayer.brainbar-daemon.plist"
    launchctl bootstrap "\$domain" "\$HOME/Library/LaunchAgents/com.brainlayer.brainbar.plist"
    launchctl kickstart -k "\$domain/com.brainlayer.brainbar-daemon"
    launchctl kickstart -k "\$domain/com.brainlayer.brainbar"
EOF
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[brainlayer-update-brainbar] ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

log() {
    printf '%s\n' "$*"
}

brainbar_cask_installed() {
    local installed_token="${CASK_TOKEN##*/}"
    case "$TEST_CASK_INSTALLED" in
        1) return 0 ;;
        0) return 1 ;;
    esac
    command -v brew >/dev/null 2>&1 && brew list --cask "$installed_token" >/dev/null 2>&1
}

brainbar_update_command() {
    if brainbar_cask_installed; then
        printf 'brew\0reinstall\0--cask\0%s\0' "$CASK_TOKEN"
    else
        printf 'brew\0install\0--cask\0%s\0' "$CASK_TOKEN"
    fi
}

brainbar_update_label() {
    local parts=()
    while IFS= read -r -d '' part; do
        parts+=("$part")
    done < <(brainbar_update_command)
    printf '%s' "${parts[*]}"
}

run_cmd() {
    log "+ $*"
    if [[ "$DRY_RUN" -eq 1 || "$DRY_RUN_COMMANDS" = "1" ]]; then
        return 0
    fi
    "$@"
}

print_plan() {
    local app_update
    app_update="$(brainbar_update_label)"
    log "BrainLayer BrainBar update plan"
    log "DRY RUN: $([[ "$DRY_RUN" -eq 1 ]] && printf yes || printf no)"
    log "BRAINBAR APP UPDATE: $app_update"
    log "Steps:"
    log "  1. + $app_update"
    log "  2. Homebrew installs the notarized BrainBar cask artifact"
    log "Recovery:"
    log "  recovery-no-sudo: restore LaunchAgents from /Applications/BrainBar.app/Contents/Resources/LaunchAgents if brew stops on root-owned leftovers."
}

main() {
    print_plan
    local command_parts=()
    while IFS= read -r -d '' part; do
        command_parts+=("$part")
    done < <(brainbar_update_command)
    if [[ "$DRY_RUN" -ne 1 && "$DRY_RUN_COMMANDS" != "1" ]] && ! command -v brew >/dev/null 2>&1; then
        echo "[brainlayer-update-brainbar] ERROR: brew is required for BrainBar cask updates" >&2
        exit 127
    fi
    run_cmd "${command_parts[@]}"
    log "BrainBar update command complete."
}

main
