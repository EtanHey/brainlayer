#!/usr/bin/env bash
# Collapse a Mac down to one canonical notarized BrainBar.app.
#
# SAFE BY DEFAULT: dry-run inventory only unless --apply is passed. Stray bundles
# are moved to a timestamped backup directory, never deleted.
set -euo pipefail

CANONICAL_APP="${BRAINLAYER_DEDUPE_BRAINBAR_CANONICAL_APP:-/Applications/BrainBar.app}"
BUNDLE_ID="com.brainlayer.brainbar"
UI_LABEL="com.brainlayer.brainbar"
DAEMON_LABEL="com.brainlayer.brainbar-daemon"
HOME_DIR="${HOME:?HOME is required}"
BACKUP_DIR="$HOME_DIR/.brainlayer/brainbar-dedupe-backup"
SEARCH_ROOTS="${BRAINLAYER_DEDUPE_BRAINBAR_SEARCH_ROOTS:-/Applications:$HOME_DIR/Applications:$HOME_DIR/Gits:$HOME_DIR/Desktop:$HOME_DIR/Downloads:/tmp:/private/tmp:/var/folders}"
APPLY=0

usage() {
    cat <<EOF
Usage: brainlayer-dedupe-brainbar.sh [--dry-run|--apply]

  --dry-run   Default. Inventory BrainBar.app copies and print the cleanup plan.
  --apply     Move stray bundles and LaunchAgents to:
              $BACKUP_DIR/<timestamp>/

Keeps the canonical notarized app at:
  $CANONICAL_APP
EOF
}

for arg in "$@"; do
    case "$arg" in
        --apply) APPLY=1 ;;
        --dry-run) APPLY=0 ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[brainlayer-dedupe-brainbar] ERROR: unknown argument: $arg" >&2
            usage >&2
            exit 2
            ;;
    esac
done

log() {
    printf '%s\n' "$*"
}

sect() {
    printf '\n==== %s ====\n' "$*"
}

run() {
    if [[ "$APPLY" -eq 1 ]]; then
        log "  + $*"
        "$@"
    else
        log "  (dry-run) would run: $*"
    fi
}

app_bundleid() {
    defaults read "$1/Contents/Info" CFBundleIdentifier 2>/dev/null || echo "?"
}

is_notarized_app() {
    local app="$1"
    [[ -d "$app" ]] || return 1
    spctl --assess --type execute "$app" >/dev/null 2>&1 || return 1
    xcrun stapler validate "$app" >/dev/null 2>&1 || return 1
}

discover_bundles() {
    local roots=()
    IFS=':' read -r -a roots <<< "$SEARCH_ROOTS"
    {
        mdfind "kMDItemCFBundleIdentifier == '$BUNDLE_ID'" 2>/dev/null || true
        find "${roots[@]}" -maxdepth 6 -name 'BrainBar.app' -type d -prune 2>/dev/null || true
    } | sort -u
}

sect "1. BrainBar.app bundles on this machine"
declare -a BUNDLES=()
while IFS= read -r bundle; do
    [[ -d "$bundle" ]] || continue
    [[ "$(app_bundleid "$bundle")" == "$BUNDLE_ID" ]] && BUNDLES+=("$bundle")
done < <(discover_bundles)

if [[ "${#BUNDLES[@]}" -eq 0 ]]; then
    log "No BrainBar.app bundles found."
else
    for bundle in "${BUNDLES[@]}"; do
        status="STRAY or unverified"
        is_notarized_app "$bundle" && status="notarized OK"
        printf '  %-60s bundle=%s -> %s\n' "$bundle" "$(app_bundleid "$bundle")" "$status"
    done
fi

sect "2. Canonical decision"
canonical_bundle_id="$(app_bundleid "$CANONICAL_APP")"
if ! is_notarized_app "$CANONICAL_APP" || [[ "$canonical_bundle_id" != "$BUNDLE_ID" ]]; then
    log "  !! Canonical app is missing or not notarized: $CANONICAL_APP"
    log "  !! Expected bundle id: $BUNDLE_ID; found: $canonical_bundle_id"
    log "  !! Refusing to clobber it from another local copy."
    log "  !! Recover through the cask:"
    log "       brew reinstall --cask etanhey/layers/brainbar"
    exit 1
fi
log "  Canonical notarized app is in place: $CANONICAL_APP"

sect "3. LaunchAgents referencing BrainBar"
declare -a AGENT_FILES=()
while IFS= read -r file; do
    [[ -n "$file" ]] && AGENT_FILES+=("$file")
done < <(
    grep -rlE 'BrainBar|brainlayer\.brainbar' \
        "$HOME_DIR/Library/LaunchAgents" /Library/LaunchAgents /Library/LaunchDaemons 2>/dev/null | sort -u || true
)

if [[ "${#AGENT_FILES[@]}" -eq 0 ]]; then
    log "  none"
else
    for file in "${AGENT_FILES[@]}"; do
        target="$(/usr/libexec/PlistBuddy -c 'Print :ProgramArguments:0' "$file" 2>/dev/null || echo '?')"
        printf '  %s\n      -> %s\n' "$file" "$target"
    done
fi

sect "4. Plan / Apply (mode: $([[ "$APPLY" -eq 1 ]] && echo APPLY || echo DRY-RUN))"
STAMP="$(date +%Y%m%d-%H%M%S)"
DEST_BACKUP="$BACKUP_DIR/$STAMP"

log "-- quit running BrainBar instances"
run osascript -e 'tell application "BrainBar" to quit' || true
run pkill -x BrainBar || true
run pkill -x BrainBarDaemon || true

log "-- back up stray bundles; never clobber canonical"
for bundle in "${BUNDLES[@]:-}"; do
    [[ -n "$bundle" ]] || continue
    [[ "$bundle" == "$CANONICAL_APP" ]] && { log "  keep canonical: $bundle"; continue; }
    run mkdir -p "$DEST_BACKUP/bundles"
    run mv "$bundle" "$DEST_BACKUP/bundles/$(echo "$bundle" | tr '/ ' '__')"
done

log "-- prune stray LaunchAgents and keep canonical app bundle resources authoritative"
for file in "${AGENT_FILES[@]:-}"; do
    [[ -n "$file" ]] || continue
    base="$(basename "$file")"
    if [[ ( "$base" == "$UI_LABEL.plist" || "$base" == "$DAEMON_LABEL.plist" ) \
        && "$file" == "$HOME_DIR/Library/LaunchAgents/$base" ]]; then
        log "  keep canonical user LaunchAgent: $file"
        continue
    fi

    run launchctl bootout "gui/$(id -u)/${base%.plist}" || true
    run mkdir -p "$DEST_BACKUP/LaunchAgents"
    run mv "$file" "$DEST_BACKUP/LaunchAgents/"
done

log "-- restore canonical LaunchAgents from app bundle resources"
AGENT_SRC="$CANONICAL_APP/Contents/Resources/LaunchAgents"
for label in "$DAEMON_LABEL" "$UI_LABEL"; do
    src="$AGENT_SRC/$label.plist"
    dst="$HOME_DIR/Library/LaunchAgents/$label.plist"
    if [[ -f "$src" ]]; then
        run mkdir -p "$HOME_DIR/Library/LaunchAgents"
        run cp "$src" "$dst"
        run launchctl bootout "gui/$(id -u)/$label" || true
        run launchctl bootstrap "gui/$(id -u)" "$dst" || true
        run launchctl kickstart -k "gui/$(id -u)/$label" || true
    else
        log "  missing bundled LaunchAgent: $src"
    fi
done

sect "5. Result"
if [[ "$APPLY" -eq 1 ]]; then
    log "Backups, if any: $DEST_BACKUP"
else
    log "DRY-RUN complete. Re-run with --apply to execute. Nothing was changed."
fi
