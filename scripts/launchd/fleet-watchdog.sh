#!/bin/sh
# BrainLayer fleet watchdog -- re-bootstraps any com.brainlayer.* LaunchAgent that got
# booted out (e.g. a nightly maintenance pause that never resumed, leaving capture,
# drain and enrichment dead for ~24h). Idempotent; safe to run every 5 minutes.
# Authored by camprClaude (M1 worker) 2026-06-25; adopted into the repo 2026-09-03 (w11b).
#
# LABEL NAMESPACE: this job runs as com.etanhey.brainlayer-fleet-watchdog, NOT
# com.brainlayer.*, on purpose. The loop below globs
# "$HOME"/Library/LaunchAgents/com.brainlayer.*.plist, so a com.brainlayer.* label would
# (a) appear in its own revival glob and (b) fall under BrainLayer's own maintenance
# pause -- the watchdog would pause itself.
#
# WARNING -- `launchctl bootout` DOES NOT HOLD while this watchdog is loaded.
# It runs on StartInterval 300, so a booted-out com.brainlayer.* label is silently
# re-bootstrapped within five minutes. On 2026-09-02 that produced a real false green on
# the M1: 1.5.11 installed, the machine still serving the 1.5.10 keg, every pid green.
# Only a label that `launchctl print-disabled gui/$UID` reports as "=> disabled"
# (or "=> true") is left alone. To quiesce for an upgrade use ONE of:
#   scripts/launchd/install.sh fleet-watchdog-quiesce  # disable + bootout THIS watchdog
#   launchctl disable "gui/$(id -u)/com.brainlayer.<label>"  # holds per label, vs every reviver
# Re-arm with `scripts/launchd/install.sh fleet-watchdog-resume` (or `launchctl enable`).

UID_="$(id -u)"
LOG_DIR="$HOME/Library/Logs/brainlayer"
LOG="$LOG_DIR/fleet-watchdog.log"
# Labels reported as operator-disabled on the previous run, one per line. Used only to
# keep a standing disable from appending an identical skip line every 300 seconds.
SKIP_STATE="$LOG_DIR/fleet-watchdog-skipped"
NL='
'

mkdir -p "$LOG_DIR" 2>/dev/null

log() {
    printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" >> "$LOG"
}

# Fail closed: with no readable disable state we cannot tell an operator disable from a
# crash, and reviving a deliberately disabled label is the worse error.
if ! disabled_listing="$(launchctl print-disabled "gui/$UID_" 2>/dev/null)"; then
    log "ERROR could not read launchd disabled state; refusing to revive anything"
    exit 1
fi

prev_skipped=""
if [ -f "$SKIP_STATE" ]; then
    prev_skipped="$NL$(cat "$SKIP_STATE" 2>/dev/null)$NL"
fi
new_skipped=""

for plist in "$HOME"/Library/LaunchAgents/com.brainlayer.*.plist; do
    [ -f "$plist" ] || continue
    label="$(basename "$plist" .plist)"
    # If launchd has no record of the label in this GUI domain, it was booted out.
    launchctl print "gui/$UID_/$label" >/dev/null 2>&1 && continue

    case "$disabled_listing" in
        *"\"$label\" => disabled"*|*"\"$label\" => true"*)
            new_skipped="$new_skipped$label$NL"
            case "$prev_skipped" in
                *"$NL$label$NL"*) ;;
                *) log "skipped $label (disabled by operator)" ;;
            esac
            continue
            ;;
    esac

    launchctl enable "gui/$UID_/$label" 2>/dev/null
    if launchctl bootstrap "gui/$UID_" "$plist" 2>/dev/null; then
        log "re-bootstrapped $label"
    else
        log "FAILED to re-bootstrap $label"
    fi
done

printf '%s' "$new_skipped" > "$SKIP_STATE" 2>/dev/null || true
