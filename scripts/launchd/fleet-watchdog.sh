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
#
# TWO deliberate-stop signals are honoured, because the fleet has two:
#   1. `launchctl disable` -- the operator signal, read via `launchctl print-disabled`.
#   2. the pause sentinel at ~/.local/share/brainlayer/pause.sentinel -- the AUTOMATED
#      signal. `brainlayer.maintenance` pauses services with a BARE bootout and no
#      `launchctl disable` (maintenance.py:_bootout_service), so mid-maintenance a label is
#      absent-and-not-disabled: exactly the state this loop reverses. Reviving enrichment or
#      drain mid-bulk-op is what AGENTS.md forbids ("Stop enrichment workers first") and how
#      the 4.7GB WAL happens. So a label named by an unexpired sentinel is left down, the way
#      maintenance.py:_resume_services does it.
# Both checks fail CLOSED: an unreadable `print-disabled`, or a sentinel that exists but does
# not parse, means nothing is revived this tick. With no readable stop-signal an operator or
# maintenance pause is indistinguishable from a crash, and reviving a deliberately stopped
# service is the worse error. The sentinel is parsed with plutil, not a regex.

UID_="$(id -u)"
LOG_DIR="$HOME/Library/Logs/brainlayer"
LOG="$LOG_DIR/fleet-watchdog.log"
# Labels reported as operator-disabled on the previous run, one per line. Used only to
# keep a standing disable from appending an identical skip line every 300 seconds.
SKIP_STATE="$LOG_DIR/fleet-watchdog-skipped"
# Same hardcoded location src/brainlayer/pause.py resolves; it is HOME-relative, not
# BRAINLAYER_DATA_DIR-relative, and this script must look exactly where the writers write.
PAUSE_SENTINEL="$HOME/.local/share/brainlayer/pause.sentinel"
# Bound the label scan; a watchdog must not be able to spin on a malformed sentinel.
PAUSE_LABEL_LIMIT=256
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

# Epoch seconds for an ISO-8601 stamp, empty when it cannot be read. Mirrors
# pause.py:_parse_iso_datetime: a naive stamp is UTC.
pause_epoch() {
    stamp="$(printf '%s' "$1" | sed -e 's/\.[0-9][0-9]*//')"
    stamp="$(printf '%s' "$stamp" | sed -e 's/Z$/+0000/' -e 's/\([+-][0-9][0-9]\):\([0-9][0-9]\)$/\1\2/')"
    case "$stamp" in
        *[+-][0-9][0-9][0-9][0-9])
            date -j -f '%Y-%m-%dT%H:%M:%S%z' "$stamp" +%s 2>/dev/null
            ;;
        *)
            TZ=UTC date -j -f '%Y-%m-%dT%H:%M:%S' "$stamp" +%s 2>/dev/null
            ;;
    esac
}

# Labels held down by an active pause sentinel, one per line.
paused_labels=""
if [ -e "$PAUSE_SENTINEL" ]; then
    # `plutil -lint` cannot read JSON (it lints plists), so the parse gate is a conversion
    # to stdout: rc 0 and a leading "{" means a real JSON object. The input file is never
    # rewritten. A sentinel that exists but is truncated, corrupt or not an object gets no
    # benefit of the doubt.
    if ! sentinel_json="$(plutil -convert json -o - -- "$PAUSE_SENTINEL" 2>/dev/null)"; then
        log "ERROR pause sentinel $PAUSE_SENTINEL does not parse; refusing to revive anything"
        exit 1
    fi
    case "$sentinel_json" in
        "{"*) ;;
        *)
            log "ERROR pause sentinel $PAUSE_SENTINEL is not a JSON object; refusing to revive anything"
            exit 1
            ;;
    esac
    pause_active=1
    if expires_at="$(plutil -extract expires_at raw -o - -- "$PAUSE_SENTINEL" 2>/dev/null)"; then
        expires_epoch="$(pause_epoch "$expires_at")"
        # An unreadable expires_at is never stale (pause.py returns stale=False), so the
        # pause keeps holding -- the safe direction.
        if [ -n "$expires_epoch" ] && [ "$(date -u +%s)" -gt "$expires_epoch" ]; then
            pause_active=0
        fi
    fi
    if [ "$pause_active" -eq 1 ]; then
        pause_index=0
        # Leading and trailing newline so a single "$NL<label>$NL" pattern matches any entry
        # and cannot match a prefix of a longer label.
        while [ "$pause_index" -lt "$PAUSE_LABEL_LIMIT" ]; do
            paused_label="$(plutil -extract "labels.$pause_index" raw -o - -- "$PAUSE_SENTINEL" 2>/dev/null)" || break
            paused_labels="$paused_labels$NL$paused_label"
            pause_index=$((pause_index + 1))
        done
        if [ -n "$paused_labels" ]; then
            paused_labels="$paused_labels$NL"
        fi
    fi
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

    skip_reason=""
    case "$disabled_listing" in
        *"\"$label\" => disabled"*|*"\"$label\" => true"*)
            skip_reason="disabled by operator"
            ;;
        *)
            case "$paused_labels" in
                *"$NL$label$NL"*)
                    skip_reason="pause sentinel is active"
                    ;;
            esac
            ;;
    esac
    if [ -n "$skip_reason" ]; then
        new_skipped="$new_skipped$skip_reason: $label$NL"
        case "$prev_skipped" in
            *"$NL$skip_reason: $label$NL"*) ;;
            *) log "skipped $label ($skip_reason)" ;;
        esac
        continue
    fi

    launchctl enable "gui/$UID_/$label" 2>/dev/null
    if launchctl bootstrap "gui/$UID_" "$plist" 2>/dev/null; then
        log "re-bootstrapped $label"
    else
        log "FAILED to re-bootstrap $label"
    fi
done

printf '%s' "$new_skipped" > "$SKIP_STATE" 2>/dev/null || true
