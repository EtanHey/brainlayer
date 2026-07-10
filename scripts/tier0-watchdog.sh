#!/bin/sh
# Tier-0 guard for the Python-backed BrainLayer health-check LaunchAgent.

set -u

TIER0_LAUNCHCTL=${TIER0_LAUNCHCTL:-/bin/launchctl}
TIER0_STAT=${TIER0_STAT:-/usr/bin/stat}
TIER0_OSASCRIPT=${TIER0_OSASCRIPT:-/usr/bin/osascript}
TIER0_CURL=${TIER0_CURL:-/usr/bin/curl}
TIER0_DATE=${TIER0_DATE:-/bin/date}
TIER0_ID=${TIER0_ID:-/usr/bin/id}
TIER0_SLEEP=${TIER0_SLEEP:-/bin/sleep}
TIER0_DIRNAME=${TIER0_DIRNAME:-/usr/bin/dirname}
TIER0_MKDIR=${TIER0_MKDIR:-/bin/mkdir}

TIER0_LABEL=${TIER0_LABEL:-com.brainlayer.health-check}
if [ -z "${TIER0_DOMAIN:-}" ]; then
    tier0_uid=$($TIER0_ID -u 2>/dev/null) || tier0_uid=
    if [ -z "$tier0_uid" ]; then
        printf '%s\n' "tier0-watchdog: unable to determine launchd user domain" >&2
        exit 2
    fi
    TIER0_DOMAIN="gui/$tier0_uid"
fi

TIER0_STATE_PATH=${TIER0_STATE_PATH:-$HOME/.local/share/brainlayer/health-check-state.json}
TIER0_HEALTH_PLIST_PATH=${TIER0_HEALTH_PLIST_PATH:-$HOME/Library/LaunchAgents/com.brainlayer.health-check.plist}
TIER0_LOG_PATH=${TIER0_LOG_PATH:-$HOME/.local/share/brainlayer/logs/tier0-watchdog.log}
TIER0_NOTIFY_ENDPOINT=${TIER0_NOTIFY_ENDPOINT:-http://localhost:3847/notify}
TIER0_STALE_SECONDS=${TIER0_STALE_SECONDS:-1200}
TIER0_ALERT_TIMEOUT_SECONDS=${TIER0_ALERT_TIMEOUT_SECONDS:-3}
TIER0_NOTIFY_TIMEOUT_SECONDS=${TIER0_NOTIFY_TIMEOUT_SECONDS:-3}

require_positive_integer() {
    variable_name=$1
    variable_value=$2
    case "$variable_value" in
        ''|*[!0-9]*|0)
            printf '%s\n' "tier0-watchdog: $variable_name must be a positive integer" >&2
            exit 2
            ;;
    esac
}

require_epoch() {
    variable_name=$1
    variable_value=$2
    case "$variable_value" in
        ''|*[!0-9]*)
            printf '%s\n' "tier0-watchdog: $variable_name must be a non-negative integer" >&2
            exit 2
            ;;
    esac
}

require_positive_integer TIER0_STALE_SECONDS "$TIER0_STALE_SECONDS"
require_positive_integer TIER0_ALERT_TIMEOUT_SECONDS "$TIER0_ALERT_TIMEOUT_SECONDS"
require_positive_integer TIER0_NOTIFY_TIMEOUT_SECONDS "$TIER0_NOTIFY_TIMEOUT_SECONDS"

if [ -n "${TIER0_NOW_EPOCH:-}" ]; then
    now_epoch=$TIER0_NOW_EPOCH
else
    now_epoch=$($TIER0_DATE +%s 2>/dev/null) || now_epoch=
fi
require_epoch TIER0_NOW_EPOCH "$now_epoch"

wait_for_alerts() {
    remaining=$TIER0_ALERT_TIMEOUT_SECONDS

    while [ "$remaining" -gt 0 ]; do
        any_running=0
        for child_pid in "$@"; do
            if kill -0 "$child_pid" 2>/dev/null; then
                any_running=1
            fi
        done
        if [ "$any_running" -eq 0 ]; then
            break
        fi
        if ! "$TIER0_SLEEP" 1 2>/dev/null; then
            remaining=0
            break
        fi
        remaining=$((remaining - 1))
    done

    for child_pid in "$@"; do
        if kill -0 "$child_pid" 2>/dev/null; then
            kill "$child_pid" 2>/dev/null || :
        fi
    done
    for child_pid in "$@"; do
        wait "$child_pid" 2>/dev/null || :
    done
}

alert_all_channels() {
    reason=$1
    notify_payload='{"title":"BrainLayer Tier-0 alert","body":"Health-check is unavailable or stale; see the Tier-0 log.","source":"alerts"}'

    (
        log_dir=$($TIER0_DIRNAME "$TIER0_LOG_PATH" 2>/dev/null) || exit 1
        "$TIER0_MKDIR" -p "$log_dir" 2>/dev/null || exit 1
        printf 'epoch=%s label=%s reason=%s\n' "$now_epoch" "$TIER0_LABEL" "$reason" >> "$TIER0_LOG_PATH"
    ) &
    log_pid=$!

    "$TIER0_OSASCRIPT" \
        -e 'display notification "Health-check is unavailable or stale; see the Tier-0 log." with title "BrainLayer Tier-0 watchdog"' \
        >/dev/null 2>&1 &
    osascript_pid=$!

    "$TIER0_CURL" -fsS \
        --max-time "$TIER0_NOTIFY_TIMEOUT_SECONDS" \
        -X POST "$TIER0_NOTIFY_ENDPOINT" \
        -H 'Content-Type: application/json' \
        --data "$notify_payload" \
        >/dev/null 2>&1 &
    curl_pid=$!

    wait_for_alerts "$log_pid" "$osascript_pid" "$curl_pid"
}

target="$TIER0_DOMAIN/$TIER0_LABEL"
failure_reason=
label_loaded=0

if "$TIER0_LAUNCHCTL" print "$target" >/dev/null 2>&1; then
    label_loaded=1
else
    failure_reason=label_unloaded
fi

if [ "$label_loaded" -eq 1 ]; then
    if [ ! -f "$TIER0_STATE_PATH" ]; then
        failure_reason=state_missing
    else
        state_mtime=$($TIER0_STAT -f %m "$TIER0_STATE_PATH" 2>/dev/null) || state_mtime=
        case "$state_mtime" in
            ''|*[!0-9]*)
                failure_reason=state_mtime_unreadable
                ;;
            *)
                if [ "$state_mtime" -gt "$now_epoch" ]; then
                    future_offset=$((state_mtime - now_epoch))
                    failure_reason="state_mtime_future offset=${future_offset}s"
                else
                    state_age=$((now_epoch - state_mtime))
                    if [ "$state_age" -ge "$TIER0_STALE_SECONDS" ]; then
                        failure_reason="state_stale age=${state_age}s threshold=${TIER0_STALE_SECONDS}s"
                    fi
                fi
                ;;
        esac
    fi
fi

if [ -z "$failure_reason" ]; then
    exit 0
fi

# Detection and all alert attempts intentionally precede every recovery command.
alert_all_channels "$failure_reason"

if [ "$label_loaded" -eq 0 ]; then
    "$TIER0_LAUNCHCTL" bootstrap "$TIER0_DOMAIN" "$TIER0_HEALTH_PLIST_PATH" >/dev/null 2>&1 || :
fi
"$TIER0_LAUNCHCTL" kickstart -k "$target" >/dev/null 2>&1 || :

exit 1
