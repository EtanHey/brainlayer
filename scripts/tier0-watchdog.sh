#!/bin/sh
# Tier-0 guard for the Python-backed BrainLayer health-check LaunchAgent.

set -u

TIER0_LAUNCHCTL=${TIER0_LAUNCHCTL:-/bin/launchctl}
TIER0_STAT=${TIER0_STAT:-/usr/bin/stat}
TIER0_DATE=${TIER0_DATE:-/bin/date}
TIER0_ID=${TIER0_ID:-/usr/bin/id}
TIER0_DIRNAME=${TIER0_DIRNAME:-/usr/bin/dirname}
TIER0_MKDIR=${TIER0_MKDIR:-/bin/mkdir}
TIER0_GREP=${TIER0_GREP:-/usr/bin/grep}

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
TIER0_ALERT_STATE_PATH=${TIER0_ALERT_STATE_PATH:-$HOME/.local/share/brainlayer/tier0-watchdog-alert-state}
TIER0_RUN_STATE_PATH=${TIER0_RUN_STATE_PATH:-$HOME/.local/share/brainlayer/tier0-watchdog-last-run}
TIER0_STALE_SECONDS=${TIER0_STALE_SECONDS:-900}
TIER0_REPEAT_ALERT_SECONDS=${TIER0_REPEAT_ALERT_SECONDS:-1800}
TIER0_MISSED_RUN_GRACE_SECONDS=${TIER0_MISSED_RUN_GRACE_SECONDS:-600}

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
require_positive_integer TIER0_REPEAT_ALERT_SECONDS "$TIER0_REPEAT_ALERT_SECONDS"
require_positive_integer TIER0_MISSED_RUN_GRACE_SECONDS "$TIER0_MISSED_RUN_GRACE_SECONDS"

if [ -n "${TIER0_NOW_EPOCH:-}" ]; then
    now_epoch=$TIER0_NOW_EPOCH
else
    now_epoch=$($TIER0_DATE +%s 2>/dev/null) || now_epoch=
fi
require_epoch TIER0_NOW_EPOCH "$now_epoch"

log_tier0_event() {
    event=$1
    log_dir=$($TIER0_DIRNAME "$TIER0_LOG_PATH" 2>/dev/null) || return 1
    "$TIER0_MKDIR" -p "$log_dir" 2>/dev/null || return 1
    printf 'epoch=%s label=%s reason=%s\n' "$now_epoch" "$TIER0_LABEL" "$event" >> "$TIER0_LOG_PATH"
}

# Seconds since this watchdog's own previous run, or "" when there is no usable record.
seconds_since_own_previous_run() {
    [ -f "$TIER0_RUN_STATE_PATH" ] || return 0
    previous_run_epoch=
    read -r previous_run_epoch < "$TIER0_RUN_STATE_PATH" || return 0
    case "$previous_run_epoch" in
        ''|*[!0-9]*) return 0 ;;
    esac
    [ "$previous_run_epoch" -le "$now_epoch" ] || return 0
    printf '%s' "$((now_epoch - previous_run_epoch))"
}

record_own_run() {
    run_state_dir=$($TIER0_DIRNAME "$TIER0_RUN_STATE_PATH" 2>/dev/null) || return 1
    "$TIER0_MKDIR" -p "$run_state_dir" 2>/dev/null || return 1
    printf '%s\n' "$now_epoch" > "$TIER0_RUN_STATE_PATH"
}

log_incident() {
    reason=$1
    log_tier0_event "$reason"
}

should_alert() {
    failure_key=$1
    if [ ! -f "$TIER0_ALERT_STATE_PATH" ]; then
        return 0
    fi

    last_alert_epoch=
    last_failure_key=
    IFS="$(printf '\t')" read -r last_alert_epoch last_failure_key < "$TIER0_ALERT_STATE_PATH" || return 0
    case "$last_alert_epoch" in
        ''|*[!0-9]*) return 0 ;;
    esac
    if [ "$last_failure_key" != "$failure_key" ] || [ "$last_alert_epoch" -gt "$now_epoch" ]; then
        return 0
    fi
    alert_age=$((now_epoch - last_alert_epoch))
    [ "$alert_age" -ge "$TIER0_REPEAT_ALERT_SECONDS" ]
}

record_alert() {
    failure_key=$1
    alert_state_dir=$($TIER0_DIRNAME "$TIER0_ALERT_STATE_PATH" 2>/dev/null) || return 1
    "$TIER0_MKDIR" -p "$alert_state_dir" 2>/dev/null || return 1
    printf '%s\t%s\n' "$now_epoch" "$failure_key" > "$TIER0_ALERT_STATE_PATH"
}

reset_alert_cooldown() {
    if [ -f "$TIER0_ALERT_STATE_PATH" ]; then
        printf '0\tok\n' > "$TIER0_ALERT_STATE_PATH"
    fi
}

seconds_since_previous_run=$(seconds_since_own_previous_run)
# Fail CLOSED: withholding is only safe while we can actually advance our own mark. If
# the run state cannot be written, the recorded epoch freezes, every later gap looks like
# a sleep, and the watchdog would withhold staleness alerts forever -- silently, which is
# the one thing a tier-0 guard must never do.
if record_own_run; then
    own_run_recorded=1
else
    own_run_recorded=0
fi

target="$TIER0_DOMAIN/$TIER0_LABEL"
failure_reason=
failure_key=
label_loaded=0

if "$TIER0_LAUNCHCTL" print "$target" >/dev/null 2>&1; then
    label_loaded=1
else
    failure_reason=label_unloaded
    failure_key=label_unloaded
fi

if [ "$label_loaded" -eq 1 ]; then
    if [ ! -f "$TIER0_STATE_PATH" ]; then
        failure_reason=state_missing
        failure_key=state_missing
    else
        state_mtime=$($TIER0_STAT -f %m "$TIER0_STATE_PATH" 2>/dev/null) || state_mtime=
        case "$state_mtime" in
            ''|*[!0-9]*)
                failure_reason=state_mtime_unreadable
                failure_key=state_mtime_unreadable
                ;;
            *)
                if [ "$state_mtime" -gt "$now_epoch" ]; then
                    future_offset=$((state_mtime - now_epoch))
                    failure_reason="state_mtime_future offset=${future_offset}s"
                    failure_key=state_mtime_future
                else
                    state_age=$((now_epoch - state_mtime))
                    if [ "$state_age" -ge "$TIER0_STALE_SECONDS" ]; then
                        failure_reason="state_stale age=${state_age}s threshold=${TIER0_STALE_SECONDS}s"
                        failure_key=state_stale
                    fi
                fi
                ;;
        esac
        if [ -z "$failure_reason" ] && "$TIER0_GREP" -Eq '"slow_check"[[:space:]]*:[[:space:]]*true' "$TIER0_STATE_PATH" 2>/dev/null; then
            failure_reason=state_slow_check
            failure_key=state_slow_check
        fi
    fi
fi

if [ -z "$failure_reason" ]; then
    reset_alert_cooldown
    exit 0
fi

# The system may have been asleep. launchd runs no StartInterval job while it sleeps, so
# neither this watchdog nor the health-check it guards gets its turn -- yet state_stale
# measures WALL-CLOCK age and cannot tell a sleeping Mac from a dead job. Our own previous
# run dates the outage: if we too were skipped for longer than a couple of intervals, the
# health-check has not had its chance yet, so withhold the staleness verdict for one cycle
# and let the next run judge a machine that was actually awake. A genuinely dead
# health-check on an awake machine still alerts, one cycle later. Only staleness is a
# function of elapsed time; label_unloaded, state_missing, state_mtime_future and
# state_slow_check are not, and are never withheld.
if [ "$failure_key" = state_stale ] && [ "$own_run_recorded" -eq 1 ] \
    && [ -n "$seconds_since_previous_run" ] \
    && [ "$seconds_since_previous_run" -ge "$TIER0_MISSED_RUN_GRACE_SECONDS" ]; then
    log_tier0_event "state_stale_withheld_after_missed_runs gap=${seconds_since_previous_run}s $failure_reason" || :
    # Kickstart anyway: on wake this refreshes the state file now instead of waiting out
    # the health-check's own interval, so the next run judges fresh state. The alert
    # cooldown is deliberately left untouched -- a real incident keeps its place.
    "$TIER0_LAUNCHCTL" kickstart -k "$target" >/dev/null 2>&1 || :
    exit 0
fi

# Detection and every due incident log intentionally precede every recovery command.
if should_alert "$failure_key"; then
    if log_incident "$failure_reason"; then
        record_alert "$failure_key" || :
    fi
fi

if [ "$label_loaded" -eq 0 ]; then
    "$TIER0_LAUNCHCTL" bootstrap "$TIER0_DOMAIN" "$TIER0_HEALTH_PLIST_PATH" >/dev/null 2>&1 || :
fi
"$TIER0_LAUNCHCTL" kickstart -k "$target" >/dev/null 2>&1 || :

exit 1
