#!/usr/bin/env bash
# Source BrainLayer's private env file before execing a launchd-managed command.
set -euo pipefail

ENV_FILE="${BRAINLAYER_ENV_FILE:-$HOME/.config/brainlayer/brainlayer.env}"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: BrainLayer env file not found at $ENV_FILE" >&2
    echo "Run 'brainlayer init' or create it from scripts/launchd/brainlayer.env.example." >&2
    exit 78
fi

env_owner_uid="$(stat -c '%u' "$ENV_FILE" 2>/dev/null || stat -f '%u' "$ENV_FILE" 2>/dev/null || true)"
current_uid="$(id -u)"
if [ -z "$env_owner_uid" ] || { [ "$env_owner_uid" != "$current_uid" ] && [ "$env_owner_uid" != "0" ]; }; then
    echo "ERROR: BrainLayer env file must be owned by the current user or root: $ENV_FILE" >&2
    exit 78
fi

if [ -n "$(find "$ENV_FILE" -prune -perm -0002 -print 2>/dev/null)" ]; then
    echo "ERROR: BrainLayer env file must not be world-writable: $ENV_FILE" >&2
    exit 78
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

is_false() {
    case "${1:-}" in
        0|false|False|FALSE|no|No|NO|off|Off|OFF|disabled|Disabled|DISABLED) return 0 ;;
        *) return 1 ;;
    esac
}

sleep_then_exit_disabled() {
    local reason="$1"
    local sleep_seconds="${BRAINLAYER_DISABLED_SLEEP_SECONDS:-3600}"
    echo "BrainLayer launchd service ${BRAINLAYER_LAUNCHD_SERVICE:-unknown} disabled by config: $reason" >&2
    if [ "$sleep_seconds" != "0" ]; then
        sleep "$sleep_seconds"
    fi
    exit 0
}

if ! is_false "${BRAINLAYER_ENRICH_ENABLED:-1}"; then
    export BRAINLAYER_AUTO_ENRICH="1"
elif [ -n "${BRAINLAYER_ENRICH_ENABLED:-}" ]; then
    export BRAINLAYER_AUTO_ENRICH="0"
fi

if [ "${BRAINLAYER_SKIP_DISABLE_GATES:-0}" != "1" ]; then
    if is_false "${BRAINLAYER_SYSTEM_ENABLED:-1}"; then
        sleep_then_exit_disabled "BRAINLAYER_SYSTEM_ENABLED"
    fi

    if [ -n "${BRAINLAYER_LAUNCHD_SERVICE:-}" ]; then
        service_key="$(printf '%s' "$BRAINLAYER_LAUNCHD_SERVICE" | tr '[:lower:]-' '[:upper:]_')"
        service_enabled_var="BRAINLAYER_LAUNCHD_${service_key}_ENABLED"
        service_enabled="${!service_enabled_var:-1}"
        if is_false "$service_enabled"; then
            sleep_then_exit_disabled "$service_enabled_var"
        fi
        if [ "$BRAINLAYER_LAUNCHD_SERVICE" = "enrichment" ] && is_false "${BRAINLAYER_ENRICH_ENABLED:-1}"; then
            sleep_then_exit_disabled "BRAINLAYER_ENRICH_ENABLED"
        fi
    fi
fi

if [ "${BRAINLAYER_REQUIRE_GOOGLE_API_KEY:-0}" = "1" ] && [ -z "${GOOGLE_API_KEY:-${GOOGLE_GENERATIVE_AI_API_KEY:-}}" ]; then
    echo "ERROR: GOOGLE_API_KEY not set by $ENV_FILE" >&2
    exit 78
fi

exec "$@"
