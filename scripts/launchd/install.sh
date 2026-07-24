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
#   ./scripts/launchd/install.sh health-check # Install stability health check only
#   ./scripts/launchd/install.sh tier0-watchdog # Install /bin/sh meta-watchdog only
#   ./scripts/launchd/install.sh throughput-watchdog # Install watcher throughput watchdog only
#   ./scripts/launchd/install.sh hotlane      # Install BrainBar hotlane embed/enrich daemon only
#   ./scripts/launchd/install.sh p0-counter   # Install daily P0 longitudinal counter only
#   ./scripts/launchd/install.sh remove       # Unload and remove all
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

stable_brainlayer_path() {
    local value="${1:-}"
    local prefix
    local after_cellar
    local after_version

    if [ -z "$value" ]; then
        printf "%s" "$value"
        return 0
    fi

    case "$value" in
        */Cellar/brainlayer/*)
            prefix="${value%%/Cellar/brainlayer/*}"
            after_cellar="${value#*/Cellar/brainlayer/}"
            after_version="${after_cellar#*/}"
            if [ "$after_version" = "$after_cellar" ]; then
                printf "%s/opt/brainlayer" "$prefix"
            else
                printf "%s/opt/brainlayer/%s" "$prefix" "$after_version"
            fi
            ;;
        *)
            printf "%s" "$value"
            ;;
    esac
}

LAUNCH_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/Library/Logs"
BRAINLAYER_LOG_DIR="$HOME/.local/share/brainlayer/logs"
BRAINLAYER_LIB_DIR="$HOME/.local/lib/brainlayer"
BRAINLAYER_DIR="$(stable_brainlayer_path "$(cd "$SCRIPT_DIR/../.." && pwd)")"
BRAINLAYER_LAUNCHD_DIR="$(stable_brainlayer_path "${BRAINLAYER_LAUNCHD_DIR:-$SCRIPT_DIR}")"
BRAINLAYER_BIN="$(stable_brainlayer_path "${BRAINLAYER_BIN:-$(which brainlayer 2>/dev/null || echo "$HOME/.local/bin/brainlayer")}")"
PYTHON_BIN="$(stable_brainlayer_path "${PYTHON_BIN:-$(command -v python3)}")"
BRAINLAYER_PYTHON="$(stable_brainlayer_path "${BRAINLAYER_PYTHON:-$PYTHON_BIN}")"
BRAINLAYER_ENV_FILE="${BRAINLAYER_ENV_FILE:-$HOME/.config/brainlayer/brainlayer.env}"
BRAINLAYER_ENV_RUN="$BRAINLAYER_LIB_DIR/brainlayer-env-run.sh"
TIER0_WATCHDOG_DST="$BRAINLAYER_LIB_DIR/tier0-watchdog.sh"
THROUGHPUT_WATCHDOG_DST="$BRAINLAYER_LIB_DIR/throughput-watchdog.py"
HOTLANE_BRAINBAR_DST="$BRAINLAYER_LIB_DIR/hotlane_brainbar_daemon.py"

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

    install -m 0755 "$src" "$BRAINLAYER_ENV_RUN" || return 1
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
    local label="com.brainlayer.${name}"
    local domain="gui/$UID/$label"
    local enable_error=""
    local retry_enable_after_bootstrap=0
    local unload_attempts="${BRAINLAYER_LAUNCHD_UNLOAD_ATTEMPTS:-20}"
    local unload_interval="${BRAINLAYER_LAUNCHD_UNLOAD_INTERVAL:-0.1}"
    local unload_attempt=1
    local confirmed_unloaded=0
    local supervisor_reloaded=0
    local initially_unloaded=0
    local initial_output=""
    local initial_pid=""
    local unload_output=""
    local current_pid=""

    if [ "$name" = "hotlane-brainbar" ]; then
        if initial_output="$(launchctl print "$domain" 2>/dev/null)"; then
            initial_pid="$(
                printf '%s\n' "$initial_output" \
                    | awk -F'= ' '/^[[:space:]]*pid = [0-9]+$/ { print $2; exit }'
            )"
        else
            initially_unloaded=1
        fi
    fi
    launchctl bootout "$domain" 2>/dev/null || true
    while [ "$unload_attempt" -le "$unload_attempts" ]; do
        if ! unload_output="$(launchctl print "$domain" 2>/dev/null)"; then
            confirmed_unloaded=1
            break
        fi
        unload_attempt=$((unload_attempt + 1))
        if [ "$unload_attempt" -le "$unload_attempts" ]; then
            sleep "$unload_interval"
        fi
    done
    if [ "$confirmed_unloaded" -ne 1 ] && [ "$name" = "hotlane-brainbar" ]; then
        current_pid="$(
            printf '%s\n' "$unload_output" \
                | awk -F'= ' '/^[[:space:]]*pid = [0-9]+$/ { print $2; exit }'
        )"
        if [ "$initially_unloaded" -eq 1 ] \
            || { [ -n "$initial_pid" ] && [ -n "$current_pid" ] && [ "$current_pid" != "$initial_pid" ]; }; then
            confirmed_unloaded=1
            supervisor_reloaded=1
            echo "WARN: $label was reloaded before the unload poll; continuing with runtime verification" >&2
        fi
    fi
    if [ "$confirmed_unloaded" -ne 1 ]; then
        echo "ERROR: $label did not unload before replacement; refusing to enable or bootstrap" >&2
        return 1
    fi
    if ! enable_error="$(launchctl enable "gui/$UID/$label" 2>&1)"; then
        if printf "%s" "$enable_error" | grep -qi "could not find service"; then
            echo "WARN: launchctl enable could not find $label before bootstrap; retrying after bootstrap" >&2
            retry_enable_after_bootstrap=1
        else
            printf "%s\n" "$enable_error" >&2
            echo "ERROR: launchctl enable failed for $label" >&2
            return 1
        fi
    fi
    if [ "$supervisor_reloaded" -ne 1 ]; then
        if ! launchctl bootstrap "gui/$UID" "$dst"; then
            if [ "$name" = "hotlane-brainbar" ] \
                && [ "$confirmed_unloaded" -eq 1 ] \
                && launchctl print "$domain" >/dev/null 2>&1; then
                echo "WARN: launchctl bootstrap raced with another supervisor for $label; service is loaded" >&2
            else
                echo "ERROR: launchctl bootstrap failed for $label" >&2
                return 1
            fi
        fi
    fi
    if [ "$retry_enable_after_bootstrap" -ne 0 ]; then
        if ! launchctl enable "gui/$UID/$label"; then
            echo "ERROR: launchctl enable failed for $label after bootstrap" >&2
            return 1
        fi
    fi
    if ! launchctl print "$domain" >/dev/null; then
        echo "ERROR: launchctl print failed for $label after bootstrap" >&2
        return 1
    fi
    echo "  Loaded: com.brainlayer.${name}"
}

unload_plist() {
    local name="$1"
    local dst="$LAUNCH_DIR/com.brainlayer.${name}.plist"
    launchctl unload "$dst" 2>/dev/null || true
    echo "  Unloaded: com.brainlayer.${name}"
}

install_hotlane_brainbar_daemon() {
    local script_src="$SCRIPT_DIR/hotlane_brainbar_daemon.py"

    if [ ! -f "$script_src" ]; then
        script_src="$SCRIPT_DIR/../hotlane_brainbar_daemon.py"
    fi
    if [ ! -f "$script_src" ]; then
        echo "ERROR: hotlane_brainbar_daemon.py not found beside or above $SCRIPT_DIR" >&2
        return 1
    fi

    install -m 0755 "$script_src" "$HOTLANE_BRAINBAR_DST" || return 1
    echo "Installed: $HOTLANE_BRAINBAR_DST"
}

verify_hotlane_runtime() {
    local label="com.brainlayer.hotlane-brainbar"
    local domain="gui/$UID/$label"
    local attempts="${BRAINLAYER_LAUNCHD_VERIFY_ATTEMPTS:-10}"
    local interval="${BRAINLAYER_LAUNCHD_VERIFY_INTERVAL:-1}"
    local attempt=1
    local output=""
    local pid=""
    local process_command=""
    local expected_command="$BRAINLAYER_PYTHON $HOTLANE_BRAINBAR_DST"
    local process_matches=0
    local previous_pid=""
    local stable_samples=0

    while [ "$attempt" -le "$attempts" ]; do
        output="$(launchctl print "$domain" 2>&1 || true)"
        pid="$(
            printf '%s\n' "$output" \
                | awk -F'= ' '/^[[:space:]]*pid = [0-9]+$/ { print $2; exit }'
        )"
        process_command=""
        process_matches=0
        if [ -n "$pid" ]; then
            process_command="$(ps -ww -p "$pid" -o command= 2>/dev/null || true)"
            case "$process_command" in
                "$expected_command"|"$expected_command "*)
                    process_matches=1
                    ;;
            esac
        fi

        if printf '%s\n' "$output" | grep -Eq '^[[:space:]]*state = running$' \
            && [ -n "$pid" ] \
            && [ "$process_matches" -eq 1 ]; then
            if [ "$pid" = "$previous_pid" ]; then
                stable_samples=$((stable_samples + 1))
            else
                previous_pid="$pid"
                stable_samples=1
            fi
            if [ "$stable_samples" -ge 2 ]; then
                echo "  Verified running: $label (pid $pid)"
                return 0
            fi
        else
            previous_pid=""
            stable_samples=0
        fi

        attempt=$((attempt + 1))
        if [ "$attempt" -le "$attempts" ]; then
            sleep "$interval"
        fi
    done

    echo "ERROR: hotlane runtime verification failed after $attempts attempts" >&2
    printf '%s\n' "$output" >&2
    if [ -n "$pid" ] && [ "$process_matches" -ne 1 ]; then
        echo "ERROR: pid $pid command does not match packaged hotlane daemon: $process_command" >&2
    fi
    launchctl bootout "$domain" >/dev/null 2>&1 || true
    return 1
}

install_plist() {
    local name="$1"
    local src="$SCRIPT_DIR/com.brainlayer.${name}.plist"
    local dst="$LAUNCH_DIR/com.brainlayer.${name}.plist"

    if [ ! -f "$src" ]; then
        echo "ERROR: $src not found"
        return 1
    fi

    if [ "$name" = "hotlane-brainbar" ]; then
        install_hotlane_brainbar_daemon || return 1
    fi

    install_env_runner || return 1
    verify_config_file || return 1

    if [ "$name" = "enrichment" ] || [ "$name" = "enrich" ]; then
        verify_gemini_env_file || return 1
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
        -e "s|__HOTLANE_BRAINBAR_DAEMON__|$HOTLANE_BRAINBAR_DST|g" \
        "$src" > "$dst" || return 1

    echo "Installed: $dst"
    echo "  Logs: $LOG_DIR/ and $BRAINLAYER_LOG_DIR/"

    if ! load_plist "$name"; then
        return 1
    fi
    if [ "$name" = "hotlane-brainbar" ] && ! verify_hotlane_runtime; then
        return 1
    fi
}

install_many() {
    local failures=0
    local name

    for name in "$@"; do
        if ! install_plist "$name"; then
            echo "ERROR: failed to install/load com.brainlayer.${name}" >&2
            failures=$((failures + 1))
        fi
    done

    if [ "$failures" -ne 0 ]; then
        echo "ERROR: failed to install/load $failures launchd service(s)" >&2
        return 1
    fi
}

install_backup_script() {
    local src="$SCRIPT_DIR/backup-daily.sh"
    local dst="$BRAINLAYER_LIB_DIR/backup-daily.sh"
    local escaped_brainlayer_dir

    if [ ! -f "$src" ]; then
        echo "ERROR: $src not found"
        return 1
    fi

    escaped_brainlayer_dir="$(printf '%s' "$BRAINLAYER_DIR" | sed 's/[\\&|]/\\&/g')" || return 1
    sed \
        -e "s|__BRAINLAYER_DIR_VALUE__|$escaped_brainlayer_dir|g" \
        "$src" > "$dst" || return 1
    chmod 755 "$dst" || return 1
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

    escaped_brainlayer_dir="$(printf '%s' "$BRAINLAYER_DIR" | sed 's/[\\&|]/\\&/g')" || return 1
    sed \
        -e "s|__BRAINLAYER_DIR_VALUE__|$escaped_brainlayer_dir|g" \
        "$src" > "$dst" || return 1
    chmod 755 "$dst" || return 1
    echo "Installed: $dst"
}

install_tier0_watchdog() {
    local script_src="$SCRIPT_DIR/tier0-watchdog.sh"
    local plist_src="$SCRIPT_DIR/com.brainlayer.tier0-watchdog.plist"
    local plist_dst="$LAUNCH_DIR/com.brainlayer.tier0-watchdog.plist"
    local escaped_home
    local escaped_tier0_watchdog_dst

    if [ ! -f "$script_src" ]; then
        script_src="$SCRIPT_DIR/../tier0-watchdog.sh"
    fi
    if [ ! -f "$script_src" ]; then
        echo "ERROR: tier0-watchdog.sh not found beside or above $SCRIPT_DIR"
        return 1
    fi
    if [ ! -f "$plist_src" ]; then
        echo "ERROR: $plist_src not found"
        return 1
    fi

    escaped_home="$(
        printf '%s' "$HOME" \
            | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g' -e 's/[\\&|]/\\&/g'
    )" || return 1
    escaped_tier0_watchdog_dst="$(
        printf '%s' "$TIER0_WATCHDOG_DST" \
            | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g' -e 's/[\\&|]/\\&/g'
    )" || return 1

    install -m 0755 "$script_src" "$TIER0_WATCHDOG_DST" || return 1
    sed \
        -e "s|__HOME__|$escaped_home|g" \
        -e "s|__TIER0_WATCHDOG_SCRIPT__|$escaped_tier0_watchdog_dst|g" \
        "$plist_src" > "$plist_dst" || return 1

    echo "Installed: $TIER0_WATCHDOG_DST"
    echo "Installed: $plist_dst"
    echo "  Logs: $LOG_DIR/ and $BRAINLAYER_LOG_DIR/"
    load_plist tier0-watchdog
}

install_throughput_watchdog() {
    local script_src="$SCRIPT_DIR/throughput-watchdog.py"
    local plist_src="$SCRIPT_DIR/com.brainlayer.throughput-watchdog.plist"
    local plist_dst="$LAUNCH_DIR/com.brainlayer.throughput-watchdog.plist"
    local escaped_env_file
    local escaped_env_run
    local escaped_home
    local escaped_python_bin
    local escaped_watchdog_dst

    if [ ! -f "$script_src" ]; then
        echo "ERROR: throughput-watchdog.py not found in $SCRIPT_DIR"
        return 1
    fi
    if [ ! -f "$plist_src" ]; then
        echo "ERROR: $plist_src not found"
        return 1
    fi

    install_env_runner || return 1
    verify_config_file || return 1

    escaped_home="$(
        printf '%s' "$HOME" \
            | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g' -e 's/[\\&|]/\\&/g'
    )" || return 1
    escaped_env_file="$(
        printf '%s' "$BRAINLAYER_ENV_FILE" \
            | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g' -e 's/[\\&|]/\\&/g'
    )" || return 1
    escaped_env_run="$(
        printf '%s' "$BRAINLAYER_ENV_RUN" \
            | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g' -e 's/[\\&|]/\\&/g'
    )" || return 1
    escaped_watchdog_dst="$(
        printf '%s' "$THROUGHPUT_WATCHDOG_DST" \
            | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g' -e 's/[\\&|]/\\&/g'
    )" || return 1
    escaped_python_bin="$(
        printf '%s' "$PYTHON_BIN" \
            | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g' -e 's/[\\&|]/\\&/g'
    )" || return 1

    install -m 0755 "$script_src" "$THROUGHPUT_WATCHDOG_DST" || return 1
    sed \
        -e "s|__HOME__|$escaped_home|g" \
        -e "s|__BRAINLAYER_ENV_FILE__|$escaped_env_file|g" \
        -e "s|__BRAINLAYER_ENV_RUN__|$escaped_env_run|g" \
        -e "s|__PYTHON_BIN__|$escaped_python_bin|g" \
        -e "s|__THROUGHPUT_WATCHDOG_SCRIPT__|$escaped_watchdog_dst|g" \
        "$plist_src" > "$plist_dst" || return 1

    echo "Installed: $THROUGHPUT_WATCHDOG_DST"
    echo "Installed: $plist_dst"
    echo "  Logs: $LOG_DIR/ and $BRAINLAYER_LOG_DIR/"
    load_plist throughput-watchdog
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
        install_many maintenance-nightly maintenance-weekly
        ;;
    health-check)
        install_plist health-check
        ;;
    tier0|tier0-watchdog)
        install_tier0_watchdog
        ;;
    throughput-watchdog)
        install_throughput_watchdog
        ;;
    hotlane|hotlane-brainbar)
        verify_gemini_env_file
        install_plist hotlane-brainbar
        ;;
    p0-counter)
        install_plist p0-counter
        ;;
    all)
        install_env_runner
        verify_config_file
        verify_gemini_env_file
        failures=0
        enrichment_ok=0
        if ! install_many index drain watch hotlane-brainbar; then
            failures=1
        fi
        if install_plist enrichment; then
            enrichment_ok=1
        else
            failures=1
        fi
        if ! install_many decay wal-checkpoint repair-fts; then
            failures=1
        fi
        if ! install_backup_script; then
            failures=1
        elif ! install_many backup-daily; then
            failures=1
        fi
        if ! install_jsonl_backup_script; then
            failures=1
        elif ! install_many jsonl-backup; then
            failures=1
        fi
        if ! install_many maintenance-nightly maintenance-weekly health-check p0-counter; then
            failures=1
        fi
        if ! install_tier0_watchdog; then
            failures=1
        fi
        if ! install_throughput_watchdog; then
            failures=1
        fi
        # Remove old enrich plist only after the replacement enrichment service loads.
        if [ "$enrichment_ok" -eq 1 ]; then
            remove_plist enrich 2>/dev/null || true
        fi
        if [ "$failures" -ne 0 ]; then
            exit 1
        fi
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
        remove_plist health-check 2>/dev/null || true
        remove_plist tier0-watchdog 2>/dev/null || true
        remove_plist throughput-watchdog 2>/dev/null || true
        remove_plist hotlane-brainbar 2>/dev/null || true
        remove_plist p0-counter 2>/dev/null || true
        rm -f "$BRAINLAYER_LIB_DIR/backup-daily.sh"
        rm -f "$BRAINLAYER_LIB_DIR/jsonl-backup.sh"
        rm -f "$TIER0_WATCHDOG_DST"
        rm -f "$THROUGHPUT_WATCHDOG_DST"
        rm -f "$HOTLANE_BRAINBAR_DST"
        ;;
    *)
        echo "Usage: $0 [index|watch|enrich|enrichment|decay|drain|hotlane|repair-fts|load [name]|unload [name]|checkpoint|backup|jsonl-backup|maintenance|maintenance-nightly|maintenance-weekly|health-check|tier0-watchdog|throughput-watchdog|p0-counter|all|remove]"
        exit 1
        ;;
esac

echo ""
echo "Done. Check logs at: $LOG_DIR/"
echo "Enrichment label: com.brainlayer.enrichment"
echo "Status: launchctl list | grep brainlayer"
