#!/usr/bin/env bash
# Drift-proof BrainBar updater.
#
# One command, idempotent, correct from ANY starting state: unmanaged app present,
# stale Homebrew registration, no registration, already current, app running.
# Never requires sudo or a TTY, so it works unattended over ssh.
#
# See docs.local/ and collab 2026-08-19-drift-proof-mac-sync.md for the ratified contract.
set -euo pipefail

# --- configuration (env overrides exist so tests can stub every external tool) -----------
CASK_TOKEN="${BRAINLAYER_UPDATE_BRAINBAR_CASK_TOKEN:-etanhey/layers/brainbar}"
CASK_NAME="${CASK_TOKEN##*/}"
APP_PATH="${BRAINLAYER_UPDATE_BRAINBAR_APP:-/Applications/BrainBar.app}"
# Rule 5: bare `brew` is not on the M1's non-interactive ssh PATH.
BREW_BIN="${BRAINLAYER_UPDATE_BREW_BIN:-/opt/homebrew/bin/brew}"
TAP_NAME="${BRAINLAYER_UPDATE_TAP_NAME:-etanhey/layers}"
TAP_DIR="${BRAINLAYER_UPDATE_TAP_DIR:-}"
TAP_BRANCH="${BRAINLAYER_UPDATE_TAP_BRANCH:-main}"
QUARANTINE_ROOT="${BRAINLAYER_UPDATE_QUARANTINE_DIR:-$HOME/.brainlayer/brainbar-caskroom-quarantine}"
SOCKET_PATH="${BRAINLAYER_UPDATE_SOCKET_PATH:-/tmp/brainbar.sock}"
LAUNCHCTL_BIN="${BRAINLAYER_UPDATE_LAUNCHCTL_BIN:-/bin/launchctl}"
DEFAULTS_BIN="${BRAINLAYER_UPDATE_DEFAULTS_BIN:-/usr/bin/defaults}"
GIT_BIN="${BRAINLAYER_UPDATE_GIT_BIN:-git}"
UI_LABEL="com.brainlayer.brainbar"
DAEMON_LABEL="com.brainlayer.brainbar-daemon"

DRY_RUN=0
VERIFY_ONLY=0
SKIP_TAP_UPDATE="${BRAINLAYER_UPDATE_SKIP_TAP_UPDATE:-0}"
SKIP_VERIFY="${BRAINLAYER_UPDATE_SKIP_VERIFY:-0}"

usage() {
    cat <<EOF
Usage: brainlayer-update-brainbar.sh [--dry-run] [--verify-only] [--skip-tap-update]

Brings this Mac's BrainBar to the canonical tapped version from any starting state,
without sudo and without a TTY. Running it twice is a no-op.

  --dry-run          Print the resolved plan; change nothing.
  --verify-only      Skip the update; just report drift and green/red status.
  --skip-tap-update  Do not git-pull the tap (offline / already fresh).

What it does:
  1. Update the $TAP_NAME tap explicitly (bare 'brew update' does NOT refresh it,
     and the tap has no upstream tracking branch, so a bare 'git pull' fails).
  2. Detect drift: 'brew list --versions --cask $CASK_NAME' vs the real
     CFBundleShortVersionString inside $APP_PATH.
  3. On drift, quarantine the stale Caskroom registration (a user-owned mv) and
     'brew install --cask --force' to ADOPT the app in place.
     It never runs 'brew upgrade'/'brew reinstall': both execute the OLD installed
     version's uninstall recipe, read from Caskroom/<cask>/.metadata/ — never the
     newly tapped one. A pre-fix 'delete:' stanza there shells to 'sudo rm', which
     has no TTY over ssh, aborts, and destroys the LaunchAgents without reinstalling.
  4. Refuse BEFORE touching anything if any path we would remove is root-owned.
  5. Verify at the end: app version, cask version, formula, process, launchd
     services, socket. Exit non-zero and say why if any of it is red.

recovery-no-sudo:
  If brew ever fails mid-uninstall on root-owned leftovers, do not rebuild locally.
  The notarized .app and Homebrew receipt may still survive. Restore the user
  LaunchAgents from the app bundle and then rerun this script:

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
        --dry-run) DRY_RUN=1 ;;
        --verify-only) VERIFY_ONLY=1 ;;
        --skip-tap-update) SKIP_TAP_UPDATE=1 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "[brainlayer-update-brainbar] ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

log() { printf '%s\n' "$*"; }
err() { printf '[brainlayer-update-brainbar] ERROR: %s\n' "$*" >&2; }
warn() { printf '[brainlayer-update-brainbar] WARN: %s\n' "$*" >&2; }

run_cmd() {
    log "+ $*"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        return 0
    fi
    "$@"
}

# --- rule 5: absolute brew ---------------------------------------------------------------
resolve_brew() {
    if [[ -x "$BREW_BIN" ]]; then
        return 0
    fi
    local fallback
    if fallback="$(command -v brew 2>/dev/null)" && [[ -n "$fallback" ]]; then
        warn "$BREW_BIN is not executable; falling back to $fallback"
        BREW_BIN="$fallback"
        return 0
    fi
    err "Homebrew not found at $BREW_BIN and not on PATH."
    err "Set BRAINLAYER_UPDATE_BREW_BIN to the absolute brew path."
    exit 127
}

# --- rule 4: explicit tap refresh ---------------------------------------------------------
resolve_tap_dir() {
    if [[ -n "$TAP_DIR" ]]; then
        return 0
    fi
    local repo user name
    repo="$("$BREW_BIN" --repository 2>/dev/null || true)"
    user="${TAP_NAME%%/*}"
    name="${TAP_NAME##*/}"
    if [[ -n "$repo" ]]; then
        TAP_DIR="$repo/Library/Taps/$user/homebrew-$name"
    fi
}

update_tap() {
    if [[ "$SKIP_TAP_UPDATE" = "1" ]]; then
        log "Tap update: skipped (--skip-tap-update)"
        return 0
    fi
    resolve_tap_dir
    if [[ -z "$TAP_DIR" || ! -d "$TAP_DIR/.git" ]]; then
        log "Tap $TAP_NAME is not tapped yet; tapping it."
        run_cmd "$BREW_BIN" tap "$TAP_NAME"
        return 0
    fi
    # The tap has no upstream tracking branch, so a bare `git pull` fails. Be explicit.
    if ! run_cmd "$GIT_BIN" -C "$TAP_DIR" pull --ff-only origin "$TAP_BRANCH"; then
        warn "Could not fast-forward $TAP_DIR from origin/$TAP_BRANCH; continuing with the tap as-is."
    fi
}

# --- state readers -------------------------------------------------------------------------
app_version() {
    [[ -d "$APP_PATH" ]] || return 1
    "$DEFAULTS_BIN" read "$APP_PATH/Contents/Info.plist" CFBundleShortVersionString 2>/dev/null
}

registered_version() {
    local line
    line="$("$BREW_BIN" list --versions --cask "$CASK_NAME" 2>/dev/null || true)"
    [[ -n "$line" ]] || return 1
    printf '%s\n' "$line" | awk '{print $2}'
}

offered_version() {
    "$BREW_BIN" info --cask --json=v2 "$CASK_TOKEN" 2>/dev/null | python3 -c '
import json, sys
try:
    payload = json.load(sys.stdin)
except Exception:
    raise SystemExit(1)
casks = payload.get("casks") or []
if not casks:
    raise SystemExit(1)
version = casks[0].get("version") or ""
if not version:
    raise SystemExit(1)
print(version)
'
}

caskroom_dir() {
    local prefix
    prefix="$("$BREW_BIN" --prefix 2>/dev/null || printf '/opt/homebrew')"
    printf '%s/Caskroom/%s\n' "$prefix" "$CASK_NAME"
}

# --- rule 3: never require sudo -------------------------------------------------------------
# Stop BEFORE destroying anything if any path we would move or replace is root-owned.
assert_no_root_owned_paths() {
    local me offender=""
    me="$(id -u)"
    local candidates=(
        "$APP_PATH"
        "$(caskroom_dir)"
        "$HOME/Library/LaunchAgents/$UI_LABEL.plist"
        "$HOME/Library/LaunchAgents/$DAEMON_LABEL.plist"
    )
    local path owner
    for path in "${candidates[@]}"; do
        [[ -e "$path" ]] || continue
        owner="$(stat -f '%u' "$path" 2>/dev/null || printf '%s' "$me")"
        if [[ "$owner" != "$me" ]]; then
            offender="$offender  $path (uid $owner)\n"
        fi
    done
    if [[ -n "$offender" ]]; then
        err "Refusing to continue: these paths are not owned by uid $me, so any"
        err "removal would shell out to sudo and abort without a TTY:"
        printf '%b' "$offender" >&2
        err "Nothing has been changed. Reclaim ownership once, then rerun:"
        err "  sudo chown -R \$(id -u):\$(id -g) <path>"
        exit 3
    fi
}

# --- drift detection (rule 1) ---------------------------------------------------------------
APP_VERSION=""
REGISTERED_VERSION=""
OFFERED_VERSION=""
DRIFT_KIND=""   # none | unmanaged | stale-ledger | missing | outdated

detect_state() {
    APP_VERSION="$(app_version || true)"
    REGISTERED_VERSION="$(registered_version || true)"
    OFFERED_VERSION="$(offered_version || true)"

    if [[ -z "$OFFERED_VERSION" ]]; then
        err "Could not read the offered version for $CASK_TOKEN from the tap."
        err "Is the tap present? Try: $BREW_BIN tap $TAP_NAME"
        exit 4
    fi

    if [[ -z "$APP_VERSION" && -z "$REGISTERED_VERSION" ]]; then
        DRIFT_KIND="missing"
    elif [[ -n "$APP_VERSION" && -z "$REGISTERED_VERSION" ]]; then
        # An app brew does not know about — exactly today's VoiceBar disease.
        DRIFT_KIND="unmanaged"
    elif [[ -z "$APP_VERSION" && -n "$REGISTERED_VERSION" ]]; then
        DRIFT_KIND="stale-ledger"
    elif [[ "$APP_VERSION" != "$REGISTERED_VERSION" ]]; then
        DRIFT_KIND="stale-ledger"
    elif [[ "$APP_VERSION" != "$OFFERED_VERSION" ]]; then
        DRIFT_KIND="outdated"
    else
        DRIFT_KIND="none"
    fi
}

print_state() {
    log "BrainBar drift-proof update"
    log "  mode:        $([[ "$DRY_RUN" -eq 1 ]] && printf 'dry-run' || { [[ "$VERIFY_ONLY" -eq 1 ]] && printf 'verify-only' || printf 'apply'; })"
    log "  brew:        $BREW_BIN"
    log "  cask:        $CASK_TOKEN"
    log "  app path:    $APP_PATH"
    log "  app version: ${APP_VERSION:-<absent>}"
    log "  registered:  ${REGISTERED_VERSION:-<not registered with brew>}"
    log "  offered:     $OFFERED_VERSION"
    log "  drift:       $DRIFT_KIND"
}

quarantine_stale_registration() {
    local caskroom stamp dest
    caskroom="$(caskroom_dir)"
    [[ -e "$caskroom" ]] || return 0
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    dest="$QUARANTINE_ROOT/$stamp"
    log "Clearing the stale Caskroom registration (user-owned mv, fully reversible)."
    run_cmd mkdir -p "$dest"
    run_cmd mv "$caskroom" "$dest/"
    log "  quarantined -> $dest/$CASK_NAME"
}

bootout_services() {
    local domain label
    domain="gui/$(id -u)"
    for label in "$UI_LABEL" "$DAEMON_LABEL"; do
        log "+ $LAUNCHCTL_BIN bootout $domain/$label (best effort)"
        if [[ "$DRY_RUN" -eq 0 ]]; then
            "$LAUNCHCTL_BIN" bootout "$domain/$label" >/dev/null 2>&1 || true
        fi
    done
}

adopt_install() {
    # --force adopts an app already sitting at the target path. Without it brew says
    # "It seems there is already an App at '<path>'" and does nothing.
    run_cmd "$BREW_BIN" install --cask --force "$CASK_TOKEN"
}

apply_update() {
    case "$DRIFT_KIND" in
        none)
            log "Already canonical at $OFFERED_VERSION — nothing to do."
            ;;
        missing)
            log "BrainBar is not installed. Installing $OFFERED_VERSION."
            adopt_install
            ;;
        unmanaged|stale-ledger|outdated)
            log "Drift detected ($DRIFT_KIND). Adopting $OFFERED_VERSION without running any uninstall recipe."
            quarantine_stale_registration
            bootout_services
            adopt_install
            ;;
        *)
            err "Unhandled drift kind: $DRIFT_KIND"
            exit 5
            ;;
    esac
}

# --- rule 6: verify at the end, fail loudly --------------------------------------------------
VERIFY_FAILED=0
check() {
    local label="$1" ok="$2" detail="$3"
    if [[ "$ok" = "1" ]]; then
        log "  [ok]   $label: $detail"
    else
        log "  [FAIL] $label: $detail"
        VERIFY_FAILED=1
    fi
}

verify() {
    local v ok
    log "Verification:"

    v="$(app_version || true)"
    [[ "$v" = "$OFFERED_VERSION" ]] && ok=1 || ok=0
    check "app version" "$ok" "${v:-<absent>} (expected $OFFERED_VERSION)"

    v="$(registered_version || true)"
    [[ "$v" = "$OFFERED_VERSION" ]] && ok=1 || ok=0
    check "cask version" "$ok" "${v:-<not registered>} (expected $OFFERED_VERSION)"

    v="$("$BREW_BIN" list --versions brainlayer 2>/dev/null | awk '{print $2}')"
    [[ -n "$v" ]] && ok=1 || ok=0
    check "brainlayer formula" "$ok" "${v:-<not installed>}"

    if pgrep -x BrainBar >/dev/null 2>&1; then ok=1; else ok=0; fi
    check "BrainBar process" "$ok" "$([[ "$ok" = 1 ]] && printf running || printf 'not running')"

    local domain label
    domain="gui/$(id -u)"
    for label in "$DAEMON_LABEL" "$UI_LABEL"; do
        if "$LAUNCHCTL_BIN" print "$domain/$label" >/dev/null 2>&1; then ok=1; else ok=0; fi
        check "launchd $label" "$ok" "$([[ "$ok" = 1 ]] && printf loaded || printf 'not loaded')"
    done

    [[ -S "$SOCKET_PATH" ]] && ok=1 || ok=0
    check "socket" "$ok" "$SOCKET_PATH $([[ "$ok" = 1 ]] && printf present || printf missing)"

    if [[ "$VERIFY_FAILED" -eq 1 ]]; then
        err "BrainBar is NOT green. See the [FAIL] lines above."
        exit 1
    fi
    log "BrainBar is green at $OFFERED_VERSION."
}

main() {
    resolve_brew
    if [[ "$VERIFY_ONLY" -eq 0 ]]; then
        update_tap
    fi
    detect_state
    print_state

    if [[ "$VERIFY_ONLY" -eq 1 ]]; then
        verify
        return 0
    fi

    if [[ "$DRIFT_KIND" != "none" && "$DRY_RUN" -eq 0 ]]; then
        assert_no_root_owned_paths
    fi

    apply_update

    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "Dry run complete. Nothing was changed."
        return 0
    fi

    if [[ "$SKIP_VERIFY" = "1" ]]; then
        log "Verification skipped (BRAINLAYER_UPDATE_SKIP_VERIFY=1)."
        return 0
    fi
    verify
}

main
