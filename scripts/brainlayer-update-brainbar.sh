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
if [[ -n "${BRAINLAYER_UPDATE_BRAINBAR_CASK_REF:-}" ]]; then
    CASK_REF="$BRAINLAYER_UPDATE_BRAINBAR_CASK_REF"
elif [[ -n "${BRAINLAYER_UPDATE_BRAINBAR_CASK_TOKEN:-}" ]]; then # deprecated, still honoured
    CASK_REF="$BRAINLAYER_UPDATE_BRAINBAR_CASK_TOKEN"
else
    CASK_REF="etanhey/layers/brainbar"
fi
CASK_NAME="${CASK_REF##*/}"
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
XATTR_BIN="${BRAINLAYER_UPDATE_XATTR_BIN:-/usr/bin/xattr}"
CODESIGN_BIN="${BRAINLAYER_UPDATE_CODESIGN_BIN:-/usr/bin/codesign}"
SPCTL_BIN="${BRAINLAYER_UPDATE_SPCTL_BIN:-/usr/sbin/spctl}"
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
  3. Prefer 'brew upgrade --cask' for an intact, consistent registration so Homebrew
     inherits the existing user's Gatekeeper approval. On drift/corrupt receipts,
     quarantine the stale registration (a reversible user-owned mv) and adopt with
     'brew install --cask --force'. Legacy uninstall delete recipes use this recovery
     path to avoid sudo. Carry approval on recovery only after signature, notarization
     and signing-identity checks; preserve quarantine provenance, never strip it.
  4. Refuse BEFORE touching anything if any path we would remove is root-owned.
  5. Verify at the end: app version, cask version, formula, canonical CLI,
     PATH resolution, process, launchd services, socket. Exit non-zero and say
     why if any of it is red.

recovery-no-sudo:
  If brew ever fails mid-uninstall on root-owned leftovers, do not rebuild locally.
  The notarized .app and Homebrew receipt may still survive. Bundled
  Contents/Resources/LaunchAgents plists are templates; do not bootstrap copies
  with unresolved placeholders. Set the paths below to the existing approved current-account
  installation and endpoint/database. Stop if that account-owned installed venv
  or the bundled renderer is unavailable; do not substitute a PATH Python or create a new setup.
  Path resolution below assumes this shell already has the approved account's configured environment;
  it does not source an env file. Stop unless the resolved path matches the approved existing database.
  Render/install the user LaunchAgents, then rerun this script:

    app="/Applications/BrainBar.app"
    venv="/absolute/path/to/account-owned/installed/venv"
    socket="/absolute/path/to/existing/approved/socket"
    approved_db="/absolute/path/to/existing/approved/database"
    domain="gui/\$(id -u)"
    test -x "\$venv/bin/python" && test -x "\$venv/bin/brainlayer" || exit
    test -f "\$app/Contents/Resources/install-services.py" || exit
    db="\$("\$venv/bin/python" -I -c 'from brainlayer.paths import resolve_db_path; print(resolve_db_path())')" || exit
    test "\$db" = "\$approved_db" && test -f "\$db" || exit
    "\$venv/bin/python" -I "\$app/Contents/Resources/install-services.py" \
      --app "\$app" --python "\$venv/bin/python" --cli "\$venv/bin/brainlayer" \
      --socket "\$socket" --db "\$db" --install || exit
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
        err "Could not fast-forward $TAP_DIR from origin/$TAP_BRANCH; aborting before using stale metadata."
        return 1
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
    "$BREW_BIN" info --cask --json=v2 "$CASK_REF" 2>/dev/null | python3 -c '
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

# Only execute a saved uninstall recipe if its JSON receipt and staged app link agree.
# Legacy Ruby-only/unreadable receipts fail closed to the existing adoption path.
registration_intact() {
    python3 - "$(caskroom_dir)" "$REGISTERED_VERSION" "$APP_PATH" "$CASK_NAME" <<'PY'
import json
import sys
from pathlib import Path

root, version, app, token = sys.argv[1:]
root, app = Path(root), Path(app)
try:
    link = root / version / app.name
    # Match Homebrew's selection across all versions by the newest timestamp.
    latest = max((root / ".metadata").glob("*/*"), key=lambda p: p.name)
    receipt = json.loads((latest / "Casks" / f"{token}.json").read_text())
    tab_path = root / ".metadata/INSTALL_RECEIPT.json"
    tab = json.loads(tab_path.read_text()) if tab_path.exists() else {}
    artifacts = receipt.get("artifacts", tab.get("uninstall_artifacts"))
    safe = all(not any(k in artifact for k in ("uninstall_preflight", "uninstall_postflight"))
               and all("delete" not in stanza for stanza in artifact.get("uninstall", []))
               for artifact in artifacts + tab.get("uninstall_artifacts", []))
    valid = (not root.is_symlink() and latest.parent.name == version
             and link.is_symlink() and link.resolve(strict=True) == app.resolve(strict=True)
             and receipt.get("token", token) == token and receipt.get("version", version) == version
             and tab.get("source", {}).get("version", version) == version
             and not tab.get("uninstall_flight_blocks", False)
             and {"app": [app.name]} in artifacts and safe)
except (OSError, ValueError, KeyError, TypeError, AttributeError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

OLD_APPROVED=0
OLD_IDENTITY=""
QUARANTINE_ATTRIBUTE=com.apple.quarantine

signing_identity() {
    local details team identifier
    details="$("$CODESIGN_BIN" -dv --verbose=4 "$APP_PATH" 2>&1)" || return 1
    team="$(printf '%s\n' "$details" | sed -n 's/^TeamIdentifier=//p')"
    identifier="$(printf '%s\n' "$details" | sed -n 's/^Identifier=//p')"
    [[ "$team" =~ ^[A-Z0-9]+$ && "$identifier" =~ ^[A-Za-z0-9.-]+$ ]] || return 1
    printf '%s;%s\n' "$team" "$identifier"
}

snapshot_approval() {
    local attributes value flags
    [[ -d "$APP_PATH" ]] || return 0
    OLD_IDENTITY="$(signing_identity || true)"
    # A failed read is unknown, never evidence that quarantine is absent.
    if attributes="$("$XATTR_BIN" "$APP_PATH" 2>/dev/null)"; then
        if ! printf '%s\n' "$attributes" | grep -Fxq "$QUARANTINE_ATTRIBUTE"; then
            OLD_APPROVED=1
        elif value="$("$XATTR_BIN" -p "$QUARANTINE_ATTRIBUTE" "$APP_PATH" 2>/dev/null)" &&
             [[ "$value" =~ ^[[:xdigit:]]{1,8}\; ]]; then
            flags="${value%%;*}"
            if (( (16#$flags & 0x40) != 0 )); then OLD_APPROVED=1; fi
        fi
    fi
}

approval_prompt() {
    warn "Approval will not be carried: $*. The first launch will prompt and must be approved by a human."
}

carry_approval() {
    local identity assessment value flags approved
    if [[ "$DRY_RUN" -eq 1 ]]; then
        if [[ "$OLD_APPROVED" -eq 1 && -n "$OLD_IDENTITY" ]]; then
            log "  approval: will be carried after signature, notarization and identity checks."
        else
            log "  approval: will not be carried; old approval or signing identity is unavailable."
        fi
        return 0
    fi
    if [[ "$OLD_APPROVED" -ne 1 ]]; then
        approval_prompt "old app was not user-approved (or approval could not be read)"
        return 0
    fi
    identity="$(signing_identity || true)"
    if [[ -z "$OLD_IDENTITY" || "$identity" != "$OLD_IDENTITY" ]]; then
        approval_prompt "signing identity changed or could not be verified"
        return 0
    fi
    if ! "$CODESIGN_BIN" --verify --deep --strict "$APP_PATH"; then
        approval_prompt "codesign verification failed"
        return 6  # Do not activate a bundle that failed signature verification.
    fi
    if ! assessment="$("$SPCTL_BIN" -a -t exec -vv "$APP_PATH" 2>&1)"; then
        approval_prompt "Gatekeeper verification failed: $assessment"
        return 6
    fi
    if ! printf '%s\n' "$assessment" | grep -Fxq "$APP_PATH: accepted" ||
       ! printf '%s\n' "$assessment" | grep -Fxq 'source=Notarized Developer ID'; then
        approval_prompt "Gatekeeper did not report accepted as Notarized Developer ID"
        return 6
    fi
    if ! value="$("$XATTR_BIN" -p "$QUARANTINE_ATTRIBUTE" "$APP_PATH" 2>/dev/null)" ||
       [[ ! "$value" =~ ^[[:xdigit:]]{1,8}\; ]]; then
        approval_prompt "new quarantine value is missing, unreadable or malformed"
        return 0
    fi
    flags="${value%%;*}"
    printf -v approved '%0*x' "${#flags}" "$((16#$flags | 0x40))"
    if "$XATTR_BIN" -w "$QUARANTINE_ATTRIBUTE" "$approved;${value#*;}" "$APP_PATH"; then
        log "Gatekeeper approval carried forward; quarantine provenance preserved."
    else
        approval_prompt "could not write the quarantine approval flag"
    fi
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
        # `-O` is "owned by the effective uid" and is built into the shell, so the
        # decision needs no external tool. `stat` differs between BSD and GNU (`-f`
        # means format on macOS and FILESYSTEM on Linux), and a safety guard must not
        # hinge on which one it got. stat is used only to name the owner in the message.
        if [[ ! -O "$path" ]]; then
            owner="$(stat -f '%u' "$path" 2>/dev/null || stat -c '%u' "$path" 2>/dev/null || printf 'unknown')"
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
    local registered_app_version offered_app_version
    APP_VERSION="$(app_version || true)"
    REGISTERED_VERSION="$(registered_version || true)"
    OFFERED_VERSION="$(offered_version || true)"

    if [[ -d "$APP_PATH" && -z "$APP_VERSION" ]]; then
        if [[ "$VERIFY_ONLY" -eq 0 && "$DRY_RUN" -eq 0 && ! -O "$APP_PATH" ]]; then
            assert_no_root_owned_paths
        fi
        err "Could not read CFBundleShortVersionString from $APP_PATH/Contents/Info.plist."
        err "Refusing to classify or replace a present app whose version is unknown."
        exit 4
    fi

    if [[ -z "$OFFERED_VERSION" ]]; then
        err "Could not read the offered version for $CASK_REF from the tap."
        err "Is the tap present? Try: $BREW_BIN tap $TAP_NAME"
        exit 4
    fi

    registered_app_version="${REGISTERED_VERSION%%,*}"
    offered_app_version="${OFFERED_VERSION%%,*}"

    if [[ -z "$APP_VERSION" && -z "$REGISTERED_VERSION" ]]; then
        DRIFT_KIND="missing"
    elif [[ -n "$APP_VERSION" && -z "$REGISTERED_VERSION" ]]; then
        # An app brew does not know about — exactly today's VoiceBar disease.
        DRIFT_KIND="unmanaged"
    elif [[ -z "$APP_VERSION" && -n "$REGISTERED_VERSION" ]]; then
        DRIFT_KIND="stale-ledger"
    elif [[ "$APP_VERSION" != "$registered_app_version" ]]; then
        DRIFT_KIND="stale-ledger"
    elif [[ "$REGISTERED_VERSION" != "$OFFERED_VERSION" || "$APP_VERSION" != "$offered_app_version" ]]; then
        DRIFT_KIND="outdated"
    else
        DRIFT_KIND="none"
    fi
}

print_state() {
    local mode=apply
    if [[ "$DRY_RUN" -eq 1 ]]; then mode=dry-run
    elif [[ "$VERIFY_ONLY" -eq 1 ]]; then mode=verify-only; fi
    log "BrainBar drift-proof update"
    log "  mode:        $mode"
    log "  brew:        $BREW_BIN"
    log "  cask:        $CASK_REF"
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
    log "Clearing the stale Caskroom registration (user-owned mv, fully reversible)."
    run_cmd mkdir -p "$QUARANTINE_ROOT"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        dest="$QUARANTINE_ROOT/$stamp.XXXXXX"
    else
        log "+ mktemp -d $QUARANTINE_ROOT/$stamp.XXXXXX"
        dest="$(mktemp -d "$QUARANTINE_ROOT/$stamp.XXXXXX")"
    fi
    run_cmd mv "$caskroom" "$dest/"
    log "  quarantined -> $dest/$CASK_NAME"
}

adopt_install() {
    # --force adopts an app already sitting at the target path. Without it brew says
    # "It seems there is already an App at '<path>'" and does nothing.
    run_cmd "$BREW_BIN" install --cask --force "$CASK_REF"
    carry_approval
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
            if [[ "$DRIFT_KIND" = outdated ]] && registration_intact; then
                log "Intact registration: using Homebrew upgrade."
                if [[ "$OLD_APPROVED" -eq 1 && -n "$OLD_IDENTITY" ]]; then
                    log "  approval: Homebrew will carry it after its signing identity checks."
                else
                    log "  approval: will not be carried; old approval or signing identity is unavailable."
                fi
                run_cmd "$BREW_BIN" upgrade --cask "$CASK_REF"
                return 0
            fi
            log "Drift detected ($DRIFT_KIND). Adopting $OFFERED_VERSION without running any uninstall recipe."
            quarantine_stale_registration
            # Keep services alive until installation succeeds. Explicit activation
            # after brew returns replaces them and verifies the running build.
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
    local v ok formula_version canonical_cli canonical_version path_cli expected_app_version
    log "Verification:"

    expected_app_version="${OFFERED_VERSION%%,*}"
    v="$(app_version || true)"
    [[ "$v" = "$expected_app_version" ]] && ok=1 || ok=0
    check "app version" "$ok" "${v:-<absent>} (expected $expected_app_version)"

    v="$(registered_version || true)"
    [[ "$v" = "$OFFERED_VERSION" ]] && ok=1 || ok=0
    check "cask version" "$ok" "${v:-<not registered>} (expected $OFFERED_VERSION)"

    formula_version="$("$BREW_BIN" list --versions brainlayer 2>/dev/null | awk '{print $2}' || true)"
    [[ -n "$formula_version" ]] && ok=1 || ok=0
    check "brainlayer formula" "$ok" "${formula_version:-<not installed>}"

    canonical_cli="${BREW_BIN%/*}/brainlayer"
    canonical_version="$("$canonical_cli" --version 2>/dev/null | awk 'NF {print $NF; exit}' || true)"
    [[ -n "$formula_version" && "$canonical_version" = "$formula_version" ]] && ok=1 || ok=0
    check "canonical brainlayer CLI" "$ok" \
        "$canonical_cli ${canonical_version:-<missing>} (expected ${formula_version:-installed formula version})"

    path_cli="$(command -v brainlayer 2>/dev/null || true)"
    if [[ -z "$path_cli" ]]; then
        log "  [WARN] brainlayer PATH: <not on PATH>; add ${canonical_cli%/*} to PATH to invoke brainlayer by name"
    elif [[ "$path_cli" != "$canonical_cli" ]]; then
        check "brainlayer PATH" 0 "$path_cli shadows $canonical_cli"
    else
        check "brainlayer PATH" 1 "$path_cli"
    fi

    if "$canonical_cli" brainbar verify-running --app "$APP_PATH"; then ok=1; else ok=0; fi
    check "running bundle identity (GitCommit/version)" "$ok" "$APP_PATH"

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

    if [[ "$DRIFT_KIND" != none ]]; then snapshot_approval; fi
    apply_update

    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "Dry run complete. Nothing was changed."
        return 0
    fi

    # Homebrew postflight deliberately does not launch the quarantined app.
    # These commands are required even when optional summary checks are skipped.
    run_cmd "${BREW_BIN%/*}/brainlayer" brainbar activate --verify
    run_cmd "${BREW_BIN%/*}/brainlayer" brainbar verify-running --app "$APP_PATH"

    if [[ "$SKIP_VERIFY" = "1" ]]; then
        log "Verification skipped (BRAINLAYER_UPDATE_SKIP_VERIFY=1)."
        return 0
    fi
    verify
}

main
