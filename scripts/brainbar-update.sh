#!/bin/bash
# Shipped installer shared by the app helper and release workers. Never invoke sudo.
set -euo pipefail
umask 077
export PATH="${BRAINBAR_UPDATE_PATH:-/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin}"
export HOMEBREW_NO_INSTALL_CLEANUP=1 HOMEBREW_NO_AUTO_UPDATE=1
export SUDO_ASKPASS=/usr/bin/false NONINTERACTIVE=1
# flock is held by a supervisor for the entire attempt, including stale mkdir recovery.
# The file is permanent: unlinking it would let contenders lock different inodes.
if [ "${1:-}" != --custody-held ]; then
  custody_root=${BRAINBAR_UPDATE_ROOT:-"$HOME/Library/Application Support/BrainBar"}
  mkdir -p "$custody_root"
  exec /usr/bin/ruby -e '
    File.open(ARGV.shift, File::RDWR|File::CREAT, 0600) do |lock|
      unless lock.flock(File::LOCK_EX|File::LOCK_NB)
        warn "Another update owns custody"; exit 75
      end
      system("/bin/bash", ARGV.shift, "--custody-held", *ARGV)
      exit($?.exitstatus || 1)
    end
  ' "$custody_root/update-custody.lock" "$0" "$@"
fi
shift
V=${1:?Usage: brainbar-update.sh VERSION OPERATION_ID}
OP=${2:?operation id required}
[[ "$V" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ && "$OP" =~ ^[A-Za-z0-9-]+$ ]] || exit 64
ROOT=${BRAINBAR_UPDATE_ROOT:-"$HOME/Library/Application Support/BrainBar"}
APP=${BRAINBAR_UPDATE_APP:-/Applications/BrainBar.app}
PLIST=${BRAINBAR_UPDATE_PLIST:-/usr/libexec/PlistBuddy}
mkdir -p "$ROOT/Backups.noindex"
LOCK="$ROOT/update.lock"
# A dead controller's unsettled attempt is retained for recovery, never silently retried.
if ! mkdir "$LOCK" 2>/dev/null; then
  REAPED=0
  if mkdir "$ROOT/update-reaper.lock" 2>/dev/null; then
   if [ -f "$LOCK/pid" ]; then
    owner=$(cat "$LOCK/pid")
    if [[ "$owner" =~ ^[0-9]+$ ]] && ! kill -0 "$owner" 2>/dev/null; then
      # Reap only our known lock files. Durable unsettled state below still blocks retries.
      rm -f "$LOCK/pid" "$LOCK/operation" "$LOCK/formula.json" "$LOCK/cask.json"
      if ! rmdir "$LOCK"; then
        rmdir "$ROOT/update-reaper.lock"
        echo 'Unknown lock contents; inspect update.lock before recovery.' >&2
        exit 75
      fi
      if mkdir "$LOCK" 2>/dev/null; then REAPED=1; fi
    fi
   fi
   rmdir "$ROOT/update-reaper.lock"
  fi
  if [ "$REAPED" = 0 ]; then
    echo 'Another update owns the lock; inspect update-state before recovery.' >&2
    exit 75
  fi
fi
printf '%s\n' "$$" > "$LOCK/pid"
printf '%s\n' "$OP" > "$LOCK/operation"
STEP=preflight
MUTATED=0
BACKUP=""
FAILURE=""
state() { printf '%s\n%s\n%s\n' "$OP" "$1" "${2:-}" > "$ROOT/update-state.new"; mv "$ROOT/update-state.new" "$ROOT/update-state"; }
cleanup() { rm -f "$LOCK/pid" "$LOCK/operation" "$LOCK/formula.json" "$LOCK/cask.json"; rmdir "$LOCK"; }
finish() {
  rc=$?
  trap - EXIT
  if [ "$rc" -ne 0 ]; then
    if [ -n "$BACKUP" ] && [ -d "$BACKUP" ]; then
      # Preserve the failed replacement rather than delete a bundle we did not create.
      if [ -e "$APP" ]; then
        failed="$ROOT/Backups.noindex/BrainBar.failed-$OP.app"
        mv "$APP" "$failed" || { state recovery-required "Restore $BACKUP to $APP"; exit 1; }
      fi
      mv "$BACKUP" "$APP" || { state recovery-required "$BACKUP"; exit 1; }
      if ! "$CLI" brainbar activate --verify --timeout 90; then
        state recovery-required "Run: $CLI brainbar activate --verify --timeout 90"
        exit 1
      fi
      # Restoration and its real activation check take priority over retention cleanup.
      if [ -n "${failed:-}" ]; then
        if [ -f "$ROOT/last-failed" ]; then
          previous_failed=$(cat "$ROOT/last-failed")
          case "$previous_failed" in "$ROOT/Backups.noindex/BrainBar.failed-"*.app)
            ruby -rfileutils -e 'FileUtils.remove_entry_secure(ARGV[0]) if File.directory?(ARGV[0])' "$previous_failed" || echo "Could not prune $previous_failed" >&2 ;;
          esac
        fi
        printf '%s\n' "$failed" > "$ROOT/last-failed"
      fi
    fi
    if [ "$MUTATED" = 0 ]; then
      state failed "${FAILURE:-Update preflight failed. Check Homebrew and try again. See update.log.}"
    else
      state failed "Step $STEP failed. Run: /opt/homebrew/bin/brainlayer brainbar activate --verify --timeout 90"
    fi
  fi
  cleanup
  exit "$rc"
}
trap finish EXIT
trap 'if [ "$MUTATED" = 0 ]; then state cancelled; exit 0; fi' TERM INT
if [ -f "$ROOT/update-state" ]; then
  previous=$(sed -n '2p' "$ROOT/update-state")
  case "$previous" in success|failed|cancelled|waiting) ;; *) echo 'Unsettled update requires recovery' >&2; cleanup; trap - EXIT; exit 75 ;; esac
fi
state preflight
busy() {
  # App keeps this snapshot fresh until the first mutation. A release worker may omit it.
  if [ -n "${BRAINBAR_UPDATE_JOBS:-}" ]; then
    ruby -rjson -e 's=JSON.parse(File.read(ARGV[0])); exit 1 unless Time.now.to_f-s.fetch("at") < 10 && s.fetch("busy").empty?' "$BRAINBAR_UPDATE_JOBS" || return 0
  fi
  pgrep -fl 'maintenance-calendar|maintenance_calendar|backup_daily|backup-daily' | grep -vE 'pytest|\.worktrees/|/t3/worktrees/' | grep -q .
}
preflight() {
  if busy; then state waiting 'Will update when background jobs finish'; exit 0; fi
  if [ -e "$ROOT/cancel-$OP" ]; then rm -f "$ROOT/cancel-$OP"; state cancelled; exit 0; fi
}
# Homebrew release/tap metadata fetch; no model call.
fetch() { brew update </dev/null; }
pin() {
  brew info --json=v2 --formula etanhey/layers/brainlayer > "$LOCK/formula.json"
  brew info --json=v2 --cask etanhey/layers/brainbar > "$LOCK/cask.json"
  ruby -rjson -e 'f=JSON.parse(File.read(ARGV[0])).fetch("formulae").fetch(0); c=JSON.parse(File.read(ARGV[1])).fetch("casks").fetch(0); abort "Tap versions changed; check again" unless f.fetch("versions").fetch("stable")==ARGV[2] && c.fetch("version")==ARGV[2]' "$LOCK/formula.json" "$LOCK/cask.json" "$V"
  rm -f "$LOCK/formula.json" "$LOCK/cask.json"
}
preflight
fetch
pin
PREFIX=$(brew --prefix)
CLI="$PREFIX/bin/brainlayer"
K="$PREFIX/Cellar/brainlayer/$V"
[ -w "$(dirname "$APP")" ] || { FAILURE="Homebrew needs permission. Run: brew upgrade --formula etanhey/layers/brainlayer; brew reinstall --cask etanhey/layers/brainbar"; echo "$FAILURE" >&2; exit 1; }
pin
preflight
STEP=formula
state "$STEP"
MUTATED=1
brew upgrade --formula etanhey/layers/brainlayer </dev/null
STEP=signatures
state "$STEP"
INSTALL=$(find "$K" -path '*/brainlayer/launchd/install.sh' -print -quit)
[ -n "$INSTALL" ]
bash "$(dirname "$INSTALL")/release-verify-signatures.sh" "$K"
STEP=consolidate
state "$STEP"
bash "$INSTALL" consolidate
STEP=backup
state "$STEP"
OLD=$("$PLIST" -c 'Print :CFBundleShortVersionString' "$APP/Contents/Info.plist")
[[ "$OLD" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
BACKUP="$ROOT/Backups.noindex/BrainBar.backup-$OLD-$OP.app"
mv "$APP" "$BACKUP"
STEP=cask
state "$STEP"
brew reinstall --cask etanhey/layers/brainbar </dev/null
[ -d "$APP" ]
STEP=activate
state "$STEP"
"$CLI" brainbar activate --verify --timeout 90
STEP=health
state "$STEP"
[ "$("$PLIST" -c 'Print :CFBundleShortVersionString' "$APP/Contents/Info.plist")" = "$V" ]
# activate --verify above performs the real MCP handshake, not a socket-presence test.
state success
# Prune only a backup explicitly recorded by a previous successful invocation.
if [ -f "$ROOT/last-backup" ]; then
  previous_backup=$(cat "$ROOT/last-backup")
  case "$previous_backup" in "$ROOT/Backups.noindex/BrainBar.backup-"*.app)
    if [ "$previous_backup" != "$BACKUP" ]; then
      ruby -rfileutils -e 'FileUtils.remove_entry_secure(ARGV[0]) if File.directory?(ARGV[0])' "$previous_backup" || echo "Could not prune $previous_backup" >&2
    fi ;;
  esac
fi
printf '%s\n' "$BACKUP" > "$ROOT/last-backup"
