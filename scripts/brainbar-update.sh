#!/bin/bash
# Shipped installer shared by the app helper and release workers. Never invoke sudo.
# brainbar-update-handoff: 1
set -euo pipefail
umask 077
export PATH="${BRAINBAR_UPDATE_PATH:-/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin}"
export HOMEBREW_NO_INSTALL_CLEANUP=1 HOMEBREW_NO_AUTO_UPDATE=1
export SUDO_ASKPASS=/usr/bin/false NONINTERACTIVE=1
# flock is shared by the supervisor and installer for the entire attempt, including stale mkdir recovery.
# The file is permanent: unlinking it would let contenders lock different inodes.
if [ "${1:-}" != --custody-held ]; then
  custody_root=${BRAINBAR_UPDATE_ROOT:-"$HOME/Library/Application Support/BrainBar"}
  mkdir -p "$custody_root"
  exec /usr/bin/ruby -e '
    File.open(ARGV.shift, File::RDWR|File::CREAT, 0600) do |lock|
      unless lock.flock(File::LOCK_EX|File::LOCK_NB)
        warn "Another update owns custody"; exit 75
      end
      lock.close_on_exec = false
      system("/bin/bash", ARGV.shift, "--custody-held", *ARGV)
      exit($?.exitstatus || 1)
    end
  ' "$custody_root/update-custody.lock" "$0" "$@"
fi
shift
# After the formula upgrade, an older helper execs the new keg's copy with --handoff: same pid,
# same inherited custody flock, same lock directory and state file. It resumes at signatures.
HANDOFF=0
if [ "${1:-}" = --handoff ]; then HANDOFF=1; shift; fi
V=${1:?Usage: brainbar-update.sh VERSION OPERATION_ID}
OP=${2:?operation id required}
[[ "$V" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ && "$OP" =~ ^[A-Za-z0-9-]+$ ]] || exit 64
ROOT=${BRAINBAR_UPDATE_ROOT:-"$HOME/Library/Application Support/BrainBar"}
APP=${BRAINBAR_UPDATE_APP:-/Applications/BrainBar.app}
PLIST=${BRAINBAR_UPDATE_PLIST:-/usr/libexec/PlistBuddy}
AGENTS=${BRAINBAR_UPDATE_AGENTS:-"$HOME/Library/LaunchAgents"}
LABELS="com.brainlayer.brainbar com.brainlayer.brainbar-daemon"
mkdir -p "$ROOT/Backups.noindex"
LOCK="$ROOT/update.lock"
if [ "$HANDOFF" = 1 ]; then
  # Custody is proven only by our own pid in the lock, written by the helper this exec replaced.
  if [ "$(cat "$LOCK/pid" 2>/dev/null)" != "$$" ] || [ "$(cat "$LOCK/operation" 2>/dev/null)" != "$OP" ]; then
    echo 'Handoff without live custody; refusing.' >&2
    exit 75
  fi
# A dead controller's unsettled attempt is retained for recovery, never silently retried.
elif ! mkdir "$LOCK" 2>/dev/null; then
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
SAVED=""
STOPPED=0
UNKNOWN=0
FAILURE=""
state() { printf '%s\n%s\n%s\n' "$OP" "$1" "${2:-}" > "$ROOT/update-state.new"; mv "$ROOT/update-state.new" "$ROOT/update-state"; }
cleanup() { rm -f "$LOCK/pid" "$LOCK/operation" "$LOCK/formula.json" "$LOCK/cask.json"; rmdir "$LOCK"; }
# A BrainBar process is one whose executable (argv[0], as launchd starts it) is inside the bundle;
# a process that only names a bundle path in a later argument (tail, codesign, grep) is not.
app_pids() {
  # The prefix travels in the environment: macOS awk rewrites its argv, so an argument would match awk itself.
  ps -axo pid=,command= | BRAINBAR_EXE_PREFIX="$APP/Contents/MacOS/" awk '
    {p = ENVIRON["BRAINBAR_EXE_PREFIX"]; c = $0; sub(/^ *[0-9]+ +/, "", c)} index(c, p) == 1 {print $1}'
}
# Every scan's exit status counts: an unknown process state never permits a bundle move.
scan() { pids=$(app_pids) || { UNKNOWN=1; echo 'Could not list processes; BrainBar state is unknown' >&2; return 1; }; }
# Stop BrainBar ourselves: Homebrew's AppleScript quit needs Automation access a launchd job never has.
stop_brainbar() {
  local label rc deadline
  STOPPED=1
  for label in $LABELS; do
    rc=0
    launchctl print "gui/$(id -u)/$label" >/dev/null 2>&1 || rc=$?
    if [ "$rc" = 0 ]; then
      launchctl bootout "gui/$(id -u)/$label" || { echo "Could not unload $label" >&2; return 1; }
    elif [ "$rc" != 113 ]; then
      UNKNOWN=1; echo "Could not inspect $label (launchctl $rc)" >&2; return 1
    fi
  done
  scan || return 1
  [ -n "$pids" ] || return 0
  kill -TERM $pids 2>/dev/null || true
  deadline=$((SECONDS + ${BRAINBAR_UPDATE_STOP_TIMEOUT:-30}))
  while [ "$SECONDS" -lt "$deadline" ]; do
    sleep 0.2
    scan || return 1
    [ -n "$pids" ] || return 0
  done
  kill -KILL $pids 2>/dev/null || true
  sleep 1
  scan || return 1
  [ -n "$pids" ] || return 0
  echo 'BrainBar did not exit' >&2
  return 1
}
# A failed plist restore cannot be fixed by re-activation; say exactly how to put them back.
restore_or_recover() {
  restore_agents && return 0
  state recovery-required "Restore plists: cp -p '$SAVED/com.brainlayer.brainbar.plist' '$SAVED/com.brainlayer.brainbar-daemon.plist' '$AGENTS/' then run: $CLI brainbar activate --verify --timeout 90"
  exit 1
}
# The cask uninstall trashes both plists; put back the saved copy of any that postflight did not write.
restore_agents() {
  local label
  [ -n "$SAVED" ] || return 0
  for label in $LABELS; do
    [ -f "$AGENTS/$label.plist" ] || cp -p "$SAVED/$label.plist" "$AGENTS/$label.plist" || return 1
  done
}
# Same gates and flag as brainlayer-update-brainbar.sh: verify signature and notarization, then
# carry the user's approval (0x40) onto the new quarantine value. Provenance is kept, never stripped.
identity() { codesign -dv --verbose=4 "$1" 2>&1 | sed -n 's/^TeamIdentifier=//p; s/^Identifier=//p' | tr '\n' ';'; }
approve() {
  local assessment attributes old new before after flags
  codesign --verify --deep --strict "$APP" || return 1
  assessment=$(spctl -a -t exec -vv "$APP" 2>&1) || { echo "Gatekeeper refused: $assessment" >&2; return 1; }
  grep -Fxq "$APP: accepted" <<<"$assessment" && grep -Fxq 'source=Notarized Developer ID' <<<"$assessment" || return 1
  attributes=$(xattr "$APP") || return 1
  grep -Fxq com.apple.quarantine <<<"$attributes" || return 0
  old=$(xattr -p com.apple.quarantine "$BACKUP" 2>/dev/null) || old=""
  new=$(xattr -p com.apple.quarantine "$APP" 2>/dev/null) || new=""
  before=$(identity "$BACKUP") || before=""
  after=$(identity "$APP") || after=""
  if [[ ! "$old" =~ ^[[:xdigit:]]{1,8}\; || ! "$new" =~ ^[[:xdigit:]]{1,8}\; ]] || (( (16#${old%%;*} & 0x40) == 0 )) ||
     [[ ! "$before" =~ ^[A-Za-z0-9.-]+\;[A-Z0-9]+\;$ ]] || [ "$before" != "$after" ]; then
    echo 'Approval not carried; the first launch will ask the user.' >&2
    return 0
  fi
  flags=${new%%;*}
  printf -v flags '%0*x' "${#flags}" "$((16#$flags | 0x40))"
  find "$APP" ! -type l -print0 | xargs -0 xattr -w com.apple.quarantine "$flags;${new#*;}" ||
    echo 'Could not write the approval flag; the first launch will ask the user.' >&2
}
finish() {
  rc=$?
  trap - EXIT
  if [ "$rc" -ne 0 ]; then
    if [ "$UNKNOWN" = 1 ]; then
      state recovery-required "BrainBar process state unknown; nothing more was moved. Check ps and launchctl, then run: $CLI brainbar activate --verify --timeout 90"
      exit 1
    fi
    if [ -n "$BACKUP" ] && [ -d "$BACKUP" ]; then
      # Same order as the update: stop whatever runs from the replacement before moving it.
      stop_brainbar || { state recovery-required "BrainBar did not stop or its state is unknown. Restore $BACKUP to $APP"; exit 1; }
      # Preserve the failed replacement rather than delete a bundle we did not create.
      if [ -e "$APP" ]; then
        failed="$ROOT/Backups.noindex/BrainBar.failed-$OP.app"
        mv "$APP" "$failed" || { state recovery-required "Restore $BACKUP to $APP"; exit 1; }
      fi
      mv "$BACKUP" "$APP" || { state recovery-required "$BACKUP"; exit 1; }
      restore_or_recover
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
    elif [ "$STOPPED" = 1 ]; then
      # Stopped but never moved: bring the untouched app back.
      restore_or_recover
      if ! "$CLI" brainbar activate --verify --timeout 90; then
        state recovery-required "Run: $CLI brainbar activate --verify --timeout 90"
        exit 1
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
if [ "$HANDOFF" = 0 ]; then
  if [ -f "$ROOT/update-state" ]; then
    previous=$(sed -n '2p' "$ROOT/update-state")
    case "$previous" in success|failed|cancelled|waiting) ;; *) echo 'Unsettled update requires recovery' >&2; cleanup; trap - EXIT; exit 75 ;; esac
  fi
  state preflight
fi
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
if [ "$HANDOFF" = 0 ]; then
  preflight
  fetch
  pin
fi
PREFIX=$(brew --prefix)
CLI="$PREFIX/bin/brainlayer"
K="$PREFIX/Cellar/brainlayer/$V"
if [ "$HANDOFF" = 0 ]; then
  [ -w "$(dirname "$APP")" ] || { FAILURE="Homebrew needs permission. Run: brew upgrade --formula etanhey/layers/brainlayer; brew reinstall --cask etanhey/layers/brainbar"; echo "$FAILURE" >&2; exit 1; }
  pin
  preflight
  STEP=formula
  state "$STEP"
  MUTATED=1
  brew upgrade --formula etanhey/layers/brainlayer </dev/null
else
  STEP=formula
  MUTATED=1
fi
INSTALL=$(find "$K" -path '*/brainlayer/launchd/install.sh' -print -quit)
[ -n "$INSTALL" ]
# The app copied its helper from the OLD keg; the rest of the update runs the new release's steps.
NEXT="$(dirname "$INSTALL")/brainbar-update.sh"
if [ "$HANDOFF" = 0 ] && [ -f "$NEXT" ] && ! cmp -s "$NEXT" "$0" && grep -qx '# brainbar-update-handoff: 1' "$NEXT"; then
  exec /bin/bash "$NEXT" --custody-held --handoff "$V" "$OP"
fi
STEP=signatures
state "$STEP"
bash "$(dirname "$INSTALL")/release-verify-signatures.sh" "$K"
STEP=consolidate
state "$STEP"
bash "$INSTALL" consolidate
STEP=backup
state "$STEP"
OLD=$("$PLIST" -c 'Print :CFBundleShortVersionString' "$APP/Contents/Info.plist")
[[ "$OLD" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
BACKUP="$ROOT/Backups.noindex/BrainBar.backup-$OLD-$OP.app"
SAVED="$ROOT/Backups.noindex/LaunchAgents-$OP"
mkdir -p "$SAVED"
for label in $LABELS; do cp -p "$AGENTS/$label.plist" "$SAVED/$label.plist"; done
STEP=stop
state "$STEP"
stop_brainbar
# Never move a running bundle: the move happens only after every BrainBar process has exited.
mv "$APP" "$BACKUP"
STEP=cask
state "$STEP"
brew reinstall --cask etanhey/layers/brainbar </dev/null
[ -d "$APP" ]
restore_agents
STEP=approval
state "$STEP"
approve
# One activation, after Homebrew has returned and both plists are in place.
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
rm -f "$SAVED"/*.plist && rmdir "$SAVED" || echo "Could not prune $SAVED" >&2
