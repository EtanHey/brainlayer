# BrainBar updates

BrainBar checks the latest GitHub release and the `etanhey/layers` formula and
cask at launch and every six hours. Settings can check on demand. All three
versions must match before an offer appears. A failed check stays in Settings.
The offer persists across launches until installed or superseded. Later hides
the header offer for the current session; Settings retains it.

The Update button copies the installed `brainlayer/launchd/brainbar-update.sh`
resource outside the keg and bootstraps an independent one-shot launchd job
(`RunAtLoad`, `LaunchOnlyOnce`, unique label per launch). Release workers
can run that same shipped script with `VERSION OPERATION_ID`. Each operation
uses one version-independent lock in `~/Library/Application Support/BrainBar`.
A Ruby supervisor and its Bash child share an inherited OS file lock for the
whole attempt, including stale mkdir-lock recovery. Killing only the supervisor
does not release custody while the installer still runs. Its permanent lock file must not be unlinked.
The helper refreshes Homebrew metadata and rechecks both tap versions immediately
before the formula upgrade. It never pulls a source checkout or takes release
receipts. Homebrew runs noninteractively, with stdin closed and askpass disabled.

The UI refuses while the consolidated maintenance calendar or its backup,
upload, nightly or weekly job observations show
work running. The helper rechecks the fresh UI snapshot and the release installer's
maintenance/backup process sample before mutation. Busy work is allowed to finish;
the offer retries at the next scheduled six-hour check, using the same operation
and copied helper. The watcher flush is not fenced.

This is a preflight sample, not an admission lease. Work can start after the
last sample and encounter the same supervisor restart as a release installation.
A separate BrainLayer protected-operation admission follow-up is required to
close that race; see the UI overhaul phase-14 lead decision 2.

Order: formula upgrade, handoff to the new keg's helper, installed-keg signature
verification, service consolidation, LaunchAgent plist copy, stop BrainBar, old-app
backup, cask reinstall, plist check, signature and Gatekeeper check with approval
carried, one activation with a real MCP handshake, installed bundle version check.
A signature failure leaves the cask and services untouched.

**Stop first.** The helper stops BrainBar itself before it touches the bundle. It
boots out `com.brainlayer.brainbar` and `com.brainlayer.brainbar-daemon`, sends
SIGTERM to every process running an executable inside `BrainBar.app`, waits up to
`BRAINBAR_UPDATE_STOP_TIMEOUT` seconds (default 30), and sends SIGKILL only after
that. It never relies on Homebrew's quit: that is an AppleScript request that needs
Automation access, which a launchd job never has (on 24 Nov 2026 it failed with
"did not quit", the running bundle was moved, and the M1's socket stayed down).
The backup move happens only after every BrainBar process has exited. If the stop
fails, the untouched app is activated again and the attempt ends `failed`.

**Plists.** The cask uninstall trashes both LaunchAgent plists and its postflight
writes new ones. The helper copies both plists into
`Backups.noindex/LaunchAgents-<operation>` before the stop and puts back any plist
the cask did not write, so the single `brainlayer brainbar activate --verify` call
always finds both. The copy is removed after a successful update.

**Approval.** After the cask step the helper runs the same gates as the release
installer (`brainlayer-update-brainbar.sh`): `codesign --verify --deep --strict` and
a Gatekeeper assessment that reports `accepted` and `Notarized Developer ID`. A
failure rolls back. When the old app carried the user's approval flag (0x40) and the
signing identity is unchanged, the flag is written onto the new quarantine value;
quarantine provenance is kept, never stripped. Otherwise the first launch asks.

**Rollback** uses the same order: stop whatever runs from the replacement, move it
aside, restore the saved bundle and any missing plist, then activate and verify
again before pruning older failed bundles. Retention failures are non-fatal. A
failed restore leaves custody and its recovery instruction in `update-state`; it
never silently retries an unsettled attempt.

**Handoff to the new release.** The app copies the helper from the INSTALLED keg,
so the copy is always the old release's script. After `brew upgrade --formula`,
the helper execs the new keg's `brainlayer/launchd/brainbar-update.sh` with
`--custody-held --handoff VERSION OPERATION_ID` when that file differs and carries
the `# brainbar-update-handoff: 1` marker. `exec` keeps the pid, so the inherited
custody flock, `update.lock/pid` and the state file carry over; the new script
refuses unless the lock names its own pid and operation, and resumes at the
signature step. An install older than this change has no handoff, so it runs its own
old steps to the end. **The first update to the release that carries this fix must
be done by the release install, `brainlayer-update-brainbar.sh`, not the in-app
Update button.**

Cancellation is accepted before formula mutation. After mutation, the operation
finishes or restores; Cancel does not kill it. The state file contains operation
identity, last step and outcome. `update.log` contains command output. Dead lock
owners can be reaped, but their unsettled state still refuses a new install.
After an interrupted swap, inspect the recorded step and saved bundle before
manually settling state. The activation recovery command is:

```sh
/opt/homebrew/bin/brainlayer brainbar activate --verify --timeout 90
```

Formula rollback is outside v1. A restored app runs with the upgraded formula.
The previous keg is retained by `HOMEBREW_NO_INSTALL_CLEANUP=1`; its rollback and
compatibility remain a release operator responsibility. A restoration failure
is reported, not represented as a healthy old daemon.

Tests inject feed, clock, launcher, brew, process sampling, app root and keg.
They never install software, restart services or connect to production data.
The opt-in `BRAINBAR_UPDATE_NETWORK_PROBE=1` test checks only the public feed.
Offscreen renders use invented versions and no windows. Installed update proof
belongs to the release mission.

The shared script must remain executable in git, the wheel and the installed
package. The app resolves and checks it off the main thread before showing Update.
Backups and one retained failed replacement live under `Backups.noindex`.

A pid-less lock or unknown contents in `update.lock` require operator inspection;
neither is assumed stale. A failed reap releases the reaper lock. Interrupted
mutation remains refused until the release operator verifies bundle, cask ledger
and daemon identity and settles the attempt. An automated `--settle` mode is
deferred from v1. Never clear custody merely because the PID is gone.

Release acceptance must prove the detached helper survives bootout of the running
app LaunchAgent and completes activation end to end. Unit tests do not establish
that process-group behavior.

The updater render regression covers resting, transient up-to-date, checking,
offer, pressed offer, Later-hidden, waiting for a job, unknown job status,
pre-mutation install, installing, cancelling, long recovery, installer unavailable,
tap pending and failed check. It renders the 440 pt Settings row and 900 pt header
in dark and light, plus the 480 pt Settings content for recovery.
