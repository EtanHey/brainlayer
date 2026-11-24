# BrainBar updates

BrainBar checks the latest GitHub release and the `etanhey/layers` formula and
cask at launch and every six hours. Settings can check on demand. All three
versions must match before an offer appears. A failed check stays in Settings.
The offer persists across launches until installed or superseded. Later hides
the header offer for the current session; Settings retains it.

The Update button copies the installed `brainlayer/launchd/brainbar-update.sh`
resource outside the keg and starts it detached from the app. Release workers
can run that same shipped script with `VERSION OPERATION_ID`. Each operation
uses one version-independent lock in `~/Library/Application Support/BrainBar`.
The helper refreshes Homebrew metadata and rechecks both tap versions immediately
before the formula upgrade. It never pulls a source checkout or takes release
receipts. Homebrew runs noninteractively, with stdin closed and askpass disabled.

The UI refuses while its backup, upload, nightly or weekly job observations show
work running. The helper rechecks the fresh UI snapshot and the release installer's
maintenance/backup process sample before mutation. Busy work is allowed to finish;
the offer retries when it becomes idle. The watcher flush is not fenced.

This is a preflight sample, not an admission lease. Work can start after the
last sample and encounter the same supervisor restart as a release installation.
A separate BrainLayer protected-operation admission follow-up is required to
close that race; see the UI overhaul phase-14 lead decision 2.

Order: formula upgrade, installed-keg signature verification, service consolidation,
old-app backup, cask upgrade, activation with a real MCP handshake, installed bundle
version check. A signature failure leaves the cask and services untouched.
App replacement or activation failure restores the saved bundle and verifies
activation again. A failed restore leaves custody and its recovery instruction
in `update-state`; it never silently retries an unsettled attempt.

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
