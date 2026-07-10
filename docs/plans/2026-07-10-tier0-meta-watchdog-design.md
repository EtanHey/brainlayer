# Tier-0 Meta-Watchdog Design

## Scope

Add a launchd job that runs under `/bin/sh` and monitors the Python-backed
`com.brainlayer.health-check` job. The watchdog must remain independent of the
interpreter class it guards, detect an unloaded health-check label or a stale
health-check state file, alert through independent local and HTTP channels, and
then heal the launchd job.

## Architecture

`scripts/tier0-watchdog.sh` is a single POSIX-shell program. It reads production
defaults from environment-overridable settings, checks the health-check label
with `launchctl print`, and reads the state-file mtime with macOS `stat`. A state
file older than 1,200 seconds is stale by default.

The launchd command contract is scoped to the current GUI user. `TIER0_DOMAIN`
defaults to `gui/$(id -u)`, and the script invokes these exact forms:

- `launchctl print "$TIER0_DOMAIN/$TIER0_LABEL"` for inspection;
- `launchctl bootstrap "$TIER0_DOMAIN" "$TIER0_HEALTH_PLIST_PATH"` when the
  label is absent;
- `launchctl kickstart -k "$TIER0_DOMAIN/$TIER0_LABEL"` after bootstrap or
  directly when the loaded label has stale or missing state.

On a detected fault, the script records the reason and attempts every alert
channel before any recovery action:

1. append a timestamped line to the Tier-0 log;
2. display a macOS notification with `osascript`;
3. POST the alert to the local notify endpoint with a short timeout.

All three alert processes are dispatched before recovery and bounded by
`TIER0_ALERT_TIMEOUT_SECONDS` (default three seconds); timed-out processes are
terminated. The HTTP request also uses curl's three-second max-time. The default
endpoint is `http://localhost:3847/notify`, with `Content-Type:
application/json` and a constant JSON object containing `title`, `body`, and
`source` fields. Keeping the JSON constant avoids shell interpolation and JSON
escaping hazards. Curl accepts only successful HTTP responses (`-f`) but every
alert attempt is fail-open, so one broken channel cannot suppress the others or
prevent recovery. If the label is absent, recovery bootstraps the installed
health-check plist and then kickstarts the label. If the label is loaded but the
state is missing or stale, recovery kickstarts it directly. A detected fault
exits nonzero even when recovery commands succeed, preserving a loud launchd
signal.

The launchd installer copies the script to
`~/.local/lib/brainlayer/tier0-watchdog.sh`, renders
`com.brainlayer.tier0-watchdog.plist`, and loads it through the existing
installer. The plist invokes only `/bin/sh` plus the installed script path.

## Testability

Tests replace command paths through explicit `TIER0_*` executable variables,
use temporary state/plist/log files, and provide a deterministic current epoch.
No drill addresses the live launchd domain or production state file.

The four drills cover:

- unloaded label: alerts, bootstraps, then kickstarts;
- stale state: alerts and kickstarts without bootstrap;
- notify endpoint failure: local log and osascript alert still occur, followed
  by recovery; the curl fixture hangs and proves the bounded alert process
  cannot delay recovery indefinitely;
- loaded label plus fresh state: no alerts and no recovery.

A separate regression covers a missing state file and asserts direct kickstart
without bootstrap. It is not one of the four named exit-gate drills.

## Alternatives Considered

- PATH-prepended command stubs were rejected because absolute production command
  defaults make command identity clearer and reduce accidental PATH coupling.
- A sourced shell adapter library was rejected because it creates an additional
  Tier-0 runtime artifact and failure surface without improving the four-drill
  contract.
