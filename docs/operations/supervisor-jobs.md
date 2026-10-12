# Consolidated daemon command manifest

Status: source wiring only. Installation and migration activate this in PR5; importing
these modules never starts jobs or changes LaunchAgents.

Run `python -m brainlayer.supervisor_jobs --manifest /absolute/jobs.json` under the
installed keg interpreter. `--socket /absolute/control.sock` overrides the shared
`BRAINLAYER_SUPERVISOR_SOCKET` resolver. TERM/INT/HUP invoke the existing bounded
shutdown and custody guard.

The manifest is a private regular file (0600), keyed by legacy launchd label. Each
record has `enabled` (boolean), `plist` (the rendered plist dictionary), and optional
`run_timeout` (positive seconds). Example shape:

```json
{"com.brainlayer.watch":{"enabled":false,"plist":{"Label":"com.brainlayer.watch","ProgramArguments":["/absolute/env-run","/absolute/brainlayer","watch"]}}}
```

PR5 records operator/configuration state before archiving plists. The daemon preserves
argv, environment gates, output/error paths, working directory and exit grace from
that snapshot. It adds the control socket path to child environments, enabling the
existing dual-mode heal adapters. Calendar and retired labels are refused here.

Resident children retain crash backoff. `StartInterval` children use monotonic deadlines,
never overlap, and wait a full interval after completion; missed ticks run once without
catch-up. A run deadline terminates and reaps its owned group, leaving sibling children
running. Exit 75 is deferred rather than counted as failed. Status retains the last exit
code and start time, and supplies the next deadline as an epoch timestamp.

Child policy is established after custody registration and before exec: file limits,
nice priority, Darwin background state and disk I/O policy. Standard children clear
inherited background state. Unknown plist keys or unsupported policy values are errors;
BrainBar bundle association belongs to the consolidated launcher, not a child process.
Each child's crash backoff preserves its ThrottleInterval. The supervisor launcher must
start at nice 0 so children can adopt their own nice values without privileged elevation.
Children receive HOME/USER/LOGNAME/SHELL/TMPDIR/encoding plus a safe PATH, then explicit
plist environment and control endpoint; arbitrary supervisor variables are excluded.

Intervals use monotonic time: elapsed-time ticks resume after wake, run once when due,
and schedule the next run without catch-up. They do not promise wall-clock cadence while
asleep. Timeout failure history resets only after a successful/deferred interval exit.
Disabled/paused intervals have no next-run deadline; cleanup refusal retains custody,
records an error and retries without stopping the supervisor or spawning duplicates.

## CLI consumers

`brainlayer daemon --manifest /absolute/jobs.json` is the named daemon launcher.
`brainlayer maintenance-calendar` runs the installed calendar plan for today's scheduled
date; `--date YYYY-MM-DD` selects a missed date. It uses the current interpreter's
sibling `brainlayer` executable and packaged launchd wrappers, with explicit `--lib-dir`,
`--state-dir` and `--logs-dir` overrides for isolated runs.

`brainlayer jobs component watch status` reads runtime evidence. While the legacy plist
exists it reads that launchd label; after archival it reads the owner-only supervisor.
`--socket` and `--manifest` select private fixtures. `restart` respects disabled/OFF/pause
refusals and never bootstraps an archived label. `env` returns the configured env-file
path and `logs` returns stdout/stderr paths; neither prints credential contents. JSON
includes the component's legacy label, evidence source and definition path. This traces
configuration provenance; it does not claim which keg a live process has mapped.
