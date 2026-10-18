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

`brainlayer daemon --manifest /absolute/jobs.json` starts the daemon. The command
`maintenance-calendar [--date YYYY-MM-DD]` passes CalendarPlan directly to its singleton with separate custody,
sibling keg CLI and packaged wrappers. Override `--lib-dir`/`--state-dir`/`--logs-dir`; set `BRAINLAYER_SUPERVISOR_SOCKET`.
`jobs component NAME status|restart|env|logs` uses legacy plists until archival, then owner-only control;
`--socket`/`--manifest` select fixtures. Restart honors OFF/disable/pause; env/log actions return paths only.
JSON traces the legacy label, evidence source and definition; live keg verification remains separate.


BrainBar cask recovery is a formula-supervised `brainbar-activation` interval worker.
The cask upgrade stops BrainBar; the supervisor re-activates it automatically after brew exits.
It attempts each Info.plist GitCommit/version once, with durable failure suppression and a
visible `brainbar-activation` job alert. Manual retry: `brainlayer brainbar activate`.
See [activation and deploy-local-prod](brainbar-service-install.md#supervisor-recovery-and-deploy-local-prod-step).
