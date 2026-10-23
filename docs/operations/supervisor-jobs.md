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

Child startup (including log opens, process creation and durable custody registration)
and custody removal run in executor workers, with one shared journal writer. Status
does not advertise a gated startup PID as ready. Pause and shutdown wait for an
in-flight startup before acknowledging group quiescence. Cancelling a worker's waiter
does not release the journal lock while its write is still running.

Heartbeats remain on the event loop with the existing 30-second guard timeout. Slow
I/O can finish without starving loop progress. Each executor operation has a
120-second budget; while an operation is overdue the supervisor withholds heartbeats,
so a permanently stuck worker also triggers the guard. A wedged event loop cannot
send heartbeats at all. The guard retains group custody and the existing bounded
quiescence/EOF cleanup protocol in both cases.

Guard readiness allows 30 seconds after process creation for interpreter/import
startup, independently of the heartbeat timeout. Timeout, EOF and an unexpected
readiness byte report distinct startup errors with parent and guard PIDs. Expected
control-client disconnects during reply/close are ignored without replaying mutations.

## CLI consumers

`brainlayer daemon --manifest /absolute/jobs.json` starts the daemon. The command
`maintenance-calendar [--date YYYY-MM-DD]` passes CalendarPlan directly to its singleton with separate custody,
sibling keg CLI and packaged wrappers. Override `--lib-dir`/`--state-dir`/`--logs-dir`; set `BRAINLAYER_SUPERVISOR_SOCKET`.
`jobs component NAME status|restart|env|logs` uses legacy plists until archival, then owner-only control;
`--socket`/`--manifest` select fixtures. Restart honors OFF/disable/pause; env/log actions return paths only.
JSON traces the legacy label, evidence source and definition; live keg verification remains separate.

The macOS daemon adds an hourly `transcript-backup-recovery` component when its
manifest contains the configured observability environment. It runs
`maintenance-calendar --recover-transcripts` with the same calendar custody directory.
Under that lock, a verified upload or verified no-op receipt less than 24 hours old
skips recovery. Missing, invalid, future, failed and local-only receipts do not.
Otherwise only `jsonl-backup` runs; indexing, maintenance and database backup are
excluded. Existing disable/pause gates still apply. Failed uploads retain failure
receipts and retry on a later interval; recovery does not replace nightly coverage
in the calendar journal. Reinstall/restart the consolidated daemon after upgrading
to activate this component; an unchanged manifest needs no rewrite.

Full calendar invocations reserve priority before acquiring custody. If recovery
is already running, the full invocation waits for its bounded runtime and shutdown
instead of losing that scheduled opportunity. Further recovery attempts and duplicate
full invocations cannot pass the reservation. Waiting happens before any calendar
step starts, so it never replays completed indexing, maintenance or backups.
The wait uses a suspend-aware elapsed deadline derived from the transcript step
and the recovery supervisor's runtime/exit bounds, plus guard startup/shutdown.
An exhausted wait records `calendar-admission-timeout` and exits nonzero without
starting workers; unresolved custody never permits concurrent work. This follows
the actual calendar invocation without assuming a particular launchd schedule.


BrainBar cask recovery is a formula-supervised `brainbar-activation` interval worker.
The cask upgrade stops BrainBar; the supervisor re-activates it automatically after brew exits.
It attempts each Info.plist GitCommit/version once, with durable failure suppression and a
visible `brainbar-activation` job alert. Manual retry: `brainlayer brainbar activate`.
See [activation and deploy-local-prod](brainbar-service-install.md#supervisor-recovery-and-deploy-local-prod-step).
