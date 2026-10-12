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
