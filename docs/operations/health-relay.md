# Independent health relay (`com.etanhey.brainlayer-health-relay`)

One LaunchAgent on the M4 reads both Macs' health files and appends one `@orc BLOCKED:` line
per new issue episode to its primary inbox. The live primary is the BrainLayer collab,
`/Users/etanheyman/Gits/orchestrator/docs.local/collab/brainlayer-health.md`.
An optional second inbox receives only prolonged, unacknowledged user-visible outages.
Source: `scripts/health_relay/health_relay.py`, stdlib only, no BrainLayer imports. It never
opens a database, reads credentials, runs `health-check`, heals, or writes producer state.

It runs **outside** the BrainLayer supervisor, so it still reports when the supervisor and every
job inside it are dead. The `com.etanhey.*` label keeps it out of the fleet watchdog's
`com.brainlayer.*` revival glob and out of `supervisor/jobs.json`.

## What it reads, per host

| Source (default under the host `dir`) | Issue it can open |
|---|---|
| `job-alerts.json` | `job-alert:<key>`, one per active key. The reason text is never copied |
| `badge-state.json` | `badge:<code>` for **critical** active issues (warnings are advisory; suppressed codes hold existing episodes open without opening new ones; `job_alert_*` codes come from job-alerts instead); `health-stale` when `generated_at` is older than `badge_max_age_seconds` (1200) or `pending_first_run` passes `expected_first_run_by` |
| `watcher-health.json` | `watcher-stale` when `updated_at` is older than `watcher_max_age_seconds` (600) |
| `supervisor/jobs.json`, `pause.sentinel` | Nothing by themselves. `enabled: false`, or a current pause naming `com.brainlayer.health-check` / `com.brainlayer.watch`, suppresses that staleness alarm |

Any read problem opens its own issue: `source-missing|unreadable|oversize|corrupt:<source>`.
A future timestamp counts as corrupt. `host-unreachable` opens after `unreachable_after` (3)
failed SSH reads in a row and closes on the next successful read. Every other issue closes only
on a **fresh, validated** read of its own source, so switching between unreadable, missing and
corrupt keeps a `source-*` issue open. A stale or `pending_first_run` badge, a stale watcher
file, or a disabled/paused job's evidence can open issues but clears none.
Absent badge and watcher files are held through `first_run_grace_seconds` (900). An absent
`job-alerts.json` is healthy only until the file has been observed in any non-absent state,
including read errors; grace does not cover it. This history rule applies to remote reads too.
For `supervisor/jobs.json` and `pause.sentinel`, absence is healthy only until that source has
been seen in any non-absent state. After a bad read, absence keeps its source-error episode open
until a validated read; absence still removes all disabling/pause controls (absent pause = resumed).
A fresh validated badge closes `badge:<code>` only when the code appears in neither `active` nor
`suppressed`. Active → suppressed → active is one episode; a code seen only in suppressed
creates no alert. Both lists must pass the same entry validation before any badge episode clears.
Every observed issue is tracked, however many there are.

Remote hosts are read with a fixed reader program on stdin over the existing SSH alias, using
`BatchMode=yes`, `StrictHostKeyChecking=yes` and `UpdateHostKeys=no`, with bounded time (45 s)
and output (2 MiB per stdout/stderr pipe, enforced during reads; overflow kills and reaps SSH).
`ssh_alias` and `remote_python` reject control characters and must fully match strict patterns.
Each remote command argument is shell-quoted because SSH joins them into a shell command.
Remote source paths are never resolved or statted locally for inbox alias checks.
Once SSH exits and stdout is complete, a lingering stderr holder cannot delay a successful read.
Config numbers are integers from 0 through 2147483647; booleans, fractional values, negatives
and out-of-range values are rejected as `config-invalid` with exit 2 before any write.
A source-less remote reader (for example a `.pyc` without its `.py`) counts as host-unreachable.
Each source file is read with `O_NOFOLLOW`, must be a regular file, and is capped at 256 KiB.

## Delivery contract

- Each episode gets an ID, `bl-<host>-<epoch>-<seq>-<issue>`. The random 128-bit epoch is persisted at
  bootstrap; sequences advance within that epoch. A new bootstrap generates a fresh epoch.
  The pending event is fsynced to `state.json` **before** any append.
- Every attempt appends **one self-contained record in a single `write`**, starting and ending
  on a line boundary:

  ```text

  ### BrainLayer health relay — <ts> — event=<id>
  @orc BLOCKED: [BrainLayer <label>] <issue> on host=<host>; source=<host>:<file>; event=<id>
  <!-- brainlayer-health:<id> -->
  ```

  A short write fails the run and is never continued: a continuation would depend on bytes
  another writer may already have followed.
- An event is acknowledged only when, after an inbox `fsync` succeeds **in that run**, a re-read
  of the inbox holds its exact BLOCKED line as a whole line and its marker as a whole line.
- If the BLOCKED line is complete but the marker is missing, the relay appends a marker-only
  record (`\n<!-- brainlayer-health:<id> -->\n`), never a suffix to someone else's bytes.
  Anything else, including a marker with no complete line, gets the full record again.
- Torn fragments are left where they are and never counted. The inbox is never truncated or
  rewritten, so other writers' bytes are always preserved.
- At most 32 events are appended per run. The rest stay pending and go out on later ticks.
- An open episode stays open until a measured healthy read. A delivered line is never retracted.
  A later failure after a healthy clear opens a new episode.
- The relay never creates the inbox. If the inbox is missing, unwritable, fails to fsync, or
  the re-read lacks a record, the run exits 1 and the event stays pending. Invalid config or a
  corrupt `state.json` exits 2. Invalid config is rejected before any write, including when the
  inbox resolves or aliases to any configured local host source, state, status, lock or atomic temporary
  files. Corrupt state is preserved byte-for-byte; the failure status has `error: "state-corrupt"` and `pending: null`.
  A lock-open failure exits 1 with `error: "lock-open-failed"` and `pending: null`; failure status
  writes are best-effort. Atomic temp names are unlinked before exclusive creation with mode 0600;
  planted symlinks and hardlinks cannot truncate their targets.
  `status.json` records `error`, `pending` and per-host reachability after each run with valid configuration.

## Tiered escalation and lead acknowledgment

Every new episode still goes to `inbox` immediately with the unchanged delivery contract above.
Without `escalation`, no second destination is read or written. To enable escalation, configure:

```json
"escalation": {
  "inbox": "/Users/etanheyman/Gits/orchestrator/docs.local/collab/orc-inbox.md",
  "after_seconds": 1800,
  "issues": [
    "host-unreachable", "health-stale", "watcher-stale",
    "job-alert:mcp-transport", "job-alert:brainbar-activation",
    "badge:brain-search-canary-failed", "source-missing:badge"
  ]
}
```

`inbox` is required in that object. The shown threshold and patterns are the defaults when omitted.
The threshold uses the same integer bounds as other config numbers. `issues` must be a list of
valid regular expressions; each must **fullmatch** the issue class. An empty list disables new
escalations. Both inboxes must be distinct, existing files at delivery time. Neither is created.
The same path-resolution, symlink/hardlink alias, local-source and state-file collision checks
apply to both; collisions fail before any write. Remote source paths are never statted locally.

The default classes are deliberately narrow: they are the real outages, where memory itself stops
working or stops being observable.

| Class | User-visible reason |
|---|---|
| `host-unreachable` | A host's memory services cannot be observed over the configured transport |
| `health-stale` | Fleet health evidence stopped advancing; outages may be invisible |
| `watcher-stale` | Conversation indexing stopped reporting progress, threatening current recall |
| `job-alert:mcp-transport` | BrainBar MCP connections or requests are failing to recover |
| `job-alert:brainbar-activation` | BrainBar activation failed, leaving the agent memory service unavailable |
| `badge:brain-search-canary-failed` | The search canary reports a failed memory retrieval path |
| `source-missing:badge` | Health evidence disappeared after grace, leaving health visibility absent |

These signals indicate a user-visible service gap, not proof of data loss.
Backup-protection alerts (`job-alert:backup-daily`, `job-alert:jsonl-backup`,
`job-alert:backup-snapshot`) stay in the primary collab: memory still works while a backup fails,
so the BrainLayer lead owns them there (orc ruling, 2026-10-30). A deployment can add them back
with an explicit `issues` list.
Internal classes such as `badge:job-scan-failed` and `badge:slow-check` stay in the primary collab.
Badge codes `lock_holder_wedge`, `queue_backed_up`, `hotlane_dead`, and `drain_no_progress`
(normalized by the relay to hyphens) also stay there: they indicate degraded operation, not a
user-visible outage by themselves, so they are excluded from default escalation.

Orc must **not** watch the primary collab: primary lines retain `@orc BLOCKED:` for compatibility,
so watching it would wake orc for every immediate alert. The BrainLayer lead watches that collab
with a filter that matches `@orc` (for example `--alias @orc`), and acknowledges an episode by
appending exactly this whole line there, in the **primary** inbox:

```text
ACK <primary-event-id>
```

Quoted, indented, suffixed, or partial matches do not count. An exact final line at EOF also counts.
CRLF ACK lines do not count; use LF line endings. Successfully sampled ACK event IDs persist in
state while their episode remains present, including when primary delivery subsequently fails,
and suppress escalation through a later primary outage. Unrelated ACK IDs are discarded.
Acknowledgment also persists in episode state; it does not
close the issue or acknowledge primary delivery. A measured recovery closes as before. Unknown
or stale evidence still cannot close anything and does not reset the episode timer.

An eligible episode open continuously for at least `after_seconds`, without a lead ACK, queues
one event `<primary-event-id>.esc`. It carries one `@orc BLOCKED:` line, the same identity fields,
and the same self-contained append-only record, durable outbox, fsync gate and marker dedup.
Both queues use the same delivery function; each can deliver up to 32 events per tick. Primary
delivery runs first. Escalation failure exits 1 with an `escalation-*` error in `status.json`;
primary delivery is retained, and `pending` counts both queues. A missing or unreadable escalation
inbox is reported even before a threshold is reached. A pending escalation retries after restart
without duplicating a complete line, including a crash after append but before state acknowledgment.
A lead ACK or measured recovery cancels an undelivered escalation on the next tick. Already
appended bytes cannot be retracted, including bytes left by a failed fsync. Escalation never
clears or reopens an episode; a new episode after recovery gets a fresh timer and escalation ID.

The escalation destination is pinned in state on first use. Changing it fails visibly with
`escalation-destination-changed` after primary delivery, preventing pending events from being
silently redirected. Removing the optional config pauses escalation and retains its pending
queue; re-enabling the same destination resumes it. If primary access or delivery fails
(`inbox-unavailable`, `inbox-unreadable`, `inbox-append-failed`, or `inbox-ack-missing`), due
escalations still run using only previously persisted ACKs. The run still exits nonzero and
`status.error` retains the original primary error, even if escalation also fails. Primary events
remain pending under the unchanged delivery contract. A missing primary scan retains its
`verify_floor`; migration ID resolution waits for primary recovery. State-integrity errors,
including `state-stale`, `state-corrupt`, and `state-commit-failed`, block escalation before any
append. A commit failure after append cannot retract that record; retries use marker dedup.
ACKs are sampled once per tick, so a concurrent ACK arriving
after that sample may race with that tick's append.

To recover from `escalation-destination-changed` when deliberately moving the destination:

1. Restore the configured escalation inbox to the pinned `escalation_inbox` in `state.json`.
   Repair access there and run normally until `escalation_outbox` is empty. Pending events may
   instead be cancelled by exact lead ACKs in the primary or measured recovery; do not discard
   them manually. If neither is possible, leave the pin in place and investigate.
2. Pause relay invocations and wait for any active run to finish. Preserve a copy of state and
   both inboxes for recovery outside the repo, with the state copy retaining private permissions.
   Recheck that `escalation_outbox` is empty while no relay can write.
3. Change only `escalation_inbox` in state to JSON `null`, using an atomic replacement with the
   original permissions. Retain the epoch, sequences, both queues, `verify_floor`, ACK IDs and
   every episode flag. Never clear `escalated`: already sent episodes must not be resent.
4. Set config `escalation.inbox` to the new distinct, existing, writable absolute inbox and
   resume. The next run pins that destination. Verify `status.error` is null and both queues
   drain normally; existing delivered episodes stay delivered. Do not truncate either inbox or
   reset the state to move a destination. Removing the optional config alone does not clear its pin.

## State recovery and rollback

A first-ever bootstrap works without `state.json` only after successfully reading an inbox with no
complete relay markers. An unreadable or missing inbox refuses bootstrap with exit 1,
`error: "state-missing"` and `pending: null`; no state is created and nothing is appended.
Restore inbox access to retry, or explicitly use `--accept-new-epoch` to authorize a new epoch.
With that flag, an unavailable inbox still leaves delivery pending until access returns.
If state is absent and the inbox already contains a complete `<!-- brainlayer-health:<id> -->`
line (including an unterminated final line), the run exits 1 with `error: "state-missing"` and
`pending: null` in `status.json`. It neither appends nor creates replacement state.

When state exists, the relay scans the inbox once at load for complete markers from that state's
epoch for every host (including SSH hosts), including an unterminated final line. A marker above
its host's persisted sequence refuses
the run with exit 1, `error: "state-stale"` and `pending: null`. Nothing is appended and state bytes
stay unchanged. If the scan cannot read the inbox, the relay persists the pre-observation per-host
sequences as `verify_floor`, retaining the earliest floor across outage runs. The next successful
scan checks against that floor and clears it only after verification. No **primary** delivery
occurs before that verification, even if inbox access returns within the same run. Observations
remain pending; the escalation safety net may still deliver due events during primary access
failure, using persisted ACKs only. A detected stale state blocks both destinations.
Never restore an older `state.json`; move stale state aside and use
`--accept-new-epoch` instead. The flag does not bypass the stale-state check while state exists.

Preserve damaged state for investigation; do not delete it as a routine repair. After investigating
state loss, an operator may explicitly authorize a new epoch:

```sh
python3 scripts/health_relay/health_relay.py --config /absolute/path/relay.json --accept-new-epoch
```

The flag permits bootstrap despite retained markers. Recurring issues get fresh event IDs and new
alerts. Lost episode history and pending events cannot be reconstructed; currently active issues
are observed again. The flag does not bypass corrupt-state validation or reset existing state.

New writes use state **version 4**: episodes persist `opened_at`, `event_id`, `acknowledged`, and
`escalated`; state also holds `escalation_outbox`, its pinned `escalation_inbox`, and the
`acknowledged_event_ids` set serialized as a list (absent in older v4 states means empty). Versions 2 and 3
migrate on the next successful write, retaining primary pending IDs, sequences, and open/closed
history. Since neither old version records episode opening time, migrated episodes start their
escalation timer at migration, conservatively waiting the full threshold. Version-2 episode IDs
retain the legacy form; newly opened episodes use the epoch form. A transient `migrated` flag
lets a v3 state that already contains v2 episodes resolve their original IDs from inbox markers
after inbox access recovers. It remains set until a marker establishes the ID; while unresolved,
an exact ACK for either the reconstructed or legacy ID suppresses escalation. Rebinding re-keys
an undelivered escalation's record and dedup ID. A record whose line was already appended
finishes delivery under its original ID before rebinding, preserving retry dedup. Migration does not resend
already delivered primary events. Invalid timestamps (including booleans, nonfinite and huge
values), flags or IDs fail closed and preserve the original state bytes.
Rolling back to a version-3-only or version-2-only relay explicitly fails closed with exit 2 and
`state-corrupt`, preserving version-4 state; use the newer relay to resume. Do not downgrade or
delete state to force a rollback.

## Limits

- `relay.lock` serializes relay runs only. Other writers of either inbox do not take it.
  Self-contained records keep relay alerts intact whatever those writers do, but a torn relay
  fragment that another writer appends to *without* a leading newline will share that writer's
  first line. The relay's own complete record still follows.
- Torn fragments from an interrupted attempt remain visible in the inbox as partial lines.

## Not covered

- A failure that starts and clears between two 5-minute polls.
- A calendar run that dies before writing an abandonment alert.
- The M4 itself, or this relay's own LaunchAgent, being down. Nothing watches the watcher.
- Critical issues that exist only in health-check output and never reach the badge.
- After an operator disables a job, its open staleness episode stays open until a fresh read:
  disabling is not evidence of recovery.

## Install plan (M4 only; after review and merge)

1. Pin the reviewed bytes outside any checkout:
   `git -C ~/Gits/brainlayer show <merge-sha>:scripts/health_relay/health_relay.py > ~/.local/lib/brainlayer-health-relay/health_relay.py`
2. Write `~/.config/brainlayer-health-relay/relay.json` (mode 0600):
   `inbox`, `state_dir`, and `hosts[]` entries with `id`, `label`, `transport` (`local`|`ssh`), `dir`,
   and optional per-source overrides in `sources`. For SSH hosts, also set `ssh_alias` and
   `remote_python`. Keep ControlPersist disabled on the `m1` alias. Use each host's **own** data directory: the `m1` alias logs in as a different
   account, so the M1 `dir` is not under `/Users/etanheyman`. If `BRAINLAYER_DB`,
   `BRAINLAYER_JOB_ALERT_PATH` or `BRAINLAYER_BADGE_STATE_PATH` are set in that host's effective
   service environment, copy them into `sources`.
3. Dry run against a disposable inbox file. Then read `status.json` and the disposable inbox.
4. Render the template: replace `__PYTHON__` with `/usr/bin/python3`, `__SCRIPT__`, `__CONFIG__`
   and `__LOG_DIR__`. Lint it with `plutil -lint`, then `launchctl bootstrap gui/$(id -u)` it.
   Prove that `status.json` `ts` advances on schedule, and that SSH works from launchd's
   environment (no interactive agent).
5. Point `inbox` at the BrainLayer collab and, if enabled, `escalation.inbox` at the orc inbox
   only within the deployment's authorization. Active issues reach the primary on the first run;
   escalation waits for the episode threshold and honors primary ACKs.
