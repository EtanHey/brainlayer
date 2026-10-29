# Independent health relay (`com.etanhey.brainlayer-health-relay`)

One LaunchAgent on the M4 (the Mac that holds `orchestrator/docs.local/collab/orc-inbox.md`)
reads both Macs' health files and appends one `@orc BLOCKED:` line per new issue episode.
Source: `scripts/health_relay/health_relay.py`, stdlib only, no BrainLayer imports. It never
opens a database, reads credentials, runs `health-check`, heals, or writes producer state.

It runs **outside** the BrainLayer supervisor, so it still reports when the supervisor and every
job inside it are dead. The `com.etanhey.*` label keeps it out of the fleet watchdog's
`com.brainlayer.*` revival glob and out of `supervisor/jobs.json`.

## What it reads, per host

| Source (default under the host `dir`) | Issue it can open |
|---|---|
| `job-alerts.json` | `job-alert:<key>`, one per active key. The reason text is never copied |
| `badge-state.json` | `badge:<code>` for **critical** active issues (warnings and suppressed are advisory; `job_alert_*` codes come from job-alerts instead); `health-stale` when `generated_at` is older than `badge_max_age_seconds` (1200) or `pending_first_run` passes `expected_first_run_by` |
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
Every observed issue is tracked, however many there are.

Remote hosts are read with a fixed reader program on stdin over the existing SSH alias, using
`BatchMode=yes`, `StrictHostKeyChecking=yes` and `UpdateHostKeys=no`, with bounded time (45 s)
and output (2 MiB per stdout/stderr pipe, enforced during reads; overflow kills and reaps SSH).
`ssh_alias` and `remote_python` reject control characters and must fully match strict patterns.
Each remote command argument is shell-quoted because SSH joins them into a shell command.
Remote source paths are never resolved or statted locally for inbox alias checks.
Once SSH exits and stdout is complete, a lingering stderr holder cannot delay a successful read.
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
stay unchanged. Never restore an older `state.json`; move stale state aside and use
`--accept-new-epoch` instead. The flag does not bypass the stale-state check while state exists.

Preserve damaged state for investigation; do not delete it as a routine repair. After investigating
state loss, an operator may explicitly authorize a new epoch:

```sh
python3 scripts/health_relay/health_relay.py --config /absolute/path/relay.json --accept-new-epoch
```

The flag permits bootstrap despite retained markers. Recurring issues get fresh event IDs and new
alerts. Lost episode history and pending events cannot be reconstructed; currently active issues
are observed again. The flag does not bypass corrupt-state validation or reset existing state.

New writes use state **version 3** and require an epoch. Version 2 is still readable and upgrades
on the next successful state write, retaining its sequences, episodes, and pending event IDs.
New events use the new epoch. Rolling back to the version-2-only relay explicitly fails closed
with exit 2 and `state-corrupt`, preserving the version-3 state; use the newer relay to resume.
Do not downgrade or delete state to force a rollback.

## Limits

- `relay.lock` serializes relay runs only. Other writers of `orc-inbox.md` do not take it.
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
5. Point `inbox` at the real `orc-inbox.md` only with authorization. Active alerts at that moment
   are delivered on the first run.
