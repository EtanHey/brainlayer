# Independent health relay

One relay on the M4 (the Mac that holds `orchestrator/docs.local/collab/orc-inbox.md`) reads
health files and appends one `@orc BLOCKED:` line per new issue episode.
Source: `scripts/health_relay/health_relay.py`, stdlib only, no BrainLayer imports. It never
opens a database, reads credentials, runs `health-check`, heals, or writes producer state.

## What it reads, per host

| Source (default under the host `dir`) | Issue it can open |
|---|---|
| `job-alerts.json` | `job-alert:<key>`, one per active key. The reason text is never copied |

Any read problem opens its own issue: `source-missing|unreadable|oversize|corrupt:<source>`.
An issue closes only on a **validated** read of its own source, so switching between
unreadable, missing and corrupt keeps a `source-*` issue open. An absent `job-alerts.json` is
healthy only until the file has been seen once. Every observed issue is tracked, however many.

## Delivery contract

- Each episode gets an ID, `bl-<host>-<seq>-<issue>`. The sequence is durable and never reused.
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
  corrupt `state.json` exits 2. `status.json` (overwritten each run) records `error` and `pending`.

## Limits

- `relay.lock` serializes relay runs only. Other writers of `orc-inbox.md` do not take it.
  Self-contained records keep relay alerts intact whatever those writers do, but a torn relay
  fragment that another writer appends to *without* a leading newline will share that writer's
  first line. The relay's own complete record still follows.
- Torn fragments from an interrupted attempt remain visible in the inbox as partial lines.
