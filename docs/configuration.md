# Configuration

BrainLayer uses one shell-compatible config file as the local source of truth:

```text
~/.config/brainlayer/brainlayer.env
```

Launchd templates source this file through `brainlayer-env-run.sh` before they
exec their service command. API keys stay out of rendered LaunchAgent plists.

## Environment Variables

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `BRAINLAYER_DB` | `~/.local/share/brainlayer/brainlayer.db` | Database file path. Set to override the default location. |

### Launchd environment

| Variable | Default | Description |
|----------|---------|-------------|
| `BRAINLAYER_SYSTEM_ENABLED` | `1` | Global launchd gate. Set to `0`/`false`/`off` to disable launchd-managed BrainLayer jobs. |
| `BRAINLAYER_ENV_FILE` | `~/.config/brainlayer/brainlayer.env` | Override the config file sourced by launchd templates. |
| `BRAINLAYER_DISABLED_SLEEP_SECONDS` | `3600` | Sleep duration for disabled KeepAlive jobs to avoid restart loops. Use `0` in tests. |
| `BRAINLAYER_SUPERVISOR_SOCKET` | unset | Opt in to owner-only Unix component control for health/watchdog checks; leave unset until consolidated installer migration. |
| `BRAINLAYER_PAUSE_SENTINEL` | existing `pause.sentinel` | Override the shared hold path for standalone watchdog adapters; explicit health-check configuration takes precedence. |

Control defaults to `~/.local/share/brainlayer/supervisor/control.sock`; the environment override is resolved identically by supervisor and adapters. Read-only health never connects to control. Healing with unavailable or malformed control returns a critical `supervisor_control_unavailable` result, preserves badge/canary checks, and records failed heals. Mutations wait for the advertised component exit grace plus five seconds.
The staged supervisor uses a 15-second shutdown cap; its future plist must allow at least 25 seconds.
Missing-count comparisons accept ticks at least 240 seconds apart. A slow healthy hotlane during big ingest can still look stalled when arrivals outrun embeddings; net counts alone cannot distinguish overload from a wedge.

### Enrichment (retired)

LLM chunk and session enrichment is retired. Provider, backend, model, concurrency,
and rate controls are no longer activation instructions.
History: [retirement details](enrichment.md).

### Google credential compatibility gate

Older installed hotlane plists can still set `BRAINLAYER_REQUIRE_GOOGLE_API_KEY=1`.
The installed `brainlayer-env-run.sh` then exits **78** before exec if neither a
`GOOGLE_API_KEY`/`GOOGLE_GENERATIVE_AI_API_KEY` value nor a Google key declaration
in the env file is present. Keep the 1Password-backed `GOOGLE_API_KEY` configuration,
the require-key flag, and this exit-78 gate until the release re-renders the installed
hotlane plists. Retirement alone does not authorize deleting or bypassing them.

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | empty | Retained Google key configuration for the installed-hotlane compatibility gate. Prefer a 1Password `op read` reference. |
| `BRAINLAYER_REQUIRE_GOOGLE_API_KEY` | `0` in env-run | When an installed plist sets `1`, require Google key configuration before executing its command. |

### Launchd Toggles

Each install-managed plist in `scripts/launchd/` sources the config file and
checks its own `BRAINLAYER_LAUNCHD_*_ENABLED` gate before exec.

| Variable | Default | Controls |
|----------|---------|----------|
| `BRAINLAYER_LAUNCHD_HOTLANE_ENABLED` | `1` | Gates `com.brainlayer.hotlane-brainbar`. |
| `BRAINLAYER_LAUNCHD_DECAY_ENABLED` | `1` | `com.brainlayer.decay` |
| `BRAINLAYER_LAUNCHD_DRAIN_ENABLED` | `1` | `com.brainlayer.drain` |
| `BRAINLAYER_LAUNCHD_WATCH_ENABLED` | `1` | `com.brainlayer.watch` |
| `BRAINLAYER_LAUNCHD_INDEX_ENABLED` | `1` | `com.brainlayer.index` |
| `BRAINLAYER_LAUNCHD_BACKUP_DAILY_ENABLED` | `1` | `com.brainlayer.backup-daily` |
| `BRAINLAYER_LAUNCHD_JSONL_BACKUP_ENABLED` | `1` | `com.brainlayer.jsonl-backup` |
| `BRAINLAYER_LAUNCHD_MAINTENANCE_NIGHTLY_ENABLED` | `1` | `com.brainlayer.maintenance-nightly` |
| `BRAINLAYER_LAUNCHD_MAINTENANCE_WEEKLY_ENABLED` | `1` | `com.brainlayer.maintenance-weekly` |
| `BRAINLAYER_LAUNCHD_REPAIR_FTS_ENABLED` | `1` | `com.brainlayer.repair-fts` |
| `BRAINLAYER_LAUNCHD_WAL_CHECKPOINT_ENABLED` | `1` | `com.brainlayer.wal-checkpoint` |

### Privacy

| Variable | Default | Description |
|----------|---------|-------------|
| `BRAINLAYER_SANITIZE_EXTRA_NAMES` | (empty) | Comma-separated names to redact from indexed content |
| `BRAINLAYER_SANITIZE_USE_SPACY` | `true` | Use spaCy NER for PII detection during indexing |

## Database Location

The database path is resolved in this order:

1. `BRAINLAYER_DB` environment variable (highest priority)
2. Canonical path `~/.local/share/brainlayer/brainlayer.db` (default)

## Data Sources

BrainLayer reads from these locations by default:

| Source | Location |
|--------|----------|
| Claude Code conversations | `~/.claude/projects/` |
| Deduplicated system prompts | `~/.local/share/brainlayer/prompts/` |
| Daemon socket | `/tmp/brainlayer.sock` |
| Historical enrichment lock | `/tmp/brainlayer-enrichment.lock` (legacy safety/cleanup only) |

### T3 projection selection

`ingest-t3` automatically prefers `~/.t3/userdata/statev2.sqlite` with the V2
projection reader. It also ingests `state.sqlite` when that file carries
projections, deduplicating shared thread/message IDs with the newer file winning.
Without V2, it falls back to `state.sqlite`. Both sources open read-only and
schema or payload failures remain fatal. An explicit `--state-db` selects only
that source and detects V2 from its projection tables when no version is supplied.
`--projection-version` overrides detection when deliberate selection is needed:

```bash
brainlayer ingest-t3 --state-db ~/.t3/userdata/statev2.sqlite --projection-version 2 --dry-run
```

V2 reads settled user/assistant messages from their JSON payloads and retains
all projects, stable message IDs, source timestamps, and `t3-thread` provenance.
Streaming messages become eligible when settled. Missing schema or invalid
payloads raise an alarm; the reader never falls back to legacy tables. When V2
succeeds, an unavailable legacy source does not block its import, but health stays
alerting with `legacy_source_failure`. Dry-run
writes the health file but does not write the destination database.

The packaged T3 LaunchAgent runs every 300 seconds and resolves default sources
on every invocation. To override the source and interval durably, add these
settings to the private BrainLayer env file, then rerender with the installed
`scripts/launchd/install.sh t3-ingest`:

```dotenv
BRAINLAYER_T3_INGEST_STATE_DB=/Users/example/.t3/userdata/statev2.sqlite
BRAINLAYER_T3_INGEST_PROJECTION_VERSION=2
BRAINLAYER_T3_INGEST_INTERVAL_SECONDS=60
```

An explicit projection-version override (V1 or V2) requires an absolute source path. The importer reads all
projects. The renderer pins `--db` to the configured `BRAINLAYER_DB` (or canonical
DB) and `--health-path` beside that DB.
`BRAINLAYER_T3_INGEST_HEALTH_PATH` may explicitly override health placement.
Invalid configuration fails before the installed plist is replaced.
The public ingest and health-check CLIs and direct health-check configuration
share this path resolver; explicit `--health-path`/`--t3-health-path` options
take precedence over the environment. Rerender the job after changing its DB
or health selection so the loaded importer and scheduled consumer agree.

The optional interval accepts 60–86400 seconds; unset restores 300 seconds.
`RunAtLoad` remains enabled. launchd runs one instance of this label, and the existing DB writer arbitration still
applies. Freshness is the interval plus import runtime, model startup, and any
writer contention, so five minutes is a scheduling target rather than a hard
searchability guarantee. Writer contention exits cleanly for the next interval
and records alerting `writer_busy` health. A short writer preflight checks IDs
before embedding, then releases the writer for model loading/embedding. The
importer reacquires it and rechecks IDs before upserting precomputed vectors;
content already persisted by another writer is skipped. Other indexing failures
still fail. Indexed chunks with unchanged scrubbed content skip both embedding
and upsert, so repeat runs do not increment seen counts or add dedupe audit rows.
Each run embeds at most 200 changed chunks across both source files. Within each
source, changed chunks are ordered newest first before applying the cap; V2
remains the preferred source. CLI output
and health report `chunks_deferred` for the remainder, which later intervals
continue using stable IDs. First V2 execution imports historical settled messages
across as many bounded runs as needed. Verify loaded argv,
schedule, health and exact persisted message IDs after a reviewed deployment.

These settings select the SQLite importer. `BRAINLAYER_T3_STATE_DB` configures
watcher provider linkage separately. Without overrides, provider linkage uses
the same automatic source selection and unions linked session IDs across files.

## Config File

Create or update the config file with:

```bash
brainlayer init
```

Secure 1Password-backed form:

```bash
GOOGLE_API_KEY="$(op read 'op://Private/Google AI/Gemini API key')"
BRAINLAYER_SYSTEM_ENABLED=1
```

Plaintext Google keys are intentionally not supported in generated BrainLayer
env files. Existing configs retain the credential reference for the compatibility
gate; it does not enable the retired enrichment feature. See
`scripts/launchd/brainlayer.env.example` for the packaged schema and launchd job gates.

## Scheduled Tasks (macOS)

BrainLayer includes launchd plist templates for automated operation:

| Service | Schedule | Description |
|---------|----------|-------------|
| `com.brainlayer.index` | Nightly | Incremental indexing of new conversations |
| `com.brainlayer.watch` | KeepAlive watcher | Watch and queue new conversation writes |
| `com.brainlayer.drain` | Queue/WatchPaths trigger | Drain queued writes as the single writer |
| `com.brainlayer.decay` | Weekly | Refresh decay metadata |
| `com.brainlayer.repair-fts` | Weekly | Read-only FTS row-count check |
| `com.brainlayer.wal-checkpoint` | Daily 09:30 | Checkpoint the WAL |
| `com.brainlayer.backup-daily` | Daily | Backup the BrainLayer DB |
| `com.brainlayer.jsonl-backup` | Daily | Backup Claude JSONL files |
| `com.brainlayer.maintenance-nightly` | Nightly | Light maintenance |
| `com.brainlayer.maintenance-weekly` | Weekly | Full maintenance |

Weekly calendar runs use index → `backup-snapshot` → full maintenance → JSONL backup,
counters, checkpoint, decay and FTS checks → `backup-daily` in upload-only mode. Snapshot and upload use the existing
calendar process-group supervisor sequentially; the upload keeps the existing job/history/log name; no detached worker is created. Nightly runs retain
maintenance first and DB upload last. `BRAINLAYER_BACKUP_PHASE=snapshot` writes a durable local
receipt after gzip CRC/ISIZE and SQLite header/sentinel checks; `upload` drains retained snapshots
from those same bytes, preserving resumable sessions and retrying completed-session verification.
The default `all` phase keeps the combined daily command.

Weekly VACUUM requires a local verified snapshot of the same database, created since local midnight
and at most six hours old. The receipt binds SHA-256, size and path; missing, altered, unverified,
future or yesterday's copies cannot authorize VACUUM. Remote upload is not part of that gate.
Calendar runs without that proof skip VACUUM with exit **76**. A standalone full run may create a
local snapshot under the daily supervisor within the remaining quiet window. A held backup lock
still waits up to two hours. Quiet-window, queue-idle and writer checks run again after backup;
post-backup gate failure skips service bootout and all DB maintenance with exit **76**.

Unuploaded gzip snapshots remain until their own upload verifies, regardless of newer snapshots;
capacity errors list retained archives. Observability's optional `db_snapshot.local_verified`,
`uploaded` and `upload_state` fields distinguish local safety from offsite coverage. Existing
`verified` keeps its uploaded-and-verified meaning. BrainBar needs the tracked Swift follow-up to
render local-verified/pending as neutral; old clients still show that state as unverified.
`--full --dry-run` reports a held backup lock without starting backup work.

Scheduled FTS row-count checks use a read-only connection and do not acquire the maintenance writer lock. Full FTS rebuilds require an explicit offline database copy path.

Install the remaining packaged agents with:

```bash
brainlayer setup
brainlayer setup --launchd
```

Keep existing Google key configuration until the release re-renders installed
hotlane plists. Do not install, load, or resume the retired enrichment service.

The install-managed plists in `scripts/launchd/` render without embedding
`GOOGLE_API_KEY`. Their ProgramArguments call the installed
`brainlayer-env-run.sh` loader, which sources `~/.config/brainlayer/brainlayer.env`
and then execs the service command.

Configuration precedence is:

1. Existing process environment.
2. Simple assignments in `~/.config/brainlayer/brainlayer.env`.
3. Built-in defaults.

The shared Python config loader does not read repo-root `.env`. Command
substitution such as `GOOGLE_API_KEY="$(op read 'op://...')"` is intentionally
left for the shell-based launchd env runner so plaintext secrets are never
written to disk by BrainLayer.

Migration for an existing hardcoded LaunchAgent: move the existing key value into
`~/.config/brainlayer/brainlayer.env` using `brainlayer setup` or
`brainlayer init`, preferably as a 1Password `op read` reference, then have the
deployment lead reinstall the repo-generated plist. Do not paste the key into
shell history, logs, PRs, or chat.


## Manual credential cleanup

`brainlayer scrub-at-rest --providers google_oauth --db /path/to/offline-copy.db --dry-run`
reports matched row counts per table, column and provider. Provider modes are
`google_oauth` (the default), `context7`, and `exa_labeled`. EXA cleanup requires
a label with an `exa` segment and `key`; separators (`_`, `.`, `-`) or a
camelCase capital can end the segment. Labels like `example_key` and bare UUIDs
are preserved. Use the same mode for the survey and apply. Remove `--dry-run` to
apply to an offline copy; `--batch-size` defaults to 100. An explicit database path
is required. A live dry run is read-only and needs no opt-in. Live apply, including
symlink/hardlink aliases, refuses without `--allow-live-db` and `--expect-rows N`.
Use the sum of **all** table row counts from the preceding dry run for N (including
FTS/history copies, not just chunks). Apply re-surveys after quiescence and refuses
if the current total differs.

The guarded mode holds the maintenance lock, requires an active enrichment pause
sentinel and a verified backup receipt for that DB no older than 24 hours, and
reuses VACUUM's quiet-window (04:00–06:00 local), idle-queue and writer gates.
`--wait-for-backup-seconds N` (default 0) lets guarded live apply poll at intervals of up to 30 seconds
for a qualifying receipt before taking the lock or quiescing services. Dry runs and
offline applies ignore it. Poll intervals shorten near the deadline. Waiting ends
at N seconds or when only 20 minutes remain in the quiet window, reserving that
time for the guarded run. A refusal after sleeping reports `verified-backup-timeout`;
if no wait is possible before the first sleep, it reports `verified-backup-required`,
as does the default. Receipt freshness uses `snapshot_created_at` (the snapshot's VACUUM time), not `attempted_at`.
It quiesces the fleet/throughput/tier-0 watchdogs, health-check healer, BrainBar UI/daemon,
hotlane, watcher, drain, index, tier-3 ingest, decay and enrichment using maintenance's service helpers. It checks
that jobs remain unloaded, BrainBar processes are gone and no writable database
descriptors remain before applying. Any failed gate aborts before writing.
Service restoration runs in `finally`; services previously down or deliberately
paused remain down, and resume failures are reported by count. Proof and rehearsal
must use copies and synthetic service stubs; canonical execution belongs to the
deployment lead's maintenance window.

The schema survey includes logical tables, FTS and repair/history copies, including
text stored in numeric-affinity columns. Only the selected provider spans change;
assignment/quarantine and other providers stay intact. Selected provider regexes
see full values so window boundaries cannot truncate spans. Changed chunk content
gets canonical hashes, SimHash bands and character counts; summaries/previews retain
all unmatched text. Existing chunk/KG/git FTS triggers run, and session FTS refreshes
explicitly. Bitemporal preimage capture and preview regeneration are transactionally
suppressed during chunk redaction and restored verbatim; independent previews keep
their unmatched text and existing history is scrubbed. Identity/reference
matches appear in dry counts but refuse apply. Unsupported virtual/external-content
index layouts refuse rather than silently skip. Unique-index collisions caused by
redaction (for example in KG entities/facts or git memories) abort the current
batch rather than deleting or merging rows. A failed batch rolls back; earlier committed batches may remain,
so rerun after resolving the error. Writer/maintenance locks serialize apply, with
bounded busy retries and checkpoints on the same writer connection before/after.
The final survey must find zero selected matches. This is logical redaction, not
physical erasure of old WAL/free pages, embeddings, source transcripts or backups.
During the entire command, SQLite diagnostics log only error codes/classes, including
file handlers; the process's normal SQLite callback is restored after stores close.
Failures retain value-free causal error categories and cleanup notes.

### Daily Drive upload recovery

`backup-daily` probes an interrupted resumable upload before retransmitting bytes.
Network failures use exponential delays (30 seconds up to 5 minutes), bounded by
`BRAINLAYER_DRIVE_UPLOAD_RETRY_BUDGET_SECONDS` (default **1200**, per period without
confirmed progress). An explicit `BRAINLAYER_DRIVE_UPLOAD_STALL_MAX_ATTEMPTS`
additionally caps attempts; without it, uploads allow 30 attempts. Metadata-read
retries retain their existing default of three attempts. HTTP 401/403 and other
client errors fail immediately; transient HTTP 429/5xx retain short bounded retries.
The whole-backup timeout still applies.

A stalled daily receipt records `resumable_upload` with the session URL,
destination folder ID, and the artifact's absolute path, byte count, and pre-upload
SHA-256. Receipts containing this private upload capability are restricted to mode 0600. The next upload run
requires the same DB, machine, target folder, provenance, and unchanged artifact,
then asks Drive for its current offset. A completed session goes through the usual
verification. An expired session (404/410) starts a fresh snapshot and upload.
See the [Drive resumable upload protocol](https://developers.google.com/workspace/drive/api/guides/manage-uploads#resume-upload).

After a newer snapshot is uploaded and verified on Drive in the same run,
reclamation only considers older dated `.db.gz` files
directly in the staging directory whose stalled receipt and current SHA-256 prove
ownership. Uploaded/verified paths, symlinks, hard links, changed bytes, malformed
receipts, and legacy receipts without identity evidence are retained. Each removal
is logged in `superseded_stalled_deleted`; uncertainty emits a warning. This does
not change Drive retention, successful-copy retention, or restore formats.
Insufficient snapshot capacity fails explicitly and names retained stalled files;
these files are never removed to make space.

Daily receipts record `snapshot_created_at` at VACUUM start; resumed receipts
preserve it. Verified-backup reuse and observability age the snapshot, rather
than its later upload attempt. Legacy dated snapshots use conservative UTC
midnight; unknown creation times fail closed. Resume requires an age below
`BRAINLAYER_BACKUP_RESUME_MAX_AGE_HOURS` (default **36**). Weekly maintenance's
`BRAINLAYER_BACKUP_REUSE_VERIFIED_MAX_AGE_HOURS` further limits that age.

The upload session is checkpointed privately before sending bytes, so even a
supervisor exit retains a probeable session. Timeout, stop, and mid-upload auth
failures preserve its identity and confirmed offset. Recovery sleeps and request
timeouts share the remaining whole-run deadline, reserving five seconds to write
the receipt. Session-creation POST network failures use the same bounded backoff;
HTTP client/auth failures still stop immediately.

A fresh snapshot cannot overwrite a same-day stalled gzip. An expired or unsafe
session with such a collision fails explicitly and retains the existing file.
A later dated run can create a replacement without changing backup filenames.

The weekly age gate is checked again after upload and verification. A snapshot
that crosses the weekly freshness limit during upload remains protected locally
and returns an unsuccessful safety receipt, so weekly VACUUM cannot proceed.
