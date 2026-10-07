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

Weekly maintenance waits up to two hours for an in-flight daily DB backup. A recent, verified daily backup is reused because daily and weekly Drive retention policies are the same. A fresh backup uses the daily process supervisor with a deadline at the end of the quiet window. After the backup wait, it reruns the quiet-window, queue-idle, and writer gates. If the wait times out or the backup is unverified, light maintenance continues, VACUUM is skipped, and the job exits **76**. If a post-backup gate fails, the job skips service bootout and all database maintenance, records the failed gate, and exits **76**. The maintenance log records the backup status. `--full --dry-run` reports a held backup lock without waiting or running backup work.

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
as does the default. Receipt freshness continues to use `attempted_at`.
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
