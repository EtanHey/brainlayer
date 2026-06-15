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

### Enrichment

| Variable | Default | Description |
|----------|---------|-------------|
| `BRAINLAYER_SYSTEM_ENABLED` | `1` | Global launchd gate. Set to `0`/`false`/`off` to disable launchd-managed BrainLayer jobs. |
| `BRAINLAYER_ENRICH_ENABLED` | `1` | Realtime enrichment gate. The launchd enrichment service sleeps/exits disabled when false. |
| `BRAINLAYER_ENRICH_MODE` | `remote` | Enrichment mode seam: `remote` for provider-backed enrichment, `local` for local model backends. |
| `BRAINLAYER_ENRICH_PROVIDER` | `gemini` | Provider selection seam. Gemini is wired today; other providers should coordinate on this key. |
| `BRAINLAYER_ENRICH_BACKEND` | `gemini` in config file, auto-detect elsewhere | Backend key. Local flows historically use `mlx`, `ollama`, or `groq`; the unified config uses `gemini` for the remote Gemini path. |
| `BRAINLAYER_ENRICH_MODEL` | `glm-4.7-flash` | Ollama model name for enrichment |
| `BRAINLAYER_MLX_MODEL` | `mlx-community/Qwen2.5-Coder-14B-Instruct-4bit` | MLX model identifier |
| `BRAINLAYER_OLLAMA_URL` | `http://127.0.0.1:11434/api/generate` | Ollama API endpoint |
| `BRAINLAYER_MLX_URL` | `http://127.0.0.1:8080/v1/chat/completions` | MLX server endpoint |
| `GOOGLE_API_KEY` | empty | Google AI API key used by realtime Gemini enrichment. Prefer a 1Password `op read` reference in the config file. |
| `BRAINLAYER_ENV_FILE` | `~/.config/brainlayer/brainlayer.env` | Override path for the config file that launchd templates source. |
| `BRAINLAYER_ENRICH_RATE` | `15` | Realtime Gemini enrichment rate used by launchd jobs. |
| `BRAINLAYER_ENRICH_CONCURRENCY` | `4` | Realtime Gemini enrichment concurrency used by launchd jobs. |
| `BRAINLAYER_MAX_COMMIT_BATCH` | `25` | Max enrichment write batch used by launchd jobs. |
| `BRAINLAYER_GEMINI_SERVICE_TIER` | `flex` | Gemini service tier used by launchd jobs. |
| `BRAINLAYER_DISABLED_SLEEP_SECONDS` | `3600` | Sleep duration for disabled KeepAlive jobs to avoid tight launchd restart loops. Use `0` in tests. |
| `BRAINLAYER_STALL_TIMEOUT` | `300` | Seconds before killing a stuck enrichment chunk |
| `BRAINLAYER_HEARTBEAT_INTERVAL` | `25` | Log progress every N chunks during enrichment |

### Launchd Toggles

Each install-managed plist in `scripts/launchd/` sources the config file and
checks its own `BRAINLAYER_LAUNCHD_*_ENABLED` gate before exec.

| Variable | Default | Controls |
|----------|---------|----------|
| `BRAINLAYER_LAUNCHD_ENRICHMENT_ENABLED` | `1` | `com.brainlayer.enrichment` |
| `BRAINLAYER_LAUNCHD_HOTLANE_ENABLED` | `1` | Reserved for the hotlane LaunchAgent when installed by the deploy path. |
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
| Enrichment lock | `/tmp/brainlayer-enrichment.lock` |

## Config File

Create or update the config file with:

```bash
brainlayer init
```

Secure 1Password-backed form:

```bash
GOOGLE_API_KEY="$(op read 'op://Private/Google AI/Gemini API key')"
BRAINLAYER_SYSTEM_ENABLED=1
BRAINLAYER_ENRICH_ENABLED=1
BRAINLAYER_ENRICH_MODE=remote
BRAINLAYER_ENRICH_PROVIDER=gemini
BRAINLAYER_ENRICH_BACKEND=gemini
BRAINLAYER_ENRICH_RATE=15
BRAINLAYER_ENRICH_CONCURRENCY=4
BRAINLAYER_MAX_COMMIT_BATCH=25
BRAINLAYER_GEMINI_SERVICE_TIER=flex
```

Plain-env fallback:

```bash
GOOGLE_API_KEY='...'
BRAINLAYER_SYSTEM_ENABLED=1
BRAINLAYER_ENRICH_ENABLED=1
BRAINLAYER_ENRICH_MODE=remote
BRAINLAYER_ENRICH_PROVIDER=gemini
BRAINLAYER_ENRICH_BACKEND=gemini
BRAINLAYER_ENRICH_RATE=15
BRAINLAYER_ENRICH_CONCURRENCY=4
BRAINLAYER_MAX_COMMIT_BATCH=25
BRAINLAYER_GEMINI_SERVICE_TIER=flex
```

Manual configs should keep the same tuning keys as the 1Password form. See
`scripts/launchd/brainlayer.env.example` for the full schema and launchd job gates.

## Scheduled Tasks (macOS)

BrainLayer includes launchd plist templates for automated operation:

| Service | Schedule | Description |
|---------|----------|-------------|
| `com.brainlayer.index` | Nightly | Incremental indexing of new conversations |
| `com.brainlayer.enrichment` | KeepAlive supervisor | Run realtime Gemini enrichment against recent chunks |
| `com.brainlayer.watch` | KeepAlive watcher | Watch and queue new conversation writes |
| `com.brainlayer.drain` | Queue/WatchPaths trigger | Drain queued writes as the single writer |
| `com.brainlayer.decay` | Weekly | Refresh decay metadata |
| `com.brainlayer.repair-fts` | Weekly | Repair FTS indexes |
| `com.brainlayer.wal-checkpoint` | Weekly | Checkpoint the WAL |
| `com.brainlayer.backup-daily` | Daily | Backup the BrainLayer DB |
| `com.brainlayer.jsonl-backup` | Daily | Backup Claude JSONL files |
| `com.brainlayer.maintenance-nightly` | Nightly | Light maintenance |
| `com.brainlayer.maintenance-weekly` | Weekly | Full maintenance |

Manual install and control:

```bash
bash scripts/launchd/install.sh enrichment
bash scripts/launchd/install.sh unload enrichment
bash scripts/launchd/install.sh load enrichment
```

The install-managed plists in `scripts/launchd/` render without embedding
`GOOGLE_API_KEY`. Their ProgramArguments call the installed
`brainlayer-env-run.sh` loader, which sources `~/.config/brainlayer/brainlayer.env`
and then execs the service command.

Migration for an existing hardcoded LaunchAgent: move the existing key value into
`~/.config/brainlayer/brainlayer.env` using `brainlayer init`, preferably as a
1Password `op read` reference, then have the deployment lead reinstall the
repo-generated plist. Do not paste the key into shell history, logs, PRs, or chat.
