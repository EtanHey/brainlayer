# Configuration

All BrainLayer configuration is via environment variables. No config files needed.

## Environment Variables

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `BRAINLAYER_DB` | `~/.local/share/brainlayer/brainlayer.db` | Database file path. Set to override the default location. |

### Enrichment

| Variable | Default | Description |
|----------|---------|-------------|
| `BRAINLAYER_ENRICH_BACKEND` | auto-detect | LLM backend: `mlx` or `ollama`. Auto-detects Apple Silicon → MLX, else Ollama. |
| `BRAINLAYER_ENRICH_MODEL` | `glm-4.7-flash` | Ollama model name for enrichment |
| `BRAINLAYER_MLX_MODEL` | `mlx-community/Qwen2.5-Coder-14B-Instruct-4bit` | MLX model identifier |
| `BRAINLAYER_OLLAMA_URL` | `http://127.0.0.1:11434/api/generate` | Ollama API endpoint |
| `BRAINLAYER_MLX_URL` | `http://127.0.0.1:8080/v1/chat/completions` | MLX server endpoint |
| `BRAINLAYER_STALL_TIMEOUT` | `300` | Seconds before killing a stuck enrichment chunk |
| `BRAINLAYER_HEARTBEAT_INTERVAL` | `25` | Log progress every N chunks during enrichment |

### Privacy

| Variable | Default | Description |
|----------|---------|-------------|
| `BRAINLAYER_SANITIZE_EXTRA_NAMES` | (empty) | Comma-separated names to redact from indexed content |
| `BRAINLAYER_SANITIZE_USE_SPACY` | `true` | Use spaCy NER for PII detection during indexing |

## Database Location

The database path is resolved in this order:

1. `BRAINLAYER_DB` environment variable (highest priority)
2. Legacy path `~/.local/share/zikaron/zikaron.db` (if it exists — for migration)
3. Canonical path `~/.local/share/brainlayer/brainlayer.db` (default for fresh installs)

## Data Sources

BrainLayer reads from these locations by default:

| Source | Location |
|--------|----------|
| Claude Code conversations | `~/.claude/projects/` |
| Deduplicated system prompts | `~/.local/share/brainlayer/prompts/` |
| Daemon socket | `/tmp/brainlayer.sock` |
| Enrichment lock | `/tmp/brainlayer-enrichment.lock` |

## Scheduled Tasks (macOS)

BrainLayer includes launchd plist templates for automated operation:

| Service | Schedule | Description |
|---------|----------|-------------|
| `com.brainlayer.index` | Every 30 minutes | Incremental indexing of new conversations |
| `com.brainlayer.enrich` | Every hour | Enrich up to 500 new chunks per run |

Install with:

```bash
brainlayer init  # Includes launchd setup option
```
