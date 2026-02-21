# Enrichment

BrainLayer enriches indexed chunks with structured metadata using a local LLM. Think of it as a librarian cataloging every conversation snippet.

## Chunk Enrichment

Each chunk gets 10 metadata fields:

| Field | Description | Example |
|-------|-------------|---------|
| `summary` | 1-2 sentence gist | "Debugging Telegram bot message drops under load" |
| `tags` | Topic tags (comma-separated) | "telegram, debugging, performance" |
| `importance` | Relevance score 1-10 | 8 (architectural decision) vs 2 (directory listing) |
| `intent` | What was happening | `debugging`, `designing`, `implementing`, `configuring`, `deciding`, `reviewing` |
| `primary_symbols` | Key code entities | "TelegramBot, handleMessage, grammy" |
| `resolved_query` | Question this answers (HyDE-style) | "How does the Telegram bot handle rate limiting?" |
| `epistemic_level` | How proven is this | `hypothesis`, `substantiated`, `validated` |
| `version_scope` | System state context | "grammy 1.32, Node 22" |
| `debt_impact` | Technical debt signal | `introduction`, `resolution`, `none` |
| `external_deps` | Libraries/APIs mentioned | "grammy, Supabase, Railway" |

### Running Enrichment

```bash
# Basic (50 chunks at a time)
brainlayer enrich

# Larger batches
brainlayer enrich --batch-size=100

# Process up to 5000 chunks
brainlayer enrich --max=5000

# With parallel workers
brainlayer enrich --parallel=3
```

### Source-Aware Thresholds

Not all chunks are worth enriching. BrainLayer automatically skips chunks that are too short:

| Source | Minimum Length | Reason |
|--------|---------------|--------|
| Claude Code | 50 characters | Code context needs substance |
| WhatsApp / Telegram | 15 characters | Short messages can still be meaningful |

Skipped chunks are tagged as `skipped:too_short` and excluded from enrichment stats.

## Session Enrichment

Session-level analysis extracts structured insights from entire conversations:

```bash
brainlayer enrich-sessions
brainlayer enrich-sessions --project my-project --since 2026-01-01
brainlayer enrich-sessions --stats   # Show progress
```

Session enrichment extracts:

- **Summary** — what the session was about
- **Decisions** — architectural and implementation choices made
- **Corrections** — mistakes caught and fixed
- **Learnings** — new knowledge gained
- **Patterns** — recurring approaches identified
- **Quality scores** — code quality, communication quality

## LLM Backends

Two local backends are supported:

| Backend | Best for | Speed | How to start |
|---------|----------|-------|-------------|
| **MLX** | Apple Silicon (M1/M2/M3) | 21-87% faster | `python3 -m mlx_lm.server --model mlx-community/Qwen2.5-Coder-14B-Instruct-4bit --port 8080` |
| **Ollama** | Any platform | ~1s/chunk (short), ~13s (long) | `ollama serve` + `ollama pull glm4` |

Backend is auto-detected: Apple Silicon defaults to MLX, everything else to Ollama. Override with:

```bash
BRAINLAYER_ENRICH_BACKEND=mlx brainlayer enrich
BRAINLAYER_ENRICH_BACKEND=ollama brainlayer enrich
```

### Performance Tips

- Set `"think": false` in Ollama API calls — GLM-4.7 defaults to thinking mode, adding 350+ tokens and 20s delay for no benefit
- Use `PYTHONUNBUFFERED=1` for log visibility in background processes
- MLX parallel workers: each gets its own DB connection (thread-local)

## Stall Detection

If a chunk takes too long (default: 5 minutes), it's automatically killed and skipped:

```bash
BRAINLAYER_STALL_TIMEOUT=300 brainlayer enrich  # 5 min default
```

Progress is logged every N chunks:

```bash
BRAINLAYER_HEARTBEAT_INTERVAL=25 brainlayer enrich  # Log every 25 chunks
```
