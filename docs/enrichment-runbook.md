# Enrichment Runbook

> How to run, monitor, and troubleshoot BrainLayer's enrichment pipeline.

---

## What Enrichment Does

Every chunk in BrainLayer starts as raw text — conversation snippets, code, error logs. Enrichment passes each chunk through a local LLM to add 10 structured metadata fields: a summary, topic tags, importance score, intent, key symbols, and more.

This metadata powers better search (filter by importance, intent, or tags), the brain graph (cluster by topic), and the dashboard analytics.

---

## Quick Start

```bash
# Make sure Ollama is running
ollama serve  # if not already running

# Run a batch (50 chunks)
cd /path/to/brainlayer
source .venv/bin/activate
brainlayer enrich

# Check progress
brainlayer enrich --stats
```

---

## Daily / Ongoing Enrichment

The `auto-enrich.sh` script handles this. Set it up with cron or launchd:

```bash
# Run every 6 hours — skips if queue is small
./scripts/auto-enrich.sh --threshold 500 --max-hours 3
```

What it does:
1. Checks how many chunks are unenriched
2. Skips if below threshold (default: 500)
3. Alerts via Telegram if queue > 5,000 (you're falling behind)
4. Starts the right backend (Ollama or MLX)
5. Runs enrichment with a time cap
6. Reports results via Telegram

---

## Choosing a Backend

| | Ollama | MLX |
|--|--------|-----|
| **Setup** | `ollama pull glm4` | `pip install mlx-lm` + download model |
| **Speed** | ~1s/chunk (short content) | 21-87% faster |
| **Memory** | ~4GB VRAM | ~8GB RAM (14B-4bit model) |
| **Parallel** | Usually 1 worker | 2-3 workers work well |
| **Env var** | `BRAINLAYER_ENRICH_BACKEND=ollama` (default) | `BRAINLAYER_ENRICH_BACKEND=mlx` |

To switch, just set the env var. Both produce the same 10-field JSON output.

---

## Cloud Backfill (Gemini Batch API)

For the initial bulk run (251K chunks), local LLM would take weeks. Instead, use Gemini 2.5 Flash-Lite Batch API:

### Cost
- ~$16 total (251K chunks)
- Gemini Flash-Lite: $0.075/MTok input, $0.30/MTok output (batch gets 50% discount)

### How to Run

```bash
cd /path/to/brainlayer
source .venv/bin/activate

# Set your Gemini API key
export GOOGLE_API_KEY=your-key-here

# Run backfill (processes ~100K chunks per batch)
python3 scripts/cloud_backfill.py

# Resume if interrupted
python3 scripts/cloud_backfill.py --resume
```

The script:
1. Exports unenriched chunks from SQLite
2. Uploads to Gemini Batch API in batches of 100K
3. Polls for completion (~30 min per batch)
4. Downloads results and imports back to SQLite
5. Logs token usage and cost

### Safety
- Only targets `WHERE enriched_at IS NULL` — never overwrites existing enrichments
- Validates a 100-chunk sample before full run
- Generates cost log for budget tracking

---

## 10-Field Schema

Each enriched chunk gets these fields:

```json
{
  "summary": "Debugging Telegram bot message drops under high load",
  "tags": "telegram, debugging, performance, grammy",
  "importance": 7,
  "intent": "debugging",
  "primary_symbols": "TelegramBot, handleMessage, grammy",
  "resolved_query": "Why does the Telegram bot drop messages during peak hours?",
  "epistemic_level": "substantiated",
  "version_scope": "grammy 1.32, Railway deployment",
  "debt_impact": "resolution",
  "external_deps": "grammy, Railway"
}
```

### Field Details

- **importance** (1-10): Directory listings get a 2, architectural decisions get an 8-9
- **intent**: One of `debugging`, `designing`, `implementing`, `configuring`, `discussing`, `deciding`, `reviewing`
- **epistemic_level**: `hypothesis` (guessing), `substantiated` (evidence-backed), `validated` (tested/confirmed)
- **debt_impact**: `introduction` (new tech debt), `resolution` (fixing debt), `none` (neutral)

---

## Troubleshooting

### Enrichment hangs or is very slow

1. **Check Ollama thinking mode**: `"think": false` must be set in the API call. Without it, GLM-4.7 adds 350+ reasoning tokens per chunk (20s vs 1s).
2. **Check DB locks**: `lsof ~/.local/share/brainlayer/brainlayer.db` — if daemon + MCP + enrichment are all running, the `busy_timeout` should handle it, but check the logs.
3. **Stale lock file**: `rm /tmp/brainlayer-enrichment.lock` if enrichment died and left a lock.

### DB locked errors

The pipeline has `busy_timeout = 5000ms` + 3-attempt retry. If you still see lock errors:
1. Check who has the DB open: `lsof ~/.local/share/brainlayer/brainlayer.db`
2. Restart the daemon: `brainlayer serve --http 8787` (it reconnects cleanly)
3. Make sure only one enrichment process runs at a time

### Enrichment produces bad JSON

The LLM sometimes returns malformed JSON. The parser tries to extract JSON from the response using brace-matching. If it fails, the chunk is skipped (counted as "failed" in batch stats). Failed chunks can be retried on the next run.

### Backup and Recovery

**Before any bulk operation**, back up the database:
```bash
# WAL-safe copy using SQLite VACUUM INTO
mkdir -p ~/.local/share/brainlayer/backups
sqlite3 ~/.local/share/brainlayer/brainlayer.db \
  "VACUUM INTO '$HOME/.local/share/brainlayer/backups/brainlayer-$(date +%Y%m%d-%H%M).db'"
```

**To restore from backup:**
```bash
# Stop daemon and any enrichment
pkill -f "brainlayer serve" || true
rm /tmp/brainlayer-enrichment.lock 2>/dev/null || true

# Copy backup over current DB
cp ~/.local/share/brainlayer/backups/brainlayer-YYYYMMDD-HHMM.db \
   ~/.local/share/brainlayer/brainlayer.db

# Restart daemon
brainlayer serve --http 8787
```

### Queue keeps growing

New Claude Code sessions add chunks constantly. If the queue grows faster than enrichment processes it:
1. Increase batch frequency (cron every 4 hours instead of 6)
2. Use MLX + parallel workers: `--parallel 3` with MLX backend
3. Run a cloud backfill to catch up
