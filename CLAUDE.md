# BrainLayer (זיכרון) - Local Knowledge Pipeline

> **Memory** for Claude Code conversations. Index, search, retrieve, and visualize knowledge from past coding sessions.

---

## Quick Start

```bash
cd ~/projects/brainlayer
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Index conversations
brainlayer index

# Start daemon (for dashboard + fast searches)
brainlayer serve --http 8787

# Search
brainlayer search "how did I implement authentication"

# Enrich with local LLM
brainlayer enrich
```

---

## Architecture (Feb 2026 - sqlite-vec)

```
~/.claude/projects/          # Source: Claude Code conversations (JSONL)
        ↓
┌─────────────────────────────────────────────────────────────┐
│  PIPELINE                                                    │
│  ┌─────────┐  ┌──────────┐  ┌───────┐  ┌───────┐  ┌───────┐│
│  │ Extract │→ │ Classify │→ │ Chunk │→ │ Embed │→ │ Index ││
│  └─────────┘  └──────────┘  └───────┘  └───────┘  └───────┘│
│                                         bge-large sqlite-vec│
│                                         1024 dims   fast DB │
└─────────────────────────────────────────────────────────────┘
        ↓
~/.local/share/brainlayer/brainlayer.db   # Storage: sqlite-vec (~1.4GB, 260K+ chunks)
        ↓
┌─────────────────────────────────────────────────────────────┐
│  POST-PROCESSING                                             │
│  ┌───────────┐  ┌──────────────┐  ┌────────────┐           │
│  │ Enrichment│  │ Brain Graph  │  │ Obsidian   │           │
│  │ (GLM-4.7) │  │ (clustering) │  │ Export     │           │
│  └───────────┘  └──────────────┘  └────────────┘           │
└─────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│  INTERFACES                                                  │
│  ┌───────┐  ┌──────────────┐  ┌───────────┐  ┌───────────┐ │
│  │  CLI  │  │ FastAPI      │  │ MCP Server│  │ Dashboard │ │
│  │       │  │ Daemon       │  │ brainlayer-  │  │ (Next.js) │ │
│  │       │  │ :8787/socket │  │ mcp       │  │ :3000     │ │
│  └───────┘  └──────────────┘  └───────────┘  └───────────┘ │
└─────────────────────────────────────────────────────────────┘
```

> **Storage:** sqlite-vec with bge-large-en-v1.5 embeddings (1024 dims). WAL mode + busy_timeout=5000ms for concurrent access.

---

## File Structure

```
brainlayer/
├── src/brainlayer/
│   ├── __init__.py
│   ├── cli/                    # CLI interface (typer)
│   │   └── __init__.py         # All CLI commands
│   ├── cli_new.py              # New unified CLI (in progress)
│   ├── client.py               # Python client for daemon API
│   ├── clustering.py           # Topic clustering (HDBSCAN + UMAP)
│   ├── daemon.py               # FastAPI HTTP daemon (25+ endpoints)
│   ├── embeddings.py           # bge-large-en-v1.5 embedding model
│   ├── index_new.py            # Unified indexer (batch + progress)
│   ├── migrate.py              # DB schema migrations
│   ├── vector_store.py         # sqlite-vec storage layer
│   ├── dashboard/              # Built-in TUI dashboard (textual)
│   │   ├── app.py
│   │   ├── search.py
│   │   └── views.py
│   ├── mcp/                    # MCP server (8 tools)
│   │   └── __init__.py
│   └── pipeline/               # Processing stages
│       ├── extract.py           # Stage 1: Parse JSONL conversations
│       ├── extract_whatsapp.py  # WhatsApp chat import
│       ├── extract_markdown.py  # Markdown file import
│       ├── extract_claude_desktop.py  # Claude Desktop import
│       ├── extract_corrections.py    # Correction detection
│       ├── classify.py          # Stage 2: Content classification
│       ├── chunk.py             # Stage 3: AST-aware chunking
│       ├── enrichment.py        # LLM enrichment (summaries, tags, importance)
│       ├── brain_graph.py       # Brain graph generation (nodes + edges)
│       ├── obsidian_export.py   # Obsidian vault export
│       ├── operation_grouping.py # read→edit→test cycle detection
│       ├── plan_linking.py      # Session → plan/phase linking
│       ├── temporal_chains.py   # Topic chain detection
│       ├── git_overlay.py       # Git diff enrichment
│       ├── semantic_style.py    # Communication style analysis
│       ├── analyze_communication.py  # Evolution analysis
│       ├── cluster_sampling.py  # Cluster-based sampling
│       ├── style_embed.py       # Style embedding
│       ├── style_index.py       # Style indexing
│       ├── unified_timeline.py  # Cross-source timeline
│       ├── time_batcher.py      # Temporal batching
│       ├── longitudinal_analyzer.py  # Long-term trend analysis
│       └── chat_tags.py         # Chat tag extraction
├── tests/
├── pyproject.toml
├── CLAUDE.md                    # This file
└── prd-json/                    # Ralph PRD (if using Ralph)
```

---

## CLI Commands

### Core

```bash
brainlayer index                           # Index all conversations
brainlayer index --project my-project      # Index specific project
brainlayer index-fast                      # Fast incremental index

brainlayer search "authentication"         # Semantic search
brainlayer search "config.py" --text       # Exact text match

brainlayer stats                           # Knowledge base statistics
brainlayer clear --yes                     # Clear database
```

### Daemon & Server

```bash
brainlayer serve                           # Start daemon (Unix socket)
brainlayer serve --http 8787               # Start daemon (HTTP mode for dashboard)
```

### Enrichment

```bash
brainlayer enrich                          # Run LLM enrichment (GLM-4.7-Flash via Ollama)
brainlayer enrich --batch-size 50          # Custom batch size
```

Enrichment adds to each chunk: summary, tags, importance score (1-10), intent classification. Uses local GLM-4.7-Flash with `"think": false` for speed (~1s/chunk for short, ~13s for long).

### Analysis & Export

```bash
brainlayer git-overlay                     # Enrich with git diff context
brainlayer group-operations                # Detect read→edit→test cycles
brainlayer topic-chains                    # Find topic continuity across sessions
brainlayer plan-linking                    # Link sessions to plans/phases
brainlayer brain-export                    # Generate brain graph JSON
brainlayer export-obsidian                 # Export to Obsidian vault
```

### Style Analysis

```bash
brainlayer analyze-style                   # Quick WhatsApp style analysis
brainlayer analyze-evolution --use-embeddings -c ~/export.json -o data/archives/style-$(date +%Y-%m-%d)
brainlayer analyze-semantic                # Semantic style profiling
brainlayer list-chats                      # List indexed chat sources
```

### Utilities

```bash
brainlayer context <chunk_id>              # Get surrounding context
brainlayer review <session_id>             # Review a session
brainlayer fix-projects                    # Normalize project names
brainlayer migrate                         # Run DB migrations
brainlayer dashboard                       # Interactive TUI dashboard
```

---

## Daemon HTTP Endpoints

The daemon (`brainlayer serve --http 8787`) exposes a FastAPI server used by the Next.js dashboard:

### Health & Stats

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/health/services` | GET | Service status (daemon, enrichment, Ollama) |
| `/stats` | GET | Knowledge base stats (chunks, projects, types) |
| `/stats/tokens` | GET | LLM token usage + costs |
| `/stats/enrichment` | GET | Enrichment progress (enriched vs total) |
| `/stats/service-runs` | GET | Recent service run logs |

### Search & Context

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search` | POST | Semantic + keyword search |
| `/context/{chunk_id}` | GET | Surrounding chunks for a result |
| `/dashboard/search` | GET | Dashboard search (GET-friendly) |
| `/session/{session_id}` | GET | Full session detail |

### Brain Graph

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/brain/graph` | GET | Full brain graph (nodes + edges) |
| `/brain/metadata` | GET | Graph metadata (node count, clusters) |
| `/brain/node/{node_id}` | GET | Single node details |

### Content & Backlog

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/content/pipeline-runs` | GET | Content pipeline execution logs |
| `/content/pipeline-stats` | GET | Pipeline routing stats |
| `/backlog/items` | GET | Backlog items (Kanban board) |
| `/backlog/items` | POST | Create backlog item |
| `/backlog/items/{id}` | PATCH | Update backlog item |
| `/backlog/items/{id}` | DELETE | Delete backlog item |

### Events

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/events/recent` | GET | Recent golem events |

---

## MCP Server (8 Tools)

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "brainlayer": {
      "command": "brainlayer-mcp",
      "args": []
    }
  }
}
```

### Available Tools

| Tool | Required Params | Optional Params | Description |
|------|----------------|-----------------|-------------|
| `brainlayer_search` | `query` | `project`, `content_type`, `num_results`, `source`, `tag`, `intent`, `importance_min` | Search past conversations |
| `brainlayer_stats` | — | — | Knowledge base statistics |
| `brainlayer_list_projects` | — | — | List indexed projects |
| `brainlayer_context` | `chunk_id` | `before` (3), `after` (3) | Surrounding context for a result |
| `brainlayer_file_timeline` | `file_path` | `project`, `limit` (50) | File interaction history across sessions |
| `brainlayer_operations` | `session_id` | — | Logical operation groups (read→edit→test) |
| `brainlayer_regression` | `file_path` | `project` | Regression analysis (what changed since last success) |
| `brainlayer_plan_links` | — | `plan_name`, `session_id`, `project` | Session ↔ plan linkage |

### Search Parameters

- **`source`**: `claude_code` (default), `whatsapp`, `youtube`, `all`
- **`content_type`**: `ai_code`, `stack_trace`, `user_message`, `assistant_text`, `file_read`, `git_diff`
- **`intent`**: `debugging`, `designing`, `configuring`, `discussing`, `deciding`, `implementing`, `reviewing`
- **`importance_min`**: 1-10 (from enrichment)
- **`tag`**: enrichment-generated tags (e.g., `bug-fix`, `authentication`, `typescript`)

---

## Enrichment Pipeline

Local LLM enrichment adds structured metadata to each chunk. Think of it as a librarian cataloging every conversation snippet — what it's about, how important it is, and how to find it later.

### Fields (10 total)

| Field | What it captures | Example |
|-------|-----------------|---------|
| `summary` | 1-2 sentence gist | "Debugging why Telegram bot drops messages under load" |
| `tags` | Topic tags (comma-separated) | "telegram, debugging, performance" |
| `importance` | 1-10 relevance score | 8 (architectural decision) vs 2 (directory listing) |
| `intent` | What was happening | `debugging`, `designing`, `implementing`, `configuring`, `discussing`, `deciding`, `reviewing` |
| `primary_symbols` | Key code entities | "TelegramBot, handleMessage, grammy" |
| `resolved_query` | Question this answers (HyDE-style) | "How does the Telegram bot handle rate limiting?" |
| `epistemic_level` | How proven is this | `hypothesis`, `substantiated`, `validated` |
| `version_scope` | What version/system state | "grammy 1.32, Node 22, pre-Railway migration" |
| `debt_impact` | Technical debt signal | `introduction`, `resolution`, `none` |
| `external_deps` | Libraries/APIs mentioned | "grammy, Supabase, Railway" |

The first 4 fields have been populated for ~11.6K chunks via local Ollama. The remaining 6 fields await cloud backfill (Gemini Batch API, ~$16 for all 251K chunks).

### Backends

Two local LLM backends available — use whichever suits your setup:

| Backend | How to start | Speed | Env var |
|---------|-------------|-------|---------|
| **Ollama** (default) | `ollama serve` + `ollama pull glm4` | ~1s/chunk (short), ~13s (long) | `BRAINLAYER_ENRICH_BACKEND=ollama` |
| **MLX** (Apple Silicon) | `python3 -m mlx_lm.server --model mlx-community/Qwen2.5-Coder-14B-Instruct-4bit --port 8080` | 21-87% faster | `BRAINLAYER_ENRICH_BACKEND=mlx` |

Both work with the same enrichment pipeline — just set the env var and go.

### Running Enrichment

```bash
# Basic (50 chunks at a time, Ollama)
brainlayer enrich

# Bigger batches, MLX, parallel workers
BRAINLAYER_ENRICH_BACKEND=mlx brainlayer enrich --batch-size=100 --parallel=3

# Process up to 5000 chunks in one run
brainlayer enrich --max=5000

# Automated scheduling (checks queue, runs if needed)
./scripts/auto-enrich.sh --threshold 500 --max-hours 3
```

### Cloud Backfill (one-time)

For the initial 251K chunk backfill, there's a Gemini Batch API script. See `docs/enrichment-runbook.md` for the full runbook.

### Concurrency Notes

- **`PRAGMA busy_timeout = 5000`** — waits up to 5s for DB locks (daemon + MCP + enrichment can all access DB)
- **Retry logic** — 3 attempts with backoff on `SQLITE_BUSY`
- **Parallel mode** — each thread gets its own DB connection (thread-local VectorStore)
- **Ollama tip:** Set `"think": false` in API calls — GLM-4.7 defaults to thinking mode, adding 350+ tokens and 20s delay for no benefit
- **Background running:** `PYTHONUNBUFFERED=1` required for log visibility in background processes

---

## Brain Graph

Generated by `brainlayer brain-export`, produces a JSON file with:
- **Nodes:** One per session, with label, project, branch, plan, chunk count
- **Edges:** Connections between related sessions (shared files, topics, plans)
- **Clusters:** HDBSCAN clustering by topic similarity

Used by the BrainLayer Dashboard 3D visualization (`react-force-graph-3d`). Can be uploaded to Supabase Storage for multi-tenant access.

---

## Obsidian Export

`brainlayer export-obsidian` generates a Markdown vault:
- One note per session with frontmatter (project, date, plan)
- Backlinks between related sessions
- Tag-based navigation
- Compatible with Obsidian graph view

---

## Data Locations

| Path | Purpose |
|------|---------|
| `~/.claude/projects/` | Source conversations (read-only) |
| `~/.local/share/brainlayer/brainlayer.db` | sqlite-vec database (~1.4GB, 260K+ chunks) |
| `~/.local/share/brainlayer/prompts/` | Deduplicated system prompts (SHA-256) |
| `/tmp/brainlayer.sock` | Daemon Unix socket |
| `/tmp/brainlayer-enrichment.lock` | Enrichment process lock file |

---

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Lint + format
ruff check src/ && ruff format src/
```

---

## Pipeline Stages

### Stage 1: Extract (`pipeline/extract.py`)
- Parse JSONL conversation files
- **Content-addressable storage** for system prompts (SHA-256 hash → dedupe)
- Detect conversation continuations (session ID + temporal proximity)
- Also: `extract_whatsapp.py`, `extract_markdown.py`, `extract_claude_desktop.py`

### Stage 2: Classify (`pipeline/classify.py`)

| Type | Value | Action |
|------|-------|--------|
| `ai_code` | HIGH | Preserve verbatim |
| `stack_trace` | HIGH | Preserve exact (never split) |
| `user_message` | HIGH | Preserve |
| `assistant_text` | MEDIUM | Preserve |
| `file_read` | MEDIUM | Context-dependent |
| `git_diff` | MEDIUM | Extract changed entities |
| `build_log` | LOW | Summarize or mask |
| `dir_listing` | LOW | Structure only |
| `noise` | SKIP | Filter out (progress, queue-operation) |

### Stage 3: Chunk (`pipeline/chunk.py`)
- **AST-aware chunking** with tree-sitter for code (~500 tokens)
- **Never split** stack traces
- **Observation masking** for large tool outputs (`[N lines elided]`)
- Turn-based chunking for conversation with 10-20% overlap

### Stage 4: Embed (`embeddings.py`)
- **bge-large-en-v1.5** via sentence-transformers (local, private)
- 1024 dimensions, 63.5 MTEB score
- ~8s model load (vs 30s with Ollama)
- MPS acceleration on Apple Silicon

### Stage 5: Index (`vector_store.py`)
- **sqlite-vec** with APSW (macOS compatible)
- WAL mode for concurrent reads
- `PRAGMA busy_timeout = 5000` for multi-process safety
- Metadata: project, content_type, source_file, char_count

---

## Communication Style Analysis

BrainLayer includes **communication pattern analysis** from WhatsApp, Claude, YouTube, and Gemini chats.

### Latest Analysis Location
```
data/archives/style-2026-01-31-2121/
├── master-style-guide.md      # Main style rules
├── per-period/                # Style evolution over time
```

### Usage
```bash
brainlayer analyze-evolution --use-embeddings -c ~/claude-export.json -o data/archives/style-$(date +%Y-%m-%d-%H%M) -y
brainlayer analyze-style                     # Quick WhatsApp-only
brainlayer analyze-semantic                  # Semantic style profiling
```

---

## Naming

**BrainLayer** (זיכרון) - Hebrew for "memory"
