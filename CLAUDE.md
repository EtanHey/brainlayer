# Zikaron (זיכרון) - Local Knowledge Pipeline

> **Memory** for Claude Code conversations. Index, search, retrieve, and visualize knowledge from past coding sessions.

---

## Quick Start

```bash
cd ~/Gits/golems/packages/zikaron
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Index conversations
zikaron index

# Start daemon (for dashboard + fast searches)
zikaron serve --http 8787

# Search
zikaron search "how did I implement authentication"

# Enrich with local LLM
zikaron enrich
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
~/.local/share/zikaron/zikaron.db   # Storage: sqlite-vec (~1.4GB, 226K+ chunks)
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
│  │       │  │ Daemon       │  │ zikaron-  │  │ (Next.js) │ │
│  │       │  │ :8787/socket │  │ mcp       │  │ :3000     │ │
│  └───────┘  └──────────────┘  └───────────┘  └───────────┘ │
└─────────────────────────────────────────────────────────────┘
```

> **Storage:** sqlite-vec with bge-large-en-v1.5 embeddings (1024 dims). WAL mode + busy_timeout=5000ms for concurrent access.

---

## File Structure

```
zikaron/
├── src/zikaron/
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
zikaron index                           # Index all conversations
zikaron index --project golems          # Index specific project
zikaron index-fast                      # Fast incremental index

zikaron search "authentication"         # Semantic search
zikaron search "config.py" --text       # Exact text match

zikaron stats                           # Knowledge base statistics
zikaron clear --yes                     # Clear database
```

### Daemon & Server

```bash
zikaron serve                           # Start daemon (Unix socket)
zikaron serve --http 8787               # Start daemon (HTTP mode for dashboard)
```

### Enrichment

```bash
zikaron enrich                          # Run LLM enrichment (GLM-4.7-Flash via Ollama)
zikaron enrich --batch-size 50          # Custom batch size
```

Enrichment adds to each chunk: summary, tags, importance score (1-10), intent classification. Uses local GLM-4.7-Flash with `"think": false` for speed (~1s/chunk for short, ~13s for long).

### Analysis & Export

```bash
zikaron git-overlay                     # Enrich with git diff context
zikaron group-operations                # Detect read→edit→test cycles
zikaron topic-chains                    # Find topic continuity across sessions
zikaron plan-linking                    # Link sessions to plans/phases
zikaron brain-export                    # Generate brain graph JSON
zikaron export-obsidian                 # Export to Obsidian vault
```

### Style Analysis

```bash
zikaron analyze-style                   # Quick WhatsApp style analysis
zikaron analyze-evolution --use-embeddings -c ~/export.json -o data/archives/style-$(date +%Y-%m-%d)
zikaron analyze-semantic                # Semantic style profiling
zikaron list-chats                      # List indexed chat sources
```

### Utilities

```bash
zikaron context <chunk_id>              # Get surrounding context
zikaron review <session_id>             # Review a session
zikaron fix-projects                    # Normalize project names
zikaron migrate                         # Run DB migrations
zikaron dashboard                       # Interactive TUI dashboard
```

---

## Daemon HTTP Endpoints

The daemon (`zikaron serve --http 8787`) exposes a FastAPI server used by the Next.js dashboard:

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
    "zikaron": {
      "command": "zikaron-mcp",
      "args": []
    }
  }
}
```

### Available Tools

| Tool | Required Params | Optional Params | Description |
|------|----------------|-----------------|-------------|
| `zikaron_search` | `query` | `project`, `content_type`, `num_results`, `source`, `tag`, `intent`, `importance_min` | Search past conversations |
| `zikaron_stats` | — | — | Knowledge base statistics |
| `zikaron_list_projects` | — | — | List indexed projects |
| `zikaron_context` | `chunk_id` | `before` (3), `after` (3) | Surrounding context for a result |
| `zikaron_file_timeline` | `file_path` | `project`, `limit` (50) | File interaction history across sessions |
| `zikaron_operations` | `session_id` | — | Logical operation groups (read→edit→test) |
| `zikaron_regression` | `file_path` | `project` | Regression analysis (what changed since last success) |
| `zikaron_plan_links` | — | `plan_name`, `session_id`, `project` | Session ↔ plan linkage |

### Search Parameters

- **`source`**: `claude_code` (default), `whatsapp`, `youtube`, `all`
- **`content_type`**: `ai_code`, `stack_trace`, `user_message`, `assistant_text`, `file_read`, `git_diff`
- **`intent`**: `debugging`, `designing`, `configuring`, `discussing`, `deciding`, `implementing`, `reviewing`
- **`importance_min`**: 1-10 (from enrichment)
- **`tag`**: enrichment-generated tags (e.g., `bug-fix`, `authentication`, `typescript`)

---

## Enrichment Pipeline

Local LLM enrichment adds metadata to indexed chunks:

| Field | Description | Source |
|-------|-------------|--------|
| `summary` | 1-2 sentence summary | GLM-4.7-Flash |
| `tags` | Comma-separated topic tags | GLM-4.7-Flash |
| `importance` | 1-10 relevance score | GLM-4.7-Flash |
| `intent` | Activity classification | GLM-4.7-Flash |

**Requirements:** Ollama running with `glm4:latest` model. Set `"think": false` in API calls — GLM-4.7-Flash defaults to thinking mode which adds 350+ reasoning tokens and takes 20s for trivial prompts.

**Concurrency:** Uses `PRAGMA busy_timeout = 5000` and 3-attempt retry logic with backoff to handle concurrent DB access from daemon + MCP + enrichment.

**Background running:** `PYTHONUNBUFFERED=1` required for log visibility in background processes.

---

## Brain Graph

Generated by `zikaron brain-export`, produces a JSON file with:
- **Nodes:** One per session, with label, project, branch, plan, chunk count
- **Edges:** Connections between related sessions (shared files, topics, plans)
- **Clusters:** HDBSCAN clustering by topic similarity

Used by the Golems Dashboard 3D visualization (`react-force-graph-3d`). Can be uploaded to Supabase Storage for multi-tenant access.

---

## Obsidian Export

`zikaron export-obsidian` generates a Markdown vault:
- One note per session with frontmatter (project, date, plan)
- Backlinks between related sessions
- Tag-based navigation
- Compatible with Obsidian graph view

---

## Data Locations

| Path | Purpose |
|------|---------|
| `~/.claude/projects/` | Source conversations (read-only) |
| `~/.local/share/zikaron/zikaron.db` | sqlite-vec database (~1.4GB, 226K+ chunks) |
| `~/.local/share/zikaron/prompts/` | Deduplicated system prompts (SHA-256) |
| `/tmp/zikaron.sock` | Daemon Unix socket |
| `/tmp/zikaron-enrichment.lock` | Enrichment process lock file |

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

Zikaron includes **communication pattern analysis** from WhatsApp, Claude, YouTube, and Gemini chats.

### Latest Analysis Location
```
data/archives/style-2026-01-31-2121/
├── master-style-guide.md      # Main style rules
├── per-period/                # Style evolution over time
```

### Usage
```bash
zikaron analyze-evolution --use-embeddings -c ~/claude-export.json -o data/archives/style-$(date +%Y-%m-%d-%H%M) -y
zikaron analyze-style                     # Quick WhatsApp-only
zikaron analyze-semantic                  # Semantic style profiling
```

---

## Naming

**Zikaron** (זיכרון) - Hebrew for "memory"
