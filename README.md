# Zikaron (זיכרון)

> **Memory** for Claude Code — Index, search, and retrieve knowledge from past AI coding sessions.

Zikaron is a local knowledge pipeline that indexes your Claude Code conversations, making them searchable via semantic embeddings. All processing happens locally using sentence-transformers and sqlite-vec, ensuring complete privacy.

## Features

- **Local, privacy-first** — All data stays on your machine, embeddings via sentence-transformers
- **Semantic search** — Find past solutions by meaning, not just keywords
- **AST-aware chunking** — Code is split at function/class boundaries using tree-sitter
- **Smart classification** — Content is categorized (code, errors, explanations) for better retrieval
- **MCP integration** — Use directly in Claude Code via Model Context Protocol
- **Communication analysis** — Extract your writing style from WhatsApp, Claude, YouTube, Gemini

---

## Architecture

```
                          ┌──────────────────────────────────────────────────────────────┐
                          │                        PIPELINE                              │
~/.claude/projects/       │                                                              │
      (JSONL)             │  ┌─────────┐  ┌──────────┐  ┌───────┐  ┌───────┐  ┌───────┐ │
          │               │  │ Extract │→ │ Classify │→ │ Chunk │→ │ Embed │→ │ Index │ │
          └──────────────→│  └─────────┘  └──────────┘  └───────┘  └───────┘  └───────┘ │
                          │       │             │            │          │          │     │
                          │   Parse JSONL    Categorize   AST-aware   bge-large sqlite-vec│
                          │   Dedupe prompts  by type     splitting   1024 dims  storage  │
                          └──────────────────────────────────────────────────────────────┘
                                                                             │
                                                                             ▼
                          ┌──────────────────────────────────────────────────────────────┐
                          │                        STORAGE                               │
                          │                                                              │
                          │  ~/.local/share/zikaron/zikaron.db  ← Vector database        │
                          │  ~/.local/share/zikaron/prompts/    ← Deduplicated prompts   │
                          │  /tmp/zikaron.sock                  ← Daemon socket          │
                          └──────────────────────────────────────────────────────────────┘
                                                                             │
                          ┌──────────────────────────────────────────────────────────────┐
                          │                       INTERFACES                             │
                          │                                                              │
                          │  ┌─────────────────┐              ┌─────────────────────┐   │
                          │  │ CLI             │              │ MCP Server          │   │
                          │  │ $ zikaron       │              │ zikaron-mcp         │   │
                          │  │   search        │              │                     │   │
                          │  │   index         │              │ Tools:              │   │
                          │  │   stats         │              │   zikaron_search    │   │
                          │  │   ...           │              │   zikaron_stats     │   │
                          │  └─────────────────┘              │   zikaron_list_...  │   │
                          │                                   └─────────────────────┘   │
                          └──────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages

### Stage 1: Extract (`pipeline/extract.py`)

Parses JSONL conversation files from `~/.claude/projects/`.

| Feature | Description |
|---------|-------------|
| **JSONL Parsing** | Uses `orjson` for fast parsing, skips malformed lines |
| **System Prompt Deduplication** | SHA-256 hash for content-addressable storage — first user message (often >2000 chars with CLAUDE.md) is deduplicated |
| **Continuation Detection** | Detects related conversations via session ID and temporal proximity |

### Stage 2: Classify (`pipeline/classify.py`)

Categorizes content blocks by type and preservation value.

| Type | Value | Handling |
|------|-------|----------|
| `ai_code` | HIGH | Preserve verbatim — Claude's code blocks |
| `stack_trace` | HIGH | Preserve exact — never split error traces |
| `user_message` | HIGH | Preserve — human questions and context |
| `assistant_text` | MEDIUM | Preserve — Claude's explanations |
| `file_read` | MEDIUM | Context-dependent tool outputs |
| `git_diff` | MEDIUM | Extract changed entities |
| `build_log` | LOW | Summarize or mask |
| `dir_listing` | LOW | Structure only |
| `noise` | SKIP | Filter out (progress bars, queue operations) |

**Markdown Types** (for `index-md`):
- `learning`, `skill`, `project_config` — HIGH value
- `research`, `prd_archive`, `documentation` — MEDIUM value
- `verification` — LOW value

### Stage 3: Chunk (`pipeline/chunk.py`)

Splits content into embedding-sized pieces (~500 tokens ≈ 2000 chars).

| Strategy | Applied To | Description |
|----------|------------|-------------|
| **AST-aware** | Code blocks | Uses tree-sitter to split at function/class boundaries |
| **Never split** | Stack traces | Preserves exact error context |
| **Observation masking** | Large outputs | First 5 + last 3 lines, middle elided |
| **Paragraph-based** | Text | Split at `\n\n` with overlap |

```python
# Chunk size constants
TARGET_CHUNK_SIZE = 2000  # chars (~500 tokens)
MIN_CHUNK_SIZE = 200
MAX_CHUNK_SIZE = 4000
```

### Stage 4: Embed (`embeddings.py`)

Generates vector embeddings using sentence-transformers (direct, no Ollama needed).

| Setting | Value |
|---------|-------|
| **Model** | `bge-large-en-v1.5` (1024 dimensions, 63.5 MTEB score) |
| **Context** | 512 tokens (~2000 chars) |
| **Hardware** | MPS (Apple Silicon) or CPU |

**Note**: bge-large-en-v1.5 provides better search quality than nomic-embed-text, with ~8s model loading vs 30s via Ollama.

### Stage 5: Index (`vector_store.py`)

Stores embeddings in sqlite-vec with metadata.

```python
# Storage location
~/.local/share/zikaron/zikaron.db

# Schema
chunks: id, content, metadata, source_file, project, content_type, value_type, char_count
chunk_vectors: chunk_id, embedding FLOAT[1024]

# Metadata per chunk
{
    "source_file": "/path/to/conversation.jsonl",
    "project": "my-project",
    "content_type": "ai_code",
    "value_type": "high",
    "char_count": 1847,
    "language": "python"  # For code chunks
}
```

---

## sqlite-vec Storage Structure

Zikaron uses sqlite-vec for fast vector storage:

```
~/.local/share/zikaron/
├── zikaron.db              # Main database (chunks + vectors)
├── prompts/                # Deduplicated system prompts
└── chromadb.backup/        # Old ChromaDB (after migration)
```

### Schema

| Table | Fields | Description |
|-------|--------|-------------|
| `chunks` | id, content, metadata, source_file, project, content_type, value_type, char_count | Chunk metadata |
| `chunk_vectors` | chunk_id, embedding FLOAT[1024] | bge-large-en-v1.5 vectors |

### Querying the Database Directly

```python
from zikaron.vector_store import VectorStore
from zikaron.embeddings import embed_query
from pathlib import Path

db_path = Path.home() / ".local/share/zikaron/zikaron.db"
store = VectorStore(db_path)

# Get stats
print(f"Total chunks: {store.count()}")
print(store.get_stats())

# Vector search
query_embedding = embed_query("authentication")
results = store.search(query_embedding=query_embedding, n_results=10)

# Text search
results = store.search(query_text="config.py", n_results=10)
```

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **~4GB RAM** for embedding model

### Installation

```bash
cd ~/Gits/golems/packages/zikaron
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# First run will download bge-large-en-v1.5 model (~1.2GB)
```

### Migration (if upgrading from ChromaDB)

```bash
zikaron migrate  # One-time conversion
# ~4-6 hours for 200k chunks with bge-large-en-v1.5
# ~1-2 hours for 50k chunks
```

### Index Your Conversations

```bash
# Index all Claude Code conversations (fast backend)
zikaron index-fast

# Index specific project only
zikaron index-fast --project my-app

# Index markdown files (learnings, skills, CLAUDE.md)
zikaron index-md ~/.claude/ --pattern "**/*.md"

# Legacy commands still work (use original ChromaDB backend)
zikaron index

# Index a specific project
zikaron index-fast --project my-project
```

### Search

```bash
# Fast semantic search (<2s with daemon running)
zikaron search-fast "how did I implement authentication"

# Text-based exact match
zikaron search-fast "config.py" --text

# Filter by project
zikaron search-fast "React hooks" --project my-project --num 10

# Legacy search (slower, uses ChromaDB)
zikaron search "query"
```

### Dashboard

```bash
# Interactive TUI dashboard
zikaron dashboard
```

### View Stats

```bash
zikaron stats
```

Output:
```
זיכרון Knowledge Base

Total Chunks: 45,231
Source Files: 892
Projects: 12
Content Types: 8

┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Project             ┃ Chunks  ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ my-project          │ 12,456  │
│ another-project     │ 8,234   │
│ claude-golem        │ 6,123   │
└─────────────────────┴─────────┘
```

---

## CLI Reference

### Fast Commands (sqlite-vec backend, recommended)

| Command | Description |
|---------|-------------|
| `zikaron search-fast QUERY` | Fast semantic search (<2s) |
| `zikaron search-fast QUERY --text` | Fast text search |
| `zikaron stats-fast` | Instant statistics |
| `zikaron index-fast [PATH]` | Index with sqlite-vec + bge-large |
| `zikaron dashboard` | Interactive TUI dashboard |
| `zikaron migrate` | Migrate ChromaDB → sqlite-vec |

### Legacy Commands (ChromaDB backend)

| Command | Description |
|---------|-------------|
| `zikaron index [PATH]` | Index JSONL conversations |
| `zikaron index-md PATH` | Index markdown files |
| `zikaron search QUERY` | Search the knowledge base |
| `zikaron stats` | Show database statistics |
| `zikaron clear --yes` | Clear the entire database |
| `zikaron fix-projects` | Fix UUID-based project names |
| `zikaron serve` | Start MCP server (stdio) |

### Index Options

```bash
zikaron index [PATH] [OPTIONS]

Options:
  -p, --project TEXT   Only index specific project folder
  -f, --force          Re-index all files (ignore cache)
```

### Search Options

```bash
zikaron search QUERY [OPTIONS]

Options:
  -n, --num INTEGER    Number of results (1-100, default: 5)
  -p, --project TEXT   Filter by project name
  -t, --type TEXT      Filter by content type
  --text               Use text-based search instead of semantic
```

### Index-md Options

```bash
zikaron index-md PATH [OPTIONS]

Options:
  -p, --pattern TEXT   Glob patterns to match (default: **/*.md)
  -e, --exclude TEXT   Directory names to exclude
  -f, --force          Re-index all files
```

---

## MCP Server Integration

Zikaron provides an MCP server for direct integration with Claude Code.

### Setup

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

Or with explicit Python path:

```json
{
  "mcpServers": {
    "zikaron": {
      "command": "/path/to/zikaron/.venv/bin/python",
      "args": ["-m", "zikaron.mcp"]
    }
  }
}
```

### Available Tools

#### `zikaron_search`

Search through past conversations and learnings.

```json
{
  "query": "how did I implement authentication",
  "project": "my-app",           // optional
  "content_type": "ai_code",     // optional
  "num_results": 5               // optional, default 5
}
```

**Content types**: `ai_code`, `stack_trace`, `user_message`, `assistant_text`, `file_read`, `git_diff`

#### `zikaron_stats`

Get knowledge base statistics.

```json
{}
```

Returns: total chunks, projects list, content types.

#### `zikaron_list_projects`

List all indexed projects.

```json
{}
```

### Usage in Claude Code

Once configured, Claude can use Zikaron automatically:

> "Search my past conversations for how I implemented Redis caching"

Claude will call `zikaron_search` and use the results to inform its response.

---

## Daemon Service (Recommended)

For instant queries (<2s), run the FastAPI daemon that keeps models pre-loaded.

### Manual Run

```bash
zikaron-daemon
```

### Auto-Start Setup (launchd)

```bash
# Install the service
python scripts/install_service.py install

# Or manually create ~/Library/LaunchAgents/com.zikaron.daemon.plist
```

The daemon listens on `/tmp/zikaron.sock` and provides instant search via pre-loaded models.

### Daemon Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `POST /search` | Vector or text search |
| `GET /stats` | Database statistics |

## File Watcher (Legacy)

For always-on indexing with the old ChromaDB backend, use the watcher script.

```bash
python scripts/watcher.py
```

---

## Communication Style Analysis

Zikaron can analyze your communication patterns from multiple sources to generate personalized AI rules.

### Data Sources

| Source | Location | Data |
|--------|----------|------|
| WhatsApp | `~/Library/.../ChatStorage.sqlite` | Tone, length, emoji, phrases |
| Claude.ai | Export from Settings > Data | Response structures, clarifications |
| YouTube | `data/youtube-comments/comments.csv` | Comment style |
| Gemini | Google Takeout | Chat patterns |

### Quick Analysis

```bash
# Basic style analysis
zikaron analyze-style

# With Claude export
zikaron analyze-style --claude-export ~/Downloads/conversations.json

# Full longitudinal analysis with embeddings
zikaron analyze-evolution \
  --use-embeddings \
  -c ~/claude-export.json \
  -o data/archives/style-$(date +%Y-%m-%d-%H%M) \
  -y
```

### Output Files

```
data/archives/style-2026-01-31-2121/
├── master-style-guide.md      # Main style rules
├── claude-ai-casual-instructions.md
├── gemini-casual-instructions.md
└── per-period/
    ├── 2026-H1-english-style.md
    ├── 2026-H1-hebrew-style.md
    └── ...
```

### Generated Rules

Copy to your AI apps' personalization settings:
- **Cursor**: `~/.cursor/rules/communication-style.md`
- **Claude.ai**: Settings → Personalization
- **Gemini**: Settings → Personalization

---

## Data Locations

| Path | Purpose |
|------|---------|
| `~/.claude/projects/` | Source: Claude Code conversations (JSONL) |
| `~/.local/share/zikaron/zikaron.db` | sqlite-vec vector database |
| `~/.local/share/zikaron/chromadb/` | Legacy ChromaDB (before migration) |
| `~/.local/share/zikaron/chromadb.backup/` | ChromaDB backup (after migration) |
| `~/.local/share/zikaron/prompts/` | Deduplicated system prompts |
| `~/.config/zikaron/chat-tags.yaml` | Relationship tags for style analysis |
| `/tmp/zikaron.sock` | FastAPI daemon Unix socket |

---

## Development

### Setup

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/

# Format
ruff format src/
```

### Project Structure

```
zikaron/
├── src/zikaron/
│   ├── __init__.py
│   ├── pipeline/
│   │   ├── extract.py          # Stage 1: Parse JSONL
│   │   ├── classify.py         # Stage 2: Content classification
│   │   ├── chunk.py            # Stage 3: AST-aware chunking
│   │   ├── embed.py            # Stage 4: Ollama embeddings
│   │   ├── index.py            # Stage 5: ChromaDB storage
│   │   ├── analyze_communication.py
│   │   ├── extract_whatsapp.py
│   │   ├── extract_claude_desktop.py
│   │   └── ...
│   ├── cli/
│   │   └── __init__.py         # Typer CLI
│   └── mcp/
│       └── __init__.py         # MCP server
├── scripts/
│   └── watcher.py              # File watcher for auto-indexing
├── tests/
├── pyproject.toml
└── CLAUDE.md
```

---

## Research Background

Zikaron's design is informed by recent research on retrieval-augmented generation:

| Finding | Implementation |
|---------|----------------|
| **Observation Masking > LLM Summarization** | Large tool outputs are masked (`[N lines elided]`) rather than summarized — 83.9% of context is observation tokens |
| **AST Chunking > Arbitrary Splitting** | tree-sitter parses code at semantic boundaries — +4.3 points Recall@5 |
| **Hybrid Retrieval** | BM25 + semantic search via ChromaDB |
| **Content-Addressable Prompts** | System prompts (often 5000+ tokens) are hashed and deduplicated |

Sources:
- Meta-RAG (JP Morgan, 2025) — Hierarchical summarization
- cAST (Carnegie Mellon, 2025) — AST-based chunking
- Complexity Trap (2025) — Observation masking
- DH-RAG (2025) — Conversation threading

---

## Naming

**Zikaron** (זיכרון) — Hebrew for "memory"

> "The golem was given life through the word *emet* (truth). Zikaron gives your AI assistant memory through indexed conversations."

---

## License

MIT
