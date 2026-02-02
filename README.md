# Zikaron (זיכרון)

> **Memory** for Claude Code — Index, search, and retrieve knowledge from past AI coding sessions.

Zikaron is a local knowledge pipeline that indexes your Claude Code conversations, making them searchable via semantic embeddings. All processing happens locally using Ollama, ensuring complete privacy.

## Features

- **Local, privacy-first** — All data stays on your machine, embeddings via Ollama
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
                          │   Parse JSONL    Categorize   AST-aware   Ollama    ChromaDB │
                          │   Dedupe prompts  by type     splitting   vectors   storage  │
                          └──────────────────────────────────────────────────────────────┘
                                                                             │
                                                                             ▼
                          ┌──────────────────────────────────────────────────────────────┐
                          │                        STORAGE                               │
                          │                                                              │
                          │  ~/.local/share/zikaron/chromadb/   ← Vector database        │
                          │  ~/.local/share/zikaron/prompts/    ← Deduplicated prompts   │
                          │                                                              │
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

### Stage 4: Embed (`pipeline/embed.py`)

Generates vector embeddings using Ollama.

| Setting | Value |
|---------|-------|
| **Model** | `nomic-embed-text` (768 dimensions) |
| **Context** | 8192 tokens (conservative 2000 char limit) |
| **Index prefix** | `search_document: ` |
| **Query prefix** | `search_query: ` |

**Important**: nomic-embed-text requires task-specific prefixes for optimal results. The pipeline handles this automatically.

### Stage 5: Index (`pipeline/index.py`)

Stores embeddings in ChromaDB with metadata.

```python
# Storage location
~/.local/share/zikaron/chromadb/

# Collection settings
metadata={"hnsw:space": "cosine"}  # Cosine similarity

# Metadata per chunk
{
    "source_file": "/path/to/conversation.jsonl",
    "project": "my-project",
    "content_type": "ai_code",
    "value": "high",
    "char_count": 1847,
    "language": "python"  # For code chunks
}
```

---

## ChromaDB Storage Structure

Zikaron uses ChromaDB with persistent SQLite storage:

```
~/.local/share/zikaron/chromadb/
├── chroma.sqlite3          # Main database (metadata, chunk mappings)
└── <collection-id>/
    └── data/
        └── *.bin           # HNSW index files (vector storage)
```

### Collection Schema

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | `{source_file}:{chunk_index}` |
| `embedding` | float[768] | nomic-embed-text vector |
| `document` | string | Chunk text content |
| `metadata` | object | project, content_type, value, char_count |

### Querying the Database Directly

```python
from zikaron.pipeline.index import get_client, get_or_create_collection

client = get_client()
collection = get_or_create_collection(client)

# Get stats
print(f"Total chunks: {collection.count()}")

# Query by metadata
results = collection.get(
    where={"project": "my-project"},
    include=["documents", "metadatas"],
    limit=10
)
```

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **Ollama** — [ollama.com](https://ollama.com/)

### Installation

```bash
cd ~/Gits/golems/packages/zikaron
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Pull the embedding model
ollama pull nomic-embed-text
```

### Index Your Conversations

```bash
# Index all Claude Code conversations
zikaron index

# Index specific project only
zikaron index --project domica

# Index markdown files (learnings, skills, CLAUDE.md)
zikaron index-md ~/.claude/ --pattern "**/*.md"
```

### Search

```bash
# Semantic search
zikaron search "how did I implement authentication"

# Text-based exact match
zikaron search "config.py" --text

# Filter by project
zikaron search "React hooks" --project union --num 10

# Filter by content type
zikaron search "deployment error" --type stack_trace
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
│ domica              │ 12,456  │
│ union               │ 8,234   │
│ claude-golem        │ 6,123   │
└─────────────────────┴─────────┘
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `zikaron index [PATH]` | Index JSONL conversations (default: ~/.claude/projects/) |
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
  "project": "domica",           // optional
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

## File Watcher (Auto-Index)

For always-on indexing, use the watcher script with launchd.

### Manual Run

```bash
python scripts/watcher.py
```

Output:
```
[Watcher] Watching: ~/.claude/projects
[Watcher] Debounce: 30s
[Watcher] Press Ctrl+C to stop

[Watcher] Queued: conversation-abc123.jsonl
[Watcher] Indexing 1 conversation(s)...
[Watcher] ✓ Indexed successfully
```

### LaunchAgent Setup

Create `~/Library/LaunchAgents/com.zikaron.watcher.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.zikaron.watcher</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOUR_USERNAME/Gits/golems/packages/zikaron/.venv/bin/python</string>
        <string>/Users/YOUR_USERNAME/Gits/golems/packages/zikaron/scripts/watcher.py</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/zikaron-watcher.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/zikaron-watcher.error.log</string>
</dict>
</plist>
```

Load:
```bash
launchctl load ~/Library/LaunchAgents/com.zikaron.watcher.plist
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
| `~/.local/share/zikaron/chromadb/` | Vector database storage |
| `~/.local/share/zikaron/prompts/` | Deduplicated system prompts |
| `~/.config/zikaron/chat-tags.yaml` | Relationship tags for style analysis |

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
