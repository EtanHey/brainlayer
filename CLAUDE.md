# Zikaron (זיכרון) - Local Knowledge Pipeline

> **Memory** for Claude Code conversations. Index, search, and retrieve knowledge from past coding sessions.

---

## Quick Start

```bash
# Install dependencies
cd ~/Gits/zikaron
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Pull embedding model
ollama pull nomic-embed-text

# Index conversations
zikaron index

# Search
zikaron search "how did I implement authentication"
```

---

## Architecture

```
~/.claude/projects/          # Source: Claude Code conversations (JSONL)
        ↓
┌─────────────────────────────────────────────────────────────┐
│  PIPELINE                                                    │
│  ┌─────────┐  ┌──────────┐  ┌───────┐  ┌───────┐  ┌───────┐│
│  │ Extract │→ │ Classify │→ │ Chunk │→ │ Embed │→ │ Index ││
│  └─────────┘  └──────────┘  └───────┘  └───────┘  └───────┘│
└─────────────────────────────────────────────────────────────┘
        ↓
~/.local/share/zikaron/chromadb/   # Storage: Vector DB
        ↓
┌─────────────────────────────────────────────────────────────┐
│  INTERFACES                                                  │
│  ┌─────────────┐              ┌─────────────────┐           │
│  │ CLI         │              │ MCP Server      │           │
│  │ zikaron     │              │ zikaron-mcp     │           │
│  └─────────────┘              └─────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages (Research-Based)

### Stage 1: Extract (`pipeline/extract.py`)
- Parse JSONL conversation files
- **Content-addressable storage** for system prompts (SHA-256 hash → dedupe)
- Detect conversation continuations (session ID + temporal proximity)

### Stage 2: Classify (`pipeline/classify.py`)
Content types with preservation rules:

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

### Stage 4: Embed (`pipeline/embed.py`)
- **nomic-embed-text** via Ollama (local, private)
- CRITICAL: Use prefixes for optimal results:
  - `search_document: ` for indexing
  - `search_query: ` for querying

### Stage 5: Index (`pipeline/index.py`)
- **ChromaDB** with persistent storage
- Cosine similarity for vector search
- Metadata: project, content_type, source_file, char_count

---

## Key Research Findings Implemented

1. **Observation Masking > LLM Summarization**
   - Simply replacing old tool outputs with `[N lines elided]` often performs as well as complex summarization
   - 83.9% of context is observation tokens - masking saves massive space

2. **AST Chunking > Arbitrary Splitting**
   - tree-sitter for semantic boundaries (functions, classes)
   - +4.3 points Recall@5 vs naive chunking

3. **Hybrid Retrieval**
   - BM25 + semantic search together
   - ChromaDB supports both

4. **Content-Addressable System Prompts**
   - First user message often 5000+ tokens of system prompt
   - Hash and deduplicate across all conversations

---

## File Structure

```
zikaron/
├── src/zikaron/
│   ├── __init__.py
│   ├── pipeline/           # Processing stages
│   │   ├── extract.py      # Stage 1: Parse JSONL, extract prompts
│   │   ├── classify.py     # Stage 2: Content classification
│   │   ├── chunk.py        # Stage 3: AST-aware chunking
│   │   ├── embed.py        # Stage 4: Ollama embeddings
│   │   └── index.py        # Stage 5: ChromaDB storage
│   ├── cli/                # CLI interface (typer)
│   │   └── __init__.py
│   └── mcp/                # MCP server for Claude
│       └── __init__.py
├── tests/
├── pyproject.toml
├── CLAUDE.md               # This file
└── prd-json/               # Ralph PRD (if using Ralph)
```

---

## CLI Commands

```bash
# Index all conversations
zikaron index

# Index specific project
zikaron index --project domica

# Search
zikaron search "authentication middleware"
zikaron search "React hooks" --project union --num 10

# Stats
zikaron stats

# Clear database
zikaron clear --yes

# Start MCP server
zikaron serve
```

---

## MCP Integration

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

### Available Tools (for Claude/Ralph)

- **`zikaron_search`**: Search past conversations
  - `query`: Natural language search
  - `project`: Filter by project name
  - `content_type`: Filter by type (ai_code, stack_trace, etc.)
  - `num_results`: Number of results (default 5)

- **`zikaron_stats`**: Knowledge base statistics

- **`zikaron_list_projects`**: List indexed projects

---

## Data Locations

| Path | Purpose |
|------|---------|
| `~/.claude/projects/` | Source conversations (read-only) |
| `~/.local/share/zikaron/chromadb/` | Vector database |
| `~/.local/share/zikaron/prompts/` | Deduplicated system prompts |

---

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/

# Format
ruff format src/
```

---

## Future Enhancements

- [ ] Watch mode for real-time indexing
- [ ] Project-specific knowledge separation
- [ ] Learning extraction (patterns, preferences)
- [ ] Integration with Obsidian for knowledge export
- [ ] Conversation summarization for long-term memory

---

## Research Sources

- Meta-RAG (JP Morgan, 2025) - Hierarchical summarization
- cAST (Carnegie Mellon, 2025) - AST-based chunking
- Complexity Trap (2025) - Observation masking
- DH-RAG (2025) - Conversation threading
- See `docs.local/research/` for full papers

---

## Naming

**Zikaron** (זיכרון) - Hebrew for "memory"

> "The golem was given life through the word *emet* (truth). Zikaron gives your AI assistant memory through indexed conversations."
