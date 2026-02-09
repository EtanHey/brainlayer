# Zikaron (זיכרון) - Local Knowledge Pipeline

> **Memory** for Claude Code conversations. Index, search, and retrieve knowledge from past coding sessions.

---

## Quick Start

```bash
# Install dependencies
cd ~/Gits/golems/packages/zikaron
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Fast search (<2s)
zikaron search-fast "how did I implement authentication"

# Interactive dashboard
zikaron dashboard
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
~/.local/share/zikaron/zikaron.db   # Storage: sqlite-vec (~1GB)
        ↓
┌─────────────────────────────────────────────────────────────┐
│  INTERFACES                                                  │
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ CLI         │  │ FastAPI Daemon  │  │ MCP Server      │ │
│  │ search      │  │ /tmp/zikaron.sock│  │ zikaron-mcp    │ │
│  │ dashboard   │  │ (<2s queries)   │  │                 │ │
│  └─────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

> **Storage:** sqlite-vec with bge-large-en-v1.5 embeddings (1024 dims).

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

### Stage 4: Embed (`embeddings.py`)
- **bge-large-en-v1.5** via sentence-transformers (local, private)
- 1024 dimensions, 63.5 MTEB score
- ~8s model load (vs 30s with Ollama)
- MPS acceleration on Apple Silicon

### Stage 5: Index (`vector_store.py`)
- **sqlite-vec** with APSW (macOS compatible)
- Fast vector similarity search
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
   - sqlite-vec supports both (FTS5 for BM25, vector for semantic)

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
│   │   └── semantic_style.py  # Style analysis
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
# Search (<2s with daemon)
zikaron search "authentication middleware"
zikaron search "config.py" --text  # Exact match

# Index conversations
zikaron index
zikaron index --project domica

# Stats
zikaron stats

# Interactive dashboard
zikaron dashboard

# Clear database
zikaron clear --yes

# Reindex all conversations
zikaron index
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
| `~/.local/share/zikaron/zikaron.db` | sqlite-vec vector database (~1GB) |
| `~/.local/share/zikaron/prompts/` | Deduplicated system prompts |
| `/tmp/zikaron.sock` | Daemon Unix socket |

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

## Communication Style Analysis

Zikaron includes **communication pattern analysis** that extracts your writing style from WhatsApp, Claude, YouTube, and Gemini chats.

### Latest Analysis Location
```
data/archives/style-2026-01-31-2121/
├── master-style-guide.md      # ← Main style rules
├── per-period/                # Style evolution over time
│   ├── 2026-H1-english-style.md
│   ├── 2026-H1-hebrew-style.md
│   └── ...
```

### Key Style Traits (Your Profile)
- **Formality: 2/10** - Extremely casual
- **Code-switching:** Hebrew ↔ English frequently
- **Laughter:** "חחח" / "חח" for humor
- **Emojis:** 🫶 sparingly but meaningfully
- **Length:** Brief, direct messages
- **Tone:** Friendly, sometimes playful sarcasm

### Usage
```bash
# Run full analysis with embeddings
zikaron analyze-evolution --use-embeddings -c ~/claude-export.json -o data/archives/style-$(date +%Y-%m-%d-%H%M) -y

# Quick WhatsApp-only analysis
zikaron analyze-style
```

### Integration with GolemsZikaron
The style analysis is used by GolemsZikaron bot to match your communication patterns when posting to Soltome. See `~/Gits/golems/packages/autonomous/SOUL.md` for the bot persona.

---

## Future Enhancements

- [ ] Watch mode for real-time indexing
- [ ] Project-specific knowledge separation
- [x] Learning extraction (communication patterns) ✅
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
