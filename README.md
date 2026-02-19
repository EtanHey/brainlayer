# BrainLayer

> Persistent memory for AI agents. Search, think, recall — across every conversation you've ever had.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-12%20tools-green.svg)](https://modelcontextprotocol.io)
[![Tests](https://img.shields.io/badge/tests-67%20passing-brightgreen.svg)](#)
[![Chunks](https://img.shields.io/badge/indexed-268K%2B%20chunks-purple.svg)](#)

---

**Your AI agent forgets everything between sessions.** Every architecture decision, every debugging session, every preference you've expressed — gone. You repeat yourself constantly.

BrainLayer fixes this. It's a **persistent memory layer** that gives any MCP-compatible AI agent (Claude, ChatGPT, Cursor, Zed, VS Code) the ability to remember, think, and recall across conversations.

```
"What approach did I use for auth last month?"     →  brainlayer_think
"Show me everything about this file's history"     →  brainlayer_recall
"What was I working on yesterday?"                 →  brainlayer_current_context
```

## How It Works

```
┌─────────────────────────────────────────────────────┐
│  AI Agent (Claude, Cursor, Zed, ChatGPT...)         │
│  "What decisions did I make about the auth system?" │
└────────────────────┬────────────────────────────────┘
                     │ MCP
                     ▼
┌─────────────────────────────────────────────────────┐
│  Think / Recall Engine                              │
│  Categorizes by intent: decisions, bugs, patterns   │
│  Groups by project, filters by importance           │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Knowledge Base (268K+ chunks)                      │
│  Hybrid search: semantic (bge-large) + keyword      │
│  10-field LLM enrichment per chunk                  │
│  SQLite + sqlite-vec — single file, zero infra      │
└─────────────────────────────────────────────────────┘
```

## Quick Start

```bash
pip install brainlayer
brainlayer init              # Interactive setup wizard
brainlayer index             # Index your Claude Code conversations
brainlayer search "how did I implement auth?"
```

### MCP Setup

Add to your editor's MCP config and your agent instantly has persistent memory:

**Claude Code** (`~/.claude.json`):
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

**Zed** (`settings.json`):
```json
{
  "context_servers": {
    "brainlayer": {
      "command": { "path": "brainlayer-mcp" }
    }
  }
}
```

**Cursor** (MCP settings):
```json
{
  "mcpServers": {
    "brainlayer": {
      "command": "brainlayer-mcp"
    }
  }
}
```

## Features

### Intelligence Layer

| Tool | What It Does |
|------|-------------|
| **`brainlayer_think`** | Given your current task, retrieves relevant past decisions, patterns, and bugs. Groups results by intent. |
| **`brainlayer_recall`** | File-based or topic-based recall. "What happened with this file?" or "What do I know about deployment?" |
| **`brainlayer_current_context`** | Lightweight — what projects, branches, and files were you working on recently? No embedding needed. |
| **`brainlayer_sessions`** | Browse recent sessions by project and date range. |

### Core Search

| Tool | What It Does |
|------|-------------|
| **`brainlayer_search`** | Hybrid semantic + keyword search with filters (project, type, source, tag, intent, importance) |
| **`brainlayer_context`** | Get surrounding conversation chunks for a search result |
| **`brainlayer_file_timeline`** | Full interaction history of a file across all sessions |
| **`brainlayer_operations`** | Logical operation groups — read/edit/test cycles |
| **`brainlayer_regression`** | What changed since a file last worked? |
| **`brainlayer_plan_links`** | Connect sessions to implementation plans |
| **`brainlayer_stats`** | Knowledge base statistics |
| **`brainlayer_list_projects`** | List all indexed projects |

### Knowledge Pipeline

- **Multi-source indexing** — Claude Code, WhatsApp, YouTube transcripts, Markdown, Claude Desktop
- **Hybrid search** — semantic vectors (bge-large-en-v1.5, 1024 dims) + FTS5 keyword, fused with Reciprocal Rank Fusion
- **10-field LLM enrichment** — summary, topics, importance, intent, symbols, sentiment, and more via local LLM (Ollama/MLX)
- **Brain graph** — HDBSCAN clustering + UMAP 3D visualization of your knowledge
- **Single-file storage** — SQLite + sqlite-vec. No Postgres, no Neo4j, no Docker required

## Why BrainLayer?

Every AI memory tool makes you choose between capability, privacy, and simplicity. BrainLayer doesn't.

| | BrainLayer | Mem0 | Zep/Graphiti | Letta | LangChain Memory |
|---|:---:|:---:|:---:|:---:|:---:|
| **Persistent memory** | Yes | Yes | Yes | Yes | Partial |
| **MCP native** | 12 tools | 1 server | 1 server | No | No |
| **Think/Recall** | Yes | No | No | No | No |
| **Self-hostable** | Yes | Secondary | Deprecated | Docker+PG | DIY |
| **Local-first** | SQLite | Cloud-first | Cloud-only | 42 PG tables | Framework |
| **Zero infra** | `pip install` | API key | API key | Docker | Multiple deps |
| **Open source** | Apache 2.0 | Apache 2.0 | Source-available | Apache 2.0 | MIT |
| **Multi-source** | 6 sources | API only | API only | API only | API only |
| **Enrichment** | 10 fields | Basic | Temporal | Self-write | None |

**Mem0** (41K stars, $24M raised) — excellent managed memory, but cloud-first. Self-hosting is an afterthought. No intelligence layer (think/recall).

**Zep/Graphiti** (22K stars) — sophisticated temporal knowledge graph, but deprecated its Community Edition. Cloud-only now.

**Letta** (14K stars, $10M raised) — ambitious agent runtime with self-modifying memory, but requires PostgreSQL with 42 tables. It's a full runtime, not a pluggable layer.

**LangChain Memory** — broadest ecosystem, but memory is bolted on. Wiring LangMem + checkpointers + backing stores requires significant integration work.

BrainLayer is the only memory layer that:
1. **Thinks before answering** — categorizes past knowledge by intent (decisions, bugs, patterns) instead of raw search results
2. **Runs on a single file** — no database servers, no Docker, no cloud accounts
3. **Works with every MCP client** — 12 tools, instant integration, zero SDK

## CLI Reference

```bash
brainlayer index              # Index new conversations
brainlayer search "query"     # Semantic + keyword search
brainlayer enrich             # Run LLM enrichment on new chunks
brainlayer stats              # Database statistics
brainlayer brain-export       # Generate brain graph data
brainlayer dashboard          # Interactive TUI dashboard
brainlayer init               # Setup wizard
brainlayer store              # Centralized file storage
brainlayer projects           # List stored projects
```

## Optional Extras

```bash
pip install "brainlayer[brain]"     # Brain graph visualization (HDBSCAN + UMAP)
pip install "brainlayer[cloud]"     # Cloud backfill (Vertex AI batch enrichment)
pip install "brainlayer[ast]"       # AST-aware code analysis (tree-sitter)
pip install "brainlayer[style]"     # Communication style analysis
```

## Architecture

BrainLayer is a Python package with no external service dependencies. Everything runs locally:

- **Storage**: SQLite + sqlite-vec (single `.db` file)
- **Embeddings**: `bge-large-en-v1.5` via sentence-transformers (1024 dimensions)
- **Search**: Hybrid vector similarity + FTS5 keyword, merged with RRF
- **Enrichment**: Local LLM via Ollama or MLX (GLM-4, Llama, etc.)
- **MCP**: stdio-based server, compatible with any MCP client
- **Clustering**: HDBSCAN + UMAP for brain graph (optional)

For a deep dive, see [docs/architecture.md](docs/architecture.md).

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Origin

BrainLayer was originally developed as "Zikaron" (Hebrew for "memory") inside a personal AI agent ecosystem. It was extracted into a standalone project because **every developer deserves persistent AI memory** — not just the ones building their own agent systems.
