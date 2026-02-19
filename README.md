# BrainLayer

> Like git for your AI conversations — the brain layer for AI development.

BrainLayer is an open-source knowledge pipeline that indexes your AI conversations (Claude Code, ChatGPT, WhatsApp, YouTube transcripts) into a searchable, enriched knowledge base. It turns months of scattered conversations into structured, queryable memory with semantic search, LLM enrichment, and brain graph visualization.

## Quick Start

```bash
pip install brainlayer
brainlayer init          # Interactive setup wizard
brainlayer index         # Index your Claude Code conversations
brainlayer search "how did I implement auth?"
```

## How It Works

```
Conversations  →  Extract  →  Chunk  →  Embed  →  Index  →  Search
(Claude Code,     (parse      (split    (bge-     (sqlite-   (semantic
 WhatsApp,        sessions    into      large-    vec +      + keyword
 YouTube)         + classify) segments) en-v1.5)  FTS5)      hybrid)
                                          ↓
                                    Enrich (10 fields via local LLM)
                                          ↓
                                    Brain Graph (cluster + visualize)
```

## Features

| Feature | Description |
|---------|-------------|
| **Semantic Search** | Vector similarity + FTS5 keyword hybrid search across all conversations |
| **10-Field Enrichment** | Local LLM (Ollama/MLX) adds summary, topics, importance, intent, symbols |
| **Brain Graph** | HDBSCAN clustering + UMAP 3D visualization of your knowledge |
| **MCP Server** | 8 tools for Claude Code, Zed, Cursor — search from your editor |
| **Style Analysis** | Communication pattern analysis across topics and languages |
| **Multi-Source** | Claude Code, WhatsApp, YouTube transcripts, Markdown, Claude Desktop |
| **Centralized Storage** | Project artifacts in `~/.local/share/brainlayer/storage/` |
| **Setup Wizard** | `brainlayer init` detects your environment and configures everything |

## MCP Integration

Add to your editor's MCP config:

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

### MCP Tools

| Tool | Description |
|------|-------------|
| `brainlayer_search` | Search past conversations with filters (project, type, source, tag, intent) |
| `brainlayer_context` | Get surrounding chunks for a search result |
| `brainlayer_stats` | Knowledge base statistics |
| `brainlayer_list_projects` | List all indexed projects |
| `brainlayer_file_timeline` | File interaction history across sessions |
| `brainlayer_operations` | Logical operation groups (read/edit/test cycles) |
| `brainlayer_regression` | What changed since a file last worked |
| `brainlayer_plan_links` | Session to plan/phase linkage |

## The Vision

**Code gets version control. Conversations don't.**

Every developer generates thousands of AI conversation turns — debugging sessions, architecture decisions, code reviews. This knowledge is scattered across chat histories, lost when contexts expire.

BrainLayer is the conversation-side complement to code version control. Where tools like Git track code changes, BrainLayer tracks the *reasoning* behind those changes.

## Optional Extras

Install with extras for additional capabilities:

```bash
pip install "brainlayer[brain]"     # Brain graph visualization (HDBSCAN + UMAP)
pip install "brainlayer[cloud]"     # Cloud backfill (Vertex AI batch enrichment)
pip install "brainlayer[ast]"       # AST-aware code analysis (tree-sitter)
pip install "brainlayer[style]"     # Communication style analysis
```

## Architecture

For a deep dive into the pipeline, embedding strategy, enrichment schema, and daemon architecture, see [docs/architecture.md](docs/architecture.md).

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

## Contributing

Contributions welcome! Please read the [architecture docs](docs/architecture.md) first.

1. Fork the repo
2. Create a feature branch
3. Write tests (TDD encouraged)
4. Submit a PR

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Origin

BrainLayer was originally developed as "Zikaron" (זיכרון, Hebrew for "memory") inside a personal AI agent ecosystem. It was extracted into a standalone project because every developer deserves persistent AI memory, not just the ones building their own agent systems.
