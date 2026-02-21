# Quick Start

## Installation

```bash
pip install brainlayer
```

### Optional extras

```bash
pip install "brainlayer[brain]"     # Brain graph visualization (HDBSCAN + UMAP)
pip install "brainlayer[cloud]"     # Cloud backfill (Gemini Batch API)
pip install "brainlayer[youtube]"   # YouTube transcript indexing
pip install "brainlayer[ast]"       # AST-aware code chunking (tree-sitter)
```

## Setup

Run the interactive wizard:

```bash
brainlayer init
```

This will:

1. Check for Claude Code conversations in `~/.claude/projects/`
2. Detect your hardware (Apple Silicon → MLX, otherwise Ollama)
3. Configure your LLM backend for enrichment
4. Create the database at `~/.local/share/brainlayer/brainlayer.db`

## Index Your Conversations

```bash
brainlayer index
```

This parses your Claude Code conversations (JSONL files), classifies content, chunks it with sentence boundaries, generates embeddings (bge-large-en-v1.5), and stores everything in the SQLite database.

## Connect to Your Editor

### Claude Code

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "brainlayer": {
      "command": "brainlayer-mcp"
    }
  }
}
```

### Cursor

Add in Cursor's MCP settings:

```json
{
  "mcpServers": {
    "brainlayer": {
      "command": "brainlayer-mcp"
    }
  }
}
```

### Zed

Add to `settings.json`:

```json
{
  "context_servers": {
    "brainlayer": {
      "command": { "path": "brainlayer-mcp" }
    }
  }
}
```

### VS Code

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "brainlayer": {
      "command": "brainlayer-mcp"
    }
  }
}
```

## Enrich (Optional)

Add structured metadata to your indexed content using a local LLM:

```bash
brainlayer enrich
```

This adds summary, tags, importance scores, intent classification, and more to each chunk. See [Enrichment](enrichment.md) for details.

## Verify

```bash
brainlayer stats              # Check your knowledge base
brainlayer search "auth"      # Test a search
```

## CLI Reference

```bash
brainlayer init               # Interactive setup wizard
brainlayer index              # Index new conversations
brainlayer search "query"     # Semantic + keyword search
brainlayer enrich             # Run LLM enrichment on new chunks
brainlayer enrich-sessions    # Session-level analysis
brainlayer stats              # Database statistics
brainlayer brain-export       # Generate brain graph JSON
brainlayer export-obsidian    # Export to Obsidian vault
brainlayer dashboard          # Interactive TUI dashboard
```
