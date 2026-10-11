# Quick Start

## Installation

```bash
pip install brainlayer
```

### Optional extras

```bash
pip install "brainlayer[brain]"     # Brain graph visualization (Leiden + UMAP)
pip install "brainlayer[youtube]"   # YouTube transcript indexing
pip install "brainlayer[ast]"       # AST-aware code chunking (tree-sitter)
pip install "brainlayer[kg]"        # GliNER entity extraction (209M params, EN+HE)
pip install "brainlayer[style]"     # ChromaDB vector store (alternative backend)
```

## Setup

Run the interactive wizard:

```bash
brainlayer setup
brainlayer init
```

On macOS, add `--launchd` to the `brainlayer setup` command to install the
packaged launchd agents.

For existing installed hotlane jobs, retain the 1Password-backed `GOOGLE_API_KEY`
and `BRAINLAYER_REQUIRE_GOOGLE_API_KEY` gate until the release re-renders their plists.
See [Configuration](configuration.md) for the env-run exit-78 compatibility gate.

This will:

1. Create `~/.config/brainlayer/brainlayer.env` without writing plaintext secrets.
2. Install launchd agents from the packaged templates when setup runs with `--launchd`.
3. Check for Claude Code conversations in `~/.claude/projects/` when using `brainlayer init`.
4. Create the database at `~/.local/share/brainlayer/brainlayer.db` during indexing.

## Index Your Conversations

```bash
brainlayer index
```

This parses your Claude Code conversations (JSONL files), classifies content, chunks it with sentence boundaries, generates embeddings (bge-large-en-v1.5), and stores everything in the SQLite database.

## Connect to Your Editor

### Claude Code, Codex, Cursor, and Gemini

Add to each agent's MCP settings under `mcpServers`:

```json
{
  "mcpServers": {
    "brainlayer": {
      "command": "socat",
      "args": ["STDIO", "UNIX-CONNECT:/tmp/brainbar.sock"]
    }
  }
}
```

BrainBar must be running and owning `/tmp/brainbar.sock`. `brainlayer setup`
rewrites any owned config still pointing at the deleted `brainlayer-mcp`
entrypoint to this socket form.

### Zed

Add the same socket command to `settings.json`:

```json
{
  "context_servers": {
    "brainlayer": {
      "command": {
        "path": "socat",
        "args": ["STDIO", "UNIX-CONNECT:/tmp/brainbar.sock"]
      }
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
      "command": "socat",
      "args": ["STDIO", "UNIX-CONNECT:/tmp/brainbar.sock"]
    }
  }
}
```

## Enrichment (retired)

LLM chunk and session enrichment is retired. Indexing, `brain_search`, and the local
knowledge graph remain available, and existing metadata remains readable.
History: [retirement details](enrichment.md).

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
brainlayer stats              # Database statistics
brainlayer brain-export       # Generate brain graph JSON
brainlayer export-obsidian    # Export to Obsidian vault
brainlayer dashboard          # Interactive TUI dashboard
```
