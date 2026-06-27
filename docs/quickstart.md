# Quick Start

## Installation

```bash
pip install brainlayer
```

### Optional extras

```bash
pip install "brainlayer[brain]"     # Brain graph visualization (Leiden + UMAP)
pip install "brainlayer[cloud]"     # Cloud backfill (Gemini Batch API)
pip install "brainlayer[youtube]"   # YouTube transcript indexing
pip install "brainlayer[ast]"       # AST-aware code chunking (tree-sitter)
pip install "brainlayer[kg]"        # GliNER entity extraction (209M params, EN+HE)
pip install "brainlayer[style]"     # ChromaDB vector store (alternative backend)
```

## Setup

Run the interactive wizard:

```bash
brainlayer setup --google-api-key-op-ref "op://Private/Google AI/Gemini API key"
brainlayer init
```

On macOS, add `--launchd` to the `brainlayer setup` command to install the
packaged launchd agents.

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
      "command": "brainlayer-mcp-stdio-bridge"
    }
  }
}
```

The bridge requires BrainBar to be running. It defaults to `/tmp/brainbar.sock`,
supports `BRAINLAYER_MCP_SOCKET` for alternate socket paths, and reconnects if
the BrainBar socket is replaced while the agent is still running.
If your editor is launched from the macOS GUI and cannot resolve
`brainlayer-mcp-stdio-bridge`, replace the command with the absolute executable
path from `command -v brainlayer-mcp-stdio-bridge`, usually
`/opt/homebrew/bin/brainlayer-mcp-stdio-bridge` for Homebrew installs.

### Zed

Add the same socket command to `settings.json`:

```json
{
  "context_servers": {
    "brainlayer": {
      "command": {
        "path": "brainlayer-mcp-stdio-bridge",
        "args": []
      }
    }
  }
}
```

Use the absolute executable path here too if Zed cannot find the command from
its GUI environment.

### VS Code

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "brainlayer": {
      "command": "brainlayer-mcp-stdio-bridge"
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
