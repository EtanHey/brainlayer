# MCP Configuration for BrainLayer

Add this to `~/.claude/settings.json` under `mcpServers`:

```json
{
  "mcpServers": {
    "brainlayer": {
      "command": "python",
      "args": ["-m", "brainlayer.mcp"],
      "cwd": "/path/to/brainlayer"
    }
  }
}
```

Or if you have brainlayer installed globally:

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

## Testing the MCP Server

1. Start the server manually to test:
   ```bash
   cd /path/to/brainlayer
   source .venv/bin/activate
   python -m brainlayer.mcp
   ```

2. In Claude Code, the tools should appear:
   - `brainlayer_search` - Search past conversations
   - `brainlayer_stats` - Knowledge base statistics
   - `brainlayer_list_projects` - List indexed projects
