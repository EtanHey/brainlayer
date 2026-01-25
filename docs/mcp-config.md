# MCP Configuration for Zikaron

Add this to `~/.claude/settings.json` under `mcpServers`:

```json
{
  "mcpServers": {
    "zikaron": {
      "command": "python",
      "args": ["-m", "zikaron.mcp"],
      "cwd": "/Users/etanheyman/Gits/zikaron"
    }
  }
}
```

Or if you have zikaron installed globally:

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

## Testing the MCP Server

1. Start the server manually to test:
   ```bash
   cd ~/Gits/zikaron
   source .venv/bin/activate
   python -m zikaron.mcp
   ```

2. In Claude Code, the tools should appear:
   - `zikaron_search` - Search past conversations
   - `zikaron_stats` - Knowledge base statistics
   - `zikaron_list_projects` - List indexed projects
