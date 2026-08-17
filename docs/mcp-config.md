# MCP Configuration for BrainLayer

One MCP server: BrainBar on `/tmp/brainbar.sock`. Agents connect with `socat`
(or the optional reconnecting `brainlayer-mcp-stdio-bridge`). The Python
`brainlayer-mcp` entrypoint is deleted — do not wire it.

Prerequisites:

- BrainBar is running and owns `/tmp/brainbar.sock`.
- `socat` is on `PATH` (Homebrew: `/opt/homebrew/bin/socat`).

Add this to Claude, Codex, Cursor, or Gemini MCP settings under `mcpServers`:

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

`brainlayer setup` rewrites any owned config still pointing at `brainlayer-mcp`
to this socket form (backs up each file first).

## Testing the MCP Server

1. Confirm BrainBar owns the MCP socket:
   ```bash
   test -S /tmp/brainbar.sock
   (printf '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}\n'; sleep 1) \
     | socat STDIO UNIX-CONNECT:/tmp/brainbar.sock \
     | tee /tmp/brainlayer-mcp-smoke.out
   grep '"id":1' /tmp/brainlayer-mcp-smoke.out
   ```

2. In Claude Code, the tools should appear:
   - `brain_search` - Unified semantic search (query, file_path, chunk_id, filters)
   - `brain_store` - Persist memories (ideas, decisions, learnings)
   - `brain_recall` - Session context and stats
