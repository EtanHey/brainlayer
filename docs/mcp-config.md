# MCP Configuration for BrainLayer

One MCP server: BrainBar on `/tmp/brainbar.sock`. Agents connect with the
shipped reconnecting `brainlayer-mcp-stdio-bridge` (no Homebrew `socat`
required; GUI hosts like Zed/VS Code/Claude Desktop do not have
`/opt/homebrew/bin` on PATH). `socat STDIO UNIX-CONNECT:/tmp/brainbar.sock`
is the documented manual alternative. The Python `brainlayer-mcp` entrypoint
is deleted — do not wire it.

Prerequisites:

- BrainBar is running and owns `/tmp/brainbar.sock`.
- `brainlayer-mcp-stdio-bridge` is on PATH (installed with the BrainLayer package).

Add this to Claude, Codex, Cursor, or Gemini MCP settings under `mcpServers`:

```json
{
  "mcpServers": {
    "brainlayer": {
      "command": "brainlayer-mcp-stdio-bridge"
    }
  }
}
```

Manual alternative when you already have `socat` on PATH (Homebrew:
`/opt/homebrew/bin/socat`):

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

`brainlayer setup` rewrites owned configs still pointing at `brainlayer-mcp`
to the socat socket form (backs up each file first). Direct
`brainlayer-mcp-stdio-bridge` wiring is already accepted and left in place.

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
