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

`brainlayer setup --migrate-mcp` rewrites owned configs still pointing at the deleted
`brainlayer-mcp` entrypoint to the socat socket form (backing up each file first). It matches by
server name and leaves entries that already use socat **or** `brainlayer-mcp-stdio-bridge`
untouched — so wiring the bridge yourself, as recommended above, is not undone by setup.

## Testing the MCP Server

1. Confirm BrainBar owns the MCP socket:
   ```bash
   test -S /tmp/brainbar.sock
   (printf '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}\n'; sleep 1) \
     | socat STDIO UNIX-CONNECT:/tmp/brainbar.sock \
     | tee /tmp/brainlayer-mcp-smoke.out
   grep '"id":1' /tmp/brainlayer-mcp-smoke.out
   ```

2. In Claude Code, the **core palette** should appear — 5 tools, not all 17:
   - `brain_search` - Search memory by topic
   - `brain_store` - Persist decisions, corrections, learnings
   - `brain_recall` - Session context and stats
   - `brain_expand` - Open one search result in full, with surrounding chunks
   - `expand_palette` - Expose the other 13 tools for this session

   Calling `expand_palette` (or starting the server with `BRAINLAYER_MCP_PROFILE=full`) returns all
   17 with their full descriptions. A gated tool called before expanding returns an error saying so.
