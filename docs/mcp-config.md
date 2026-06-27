# MCP Configuration for BrainLayer

Use the BrainBar socket bridge for agent MCP wiring. This avoids GUI `PATH`
drift in Finder-launched apps because agents connect to the already-running
BrainBar daemon instead of spawning a write-capable Python MCP process. Use the
reconnecting stdio bridge rather than raw `socat`; it keeps the MCP transport
alive while the BrainBar socket is replaced and reconnects automatically.

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

By default the bridge connects to `/tmp/brainbar.sock`. To point it at a
different front socket, set `BRAINLAYER_MCP_SOCKET` in that MCP entry's env.

The Python `brainlayer-mcp` entrypoint is still packaged for development and
formula installs, but it is not the recommended agent wiring path.

## Testing the MCP Server

1. Confirm BrainBar owns the MCP socket:
   ```bash
   test -S /tmp/brainbar.sock
   (printf '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}\n'; sleep 1) \
     | brainlayer-mcp-stdio-bridge
   ```

2. In Claude Code, the tools should appear:
   - `brain_search` - Unified semantic search (query, file_path, chunk_id, filters)
   - `brain_store` - Persist memories (ideas, decisions, learnings)
   - `brain_recall` - Proactive retrieval (context, sessions, summaries)

   *Old `brainlayer_*` names still work as backward-compat aliases.*
