# MCP Tools Reference

BrainLayer exposes **3 MCP tools** (Phase 4 consolidation).

## brain_search

Unified semantic search — pass `query`, `file_path`, `chunk_id`, or filters. Auto-routes to the right view.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | No* | Search query (semantic + keyword) |
| `file_path` | string | No* | File to get timeline/history for |
| `chunk_id` | string | No* | Chunk ID for surrounding context |
| `project` | string | No | Filter by project |
| `content_type` | string | No | Filter: `ai_code`, `stack_trace`, `user_message`, etc. |
| `num_results` | integer | No | Max results (default: 5, max: 100) |
| `source` | string | No | Filter: `claude_code`, `whatsapp`, `youtube`, `all` |
| `tag` | string | No | Filter by enrichment tag |
| `intent` | string | No | Filter: `debugging`, `designing`, `implementing`, etc. |
| `importance_min` | integer | No | Minimum importance (1-10) |

*At least one of `query`, `file_path`, or `chunk_id` typically used; filters apply when relevant.

**Returns:** Markdown with matched chunks, context, or timeline depending on input.

---

## brain_store

Persist a memory (idea, decision, learning, mistake, etc.) for future retrieval. Auto-type and auto-importance from content.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | string | Yes | The memory content to store |
| `type` | string | No | Memory type: idea, mistake, decision, learning, todo, bookmark, note, journal (auto-detected if omitted) |
| `project` | string | No | Project to scope the memory |
| `tags` | array[string] | No | Tags for categorization |
| `importance` | integer | No | Importance score 1-10 (auto-detected if omitted) |

**Returns:** Chunk ID and related existing memories.

**Use when:** An agent discovers something worth remembering for future sessions.

---

## brain_recall

Proactive retrieval — current context, sessions, session summaries. Mode defaults to "context".

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `mode` | string | No | `context` (default), `sessions`, `session_summary` |
| `session_id` | string | No | For session_summary mode |
| `project` | string | No | Filter by project |
| `days` | integer | No | Days back for sessions (default: 7, max: 365) |
| `hours` | integer | No | Hours back for context (default: 24) |
| `limit` | integer | No | Max sessions (default: 20, max: 100) |

**Returns:** Structured summary of recent activity, session list, or session-level analysis.

**Use when:** Starting a conversation, reviewing sessions, or understanding current state.

---

## Aliases

Old `brainlayer_*` names still work as backward-compat aliases.

- `brain_search` aliases: `brainlayer_search`, `brainlayer_context`, `brainlayer_stats`, `brainlayer_list_projects`, `brainlayer_file_timeline`, `brainlayer_operations`, `brainlayer_regression`, `brainlayer_plan_links`, `brainlayer_think`
- `brain_store` alias: `brainlayer_store`
- `brain_recall` aliases: `brainlayer_recall`, `brainlayer_current_context`, `brainlayer_sessions`, `brainlayer_session_summary`
