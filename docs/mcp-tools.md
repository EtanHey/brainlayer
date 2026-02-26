# MCP Tools Reference

BrainLayer exposes **7 MCP tools** — 3 core (search/store/recall) + 4 knowledge graph (digest/entity/update/get_person).

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

**Annotations:** `readOnlyHint: true`

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
| `entity_id` | string | No | Link memory to a KG entity |

**Returns:** Chunk ID and related existing memories.

**Use when:** An agent discovers something worth remembering for future sessions.

---

## brain_recall

Proactive retrieval — current context, sessions, session summaries. Mode defaults to "context".

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `mode` | string | No | `context` (default), `sessions`, `session_summary`, `operations`, `plan`, `summary`, `stats` |
| `session_id` | string | No | For session_summary mode |
| `project` | string | No | Filter by project |
| `days` | integer | No | Days back for sessions (default: 7, max: 365) |
| `hours` | integer | No | Hours back for context (default: 24) |
| `limit` | integer | No | Max sessions (default: 20, max: 100) |

**Returns:** Structured summary of recent activity, session list, or session-level analysis.

**Annotations:** `readOnlyHint: true`

**Use when:** Starting a conversation, reviewing sessions, or understanding current state.

---

## brain_digest

Ingest raw content (transcripts, docs, articles, conversation blocks). Runs entity extraction, relation extraction, sentiment analysis, and action item detection.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | string | Yes | Raw content to digest |
| `source` | string | No | Source label (default: `digest`) |
| `project` | string | No | Project to scope |
| `participants` | array[string] | No | Known participants for entity linking |

**Returns:** Digest summary with extracted entities, relations, sentiment, action items, decisions, and questions.

**Use when:** Ingesting meeting transcripts, long documents, or conversation logs. Batch 5-10 messages for short chat — don't call per-message.

---

## brain_entity

Look up a known entity in the knowledge graph. Returns entity type, relations, and evidence chunks.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Entity name or search query |
| `entity_id` | string | No | Direct entity ID lookup |

**Returns:** Entity details (type, metadata), relations to other entities, and linked chunks.

**Annotations:** `readOnlyHint: true`

**Use when:** Looking up people, projects, or concepts in the knowledge graph.

---

## brain_update

Update, archive, or merge existing memories.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `action` | string | Yes | `update`, `archive`, or `merge` |
| `chunk_id` | string | Yes | Target chunk ID |
| `content` | string | No | New content (re-embeds if changed). For `update` action. |
| `tags` | array[string] | No | New tags. For `update` action. |
| `importance` | integer | No | New importance. For `update` action. |
| `merge_chunk_ids` | array[string] | No | Chunk IDs to archive (kept chunk = `chunk_id`). For `merge` action. |

**Returns:** Confirmation of the action taken.

**Use when:** Preferences change ("actually Sundays work now"), deduplicating similar memories, or soft-deleting outdated information.

---

## brain_get_person

Look up a person by name — returns entity details, recent interactions, preferences, and related memories. Optimized for real-time lookups (~200-500ms).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | Person's name |
| `project` | string | No | Scope to a project |

**Returns:** Person entity with metadata, recent chunks, relations, and preferences.

**Use when:** Need quick context about a person before a meeting, during a conversation, or for personalization.

---

## Aliases

Old `brainlayer_*` names still work as backward-compat aliases.

- `brain_search` aliases: `brainlayer_search`, `brainlayer_context`, `brainlayer_stats`, `brainlayer_list_projects`, `brainlayer_file_timeline`, `brainlayer_operations`, `brainlayer_regression`, `brainlayer_plan_links`, `brainlayer_think`
- `brain_store` alias: `brainlayer_store`
- `brain_recall` aliases: `brainlayer_recall`, `brainlayer_current_context`, `brainlayer_sessions`, `brainlayer_session_summary`
