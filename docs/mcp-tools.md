# MCP Tools Reference

BrainLayer exposes 14 MCP tools, organized into two categories.

## Intelligence Layer

These tools go beyond raw search — they understand intent and context.

### brainlayer_think

Given your current task context, retrieves relevant past decisions, patterns, and bugs. Groups results by intent category.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `context` | string | Yes | What you're currently working on |
| `project` | string | No | Filter to a specific project |
| `max_results` | integer | No | Maximum results (default: 10) |

**Returns:** Markdown with results grouped by category (decisions, bugs, patterns, implementations).

**Use when:** Starting a new task, making architectural decisions, or debugging.

---

### brainlayer_recall

File-based or topic-based recall. "What happened with this file?" or "What do I know about deployment?"

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file_path` | string | No* | File to recall history for |
| `topic` | string | No* | Topic to recall knowledge about |
| `project` | string | No | Filter to a specific project |
| `max_results` | integer | No | Maximum results (default: 10) |

*At least one of `file_path` or `topic` is required.

**Returns:** Markdown with relevant chunks, session context, and file interactions.

**Use when:** Investigating a file's history, or gathering knowledge about a specific topic.

---

### brainlayer_current_context

Lightweight — what projects, branches, files, and active plan were you working on recently? No embedding needed.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `hours` | integer | No | How many hours back to look (default: 24) |

**Returns:** Structured summary of recent activity.

**Use when:** Starting a conversation, to understand current state.

---

### brainlayer_sessions

Browse recent sessions by project and date range.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `project` | string | No | Filter to a specific project |
| `days` | integer | No | How many days back (default: 7, max: 365) |
| `limit` | integer | No | Maximum sessions (default: 20, max: 100) |

**Returns:** Markdown list of sessions with ID, project, branch, plan, and start time.

---

### brainlayer_session_summary

Session-level analysis: decisions made, corrections, learnings, quality scores.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `session_id` | string | Yes | Session ID to summarize |

**Returns:** Markdown with decisions, corrections, learnings, patterns, and quality metrics.

**Use when:** Reviewing what happened in a specific session.

---

### brainlayer_store

Persist a memory (idea, decision, learning, mistake, etc.) for future retrieval.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | string | Yes | The memory content to store |
| `type` | string | Yes | Memory type: idea, mistake, decision, learning, todo, bookmark, note, journal |
| `project` | string | No | Project to scope the memory |
| `tags` | array[string] | No | Tags for categorization |
| `importance` | integer | No | Importance score 1-10 |

**Returns:** Chunk ID and related existing memories.

**Use when:** An agent discovers something worth remembering for future sessions.

---

## Search & Context

Core search and retrieval tools.

### brainlayer_search

Hybrid semantic + keyword search with filters.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Search query |
| `project` | string | No | Filter by project |
| `content_type` | string | No | Filter: `ai_code`, `stack_trace`, `user_message`, `assistant_text`, `file_read`, `git_diff` |
| `num_results` | integer | No | Results to return (default: 5, max: 100) |
| `source` | string | No | Filter: `claude_code`, `whatsapp`, `youtube`, `all` |
| `tag` | string | No | Filter by enrichment tag |
| `intent` | string | No | Filter: `debugging`, `designing`, `configuring`, `discussing`, `deciding`, `implementing`, `reviewing` |
| `importance_min` | integer | No | Minimum importance score (1-10) |

**Returns:** Matched chunks with content, metadata, similarity scores.

---

### brainlayer_context

Get surrounding conversation chunks for a search result.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `chunk_id` | string | Yes | Chunk ID from a search result |
| `before` | integer | No | Chunks before (default: 3, max: 50) |
| `after` | integer | No | Chunks after (default: 3, max: 50) |

**Returns:** The target chunk plus surrounding conversation context.

---

### brainlayer_file_timeline

Full interaction history of a file across all sessions.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file_path` | string | Yes | File path to look up |
| `project` | string | No | Filter by project |
| `limit` | integer | No | Maximum entries (default: 50) |

**Returns:** Chronological timeline of all interactions with the file.

---

### brainlayer_operations

Logical operation groups — read/edit/test cycles within a session.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `session_id` | string | Yes | Session to analyze |

**Returns:** Grouped operations showing the read→edit→test workflow patterns.

---

### brainlayer_regression

What changed since a file last worked? Diff-based regression analysis.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file_path` | string | Yes | File to analyze |
| `project` | string | No | Filter by project |

**Returns:** Changes since the file's last known-good state.

---

### brainlayer_plan_links

Connect sessions to implementation plans and phases.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `plan_name` | string | No | Filter by plan name |
| `session_id` | string | No | Filter by session |
| `project` | string | No | Filter by project |

**Returns:** Session-to-plan linkage with phase information.

---

### brainlayer_stats

Knowledge base statistics.

**Returns:** Total chunks, projects, content types, enrichment progress, and source breakdown.

---

### brainlayer_list_projects

List all indexed projects.

**Returns:** Project names with chunk counts.
