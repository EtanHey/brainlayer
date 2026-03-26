# BrainLayer (זיכרון) - Local Knowledge Pipeline

> Memory for Claude Code conversations: index, search, enrich, and visualize sessions.

## Purpose (WHY)
- Build a local, private knowledge base from Claude Code sessions.
- Provide fast search, context retrieval, and exports for downstream tools.

---

## BrainBar Stub Warnings

BrainBar Swift daemon has 3 STUB tools returning fake success:
- brain_update, brain_expand, brain_tags — BROKEN (return success, save nothing)
- Working: brain_search, brain_store, brain_recall, brain_entity, brain_digest, brain_get_person

---

## Stack (WHAT)
- Python package + Typer CLI in `src/brainlayer/`
- sqlite-vec storage via APSW (`vector_store.py`)
- bge-large-en-v1.5 embeddings (`embeddings.py`)
- FastAPI daemon (`daemon.py`), MCP server (`mcp/`)
- Textual TUI (`dashboard/`) and Next.js dashboard
- Source data: JSONL in `~/.claude/projects/`

## Workflow (HOW)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
brainlayer index
brainlayer serve
brainlayer search "how did I implement authentication"
brainlayer enrich
```
- Tests: `pytest`
- Lint/format: `ruff check src/ && ruff format src/`

## Pipeline Overview
- Extract -> Classify -> Chunk -> Embed -> Index
- Post-processing: Enrichment, Brain Graph, Obsidian export
- Storage: `~/.local/share/brainlayer/brainlayer.db` (canonical path, ~8GB)
- DB path resolved by `paths.py:get_db_path()` — env var override or canonical path
- All scripts and CLI use `paths.py` for DB path resolution
- Concurrency: retry on `SQLITE_BUSY`; each worker uses its own connection

## Classification & Chunking Rules
- Preserve verbatim: `ai_code`, `stack_trace`, `user_message`
- Skip/summarize: `noise` (skip), `build_log` (summarize), `dir_listing` (structure only)
- Chunking: AST-aware (tree-sitter); never split stack traces; mask large tool output

## Enrichment
- Primary backend: **Groq** (cloud, configured in launchd plist)
- Fallback: Gemini via `enrichment_controller.py`, Ollama as offline last-resort
- Override with `BRAINLAYER_ENRICH_BACKEND=ollama|mlx|groq`
- Rate configurable via `BRAINLAYER_ENRICH_RATE` env var (default 0.2 = 12 RPM)
- Adds metadata (summary, tags, importance, intent); session enrichment captures decisions/corrections

## Interfaces
- Daemon API (core): `/health`, `/stats`, `/search`, `/context/{chunk_id}`, `/session/{session_id}`
- Brain graph API: `/brain/graph`, `/brain/node/{node_id}`
- Backlog API: `/backlog/items` (GET/POST/PATCH/DELETE)
- MCP tools (11): `brain_search`, `brain_store`, `brain_recall`, `brain_entity`, `brain_expand`, `brain_update`, `brain_digest`, `brain_get_person`, `brain_tags`, `brain_supersede`, `brain_archive` (legacy `brainlayer_*` aliases still work)
- MCP server entrypoint: `brainlayer-mcp`

## Exports
- `brainlayer brain-export` -> graph JSON for dashboard
- `brainlayer export-obsidian` -> Markdown vault (backlinks + tags)

## Real-time JSONL Watcher
- `brainlayer watch` — persistent watcher for `~/.claude/projects/*.jsonl`
- LaunchAgent: `com.brainlayer.watch.plist` (KeepAlive, Nice=10)
- 4-layer content filters: entry type whitelist → classify → chunk min-length → system-reminder strip
- Offset persistence: `~/.local/share/brainlayer/offsets.json` (survives restarts)
- Rewind detection: file shrink = checkpoint restore → soft-archives reverted chunks
- Axiom telemetry: startup, flush, error, heartbeat (60s) to `brainlayer-watcher` dataset
- Source: `watcher.py` (tailer + indexer), `watcher_bridge.py` (pipeline integration)

## Chunk Lifecycle
- Columns: `superseded_by`, `aggregated_into`, `archived_at` on chunks table
- Default search excludes lifecycle-managed chunks; `include_archived=True` shows history
- `brain_supersede`: safety gate for personal data (journals, notes, health/finance)
- `brain_archive`: soft-delete with timestamp
- `brain_store` gains `supersedes` param for atomic store-and-replace

## Session Dedup Coordination
- `/tmp/brainlayer_session_{id}.json` — shared between SessionStart and UserPromptSubmit hooks
- SessionStart writes injected chunk_ids; UserPromptSubmit skips already-injected
- Handoff detection: prompts with "handoff", "session-handoff" skip auto-search
- Module: `hooks/dedup_coordination.py`

## Data & Locks
- DB: `~/.local/share/brainlayer/brainlayer.db`
- Watcher offsets: `~/.local/share/brainlayer/offsets.json`
- Prompts cache: `~/.local/share/brainlayer/prompts/`
- Watcher logs: `~/.local/share/brainlayer/logs/watch.{log,err}`
- Socket: `/tmp/brainlayer.sock`
- Enrichment lock: `/tmp/brainlayer-enrichment.lock`
- Session dedup: `/tmp/brainlayer_session_*.json`

## Bulk DB Operations (SAFETY)
1. **Stop enrichment workers first** — never run bulk ops while enrichment is writing (causes WAL bloat + potential freeze)
2. **Checkpoint WAL** before and after: `PRAGMA wal_checkpoint(FULL)`
3. **Drop FTS triggers** before bulk deletes — `chunks_fts_delete` trigger is a massive perf killer. Recreate after.
4. **Batch deletes** in 5-10K chunks, checkpoint every 3 batches
5. Never delete from `chunks` while FTS trigger is active on large datasets

## Naming
- BrainLayer (זיכרון) = "memory"
