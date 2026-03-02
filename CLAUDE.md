# BrainLayer (זיכרון) - Local Knowledge Pipeline
@~/Gits/orchestrator/standards/autonomous-workflow.md

> Memory for Claude Code conversations: index, search, enrich, and visualize sessions.

## Purpose (WHY)
- Build a local, private knowledge base from Claude Code sessions.
- Provide fast search, context retrieval, and exports for downstream tools.

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
- Storage: currently `~/.local/share/zikaron/zikaron.db` (legacy path, 5GB+). Target: `~/.local/share/brainlayer/brainlayer.db`
- DB path resolved by `paths.py:get_db_path()` — prefers zikaron if it exists, falls through to brainlayer
- 8 scripts in `scripts/` hardcode `brainlayer.db` path — need fixing during DB consolidation
- Concurrency: retry on `SQLITE_BUSY`; each worker uses its own connection

## Classification & Chunking Rules
- Preserve verbatim: `ai_code`, `stack_trace`, `user_message`
- Skip/summarize: `noise` (skip), `build_log` (summarize), `dir_listing` (structure only)
- Chunking: AST-aware (tree-sitter); never split stack traces; mask large tool output

## Enrichment
- Primary backend: **MLX** (`Qwen2.5-Coder-14B-Instruct-4bit`) on Apple Silicon (port 8080)
- Fallback: Ollama (`glm-4.7-flash`) on port 11434, auto-switches after 3 consecutive MLX failures
- Override with `BRAINLAYER_ENRICH_BACKEND=ollama|mlx|groq`
- Worker script: `golems/scripts/enrichment-lazy.sh` (launchd, nice=20, batch=50)
- Adds metadata (summary, tags, importance, intent); session enrichment captures decisions/corrections
- Known issue: 72K empty 0-char chunks from tool-call-only turns pollute the queue (FIXED 2026-03-01, deleted)

## Interfaces
- Daemon API (core): `/health`, `/stats`, `/search`, `/context/{chunk_id}`, `/session/{session_id}`
- Brain graph API: `/brain/graph`, `/brain/node/{node_id}`
- Backlog API: `/backlog/items` (GET/POST/PATCH/DELETE)
- MCP tools (8): `brain_search`, `brain_store`, `brain_recall`, `brain_entity`, `brain_expand`, `brain_update`, `brain_digest`, `brain_get_person` (legacy `brainlayer_*` aliases still work)
- MCP server entrypoint: `brainlayer-mcp`

## Exports
- `brainlayer brain-export` -> graph JSON for dashboard
- `brainlayer export-obsidian` -> Markdown vault (backlinks + tags)

## Data & Locks
- DB: `~/.local/share/brainlayer/brainlayer.db`
- Prompts cache: `~/.local/share/brainlayer/prompts/`
- Socket: `/tmp/brainlayer.sock`
- Enrichment lock: `/tmp/brainlayer-enrichment.lock`

## Bulk DB Operations (SAFETY)
1. **Stop enrichment workers first** — never run bulk ops while enrichment is writing (causes WAL bloat + potential freeze)
2. **Checkpoint WAL** before and after: `PRAGMA wal_checkpoint(FULL)`
3. **Drop FTS triggers** before bulk deletes — `chunks_fts_delete` trigger is a massive perf killer. Recreate after.
4. **Batch deletes** in 5-10K chunks, checkpoint every 3 batches
5. Never delete from `chunks` while FTS trigger is active on large datasets

## Naming
- BrainLayer (זיכרון) = "memory"
