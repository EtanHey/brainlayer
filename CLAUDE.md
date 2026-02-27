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
brainlayer serve --http 8787
brainlayer search "how did I implement authentication"
brainlayer enrich
```
- Tests: `pytest`
- Lint/format: `ruff check src/ && ruff format src/`

## Pipeline Overview
- Extract -> Classify -> Chunk -> Embed -> Index
- Post-processing: Enrichment, Brain Graph, Obsidian export
- Storage: `~/.local/share/brainlayer/brainlayer.db` (sqlite-vec, WAL, `busy_timeout=5000`)
- Concurrency: retry on `SQLITE_BUSY`; each worker uses its own connection

## Classification & Chunking Rules
- Preserve verbatim: `ai_code`, `stack_trace`, `user_message`
- Skip/summarize: `noise` (skip), `build_log` (summarize), `dir_listing` (structure only)
- Chunking: AST-aware (tree-sitter); never split stack traces; mask large tool output

## Enrichment
- Backends: Ollama (`glm4`) or MLX (`BRAINLAYER_ENRICH_BACKEND=ollama|mlx`)
- Set `"think": false` for GLM-4.7 speed
- Adds metadata (summary, tags, importance, intent); session enrichment captures decisions/corrections

## Interfaces
- Daemon API (core): `/health`, `/stats`, `/search`, `/context/{chunk_id}`, `/session/{session_id}`
- Brain graph API: `/brain/graph`, `/brain/node/{node_id}`
- Backlog API: `/backlog/items` (GET/POST/PATCH/DELETE)
- MCP tools: `brain_search`, `brain_store`, `brain_recall` (legacy `brainlayer_*` aliases)
- MCP server entrypoint: `brainlayer-mcp`

## Exports
- `brainlayer brain-export` -> graph JSON for dashboard
- `brainlayer export-obsidian` -> Markdown vault (backlinks + tags)

## Data & Locks
- DB: `~/.local/share/brainlayer/brainlayer.db`
- Prompts cache: `~/.local/share/brainlayer/prompts/`
- Socket: `/tmp/brainlayer.sock`
- Enrichment lock: `/tmp/brainlayer-enrichment.lock`

## Naming
- BrainLayer (זיכרון) = "memory"
