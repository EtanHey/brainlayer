# BrainLayer — a letter from Etan

> Ratified row-by-row by voice, 2026-08-07. Record: `docs.local/GRILL-RATIFIED-2026-08-06.md`.

BrainLayer is the fleet's memory — both sides of it. My prompts, intent, and corrections on one
side; what you all tried, what worked, what eroded and why on the other — the fleet's accumulated
experience, so agents learn from each other's experience instead of re-deriving what the fleet
already paid to learn. More than storage: it should give agents the
understanding of WHY to go one way and not the other.

**Everything traces to its origin.** From any memory you can walk back to the conversation it came
from. Prefer pointers to the source of truth over hoarding copies.

**Every indexed source carries its class.** CLI coding agents (Claude Code, Codex, antigravity),
desktop apps (Claude Desktop, Gemini, ChatGPT), and subagents — normal subagents versus
brain-workers. The class is labeled at ingest and decides default-search visibility and importance.
Desktop apps: indexed, hidden from default search, opt-in only — and the opt-in switch stays
unadvertised so benchmark lanes can't trip it. Normal subagents: indexed, labeled, expandable as a
subagent thread. Brain-workers: always saved, never in default searches, never demoting other
results — `[OPEN]` whether they index at all, and they should ship with BrainLayer itself. All
transcripts get saved, so every memory can trace back. A new source enters only by deliberate
wiring; nothing is auto-detected.

**Store discipline.** Verify, then store. Do the thing, then store it — never store-before-work as
ritual. When something you stored turns out wrong, UPDATE it, don't store again beside it. If it's
my intent you corrected, come back and ask me if you got it right — and make sure your question
actually reached me: a question buried under recon output was never asked. Standing rules carry
their date and expiry — a stale truth must not masquerade as current. `[NOT A FEATURE YET — this is where I'm driving this]`
Eventually BrainLayer should be able to sleep like a human brain does: during the night or
hibernation periods it runs local or large-context models to figure out what the current truths
still are, and learns. That's what a brain does when it sleeps.

**Never silently degrade. Never auto-delete personal data.** Transcripts move to the archive only
after they're embedded, and only if the usage readers still see everything. Test data changes
against a copy of the real database before merging. Merged is not deployed — verify the thing that
executes. Never ship my database inside a package.

This letter is for agents building BrainLayer. The rules for agents USING it live in the tool
descriptions — keep those true.

— Etan

---

# BrainLayer Agent Notes (operational)

BrainLayer is the memory layer for the entire ecosystem. If it breaks, every golem degrades into a vanilla LLM with no durable recall.

## Review Guidelines
- Treat retrieval correctness, write safety, and MCP stability as critical-path concerns.
- Prefer finding regressions in search quality, lock handling, and tool contracts before style or refactor nits.
- Flag risky DB or concurrency changes explicitly. Do not hand-wave lock behavior.

## Key Paths
- `src/brainlayer/`
- `scripts/`
- `tests/`

## Database
- Canonical DB: `~/.local/share/brainlayer/brainlayer.db`

## MCP Tools
- `brain_search`
- `brain_store`
- `brain_recall`
- `brain_expand`
- `brain_digest`
- `brain_entity`
- `brain_update`
- `brain_tags`

## Concurrency Rules
- One write at a time.
- Reads are safe.
- `brain_digest` is write-heavy; do not run it in parallel with other MCP work.

## Tests
- Run `pytest` before claiming behavior changed safely.
- Current suite size: 929 tests.

## PR Workflow
- Request `@codex review`.
- Request a lead-routed Claude pair review through the active collab lane.
- Bugbot is out of quota and the Greptile trial has expired; do not route mandatory reviews to either.

## Known Issues
- DB locking during enrichment.
- WAL can grow to 4.7GB.

<!-- IDENTITY: brainlayer, owner=EtanHey, purpose=the fleet's memory — see the letter at the top of this file -->
## BrainBar Native Tools

Current native Swift BrainBar tools (PR #135, 2026-03-30):
- **brain_search** — FTS5 hybrid search with BM25 ranking, AND matching, ANSI formatted output
- **brain_store** — Store chunks with tags and importance
- **brain_recall** — Stats mode (counts, enrichment %) or context mode (session lookup by conversation_id)
- **brain_entity** — Entity lookup (exact/LIKE) with relations from kg_relations
- **brain_digest** — Rule-based entity extraction (capitalized names, PascalCase, URLs, code paths)
- **brain_update** — Update chunk importance and/or tags by chunk_id
- **brain_expand** — Get chunk + surrounding session context (before/after)
- **brain_tags** — List unique tags with counts, filter by query prefix

---

<!-- ARCHITECTURE: Python/Typer CLI, sqlite-vec storage via APSW, bge-large embeddings, FastAPI daemon, MCP server, Textual TUI + Next.js dashboard -->
## Stack (WHAT)
- Python package + Typer CLI in `src/brainlayer/`
- sqlite-vec storage via APSW (`vector_store.py`)
- bge-large-en-v1.5 embeddings (`embeddings.py`)
- FastAPI daemon (`daemon.py`), MCP server (`mcp/`)
- Textual TUI (`dashboard/`) and Next.js dashboard
- Source data: JSONL in `~/.claude/projects/`

<!-- COMMANDS: `python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"` | test: `pytest` | lint: `ruff check src/ && ruff format src/` | run: `brainlayer index && brainlayer serve` -->
## Workflow (HOW)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
brainlayer index
brainlayer serve
brainlayer search "how did I implement authentication"
brainlayer enrich
```
- Lint/format: `ruff check src/ && ruff format src/`
- Pre-push: `.githooks/pre-push` runs `scripts/run_tests.sh` with `BRAINLAYER_PREPUSH=1`; full
  runs are deduped by git tree hash in `.git/brainlayer-prepush-cache`.
- Scoped worker pushes: use `BRAINLAYER_PREPUSH_SCOPE=changed-only git push` to map changed files
  to focused pytest targets while keeping the lightweight registration, isolated, bun, and shell
  gates.
- Worker pre-push must not run real-DB files `tests/test_vector_store.py` or
  `tests/test_engine.py`; run them only as deliberate production-DB checks.

## Pipeline Overview
- Extract -> Classify -> Chunk -> Embed -> Index
- Post-processing: Enrichment, Brain Graph, Obsidian export
- Storage: `~/.local/share/brainlayer/brainlayer.db` (canonical path, ~8GB)
- DB path resolved by `paths.py:get_db_path()` — env var override or canonical path
- All scripts and CLI use `paths.py` for DB path resolution
- Concurrency: retry on `SQLITE_BUSY`; each worker uses its own connection

## P1 Pipeline Contracts
- BL-10 source denylist is implemented in `src/brainlayer/ingest_denylist.py`. By default, provider
  sessions and ordinary Claude subagents ingest. **Exclusion is scoped to agents whose job is to READ
  OTHER AGENTS' MEMORY OR JSONLs — `brain-worker` sub-agents, and `session-miner` / `weave` workflow
  agents. It is NOT all workflows.** Etan, 2026-08-03: *"That's very specific. It's not all workflows.
  I don't know why y'all keep thinking it's all workflows."* and *"the Weaver's workflow is built of
  session miners."*
- **brain-worker and miner sessions are KEPT but NOT INDEXED.** Exclusion never deletes the source
  JSONL — it means exclusion from the index only. `BRAINLAYER_INGEST_DENYLIST` remains an explicit
  deployment override.
- **brain-worker ≠ brainlayer worker (Etan, ratified by voice 2026-08-08):** a *brain-worker* is
  the ingest-excluded SUBAGENT type (Claude Agent-tool type today; other harnesses hopefully later)
  — its transcript stays out of the index BY CLASS, which is its entire point. A *brainlayer
  worker* is a cmux pane of a brainlayerGolem and ingests normally. Never substitute a visible pane
  when ingest-exclusion is the goal; to "paste" to a subagent, paste to the lead, who relays.
- ⚠️ **Known code drift, do not read this file as describing shipped behaviour:**
  `ingest_denylist.py:14` is `("~/.claude/projects/**/wf_*/**",)`, which excludes **every** workflow
  path and silently discards legitimate workflow-agent memory. The rule above is the ruling; the code
  is the defect. Code fix owned by the brainlayer lane.
- Go-forward secret scrubbing runs in `src/brainlayer/pipeline/secret_scrub.py` from `src/brainlayer/watcher_bridge.py` before chunk persistence. Provider-prefixed and labeled high-entropy secrets are redacted; unlabeled high-entropy tokens are recorded in quarantine metadata.
- MCP search uses a fixed-size readonly WAL `VectorStore` pool in `src/brainlayer/mcp/_shared.py`. `BRAINLAYER_READ_POOL_SIZE` defaults to 8, or 4 on detected Apple M1; M1 machines can keep the lower override explicitly. Checkout beyond the fixed pool blocks up to `BRAINLAYER_READ_BUSY_TIMEOUT_MS`, and startup rejects `pool_size * BRAINLAYER_READ_CACHE_KB` above about 768MB.

<!-- ARCHITECTURE: classification preserves ai_code/stack_trace/user_message verbatim; skips noise; AST-aware chunking via tree-sitter; never split stack traces -->
## Classification & Chunking Rules
- Preserve verbatim: `ai_code`, `stack_trace`, `user_message`
- Skip/summarize: `noise` (skip), `build_log` (summarize), `dir_listing` (structure only)
- Chunking: AST-aware (tree-sitter); never split stack traces; mask large tool output

<!-- ARCHITECTURE: primary enrichment backend=Groq (cloud); fallback=Gemini via enrichment_controller.py; override via BRAINLAYER_ENRICH_BACKEND env var -->
## Enrichment
- Primary backend: **Groq** (cloud, configured in launchd plist)
- Fallback: Gemini via `enrichment_controller.py`, Ollama as offline last-resort
- Override with `BRAINLAYER_ENRICH_BACKEND=ollama|mlx|groq`
- Rate configurable via `BRAINLAYER_ENRICH_RATE` env var (default 0.2 = 12 RPM)
- Adds metadata (summary, tags, importance, intent); session enrichment captures decisions/corrections

<!-- MCP-SERVERS: add new MCP tool entries to mcp/ dir; entrypoint is `brainlayer-mcp`; 13 tools: brain_search, brain_store, brain_recall, brain_resume, brain_entity, brain_expand, brain_update, brain_digest, brain_get_person, brain_enrich, brain_tags, brain_supersede, brain_archive -->
## Interfaces
- Daemon API (core): `/health`, `/stats`, `/search`, `/context/{chunk_id}`, `/session/{session_id}`
- Brain graph API: `/brain/graph`, `/brain/node/{node_id}`
- Backlog API: `/backlog/items` (GET/POST/PATCH/DELETE)
- MCP tools (13): `brain_search`, `brain_store`, `brain_recall`, `brain_resume`, `brain_entity`, `brain_expand`, `brain_update`, `brain_digest`, `brain_get_person`, `brain_enrich`, `brain_tags`, `brain_supersede`, `brain_archive` (legacy `brainlayer_*` aliases still work; note: `brain_update`, `brain_expand`, `brain_tags` are deprecated in the Python MCP path and return errors — use the BrainBar native path for these)
- MCP server entrypoint: `brainlayer-mcp`

<!-- COMMANDS: `brainlayer brain-export` → graph JSON for dashboard | `brainlayer export-obsidian` → Markdown vault with backlinks + tags -->
## Exports
- `brainlayer brain-export` -> graph JSON for dashboard
- `brainlayer export-obsidian` -> Markdown vault (backlinks + tags)

<!-- ARCHITECTURE: real-time watcher via LaunchAgent (com.brainlayer.watch.plist), 4-layer content filters, offset persistence, rewind detection, Axiom telemetry -->
## Real-time JSONL Watcher
- `brainlayer watch` — persistent watcher for `~/.claude/projects/*.jsonl`
- LaunchAgent: `com.brainlayer.watch.plist` (KeepAlive, Nice=10)
- 4-layer content filters: entry type whitelist → classify → chunk min-length → system-reminder strip
- Offset persistence: `~/.local/share/brainlayer/offsets.json` (survives restarts)
- Rewind detection: file shrink = checkpoint restore → soft-archives reverted chunks
- Axiom telemetry: startup, flush, error, heartbeat (60s) to `brainlayer-watcher` dataset
- Source: `watcher.py` (tailer + indexer), `watcher_bridge.py` (pipeline integration)

<!-- ARCHITECTURE: chunk lifecycle columns superseded_by/aggregated_into/archived_at; brain_supersede has safety gate for personal data; brain_archive is soft-delete -->
## Chunk Lifecycle
- Columns: `superseded_by`, `aggregated_into`, `archived_at` on chunks table
- Default search excludes lifecycle-managed chunks; `include_archived=True` shows history
- `brain_supersede`: safety gate for personal data (journals, notes, health/finance)
- `brain_archive`: soft-delete with timestamp
- `brain_store` gains `supersedes` param for atomic store-and-replace

<!-- HOOKS: SessionStart writes injected chunk_ids to /tmp/brainlayer_session_{id}.json; UserPromptSubmit skips already-injected; module: hooks/dedup_coordination.py -->
## Session Dedup Coordination
- `/tmp/brainlayer_session_{id}.json` — shared between SessionStart and UserPromptSubmit hooks
- SessionStart writes injected chunk_ids; UserPromptSubmit skips already-injected
- Handoff detection: prompts with "handoff", "session-handoff" skip auto-search
- Module: `hooks/dedup_coordination.py`

<!-- PATHS: DB=~/.local/share/brainlayer/brainlayer.db | offsets=~/.local/share/brainlayer/offsets.json | logs=~/.local/share/brainlayer/logs/watch.{log,err} | socket=/tmp/brainlayer.sock | lock=/tmp/brainlayer-enrichment.lock -->
## Data & Locks
- Backup log: real runs append JSONL to `~/.local/share/brainlayer/logs/backup-daily.log` with
  `backup_log_provenance=real`; pytest sets `BRAINLAYER_BACKUP_LOG_PATH` and
  `BRAINLAYER_BACKUP_LOG_PROVENANCE=pytest` so tests cannot refresh the production heartbeat log.
- Watcher offsets: `~/.local/share/brainlayer/offsets.json`
- Prompts cache: `~/.local/share/brainlayer/prompts/`
- Watcher logs: `~/.local/share/brainlayer/logs/watch.{log,err}`
- Socket: `/tmp/brainlayer.sock`
- Enrichment lock: `/tmp/brainlayer-enrichment.lock`
- Session dedup: `/tmp/brainlayer_session_*.json`

<!-- ANTI-PATTERNS: never run bulk ops while enrichment is writing; never delete from chunks while FTS trigger active on large datasets; always stop workers + checkpoint WAL first -->
## Bulk DB Operations (SAFETY)
1. **Stop enrichment workers first** — never run bulk ops while enrichment is writing (causes WAL bloat + potential freeze)
2. **Checkpoint WAL** before and after: `PRAGMA wal_checkpoint(FULL)`
3. **Batch deletes** in 5-10K chunks, checkpoint every 3 batches

## Memory Tools
- Always `brain_search` before answering questions about project history, architecture, or past decisions
- `brain_store` after making decisions, hitting bugs, or receiving corrections
- Don't rely solely on hook-injected context -- it's a hint, not comprehensive

## Naming
- BrainLayer (זיכרון) = "memory"

## Rulings that bind every agent here (ratified by voice 2026-08-02)

- **Worktrees: `project` is always the REPO** (canonical); the branch is a separate field.
  A branch name is never a project.
- **Rules ride in instruction files** (this file + `CLAUDE.md`), deterministically. BrainLayer
  injects CONTEXT only — it is **not a source of law**, and no third curated instruction source gets created.
- **Agent-authored chunks are the normal case** — *"rules cite no source"* is a **non-finding**;
  never report it.
- **Pre-merge live-check against a DB copy** is required before any change that touches stored data.
  *(Etan's addition at ratification, 2026-08-02.)*
