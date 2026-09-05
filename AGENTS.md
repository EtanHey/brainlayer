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
Served by BrainBar (`MCPRouter.toolDefinitions`), 17 total. A session boots into the **core
palette** — `brain_search`, `brain_store`, `brain_recall`, `brain_expand`, plus `expand_palette`.
Call `expand_palette` or set `BRAINLAYER_MCP_PROFILE=full` for the rest.

- Core: `brain_search`, `brain_store`, `brain_recall`, `brain_expand`
- Gated: `brain_entity`, `brain_get_person`, `brain_tags`, `brain_digest`, `brain_update`,
  `brain_enrich`, `brain_supersede`, `brain_archive`, `brain_subscribe`, `brain_unsubscribe`,
  `brain_ack`, `brain_backup_vacuum_into`, `brain_maintenance_rebuild_trigram`
- `brain_backup_vacuum_into` is callable regardless of profile (owner-only socket is the trust
  boundary), but stays out of the advertised core inventory.
- `brain_resume` exists only in the Python library handlers under `src/brainlayer/mcp/`. BrainBar
  does not serve it, so agents cannot call it.

## Concurrency Rules
- One write at a time.
- Reads are safe.
- `brain_digest` is write-heavy; do not run it in parallel with other MCP work.

## Tests
- Run `pytest` before claiming behavior changed safely.
- Current suite size: 4,386 Python tests (`pytest tests/ --collect-only -q`) + 890 Swift tests in `brain-bar/Tests/`.
- **Suite hygiene, enforced not just written down: no test loads an embedding model, and no test
  opens the canonical DB.** `tests/conftest.py` arms both guards for every unmarked test —
  `BRAINLAYER_FORBID_EMBEDDING_MODEL=1` (checked at every model-load site, and inherited by
  subprocesses) and a refusal on `sqlite3.connect`/`apsw.Connection` against
  `~/.local/share/brainlayer`. The only escape is a declared marker:
  `@pytest.mark.embedding_model` for a test that really needs a real model (deselected by
  `scripts/run_tests.sh`, still run in CI, which warms the HF cache on purpose), or
  `integration`/`live` for a deliberate production-DB check.
- **The guard's one contract:** in `tests/`, reach an embedding model through
  `brainlayer.embeddings` or a module attribute — never `from sentence_transformers import X`. A
  direct alias binds the real class at import time, before any fixture, and reaches neither the
  module patch nor the env check (which lives at BrainLayer's load sites, not inside a third-party
  constructor). `test_no_test_module_binds_an_embedding_model_class_directly` fails on that shape.
- `BRAINLAYER_PREPUSH_SCOPE=changed-only` never escalates to the full suite on an empty change set;
  it skips loudly and says nothing was measured.

## Release safety
- Every Homebrew release must run `scripts/release-verify-signatures.sh <keg-path>` after installation.
- Any invalid `*.so` or `*.dylib` blocks release/deploy; never restart services until the gate passes.
- A `Casks/brainbar.rb` version lower than the package version is allowed only when `BRAINLAYER_VERSION_CHECK_CASK_LAG_REASON="no BrainBar release for <ver>"` is set for `scripts/brainlayer-version-check.sh`; a cask ahead of the package still fails hard.

## PR Workflow
- Request `@codex review`.
- Request a lead-routed Claude pair review through the active collab lane.
- Do not route mandatory reviews to Bugbot or Greptile.

## Known Issues
- DB locking during enrichment.
- WAL can grow to 4.7GB.

<!-- IDENTITY: brainlayer, owner=EtanHey, purpose=the fleet's memory — see the letter at the top of this file -->
## BrainBar Native Tools

BrainBar is the only agent MCP transport. The authoritative tool list, descriptions, schemas, and
ToolAnnotations live in `brain-bar/Sources/BrainBar/MCPRouter.swift` (`toolDefinitions`) — read that,
not a copy here. Two contracts an agent must know:

- **`brain_store` outcomes (#725):** `STORED|DUPLICATE|MERGED|DEFERRED` are all success — do NOT
  re-store; `DEFERRED` is queued and will be persisted. `REJECTED|ERROR` stored nothing and return
  no `status` and no `chunk_id`. A committed write can never answer `REJECTED` or `ERROR`.
- **Descriptions survive the wire (#727):** oversized `tools/list` responses compact as a ladder —
  annotations and inputSchema prose go first; only then are descriptions *shortened* (marked
  `…[truncated]`, with a `result._meta["brainlayer/descriptionsTruncated"]` notice naming each one).
  Descriptions are never removed; below the floor the response ships over-limit with its contract
  intact and logs why.

---

<!-- ARCHITECTURE: Python/Typer CLI, sqlite-vec storage via APSW, bge-large embeddings, BrainBar Swift MCP daemon over /tmp/brainbar.sock, library MCP handlers under mcp/, Textual TUI + Next.js dashboard -->
## Stack (WHAT)
- Python package + Typer CLI in `src/brainlayer/`
- sqlite-vec storage via APSW (`vector_store.py`)
- bge-large-en-v1.5 embeddings (`embeddings.py`)
- BrainBar (Swift) owns the MCP daemon and `/tmp/brainbar.sock`; Python keeps library handlers under `mcp/`
- The FastAPI daemon was removed (`fix(arch): remove FastAPI daemon`, 692839cc) — there is no HTTP API
- Textual TUI (`dashboard/`) and Next.js dashboard
- Source data: JSONL in `~/.claude/projects/`

<!-- COMMANDS: `python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"` | test: `pytest` | lint: `ruff check src/ tests/ && ruff format src/ tests/` | run: `brainlayer index && brainlayer setup` -->
## Workflow (HOW)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
brainlayer index
brainlayer setup
brainlayer search "how did I implement authentication"
brainlayer enrich
```
- Lint/format: `ruff check src/ tests/ && ruff format src/ tests/`
- Pre-push: `.githooks/pre-push` runs `scripts/run_tests.sh` with `BRAINLAYER_PREPUSH=1`; full
  runs are deduped by git tree hash in `.git/brainlayer-prepush-cache`.
- A **tag** push has no branch to diff against, so the hook reads the pushed refs off its stdin and
  scopes the run to `<previous release tag>..<tag>` via `BRAINLAYER_CHANGED_FILES_RANGE`. The
  predecessor is resolved with `--match 'v*'`: an intervening non-release tag would start the range
  short and leave real commits unmapped. It refuses to narrow — and prints the reason in the push
  output — in five cases: a branch rides along; `BRAINLAYER_PREPUSH_SCOPE` was set explicitly
  (**explicit beats default, in both directions** — an explicit `full` is never narrowed, and an
  explicit `changed-only` is not handed a range either); `BRAINLAYER_CHANGED_FILES`/`_RANGE` is
  already set; more than one tag is pushed at once; the tag has no previous `v*` predecessor. In
  all five the scope is left exactly as the caller set it, and none of them are silent.
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
- Rate configurable via `BRAINLAYER_ENRICH_RATE` env var (default 5.0 req/s = 300 RPM; see `enrichment_controller.py:RATE_LIMITS`)
- Adds 15 metadata fields (summary, key_facts, tags, importance, intent, primary_symbols, resolved_queries, epistemic_level, version_scope, debt_impact, external_deps, entities, sentiment_label, sentiment_score, sentiment_signals); session enrichment captures decisions/corrections

<!-- MCP-SERVERS: agent MCP is BrainBar on /tmp/brainbar.sock (brainlayer-mcp-stdio-bridge, or socat STDIO UNIX-CONNECT); brainlayer-mcp Python entrypoint DELETED; no HTTP daemon API; library handlers live under mcp/; 17 tools in MCPRouter.toolDefinitions, core palette = brain_search/brain_store/brain_recall/brain_expand + expand_palette -->
## Interfaces
- **There is no HTTP API.** The FastAPI daemon and its `/health`, `/stats`, `/search`, `/brain/graph`,
  `/backlog/items` routes were removed. Anything still calling them is calling a surface that is gone.
- MCP tools: 17, defined in `brain-bar/Sources/BrainBar/MCPRouter.swift` — see **MCP Tools** above for
  the core/gated split. Legacy `brainlayer_*` names are still handled by the Python library handlers
  under `src/brainlayer/mcp/`, but BrainBar does not serve them.
- `brain_expand` and `brain_tags` are deprecated in the Python MCP path and return errors there; the
  BrainBar path serves both normally.
- MCP server: BrainBar on `/tmp/brainbar.sock` only. Preferred wiring is
  `{"command":"brainlayer-mcp-stdio-bridge"}` (ships with the package, reconnects across BrainBar
  restarts, needs nothing else on PATH); `{"command":"socat","args":["STDIO","UNIX-CONNECT:/tmp/brainbar.sock"]}`
  also works. The Python `brainlayer-mcp` entrypoint is deleted. `brainlayer setup --migrate-mcp`
  rewrites owned legacy configs to the **socat** form and leaves existing bridge entries in place.

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
- Axiom telemetry: startup, flush, error, heartbeat (60s threshold, but it is checked once per poll, so at the 30s poll default the real spacing is ~60-95s — measured 91s; do not alert on a 60s cadence) to `brainlayer-watcher` dataset
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

<!-- PATHS: DB=~/.local/share/brainlayer/brainlayer.db | offsets=~/.local/share/brainlayer/offsets.json | logs=~/Library/Logs/brainlayer/watch.{out,err}.log | socket=/tmp/brainlayer.sock | lock=/tmp/brainlayer-enrichment.lock -->
## Data & Locks
- Backup log: real runs append JSONL to `~/.local/share/brainlayer/logs/backup-daily.log` with
  `backup_log_provenance=real`; pytest sets `BRAINLAYER_BACKUP_LOG_PATH` and
  `BRAINLAYER_BACKUP_LOG_PROVENANCE=pytest` so tests cannot refresh the production heartbeat log.
- Watcher offsets: `~/.local/share/brainlayer/offsets.json`
- Prompts cache: `~/.local/share/brainlayer/prompts/`
- Watcher logs: `~/Library/Logs/brainlayer/watch.{out,err}.log`
  (moved from `~/.local/share/brainlayer/logs/watch.{log,err}` by `3dca26a2`, 2026-05-18;
  the old files are frozen at that date and are NOT a health signal)
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
