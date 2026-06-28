# MODULE-MAP — BrainLayer engine/package split (Phase 0, contract-only)

> Author: brainlayerClaude-LEAD-v4, 2026-05-29. **v2 — corrected after a 3-lens critique** (boundary / contract-completeness / migration-risk; critiques stored this session). Supersedes MODULE-MAP-draft.md. Status: contract-only (Etan-agreed); no physical extraction this turn.

## ⚠️ Corrections the critique forced (the draft was wrong on these — read first)
1. **The live MCP server is BrainBar (Swift), NOT the Python `mcp/`.** `.mcp.json` connects agents via `brainlayer-mcp-stdio-bridge` → the configured BrainBar socket (`/tmp/brainbar.sock` by default; BrainBar listens on `BRAINBAR_SOCKET_PATH`, and the bridge dials `BRAINLAYER_MCP_SOCKET`) → `brain-bar/Sources/BrainBar/MCPRouter.swift` (17 tools). For a non-default socket, set both env vars to the same path. Python `src/brainlayer/mcp/` (13 tools, incl. `brain_resume`; entry `brainlayer-mcp`) is the SECONDARY/legacy transport. The contract pins **BrainBar's MCPRouter** as the surface of record while the stdio bridge keeps agent transports alive across socket replacement.
2. **The DB schema is DUAL-OWNED and ALREADY DRIFTING.** BrainBar uses **raw SQLite3 (not GRDB)** and independently CREATEs + migrates `chunks`, `chunks_fts`, `kg_*`, triggers — under the *same* migration name `atomic_brick_chunks_v1` as Python. Live drift: `chunks_fts` = **5 cols in Swift vs 7 in Python** (`key_facts`, `resolved_queries` Python-only). This is the single highest-risk surface and must be pinned.
3. **The launch-mode contract is ALREADY pinned** — PR #361 (`BrainBarDaemonLaunchModeTests.swift`, HEAD d10bcaf6) shipped it. NOT a TODO. The real contract = UserDefaults `brainbar.launchMode` + env `BRAINBAR_LAUNCH_MODE` (rawValues `app-window`/`menu-item-daemon`); the enum lives only in the Swift target (no separate Python parser).
4. **FABRICATION CORRECTED:** the draft called root `brainlayer.db` a "stray 9.6GB copy" — it is **0 bytes** (tracked-but-should-be-ignored). The 9.6GB is the real DB at `~/.local/share/brainlayer/`. (`/never-fabricate` miss — owned.)

## Three packages (shape CONFIRMED sound; specifics corrected)

### 1. `brain-engine` (Python) — write path, enrichment, search, KG, secondary MCP transport
`src/brainlayer/` core: write path (`vector_store`, `store`, `storage`, `drain`, `queue_io`, `queue_merge`, `dedupe`), search (`search_repo`, `search_profile`, `embeddings`, `engine`, `clustering`), enrichment (`enrichment_controller`, `cloud_backfill`, `calibrate`, `decay*`), KG (`kg_repo`, `kg_promotion`, `kg/`), ingest/classify (`classify`, `chunk_origin`, `ingest_guard`, `lexical_defense`, `phonetic`, `ingest/`, `pipeline/`), watcher (`watcher`, `watcher_bridge`), support (`types`, `_helpers`, `config`, `paths`, `claude_paths`, `scoping`, `telemetry`, `migrate`, `session_repo`, `agent_profiles`, `index_new`, `git_learning`, `parent_death`), `mcp/` (secondary transport), `eval/`, `data/`, `migrations/`. **Also engine (critique fix):** `src/brainlayer/hooks/indexer.py` (RealtimeIndexer), `brainbar_hybrid_helper.py` (the bridge — see contract).
- **Boundary rule (Decision A — orc lean 2026-05-29, HELD pending Etan confirm):** brain-engine = **PURE LIBRARY**. The CLI (`cli/`, `cli_new.py`) + Textual TUI (`dashboard/`) move to **brainlayer-root** (or a thin surface pkg) — mirroring the BrainBar UI-separation logic. **Physical move DEFERRED** (contract-only turn); recorded as the intended boundary.

### 2. `BrainBar` (Swift) — standalone package `brain-bar/`, AND the live MCP server + a co-owner of the DB schema
Self-contained Swift (`BrainBar`, `BrainBarDaemon`, `BrainBarLifecycle`). Owns the configured BrainBar MCP socket (`/tmp/brainbar.sock` by default), the brain-bus event stream, the offline pending-stores write buffer, and its own copy of the schema DDL/migrations. **Not a read-only consumer** — a co-writer.

### 3. `brainlayer-root` (orchestration/packaging/docs)
`launchd/`, `scripts/`, `.githooks/`, `.github/`, top-level `hooks/` (stdlib-only Claude Code hooks — EXCEPT `brainlayer-prompt-search.py`, which imports the engine → either relocate to engine or root-depends-on-engine), packaging (`pyproject.toml`, `uv.lock`), config (`.mcp.json`, `server.json`, `greptile.json`, `macroscope.md`), docs (`README`/`CHANGELOG`/`CONTRIBUTING`/`LICENSE`/`AGENTS`/`CLAUDE`/`GEMINI`.md, `docs/`, `mkdocs.yml`, `overrides/`, `site/`, `landing/`, `extensions/`), `CODEOWNERS`.
- **De-dup:** `scripts/wal_checkpoint.py` must be a thin wrapper over the engine's `wal_checkpoint.py` (one canonical home), not a second copy.
- **CLEANUP (gitignore/remove, not a package):** root `brainlayer.db` (0-byte, `git rm --cached`), `dist/`, `.perf/`, caches, `prd-json/`, `grill*/`, `progress.txt`, `.tmp-*`, `2026-*.txt`, and the **~15 `BUGBOT_*.md`/`findings.md`** pile → `docs/reviews/` or delete.

## The engine↔UI contract (CORRECTED — ~7 surfaces, not 4)
1. **Shared SQLite file + schema DDL/migrations** (dual-owned; pin the column sets + the `atomic_brick_chunks_v1` migration; FIX the `chunks_fts` 5-vs-7 drift). DB path `~/.local/share/brainlayer/brainlayer.db` (+ `BRAINLAYER_DB`).
2. **BrainBar MCPRouter tool surface** over the configured BrainBar socket (17 tools) — the surface of record; reconcile vs Python's 13.
3. **Hybrid-helper subprocess + socket** (BrainBar → Python): invocation `python3 -m brainlayer.brainbar_hybrid_helper --socket-path … --db-path …` with env triple `BRAINLAYER_REPO_ROOT` / `PYTHONPATH=<root>/src` / `BRAINBAR_PYTHON`; NDJSON `{method,arguments}`→`{ok,text,metadata}` (arg keys enumerated in `brainbar_hybrid_helper.py`). **Undocumented today → needs a golden-fixture contract test.**
4. **Reverse socket** `/tmp/brainbar.sock` (Python `backup_daily.py` → BrainBar `vacuum_into`).
5. **brain-bus event stream** (`notifications/brain-bus`, method `watch-brain-bus`): `queue_depth`/`enrich_status`/`last_chunk_id`/`db_busy`/`health_tick` (PR #360 consumer).
6. **`injection_events` shared table** — written by `hooks/brainlayer-prompt-search.py`, read by BrainBar.
7. **Launch-mode** (already pinned, PR #361) — keep as a contract test, don't re-author.

## Migration risk = HIGH (for the LATER physical extraction)
Driver: BrainBar hard-binds to the engine's **source-tree layout** (`<repoRoot>/src` via subprocess), and ~30 scripts + 7 launchd plists hardcode `PYTHONPATH=.../src`. Extraction breaks the UI's search + every daemon unless done atomically.
**Guardrails the contract-only turn locks NOW:** (a) lock dist name = `brainlayer` (invariant); (b) pin the hybrid-helper invocation+wire contract in `contracts/` with a fixture test; (c) record MCP-server-of-record = BrainBar; (d) extend `tests/test_launchd_hygiene.py` + `tests/test_brainbar_build_app_guards.py` to assert no DEV/worktree `/src` path leaks into the canonical LaunchAgent env (defeats the "competing trees" hazard — the binary-flashing is already guarded by `build-app.sh` DEV-bundle/canonical rules). Safe extraction sequence: keep name → swap all `PYTHONPATH=/src` to a real editable install in ONE change (incl. Swift resolver prefers installed pkg, `/src` fallback) → pin helper protocol → extract → verify all daemons + 5 contract tests.

## Open questions for agreement
- **A.** CLI/TUI (`cli/`, `cli_new.py`, `dashboard/`) → in brain-engine or root? **RESOLVED (provisional, orc 2026-05-29):** engine = pure library → CLI/TUI to ROOT. Physical move HELD pending Etan final confirm. Does not affect the contract-only turn.
- **B.** Fix the `chunks_fts` 5-vs-7 drift now (small PR) or fold into the extraction? (Recommend: now — it's a live data-integrity risk.)
- **C.** Reconcile the 17(Swift)/13(Python) MCP tool sets — which is canonical, what retires?

## Phase-0 close = contract-only deliverables
1. This `MODULE-MAP.md` (committed to brainlayer repo, after A/B/C agreed).
2. `contracts/` gains: the hybrid-helper invocation+wire spec, the MCP-server-of-record note, the schema column-set + `chunks_fts` drift fix.
3. Contract tests: hybrid-helper golden-fixture; launchd-hygiene no-leak; (launch-mode already covered by PR #361).
4. NO physical extraction this turn.
