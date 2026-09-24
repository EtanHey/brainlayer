# Changelog

## [Unreleased]

### Added
- `brain_expand` MCP tool — expand a chunk_id with N surrounding chunks for full context retrieval
- Groq enrichment backend support (`GROQ_API_KEY`, `BRAINLAYER_GROQ_URL`, `BRAINLAYER_GROQ_MODEL`)
- `[style]` optional extra: ChromaDB vector store as alternative backend
- `faiss-cpu` added to `[brain]` optional extra for fast ANN search
- Deferred embedding in `brain_store` — async embedding pipeline for faster writes (#76)
- Eval suite + entity injection in UserPromptSubmit hook (#72)
- C6 eval coverage expansion — 18 new test cases for search scenarios (#75)
- C7 failed query mining script — automated discovery of low-score searches (#74)
- C1+C2 lifecycle hardening — stale MCP process cleanup + WAL checkpoint on start (#73)
- Post-RRF importance and recency reranking (US-002) — boosts high-importance and recent chunks
- FTS5 expansion to index `summary`, `tags`, `resolved_query` fields (US-001)
- KG entity quality — validation, prompts, and cleanup pipeline (#69)
- KG rebuild pipeline with audit fixes (119 entities, 153K entity-chunk links) (#67)
- Groq rate limiter for enrichment backend (#68)

### Fixed
- `brain_graph.py`: replaced correlated subquery with literal `'claude_code'` — eliminates N+1 subquery on large DBs (PR #62)
- Removed `tree-sitter>=0.21.0` from core dependencies
- `brainlayer serve` docs: removed non-existent `--http` flag
- `brain_expand` for manual chunk IDs — fixed lookup path (#78)
- Consolidated DB paths — single canonical `~/.local/share/brainlayer/brainlayer.db` across all scripts (#77)
- `brain_search` now returns `chunk_id` in results (#66)
- DB lock resilience — retry + queue for MCP writes and reads (#65)
- `format` parameter renamed to avoid Python builtin shadowing (BUG-005)
- Entity merge in cleanup script — UNIQUE constraint violation (#a125d68)
- Search validation hardened with backfill coverage (a108aff)

### Changed
- MCP tool count updated from 7 to 8 (brain_expand added)
- Brain graph clustering documented as Leiden + UMAP (was incorrectly listed as HDBSCAN + UMAP)
- Test count: 698 → 929 (after eval suite expansion and search hardening)
- `[kg]` optional extra documented in README and Optional Extras section
- Search: hybrid RRF reranking now incorporates importance and recency signals
- Embeddings: deferred from synchronous to async pipeline

## [1.5.39] - 2026-09-24

- #940: BrainBar's Jobs panel shows each job's real status, last run and next run, with a one-line cause on every "Needs attention" badge; a launchd counter reset reads "Awaiting next run".
- #941: BrainBar reports FTS5 and Trigram coverage against the indexes search actually routes to, and shows vector drain as "Not measured" instead of an estimate borrowed from enrichment.
- #943: When observability is stale, BrainBar keeps the last measured counts, labelled with their measurement time, and shows paused enrichment as "Enrichment paused · N queued".
- #944: Remove BrainBar's Search, Quick Capture and Knowledge Graph UI; MCP search and store are unchanged.
- #945: The Hotlane toggle in BrainBar Settings now actually enables and disables the `com.brainlayer.hotlane-brainbar` job.
- #946: Document which launchd job the Hotlane toggle controls.
- #947: Hotlane keeps its backlog cursor when a write hits a busy or locked database, and its schedule allows a 16-vector batch every 7 seconds.
- #948: Database backups ask macOS to purge purgeable space before the snapshot when raw free space is short, and refuse before writing if it still is not enough.
- #949: Remove the code left unused by the retired BrainBar Search, Quick Capture and Knowledge Graph UI.
- #950: The unit test suite can no longer reach the production BrainBar socket.
- #951: Database backups record a launchd stop and a lock refusal as themselves, and wait up to 30 seconds for purged space to be released.
- #952: A `brain_store` made while a backup snapshot is running is queued and saved once the backup finishes (`STORED (deferred)`), instead of failing.
- #953: Remove personal-data exports from the repository, and add a CI guard against adding them back.
- #954: A Drive backup upload that finished is no longer recorded as failed when the connection resets afterwards; interrupted uploads ask Drive for its offset before re-sending.
- #957: The embedding hotlane runs at Standard priority (Nice 10) instead of Background, which held embedding at scheduler priority 4; the hotlane plist is re-rendered only by `install.sh hotlane-brainbar`, not by a formula upgrade.

## [1.5.38] - 2026-09-23

- #939: Make the database backup's free-space check measure macOS available capacity with a protected raw floor, and give transcript backups a 2-hour limit.

## [1.5.37] - 2026-09-23

- #905: Quarantine the #891 socket-notification CI flake behind an opt-in test.
- #927: Keep long BrainBar Details facts visible in flat rows.
- #932: Add semantic BrainBar status tokens and enforce the 9 pt text floor.
- #934: Stabilize the BrainBar panel frame and scroll anchor across Details toggles.
- #936: Keep BrainBar dashboard card shapes steady across data states.
- #937: Open BrainBar Dashboard and Settings through URL routes.

## [1.5.36] - 2026-09-23

- #909: Expose backup import dependency failures in observability diagnostics.
- #910: Respect deliberately paused launchd jobs in health checks.
- #911: Accept an empty corpus in the canary health check.
- #912: Use the query-embedding interface for digest passage retrieval.
- #913: Keep BrainBar ingest charts readable at rest.
- #914: Declare required runtime dependencies and gate installed module and hook imports.
- #915: Contain inherited Git repository variables in hook tests.
- #916: Unify Settings inside the BrainBar panel.
- #917: Isolate Git reads when maintenance classifies installed code.
- #918: Preserve the SQLite error when an index dedupe savepoint disappears.
- #919: Retain BrainBar coverage fills while toggling Details.
- #920: Add the BrainBar Settings sidebar and section layout.
- #921: Restart loaded BrainLayer jobs after upgrade with verification.
- #922: Cap installed BrainLayer job logs.
- #923: Align BrainBar badge paths and confirm API-key replacement.
- #924: Report stalled drain progress without hiding a stale heartbeat.
- #925: Show measured search and store receipts in BrainBar.
- #926: Disclose Settings locality and model residency in BrainBar.
- #928: Harden BrainBar's first-run badge and API-key confirmation.
- #929: Self-heal failed BrainLayer jobs with badge escalation.
- #930: Preserve self-heal state across uncertain health checks.
- #931: Present honest signal-coverage counts in BrainBar.
- #933: Verify installed keg native-library loadability alongside signatures.

## [1.0.0] - 2026-02-19

### Added
- Initial open-source release as BrainLayer (formerly Zikaron)
- Semantic search across AI conversation history (sqlite-vec + bge-large-en-v1.5)
- 10-field LLM enrichment pipeline (Ollama / MLX backends)
- Brain graph visualization (HDBSCAN clustering + UMAP 3D layout)
- MCP server with 7 tools (+ 14 backward-compatible aliases) for Claude Code, Zed, Cursor
- Interactive setup wizard (`brainlayer init`)
- Centralized artifact storage (`~/.local/share/brainlayer/storage/`)
- Multi-source indexing: Claude Code, WhatsApp, YouTube, Markdown, Claude Desktop
- Communication style analysis pipeline
- Obsidian vault export
- FastAPI daemon with 25+ HTTP endpoints
- GitHub Actions CI/CD with PyPI publishing
- PII sanitization pipeline for safe cloud processing
- Source-aware enrichment thresholds
