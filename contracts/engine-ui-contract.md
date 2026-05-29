# Engine UI Contract

Phase 0 contract-only boundary for the future BrainLayer package split. This pins the live engine UI surfaces without moving modules or extracting packages.

## 1. Shared SQLite Schema

The canonical database path is `~/.local/share/brainlayer/brainlayer.db`, overridden by `BRAINLAYER_DB` in Python (`src/brainlayer/paths.py`). BrainBar and Python both create and migrate shared tables, so schema changes must stay dual-owned.

Pinned shared surfaces:

- `chunks`: Python creates the base table at `src/brainlayer/vector_store.py:544` and adds enrichment columns including `resolved_query`, `key_facts`, and `resolved_queries` at `src/brainlayer/vector_store.py:606`. BrainBar creates the same live columns at `brain-bar/Sources/BrainBar/BrainDatabase.swift:369` and upgrades existing DBs at `brain-bar/Sources/BrainBar/BrainDatabase.swift:2257`.
- `chunks_fts`: the column contract is `content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id UNINDEXED`. Python pins it at `src/brainlayer/vector_store.py:815`; BrainBar pins it at `brain-bar/Sources/BrainBar/BrainDatabase.swift:25`.
- `chunks_fts` rebuild and triggers: Python drops/recreates old FTS schemas at `src/brainlayer/vector_store.py:821` and installs 7-column insert/update triggers at `src/brainlayer/vector_store.py:866` and `src/brainlayer/vector_store.py:922`. BrainBar drops/recreates old FTS schemas at `brain-bar/Sources/BrainBar/BrainDatabase.swift:2503`, repopulates at `brain-bar/Sources/BrainBar/BrainDatabase.swift:2520`, and installs 7-column insert/update triggers at `brain-bar/Sources/BrainBar/BrainDatabase.swift:2526` and `brain-bar/Sources/BrainBar/BrainDatabase.swift:2541`.
- `chunks_fts_trigram`: Python creates the 7-column trigram table at `src/brainlayer/vector_store.py:840` and installs its 7-column triggers at `src/brainlayer/vector_store.py:884` and `src/brainlayer/vector_store.py:943`. BrainBar creates/rebuilds it at `brain-bar/Sources/BrainBar/BrainDatabase.swift:2562`, installs 7-column triggers at `brain-bar/Sources/BrainBar/BrainDatabase.swift:2572` and `brain-bar/Sources/BrainBar/BrainDatabase.swift:2587`, and repopulates full/table batches at `brain-bar/Sources/BrainBar/BrainDatabase.swift:2596` and `brain-bar/Sources/BrainBar/BrainDatabase.swift:2648`.
- `kg_*`: BrainBar owns `kg_entities`, `kg_relations`, `kg_entity_chunks`, and `kg_entity_aliases` creation at `brain-bar/Sources/BrainBar/BrainDatabase.swift:456`, `brain-bar/Sources/BrainBar/BrainDatabase.swift:471`, `brain-bar/Sources/BrainBar/BrainDatabase.swift:484`, and `brain-bar/Sources/BrainBar/BrainDatabase.swift:493`. Python owns the matching KG schema at `src/brainlayer/vector_store.py:1294`, `src/brainlayer/vector_store.py:1312`, `src/brainlayer/vector_store.py:1332`, and `src/brainlayer/vector_store.py:1379`.
- Atomic migration marker: both owners use `atomic_brick_chunks_v1`; Python checks/inserts it at `src/brainlayer/vector_store.py:579` and `src/brainlayer/vector_store.py:758`, while BrainBar checks/inserts it at `brain-bar/Sources/BrainBar/BrainDatabase.swift:2255` and `brain-bar/Sources/BrainBar/BrainDatabase.swift:2373`.

## 2. BrainBar MCP Router Over `/tmp/brainbar.sock`

BrainBar is the MCP surface of record. Its router dispatches 17 tools at `brain-bar/Sources/BrainBar/MCPRouter.swift:195` and defines the tool schemas at `brain-bar/Sources/BrainBar/MCPRouter.swift:950`.

Canonical BrainBar tools:

`brain_search`, `brain_store`, `brain_get_person`, `brain_recall`, `brain_entity`, `brain_digest`, `brain_update`, `brain_expand`, `brain_tags`, `brain_supersede`, `brain_archive`, `brain_enrich`, `brain_subscribe`, `brain_unsubscribe`, `brain_ack`, `brain_maintenance_rebuild_trigram`, `brain_backup_vacuum_into`.

Python `src/brainlayer/mcp/` remains the secondary transport with 13 tools defined at `src/brainlayer/mcp/__init__.py:387`, `src/brainlayer/mcp/__init__.py:543`, `src/brainlayer/mcp/__init__.py:568`, `src/brainlayer/mcp/__init__.py:672`, `src/brainlayer/mcp/__init__.py:701`, `src/brainlayer/mcp/__init__.py:862`, `src/brainlayer/mcp/__init__.py:905`, `src/brainlayer/mcp/__init__.py:974`, `src/brainlayer/mcp/__init__.py:999`, `src/brainlayer/mcp/__init__.py:1042`, `src/brainlayer/mcp/__init__.py:1080`, `src/brainlayer/mcp/__init__.py:1113`, and `src/brainlayer/mcp/__init__.py:1135`.

Differences:

- BrainBar-only: `brain_subscribe`, `brain_unsubscribe`, `brain_ack`, `brain_maintenance_rebuild_trigram`, `brain_backup_vacuum_into`.
- Python-only: `brain_resume`.

## 3. Hybrid Helper Subprocess And Socket

BrainBar starts the helper from `HybridSearchHelperClient` with `-m brainlayer.brainbar_hybrid_helper --socket-path ... --db-path ...`; see the invocation at `brain-bar/Sources/BrainBar/HybridSearchHelperClient.swift:189`. If `pythonExecutable` resolves to `/usr/bin/env`, BrainBar invokes `python3`; otherwise it invokes the resolved Python executable directly.

Environment contract:

- `BRAINBAR_PYTHON` overrides Python executable resolution at `brain-bar/Sources/BrainBar/HybridSearchHelperClient.swift:53`.
- Installed package imports are the default. `BRAINLAYER_REPO_ROOT` is used to resolve `<repo>/src` for `PYTHONPATH` only when `BRAINLAYER_SOURCE_FALLBACK=1`, preserving a deliberate source-tree fallback at `brain-bar/Sources/BrainBar/HybridSearchHelperClient.swift:78`.
- `PYTHONPATH` priority: if `PYTHONPATH` is already set in the environment, it is preserved and takes precedence over the fallback and default installed-package behavior. Order of precedence: (1) existing `PYTHONPATH`, (2) `BRAINLAYER_SOURCE_FALLBACK=1` resolves `<repo>/src`, (3) nil uses the installed package.
- `PYTHONPATH` is passed through or set before launch at `brain-bar/Sources/BrainBar/HybridSearchHelperClient.swift:213`.
- `BRAINLAYER_DB` is set to the selected DB path at `brain-bar/Sources/BrainBar/HybridSearchHelperClient.swift:214` and by the helper at `src/brainlayer/brainbar_hybrid_helper.py:258`.

Helper CLI and wire contract:

- CLI args: `--socket-path` is required and `--db-path` is optional at `src/brainlayer/brainbar_hybrid_helper.py:246`.
- The helper binds a Unix socket, chmods it `0600`, listens, and warms search at `src/brainlayer/brainbar_hybrid_helper.py:85`.
- Request envelope: one NDJSON object with `{ "method": "brain_search", "arguments": { ... } }`. Unsupported methods are rejected at `src/brainlayer/brainbar_hybrid_helper.py:160`.
- Response envelope: `{ "ok": true, "text": "...", "metadata": { ... } }`; `isError` is present only when the underlying MCP result reports an error at `src/brainlayer/brainbar_hybrid_helper.py:166`.
- Accepted argument keys are read in `src/brainlayer/brainbar_hybrid_helper.py:178`: `_profile_query_id`, `source`, `query`, `project`, `tag`, `importance_min`, `agent_id`, `num_results`, `max_results`, and `detail`.

## 4. Reverse Socket Backup

Python daily backup calls BrainBar over `/tmp/brainbar.sock` instead of copying the live DB directly. The default socket path is `src/brainlayer/backup_daily.py:33`; the request uses MCP `tools/call` with `brain_backup_vacuum_into` and `target_path` at `src/brainlayer/backup_daily.py:119`. BrainBar exposes that tool schema at `brain-bar/Sources/BrainBar/MCPRouter.swift:1157` and dispatches it at `brain-bar/Sources/BrainBar/MCPRouter.swift:229`.

## 5. Brain Bus Events

Brain bus is a BrainBar socket stream, not a Python MCP tool. Clients send method `watch-brain-bus` at `brain-bar/Sources/BrainBar/BrainBusClient.swift:123`; the server handles it at `brain-bar/Sources/BrainBar/BrainBarServer.swift:482` and emits JSON-RPC notifications with method `notifications/brain-bus` at `brain-bar/Sources/BrainBar/BrainBarServer.swift:986`.

Pinned event types and JSON keys come from `brain-bar/Sources/BrainBar/BrainBusEvent.swift:3` and `brain-bar/Sources/BrainBar/BrainBusEvent.swift:21`:

- Types: `queue_depth`, `enrich_status`, `last_chunk_id`, `db_busy`, `health_tick`.
- Keys: `type`, `sequence`, `generated_at`, `queue_depth`, `enrich_status`, `last_chunk_id`, `db_busy`, `open_connections`.

## 6. `injection_events` Shared Table

Hooks write this table; BrainBar reads it for the live viewer.

- BrainBar creates `injection_events` at `brain-bar/Sources/BrainBar/BrainDatabase.swift:520`, writes local events at `brain-bar/Sources/BrainBar/BrainDatabase.swift:4678`, and reads live events at `brain-bar/Sources/BrainBar/BrainDatabase.swift:4699`.
- The prompt hook ensures hook-specific columns and inserts rows at `hooks/brainlayer-prompt-search.py:905` and `hooks/brainlayer-prompt-search.py:940`.

## 7. Launch Mode

Launch mode is already pinned by `brain-bar/Tests/BrainBarDaemonTests/BrainBarDaemonLaunchModeTests.swift:4`. The contract is the persisted `brainbar.launchMode` key plus `BRAINBAR_LAUNCH_MODE` raw values `app-window` and `menu-item-daemon`, verified at `brain-bar/Tests/BrainBarDaemonTests/BrainBarDaemonLaunchModeTests.swift:5`, `brain-bar/Tests/BrainBarDaemonTests/BrainBarDaemonLaunchModeTests.swift:41`, and `brain-bar/Tests/BrainBarDaemonTests/BrainBarDaemonLaunchModeTests.swift:54`.
