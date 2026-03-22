# ADR-0001: sqlite-vec over a dedicated vector database

**Status:** Accepted

**Date:** 2025-12 (approximate — decision predates ADR documentation)

**Deciders:** EtanHey

## Context

BrainLayer needs a vector store to hold 1024-dimensional embeddings (bge-large-en-v1.5) and support nearest-neighbour search for semantic retrieval. The main candidates were:

| Option | Deployment | Pros | Cons |
|--------|-----------|------|------|
| **pgvector** (Postgres) | Separate service | Mature, ACID, rich SQL | Requires a running Postgres instance |
| **ChromaDB** | Embedded or client/server | Python-native, familiar API | Separate process, own storage format |
| **Pinecone / Weaviate** | Cloud SaaS | Managed, scalable | Cloud dependency, latency, cost, privacy |
| **sqlite-vec** (via APSW) | Embedded, single file | Zero-service, portable, WAL concurrency | Brute-force KNN, limited scale |

BrainLayer's design goals are:

1. **Local-first, zero cloud dependencies** — users must be able to run entirely offline.
2. **Single-file portability** — the entire knowledge base should be one file you can copy, back up, or move.
3. **Multi-process safety** — the MCP server, enrichment workers, and CLI must access the same DB concurrently.
4. **No Docker or service orchestration** — `pip install brainlayer` should be the only setup step.

## Decision

Use **sqlite-vec** as the vector store, accessed through **APSW** (Another Python SQLite Wrapper) instead of the stdlib `sqlite3` module.

Key implementation details:

- **APSW** is required because sqlite-vec is loaded as a runtime extension (`sqlite_vec.load(conn)`), which the stdlib `sqlite3` module does not support reliably across platforms.
- **WAL mode** is set on every connection (`PRAGMA journal_mode=WAL`) for concurrent readers.
- **`busy_timeout = 30000ms`** is set before any other APSW hooks fire (including `bestpractice.recommended`) to prevent `BusyError` during `PRAGMA optimize` under contention.
- Embeddings are stored in a dedicated `chunk_vectors` virtual table, separate from the `chunks` metadata table.
- Full-text search uses a separate **FTS5** virtual table (`chunks_fts`), kept in sync via triggers.

## Consequences

### Positive

- **Zero infrastructure** — no database server to install, configure, or keep running. `pip install brainlayer` is the complete setup.
- **Single-file backup** — the entire knowledge base (268K+ chunks, vectors, FTS index, KG data) lives in one `~/.local/share/brainlayer/brainlayer.db` file.
- **WAL concurrency** — multiple processes (MCP server, enrichment workers, CLI) can read simultaneously. Writes serialize naturally via SQLite's locking.
- **APSW advantages** — reliable extension loading, connection hooks for busy_timeout ordering, `bestpractice` module for automatic `PRAGMA optimize`.
- **Portability** — works on macOS, Linux, and Windows without platform-specific setup.

### Negative

- **Brute-force KNN at scale** — sqlite-vec performs exact nearest-neighbour search, not approximate. At 268K chunks (current scale) search completes within hundreds of milliseconds. At 1M+ chunks this will degrade and may require migration to an ANN index or sharding strategy.
- **Single-writer bottleneck** — SQLite allows only one writer at a time. Enrichment workers that hold the write lock for extended periods can block MCP writes, requiring the pending-queue pattern (`pending-stores.jsonl`) as a workaround.
- **WAL growth** — under heavy write load (e.g., bulk enrichment), the WAL file can grow to several GB. Manual `PRAGMA wal_checkpoint(FULL)` is needed after bulk operations.
- **APSW dependency** — APSW is less commonly used than the stdlib `sqlite3` module. Contributors need to understand APSW's connection hook system and its differences from the stdlib.

### Neutral

- The choice does not preclude adding an optional pgvector or ChromaDB backend later. The `VectorStore` class could be refactored behind an interface if scaling demands it.
- FTS5 triggers add complexity (7 triggers for keeping `chunks_fts` and `chunk_tags` in sync) but are necessary for hybrid search correctness.
