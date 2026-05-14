# BrainLayer Writer Arbitration

BrainLayer uses a single-writer arbitration path for background producers that can otherwise fight over SQLite's write lock.

## Operator Contract

- `BRAINLAYER_ARBITRATED=1` makes producers enqueue writes instead of writing directly to the database. The current launchd templates set this for watch and enrichment.
- `BRAINLAYER_DRAIN_EMBED=0` disables post-drain embedding, mainly for tests or emergency operator debugging. Production should leave it enabled so queued `brain_store` chunks reach semantic search.
- The unified durable queue lives at `~/.brainlayer/queue/`. Each event is one JSONL file named by source, timestamp, and UUID.
- `com.brainlayer.drain.plist` runs `scripts/drain_daemon.py`, which drains the queue every 500ms under `BEGIN IMMEDIATE`.
- The drain daemon opens SQLite with APSW, loads `sqlite-vec`, writes chunks/enrichment updates, and embeds queued `brain_store` chunks into `chunk_vectors`/`chunk_vectors_binary` before removing their queue files.
- Legacy `pending-stores.jsonl` is migrated by `brainlayer flush`; migration assigns stable chunk IDs so rerunning after a crash is `INSERT OR IGNORE` safe.

## FTS Repair

- Startup no longer performs large synchronous trigram repairs.
- Run `brainlayer repair-fts` for an explicit `chunks_fts_trigram` rebuild.
- `scripts/launchd/com.brainlayer.repair-fts.plist` schedules that repair weekly.
- Set `BRAINLAYER_REPAIR=1` only for an operator-controlled process that should run the repair during VectorStore initialization.
