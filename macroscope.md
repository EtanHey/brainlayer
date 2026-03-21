# Macroscope — BrainLayer Code Review Rules

Rules checked by Macroscope during PR reviews. Organized by category.

---

## Security

- **No hardcoded API keys.** `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, and similar must come from environment variables or 1Password — never inline in source.
- **BrainBar stub tools return fake success.** `brain_digest`, `brain_update`, `brain_expand`, and `brain_tags` are broken stubs. Never trust their output in tests or production code. Only `brain_search`, `brain_store`, `brain_recall`, and `brain_entity` are functional.

## Architecture

- **SQLite WAL mode is mandatory.** All connections must set `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout` >= 3000ms. This prevents writer starvation and SQLITE_BUSY errors under concurrent access.
- **FTS5 and vector embeddings must stay in sync.** After any schema change, bulk delete, or bulk insert, rebuild the FTS index. Drop FTS triggers before bulk deletes (the `chunks_fts_delete` trigger kills performance on large datasets) and recreate them after.
- **New MCP tools must be registered in two places.** Add the handler in `MCPRouter.swift` AND advertise the tool in server capabilities. Missing either causes silent failures.
- **Enforce content_hash uniqueness.** The `content_hash` column has a UNIQUE index. All inserts must use `INSERT OR IGNORE` (or equivalent) to respect this constraint — duplicate inserts must not raise errors.
- **Pub/sub cursors use rowid, not timestamps.** Subscriptions track position with rowid-based cursors. Timestamp-based cursors drift under concurrent writes and batch inserts.

## Testing

- **Swift tests:** Run with `swift test --package-path brain-bar`. All BrainBar changes must pass before merge.
- **Python enrichment dry-run:** Enrichment scripts must support a `--test` flag for dry-run validation. Never merge an enrichment script that can only be tested against production data.
- **Every new brain_* MCP tool needs a test.** At minimum one `MCPRouterTest` covering the happy path. Stub tools still need tests proving they return the expected (stub) response.
- **Use temporary in-memory DBs for integration tests.** Never point tests at the production database (`~/.local/share/brainlayer/brainlayer.db`). Use `:memory:` or a temp file that is cleaned up after the run.

## Style

- **Python CLI: use argparse.** All CLI parameters must be declared via `argparse` (or Typer/Click decorators) — no hardcoded paths or magic constants buried in `main()`.
- **Swift: no untyped Any.** Avoid `Any` unless explicitly unavoidable. When `Any` is required (e.g., JSON parsing boundaries), add a comment justifying why a concrete type is not possible.
- **Tags are JSON arrays.** Store tags as `["tag1", "tag2"]`, never as comma-separated strings. All code that reads or writes tags must expect array format.
