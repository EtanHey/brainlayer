# BrainLayer Agent Notes

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
- Request `@cursor` and `@bugbot` review.

## Known Issues
- DB locking during enrichment.
- WAL can grow to 4.7GB.
