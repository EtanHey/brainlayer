# Roadmap

Planned features and improvements for BrainLayer.

## Near-Term

### Boot Context Loading
Load relevant memories automatically when a new Claude Code session starts, replacing manual CLAUDE.md context with dynamically retrieved knowledge.

### Compact Search
Optimized search mode that returns condensed results for context-constrained environments (e.g., during conversation compaction).

### Pinned Memories
Allow users to pin critical memories (architectural decisions, conventions, preferences) so they always surface in context retrieval regardless of age or decay.

## Medium-Term

### MCP Registry Listing
Publish BrainLayer to the official [MCP Registry](https://registry.modelcontextprotocol.io) for one-click installation.

### Demo / Showcase
Terminal recording (VHS) demonstrating search, store, recall, and entity lookup in a real workflow.

### Architecture Decision Records
Formalize key design decisions (sqlite-vec, RRF scoring, enrichment pipeline) as ADRs in `docs/adr/`.

## Phase Plans

Detailed implementation plans for specific features:

- [Phase 3: Brain Digest](plans/2026-02-25-phase-3-brain-digest.md) — entity extraction, relations, sentiment analysis
- [Phase 6: Sentiment Analysis](plans/2026-02-25-phase-6-sentiment.md) — communication style and sentiment pipeline
