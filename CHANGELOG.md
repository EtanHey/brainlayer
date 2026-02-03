# Changelog

All notable changes to Zikaron (זיכרון) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Phase 1-2: Performance Revolution** (Feb 2026)
  - **sqlite-vec** replaces ChromaDB: 12x faster cold start (180s → 15s)
  - **bge-large-en-v1.5** (1024 dims) replaces nomic-embed-text (768 dims): better search quality, 63.5 MTEB score
  - **sentence-transformers** replaces Ollama: direct model loading, no network overhead
  - **FastAPI daemon service**: pre-loaded models, <2s warm queries
  - **APSW** for macOS: SQLite extension support (enable_load_extension)
  - New commands: `search-fast`, `stats-fast`, `index-fast`, `dashboard`, `migrate`
  - Migration tool: `zikaron migrate` converts ChromaDB → sqlite-vec
  - Unix socket communication for instant daemon queries

- **YouTube comments** – Load from Google Takeout or `data/youtube-comments/comments.csv`
- **Casual-style instructions** – Generated `claude-ai-casual-instructions.md` and `gemini-casual-instructions.md` for Claude.ai and Gemini personalization (copy-paste into Settings)
- **Archive script** – `scripts/archive-style-analysis.sh` to archive analysis runs from `/tmp` to `data/archives/`
- **Takeout cleanup** – `scripts/extract-youtube-and-cleanup.sh` extracts YouTube comments and deletes Takeout zips to free space
- **Longitudinal analysis** (`zikaron analyze-evolution`) – Half-year batches, LLM style reports, evolution analysis
- **Chat-based analysis** – Relationship tagging via `chat-tags.yaml`
- **Embedding-enhanced sampling** – StyleDistance for style-aware clustering (`--use-embeddings`)
- **Unified timeline** – WhatsApp, Claude, Gemini, YouTube sources
- **`list-chats`** – List unique chats for tagging

### Changed

- ChromaDB: batched deletes for collections >5,461 docs (avoids "Cannot submit more than 5,461 embeddings at once")
- Progress display: fixed `[15/14]` → correct `[15/16]`, `[16/16]` for evolution + master guide steps
- Gemini parser: support Google Takeout schema (header, subtitles, title); filter for Gemini/AI Mode/Assistant
- CLI: iterates all Takeout zips for Gemini/YouTube; loads from `data/youtube-comments/` when zips removed

### Fixed

- ChromaDB batch limit on `clear_style_collection`
- Progress counter for post-batch steps (evolution, master guide)

### Documentation

- `docs/analysis-archives.md` – How to archive runs
- `docs/google-takeout-cleanup.md` – Extract YouTube, delete zips
- `docs/github-issues-draft.md` – Feature ideas for manual issue creation

---

## [0.1.0]

### Added

- Claude Code conversation indexing from `~/.claude/projects/`
- ChromaDB vector storage with nomic-embed-text via Ollama
- Extract → Classify → Chunk → Embed → Index pipeline
- AST-aware code chunking (tree-sitter)
- MCP server (`zikaron-mcp`) for Claude integration
- CLI: `index`, `search`, `stats`, `clear`, `fix-projects`, `serve`
- `analyze-style` for WhatsApp/Claude style extraction

[Unreleased]: https://github.com/EtanHey/zikaron/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/EtanHey/zikaron/releases/tag/v0.1.0
