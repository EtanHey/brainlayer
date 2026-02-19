# Session Enrichment Research

> Research collected Feb 2026 for BrainLayer's session-level enrichment feature.

## Context

BrainLayer currently has **chunk-level enrichment** (summary, tags, importance, intent per ~2000-char chunk). Session-level enrichment analyzes full conversations as a unit to extract things invisible from individual chunks: corrections, decisions, patterns, frustrations, learnings.

## Research Files

| File | Topic | Source |
|------|-------|--------|
| [prompts.md](prompts.md) | 5 research prompts used for web AI research | Claude Code |
| [session-enrichment-architecture.md](session-enrichment-architecture.md) | Schema design, pipeline, metadata fields | Web AI research |
| [brainstore-write-tools.md](brainstore-write-tools.md) | Write-side MCP tool (`brainlayer_store`) design | Web AI research |
| [conversation-reconstruction.md](conversation-reconstruction.md) | JSONL fork/branch detection, UUID tree reconstruction | Web AI research |
| [auto-extracting-learnings.md](auto-extracting-learnings.md) | Mining corrections and patterns from 268K chunks | Web AI research |
| [cli-ux-patterns.md](cli-ux-patterns.md) | CLI display design with rich library | Web AI research |

## Key Findings

### Session JSONL Structure
- Claude Code sessions use `uuid`/`parentUuid` forming a tree
- ~92% of messages are parallel tool events (progress + tool_result), not real conversation branches
- Real conversation = trace `parentUuid` from last message to root
- Compaction events marked by "This session is being continued from a previous conversation"
- Real user rewinds (checkpoints) are rare (~5-8 per long session)

### Two-Layer Enrichment
1. **Chunk-level** (exists): summary, tags, importance, intent per chunk
2. **Session-level** (proposed): corrections, decisions, patterns, learnings per conversation

### brainStore Write Tool
- Quick MCP tool for any Claude Code session to store ideas, mistakes, decisions, notes
- Replaces golems' `/learn-mistake` skill with proper database-backed system
- Available to ALL Claude Code sessions, not just golems

## Next Steps

- Write formal design doc in `brainlayer/docs/session-enrichment-design.md`
- Implement session reconstruction algorithm
- Add `brainlayer_store` MCP tool
- Build session enrichment pipeline (Gemini Flash for long sessions)
