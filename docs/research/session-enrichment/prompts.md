# Session Enrichment Research Prompts

> Paste these into web AI (Claude.ai, Gemini, ChatGPT) for research.
> Each prompt is self-contained with full context.

---

## Prompt 1: Session-Level Enrichment Schema

```
I'm building BrainLayer — an open-source persistent memory layer for AI agents (Claude Code specifically). It indexes Claude Code conversation transcripts into a SQLite database with sqlite-vec for semantic search.

Currently we have CHUNK-LEVEL enrichment — each 2000-char chunk gets: summary, tags, importance (1-10), intent (debugging/implementing/discussing/etc), key_symbols, language.

I want to add SESSION-LEVEL enrichment — analyzing full conversations as a unit. This would extract things you can't see from individual chunks:

- What corrections did the user make? (Claude did X, user said "no, do Y")
- What decisions were made and WHY?
- What patterns of frustration emerged?
- What worked well vs poorly?
- What instructions did the user repeat (Claude forgot/ignored)?
- What new knowledge was established?

**Technical context:**
- Sessions are JSONL files, 100-4000+ messages each
- Message types: user, assistant, progress (tool calls), system
- Sessions have a UUID tree (parentUuid chains) — but "forks" are mostly just parallel tool results sharing a parent, not real conversation branches
- Compaction events are marked by "This session is being continued from a previous conversation" in user messages
- We already have 268K chunks across 800+ sessions
- Local LLM available (Ollama with GLM-4.7-Flash, ~8K context) but also Gemini Flash (free, 1M context)

**Questions:**
1. What metadata fields should a session enrichment record have? Design the schema.
2. How should we handle sessions that are 200K+ tokens? Staged approach? Sliding window? Map-reduce?
3. Should corrections/learnings be their own entity (separate from session summary) so they're independently searchable?
4. How would you structure the LLM prompt that analyzes a session?
5. What about linking learnings across sessions? (Same mistake in session A and session C = pattern)
6. How does this relate to existing chunk enrichment — should session enrichment reference chunk IDs?
```

---

## Prompt 2: brainStore — Write-Side MCP Tool Design

```
I'm building BrainLayer — an open-source memory layer for AI agents, exposed as an MCP (Model Context Protocol) server. Currently all 12 MCP tools are READ-ONLY (search, think, recall, context, etc).

I want to add a WRITE-SIDE tool called brainlayer_store (or brainStore) that lets any Claude Code session quickly store:
- Feature ideas that come up during work
- Mistakes/corrections (like "/learn-mistake" but automatic)
- Decisions made ("we chose X over Y because Z")
- Quick notes ("this API is flaky, retry needed")

**Current architecture:**
- Python MCP server using `mcp` library
- SQLite database with `apsw` (268K chunks, ~3.8GB)
- Embeddings via sentence-transformers (bge-large-en-v1.5, 1024 dims)
- Enrichment via local Ollama (GLM-4.7-Flash)

**Design questions:**
1. Schema: What table structure? Should stored items go into the existing `chunks` table or a new `notes`/`store` table?
2. Should items be embedded at write time (slow, ~1-2s) or queued for batch embedding later?
3. Categories: What types make sense? (idea, mistake, decision, learning, todo, bookmark?)
4. Should brainlayer_store return confirmation or also return related existing memories? ("You stored X. FYI, you noted something similar 3 days ago: Y")
5. MCP tool signature — what parameters? (content, type, project, tags, priority?)
6. How does this integrate with the existing search? Should stored items be searchable via brainlayer_search?
7. Security: Can any MCP client write? Should there be auth/scoping?
8. Aggregation: How to cluster/summarize stored items periodically (nightly batch)?

**Context:** We already have a "/learn-mistake" skill in our golems system that stores mistakes to JSON files and clusters them nightly using embeddings. brainStore would replace that with a proper database-backed system available to ALL Claude Code sessions, not just golems.
```

---

## Prompt 3: Conversation Fork/Branch Detection

```
I'm analyzing Claude Code session transcripts (JSONL format) for a memory system. Each message has a `uuid` and `parentUuid` field, forming a tree structure.

In a real session I analyzed:
- 4,118 messages with UUIDs
- 321 on the "active path" (tracing from last message back to root)
- 3,797 "abandoned" messages (92%!)
- 539 apparent "fork points" (parent with >1 child)

But investigating further:
- Only 8 are compaction events (marked by "This session is being continued from a previous conversation")
- The other 531 "forks" are just parallel events — tool results and progress events sharing the same parent assistant message. Not real conversation branches.
- Real user rewinds (checkpoints) are rare (~5-8 per long session)

**For session-level enrichment, I need to:**
1. Reconstruct the actual linear conversation from the JSONL tree
2. Distinguish: compaction (content was summarized), real rewind (user changed direction), parallel tool events (not a fork)
3. Handle compaction gracefully — the summary replaces earlier content, but the enrichment should know context was lost

**Questions:**
1. What's the best algorithm to reconstruct the "true" conversation from this tree?
2. Should abandoned branches be analyzed too? (They show what Claude tried before the user corrected)
3. How to detect real user rewinds vs compaction vs parallel events?
4. For session enrichment: should we enrich each "segment" between compactions separately, or try to reconstruct the full session?
5. Is there prior art in conversation analysis / dialogue systems for handling branching conversations?
```

---

## Prompt 4: Auto-Extracting Learnings from 268K Chunks

```
I have a SQLite database with 268,864 indexed chunks from Claude Code conversations. Each chunk has:
- content (the actual text, 50-2000 chars)
- content_type (user_message, assistant_text, ai_code, stack_trace, file_read, git_diff)
- source (claude_code, whatsapp, youtube)
- intent (debugging, implementing, discussing, deciding, configuring, reviewing)
- importance (1-10 float)
- tags (JSON array of topic tags)
- summary (1-sentence summary)
- project (which codebase)
- created_at (timestamp)

I want to automatically extract "learnings" — patterns like:
- Repeated corrections: user keeps telling Claude the same thing
- Common mistakes: same type of error across sessions
- Established rules: "always do X, never do Y" type instructions
- Tool preferences: "use Read not cat", "use bun not npm"
- Architecture decisions: "we chose X because Y"

**The challenge:** Individual chunks lack conversation context. A user message saying "no, RTL" only makes sense next to the assistant message that got it wrong.

**Approaches I'm considering:**
1. **Keyword mining:** Find chunks with correction signals ("no", "wrong", "I said", "not what I meant", "actually", "stop") and pull surrounding context
2. **High-importance user messages:** Filter importance >= 7, intent = "discussing" or "deciding"
3. **Session-level analysis:** Process full sessions (see separate prompt) to extract corrections in context
4. **Clustering:** Embed all user corrections, cluster by similarity, surface repeated patterns
5. **Diff analysis:** Compare what user asked vs what Claude produced (using git_diff chunks nearby)

**Questions:**
1. Which approach (or combination) would give the best signal-to-noise ratio?
2. How to distinguish genuine corrections from normal conversation flow?
3. How to cluster learnings across 800+ sessions into actionable rules?
4. Output format: How should extracted learnings be structured for auto-generating CLAUDE.md rules?
5. How to handle learnings that contradict each other (user changed their mind over time)?
6. Scale: Can this run on 268K chunks locally, or does it need sampling?
```

---

## Prompt 5: CLI UX for Memory/Knowledge Tools

```
I'm designing the CLI experience for BrainLayer, an open-source memory layer for AI agents. Current CLI commands are bare-bones — `brainlayer search "query"` returns results with score, project, and raw content.

The database has rich enrichment data that isn't being displayed:
- summary (1-sentence), tags (topic array), importance (1-10), intent (debugging/implementing/etc)
- 144K enriched chunks out of 268K total
- Per-source data: claude_code (250K chunks), whatsapp (16K, with contact_name), youtube (2K)

**Current display (ugly):**
```
1. (score: 0.823) (golems)
Raw chunk content here up to 500 chars...
ID: chunk_abc123
```

**I want to design a much nicer CLI with:**
1. Rich display using the `rich` Python library (tables, colors, panels)
2. Enrichment data visible (tags as colored badges, importance as stars/bars, intent as icons)
3. Smart truncation (show summary instead of raw content when available)
4. Multiple display modes (compact list, detailed cards, table format)
5. Interactive features (filter by tag, sort by importance, drill into context)
6. `brainlayer insights` — show extracted learnings/patterns
7. `brainlayer stats --detailed` — per-project, per-source breakdowns with charts
8. `brainlayer timeline` — show activity over time (like GitHub contribution graph but for knowledge)

**Constraints:**
- Must work in standard terminal (no TUI framework, just rich library)
- Fast — search results should appear in <2 seconds
- The library already uses `typer` for CLI and `rich` for output

**Questions:**
1. What's the ideal search result layout? Show me ASCII mockups.
2. How should enrichment data be prioritized in limited terminal space?
3. What CLI commands/subcommands make sense for a knowledge tool?
4. Any prior art in CLI tools with great search UX? (ripgrep, fzf, etc)
5. How to handle the transition between "quick search" and "deep exploration"?
```

---

## How to Use These

1. **Start with Prompt 1** (schema design) — this shapes everything else
2. **Prompt 2** (brainStore) can run in parallel — it's independent
3. **Prompt 3** (fork detection) informs the implementation of Prompt 1
4. **Prompt 4** (auto-learnings) builds on top of session enrichment
5. **Prompt 5** (CLI UX) can run anytime — it's the presentation layer

Save responses to `docs/research/session-enrichment/` for reference.
