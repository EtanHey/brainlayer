# Zikaron Dashboard & UX Research

## Problem Statement

Zikaron has no visibility into what's indexed, no way to filter/categorize results, and returns low-quality results. We need a dashboard that makes the memory layer useful and transparent.

---

## Requirements (EARS Notation)

### R1: Statistics Visibility
**WHEN** a user opens the dashboard
**THE SYSTEM** shall display total indexed items, breakdown by source, and last index time
**SO THAT** the user understands what data is available

### R2: Search with Filters
**WHEN** a user searches with filters (source, date range, content type)
**THE SYSTEM** shall apply filters before vector search
**AND** display results with relevant context and scores

### R3: Result Quality
**WHEN** search results are returned
**THE SYSTEM** shall rank by relevance AND recency
**AND** show why each result matched (highlighted terms, metadata)

### R4: Data Management
**WHEN** a user wants to delete or re-index content
**THE SYSTEM** shall provide controls to prune old data, re-index sources, or clear collections

---

## Research Architecture

### Phase 1: Competitor Analysis
```
┌─────────────────────────────────────────────────────────┐
│  TOOLS TO ANALYZE                                        │
│  - Khoj (open source personal AI)                       │
│  - Obsidian Copilot / Smart Connections                 │
│  - Rewind.ai (screen capture + search)                  │
│  - mem0 (memory layer for AI agents)                    │
│  - Notion AI search                                     │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│  FOR EACH TOOL DOCUMENT                                  │
│  - How does search UX work?                             │
│  - What filters are available?                          │
│  - How are results displayed?                           │
│  - What stats/visibility does it provide?               │
│  - What can we steal?                                   │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│  OUTPUT: docs.local/research/dashboard/competitors.md   │
└─────────────────────────────────────────────────────────┘
```

### Phase 2: Search Quality Research
```
┌─────────────────────────────────────────────────────────┐
│  RETRIEVAL IMPROVEMENTS                                  │
│  - Hybrid search (BM25 + semantic)                      │
│  - Reranking with cross-encoders                        │
│  - Query understanding / intent classification          │
│  - Temporal weighting (recency boost)                   │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│  CHUNKING STRATEGIES                                     │
│  - AST-based for code (tree-sitter)                     │
│  - Turn-based for chat                                  │
│  - Section-based for markdown                           │
│  - Current chunk size analysis                          │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│  OUTPUT: docs.local/research/dashboard/search-quality.md│
└─────────────────────────────────────────────────────────┘
```

### Phase 3: UI Framework Research
```
┌─────────────────────────────────────────────────────────┐
│  CLI-FIRST OPTIONS                                       │
│  - Ink (React for CLI) - already used in golems        │
│  - Textual (Python TUI) - rich widgets                  │
│  - Blessed/neo-blessed (Node.js)                        │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│  WEB OPTIONS                                             │
│  - Streamlit (Python, quick dashboards)                 │
│  - FastAPI + htmx (minimal JS)                          │
│  - Next.js (if we want rich interactivity)              │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│  DECISION CRITERIA                                       │
│  - Python vs TypeScript (zikaron is Python)             │
│  - CLI-first vs web-first vs both                       │
│  - Integration with existing golems dashboard plans     │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│  OUTPUT: docs.local/research/dashboard/ui-framework.md  │
└─────────────────────────────────────────────────────────┘
```

### Phase 4: Unified Dashboard Design
```
┌─────────────────────────────────────────────────────────┐
│  INFORMATION ARCHITECTURE                                │
│  - Home: stats overview, quick search                   │
│  - Memory: search + filters + results                   │
│  - Jobs: pipeline view (if integrating JobGolem)        │
│  - Golems: status, nightshift PRs, notifications        │
│  - Settings: collections, re-index, prune               │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│  WIREFRAMES                                              │
│  - ASCII wireframes for each view                       │
│  - User flows (search, manage, configure)               │
│  - Mobile considerations (Telegram integration?)        │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│  OUTPUT: docs.local/research/dashboard/design.md        │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation Tasks

| ID | Task | Depends On | Output |
|----|------|------------|--------|
| T1 | Analyze 5 competitor tools | - | competitors.md |
| T2 | Research hybrid search & reranking | - | search-quality.md (section 1) |
| T3 | Research chunking strategies | - | search-quality.md (section 2) |
| T4 | Evaluate CLI UI frameworks | - | ui-framework.md (section 1) |
| T5 | Evaluate web UI frameworks | - | ui-framework.md (section 2) |
| T6 | Recommend framework for zikaron | T4, T5 | ui-framework.md (section 3) |
| T7 | Design information architecture | T1, T6 | design.md (section 1) |
| T8 | Create wireframes | T7 | design.md (section 2) |
| T9 | Write synthesis with recommendations | T1-T8 | synthesis.md |

---

## Success Criteria

- [ ] Documented 5+ competitor approaches with learnings
- [ ] Identified hybrid search approach to improve result quality
- [ ] Recommended chunking strategy for code vs chat vs docs
- [ ] Recommended UI framework with justification
- [ ] Information architecture for unified dashboard
- [ ] Wireframes for key views (home, search, settings)

---

## Non-Goals (Out of Scope)

- Implementing any changes (research only)
- Performance optimization (separate spec)
- Data source parsing details
- MCP server changes

---

## Coordination Note

This spec runs in parallel with `zikaron-performance.md`. No dependencies between them - one focuses on backend architecture, this focuses on frontend/UX.
