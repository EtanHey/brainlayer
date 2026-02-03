# Zikaron Performance & Architecture Research

## Problem Statement

Zikaron search takes 3+ minutes on a 3.2GB ChromaDB database. This makes it unusable for interactive workflows. We need to understand the bottlenecks and design a fast, usable architecture.

---

## Requirements (EARS Notation)

### R1: Performance Profiling
**WHEN** a user runs `zikaron search "<query>"`
**THE SYSTEM** shall complete the search in under 5 seconds for warm queries
**AND** under 30 seconds for cold start queries

### R2: Collection Separation
**WHEN** a user has mixed data sources (Claude Code, WhatsApp, Gemini)
**THE SYSTEM** shall separate data into distinct collections
**SO THAT** queries can target specific collections for faster results

### R3: Daemon Service
**WHEN** the user's machine boots
**THE SYSTEM** shall pre-load embedding models and database connections
**SO THAT** subsequent searches are instant (warm queries)

### R4: Incremental Indexing
**WHEN** new conversations are added to source directories
**THE SYSTEM** shall only index new/changed files
**AND NOT** re-process already-indexed content

---

## Research Architecture

### Phase 1: Diagnose (Profile Current State)
```
┌─────────────────────────────────────────────────────────┐
│  INPUTS                                                  │
│  - Current zikaron codebase                             │
│  - 3.2GB ChromaDB at ~/.local/share/zikaron/chromadb/   │
│  - Sample queries for benchmarking                      │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│  ANALYSIS                                                │
│  1. cProfile/line_profiler on search path               │
│  2. Measure: model load, DB init, embedding, similarity │
│  3. Memory profiling (tracemalloc)                      │
│  4. Identify top 5 bottlenecks                          │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│  OUTPUT: docs.local/research/performance/profiling.md   │
└─────────────────────────────────────────────────────────┘
```

### Phase 2: Evaluate Alternatives
```
┌─────────────────────────────────────────────────────────┐
│  EMBEDDING MODELS                                        │
│  - nomic-embed-text (current)                           │
│  - all-MiniLM-L6-v2 (smallest)                          │
│  - bge-small-en-v1.5                                    │
│  - gte-small                                            │
│  Metrics: load time, embed speed, quality (10 queries)  │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│  VECTOR DATABASES                                        │
│  - ChromaDB (current)                                   │
│  - sqlite-vec                                           │
│  - LanceDB                                              │
│  - DuckDB + vss extension                               │
│  - FAISS (no DB, just numpy)                            │
│  Metrics: cold start, query time, memory for 3GB data   │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│  OUTPUT: docs.local/research/performance/alternatives.md│
└─────────────────────────────────────────────────────────┘
```

### Phase 3: Design New Architecture
```
┌─────────────────────────────────────────────────────────┐
│  DAEMON SERVICE DESIGN                                   │
│  - FastAPI/uvicorn persistent service                   │
│  - launchd plist for auto-start                         │
│  - Unix socket or localhost:PORT for CLI connection     │
│  - Memory budget analysis                               │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│  COLLECTION STRATEGY                                     │
│  - work: Claude Code conversations                      │
│  - social: WhatsApp, personal chats                     │
│  - research: papers, docs, notes                        │
│  - Metadata filtering vs separate DBs                   │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│  INCREMENTAL INDEXING                                    │
│  - File hash/mtime tracking                             │
│  - Delta detection algorithm                            │
│  - ChromaDB upsert vs delete+insert                     │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│  OUTPUT: docs.local/research/performance/architecture.md│
└─────────────────────────────────────────────────────────┘
```

---

## Implementation Tasks

| ID | Task | Depends On | Output |
|----|------|------------|--------|
| T1 | Profile current search with cProfile | - | profiling.md |
| T2 | Benchmark 4 embedding models | - | alternatives.md (section 1) |
| T3 | Benchmark 5 vector DB options | T1 | alternatives.md (section 2) |
| T4 | Design daemon service architecture | T1, T2, T3 | architecture.md (section 1) |
| T5 | Design collection separation strategy | T1 | architecture.md (section 2) |
| T6 | Design incremental indexing system | T1 | architecture.md (section 3) |
| T7 | Write synthesis with recommendations | T4, T5, T6 | synthesis.md |

---

## Success Criteria

- [ ] Identified why search takes 3+ minutes (root cause)
- [ ] Recommended embedding model with <1s load time
- [ ] Recommended vector DB with <5s cold start for 3GB
- [ ] Daemon service design that enables <1s warm queries
- [ ] Collection strategy that reduces search scope by 70%+
- [ ] Incremental indexing design that avoids full re-index

---

## Non-Goals (Out of Scope)

- Implementing any changes (research only)
- Dashboard/UI design (separate spec)
- Data source parsing (WhatsApp format, etc.)
- Privacy/security architecture
