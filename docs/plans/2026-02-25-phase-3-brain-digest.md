# Phase 3: Brain Digest + Brain Entity — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `brain_digest` MCP tool (structured content ingestion with entity/relation/action extraction) and `brain_entity` MCP tool (entity lookup with evidence).

**Architecture:** brain_digest creates a new chunk from input content, runs Phase 2's entity extraction pipeline on it, extracts action items/decisions/questions via LLM, applies Phase 6's sentiment analysis, stores everything in KG tables with confidence tiers and user_verified flags. brain_entity is a read-only lookup that returns structured entity info with relations and evidence chunks.

**Tech Stack:** Python, APSW/sqlite-vec, Phase 2 extraction pipeline, Phase 6 sentiment, Ollama/MLX LLM

---

### Task 1: Add user_verified column to KG tables

**Files:**
- Modify: `src/brainlayer/vector_store.py` (add columns to kg_entities and kg_relations)
- Test: `tests/test_phase3_digest.py` (new file)

### Task 2: Digest pipeline module — extract structured knowledge from text

**Files:**
- Create: `src/brainlayer/pipeline/digest.py`
- Test: `tests/test_phase3_digest.py`

Core function: `digest_content(content, store, embed_fn, ...)` that:
1. Creates a chunk with source="digest"
2. Runs entity extraction (Phase 2's process_chunk + store_extraction_result)
3. Runs sentiment analysis (Phase 6's analyze_sentiment)
4. Extracts action_items, decisions, questions via LLM
5. Applies confidence tiers
6. Returns structured DigestResult

### Task 3: brain_digest MCP tool

**Files:**
- Modify: `src/brainlayer/mcp/__init__.py` (add brain_digest tool + handler)
- Test: `tests/test_phase3_digest.py`

### Task 4: brain_entity MCP tool

**Files:**
- Modify: `src/brainlayer/mcp/__init__.py` (add brain_entity tool + handler)
- Test: `tests/test_phase3_digest.py`

### Task 5: CLI command + integration test

**Files:**
- Modify: `src/brainlayer/cli/__init__.py` (add digest command)
- Test: `tests/test_phase3_digest.py`

### Task 6: Baseline tests + lint + PR
