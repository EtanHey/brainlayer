# Phase 6: Sentiment + Conversation Analysis — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract frustration, positive, confusion, and satisfaction signals from conversation chunks using two-tier detection (rule-based + LLM enrichment).

**Architecture:** 3 new columns on `chunks` table (`sentiment_label`, `sentiment_score`, `sentiment_signals`). Tier 1: fast rule-based pattern matching on `user_message` chunks (bilingual EN+HE). Tier 2: add sentiment fields to existing LLM enrichment prompt. New `sentiment` filter on `brain_search` MCP tool. New `brainlayer sentiment` CLI command for batch backfill.

**Tech Stack:** Python, sqlite-vec (APSW), regex patterns, existing enrichment pipeline (Ollama/MLX)

---

### Task 1: Schema — Add sentiment columns to chunks table

**Files:**
- Modify: `src/brainlayer/vector_store.py:116-151` (column definitions + indexes)
- Test: `tests/test_phase6_sentiment.py` (new file)

**Step 1: Write the failing test**

```python
# tests/test_phase6_sentiment.py
"""Phase 6: Sentiment + Conversation Analysis tests."""
import json
from pathlib import Path
from brainlayer.vector_store import VectorStore


def test_sentiment_columns_exist(tmp_path):
    """Sentiment columns are created in chunks table."""
    store = VectorStore(tmp_path / "test.db")
    cursor = store.conn.cursor()
    cols = {row[1] for row in cursor.execute("PRAGMA table_info(chunks)")}
    assert "sentiment_label" in cols
    assert "sentiment_score" in cols
    assert "sentiment_signals" in cols


def test_sentiment_index_exists(tmp_path):
    """Sentiment label index is created for fast filtering."""
    store = VectorStore(tmp_path / "test.db")
    cursor = store.conn.cursor()
    indexes = {row[1] for row in cursor.execute("PRAGMA index_list(chunks)")}
    assert "idx_chunks_sentiment" in indexes
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/etanheyman/Gits/brainlayer/.claude/worktrees/phase-6-sentiment && python3 -m pytest tests/test_phase6_sentiment.py -v`
Expected: FAIL — columns don't exist yet

**Step 3: Write minimal implementation**

In `vector_store.py`, add to the column migration list (~line 138):
```python
            # Phase 6: Sentiment columns
            ("sentiment_label", "TEXT"),   # frustration|confusion|positive|satisfaction|neutral
            ("sentiment_score", "REAL"),   # -1.0 (frustration) to +1.0 (positive)
            ("sentiment_signals", "TEXT"), # JSON array of matched signals
```

Add index (~line 152):
```python
            ("idx_chunks_sentiment", "sentiment_label"),
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_phase6_sentiment.py -v`
Expected: PASS

**Step 5: Commit**

---

### Task 2: Rule-based sentiment analyzer module

**Files:**
- Create: `src/brainlayer/pipeline/sentiment.py`
- Test: `tests/test_phase6_sentiment.py` (add tests)

**Step 1: Write failing tests**

```python
def test_analyze_sentiment_frustration():
    """Detects frustration in text."""
    from brainlayer.pipeline.sentiment import analyze_sentiment
    result = analyze_sentiment("what the fuck, this is broken again")
    assert result["label"] == "frustration"
    assert result["score"] < -0.5
    assert len(result["signals"]) > 0


def test_analyze_sentiment_positive():
    """Detects positive sentiment."""
    from brainlayer.pipeline.sentiment import analyze_sentiment
    result = analyze_sentiment("wow this is amazing, works perfectly!")
    assert result["label"] == "positive"
    assert result["score"] > 0.5


def test_analyze_sentiment_confusion():
    """Detects confusion."""
    from brainlayer.pipeline.sentiment import analyze_sentiment
    result = analyze_sentiment("I don't understand, wait what? how does this work?")
    assert result["label"] == "confusion"
    assert result["score"] < 0


def test_analyze_sentiment_satisfaction():
    """Detects satisfaction (task done, thanks)."""
    from brainlayer.pipeline.sentiment import analyze_sentiment
    result = analyze_sentiment("perfect, thanks! that's exactly what I needed")
    assert result["label"] == "satisfaction"
    assert result["score"] > 0.5


def test_analyze_sentiment_neutral():
    """Neutral text gets neutral label."""
    from brainlayer.pipeline.sentiment import analyze_sentiment
    result = analyze_sentiment("please read the file at src/main.py")
    assert result["label"] == "neutral"
    assert -0.3 <= result["score"] <= 0.3


def test_analyze_sentiment_hebrew_frustration():
    """Detects Hebrew frustration markers."""
    from brainlayer.pipeline.sentiment import analyze_sentiment
    result = analyze_sentiment("לעזאזל, זה לא עובד בכלל")
    assert result["label"] == "frustration"
    assert result["score"] < -0.5


def test_analyze_sentiment_hebrew_positive():
    """Detects Hebrew positive markers."""
    from brainlayer.pipeline.sentiment import analyze_sentiment
    result = analyze_sentiment("מדהים! עובד מעולה")
    assert result["label"] == "positive"
    assert result["score"] > 0.5


def test_analyze_sentiment_returns_matched_signals():
    """Signals list contains the actual matched patterns."""
    from brainlayer.pipeline.sentiment import analyze_sentiment
    result = analyze_sentiment("damn this bug is so frustrating")
    assert "damn" in result["signals"] or "frustrating" in result["signals"]


def test_analyze_sentiment_empty_text():
    """Empty text returns neutral."""
    from brainlayer.pipeline.sentiment import analyze_sentiment
    result = analyze_sentiment("")
    assert result["label"] == "neutral"
    assert result["score"] == 0.0
    assert result["signals"] == []
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_phase6_sentiment.py::test_analyze_sentiment_frustration -v`
Expected: FAIL — module doesn't exist

**Step 3: Implement sentiment analyzer**

Create `src/brainlayer/pipeline/sentiment.py` with:
- `FRUSTRATION_PATTERNS`: EN + HE regex patterns
- `POSITIVE_PATTERNS`: EN + HE regex patterns
- `CONFUSION_PATTERNS`: EN + HE regex patterns
- `SATISFACTION_PATTERNS`: EN + HE regex patterns
- `analyze_sentiment(text: str) -> dict`: Returns `{"label": str, "score": float, "signals": list[str]}`
- Score calculation: weighted sum of matched patterns, normalized to [-1, 1]
- Label thresholds: frustration (<-0.3), confusion (-0.3 to 0 with confusion patterns), positive (>0.3), satisfaction (>0.3 with satisfaction patterns), neutral (else)

**Step 4: Run all sentiment tests**

Run: `python3 -m pytest tests/test_phase6_sentiment.py -v`
Expected: ALL PASS

**Step 5: Commit**

---

### Task 3: VectorStore — update_sentiment method + sentiment filter on search

**Files:**
- Modify: `src/brainlayer/vector_store.py` (add `update_sentiment()`, add `sentiment_filter` to `search()` and `hybrid_search()`)
- Test: `tests/test_phase6_sentiment.py` (add tests)

**Step 1: Write failing tests**

```python
def test_update_sentiment(tmp_path):
    """Can write and read back sentiment metadata."""
    store = VectorStore(tmp_path / "test.db")
    # Insert a test chunk
    store.add_chunks(
        ids=["test-1"],
        documents=["what the fuck is this error"],
        metadatas=[{"source_file": "test.jsonl", "project": "test"}],
        embeddings=[[0.1] * 1024],
    )
    store.update_sentiment("test-1", "frustration", -0.8, ["wtf"])
    cursor = store.conn.cursor()
    row = list(cursor.execute(
        "SELECT sentiment_label, sentiment_score, sentiment_signals FROM chunks WHERE id = ?",
        ["test-1"]
    ))[0]
    assert row[0] == "frustration"
    assert row[1] == -0.8
    assert json.loads(row[2]) == ["wtf"]


def test_search_sentiment_filter(tmp_path):
    """Search can filter by sentiment_label."""
    store = VectorStore(tmp_path / "test.db")
    emb = [0.1] * 1024
    store.add_chunks(
        ids=["pos-1", "neg-1"],
        documents=["great work", "this is broken"],
        metadatas=[
            {"source_file": "a.jsonl", "project": "test"},
            {"source_file": "b.jsonl", "project": "test"},
        ],
        embeddings=[emb, emb],
    )
    store.update_sentiment("pos-1", "positive", 0.8, ["great"])
    store.update_sentiment("neg-1", "frustration", -0.7, ["broken"])
    results = store.search(
        query_embedding=emb, n_results=10, sentiment_filter="frustration"
    )
    ids = [m.get("id") or m.get("chunk_id") for m in results["metadatas"][0]]
    # Only frustration chunks returned
    assert "neg-1" in [r for r in results["documents"][0] if "broken" in r] or len(results["documents"][0]) == 1
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement**

- Add `update_sentiment(chunk_id, label, score, signals)` method on VectorStore (same retry pattern as `update_enrichment`)
- Add `sentiment_filter: Optional[str] = None` param to `search()` and `hybrid_search()`
- Add WHERE clause: `c.sentiment_label = ?` when filter is set

**Step 4: Run tests, verify pass**

**Step 5: Commit**

---

### Task 4: MCP — Add sentiment filter to brain_search

**Files:**
- Modify: `src/brainlayer/mcp/__init__.py` (add `sentiment` param to brain_search tool schema + dispatcher)
- Test: `tests/test_phase6_sentiment.py` (add test)

**Step 1: Write failing test**

```python
def test_brain_search_schema_has_sentiment():
    """brain_search tool schema includes sentiment parameter."""
    import asyncio
    from brainlayer.mcp import server
    # Get tool list
    tools = asyncio.run(server.list_tools())
    brain_search = next(t for t in tools if t.name == "brain_search")
    props = brain_search.inputSchema.get("properties", {})
    assert "sentiment" in props
    assert props["sentiment"]["type"] == "string"
```

**Step 2: Run test to verify it fails**

**Step 3: Implement**

In `mcp/__init__.py`:
- Add `sentiment` to the brain_search tool schema (enum: frustration, confusion, positive, satisfaction, neutral)
- Pass `sentiment` through `_brain_search()` → `_search()` → `store.hybrid_search(sentiment_filter=...)`

**Step 4: Run tests, verify pass**

**Step 5: Commit**

---

### Task 5: CLI — `brainlayer sentiment` batch command

**Files:**
- Modify: `src/brainlayer/cli/__init__.py` (add `sentiment` command)
- Test: `tests/test_phase6_sentiment.py` (add test)

**Step 1: Write failing test**

```python
def test_sentiment_cli_processes_chunks(tmp_path):
    """CLI sentiment command processes user_message chunks."""
    from brainlayer.pipeline.sentiment import batch_analyze_sentiment
    store = VectorStore(tmp_path / "test.db")
    store.add_chunks(
        ids=["u1"],
        documents=["this is incredibly frustrating!"],
        metadatas=[{"source_file": "t.jsonl", "project": "test", "content_type": "user_message"}],
        embeddings=[[0.1] * 1024],
    )
    # Manually set content_type since add_chunks may not set it
    store.conn.cursor().execute("UPDATE chunks SET content_type = 'user_message' WHERE id = 'u1'")
    processed = batch_analyze_sentiment(store, batch_size=10, max_chunks=100)
    assert processed >= 1
    row = list(store.conn.cursor().execute(
        "SELECT sentiment_label FROM chunks WHERE id = 'u1'"
    ))[0]
    assert row[0] == "frustration"
```

**Step 2: Run test to verify it fails**

**Step 3: Implement**

- `batch_analyze_sentiment(store, batch_size, max_chunks)` in `sentiment.py`: queries chunks WHERE content_type = 'user_message' AND sentiment_label IS NULL, runs `analyze_sentiment()`, calls `store.update_sentiment()`
- CLI `sentiment` command in `cli/__init__.py`: wrapper with rich progress bar

**Step 4: Run tests, verify pass**

**Step 5: Commit**

---

### Task 6: Enrichment integration — Add sentiment to LLM enrichment prompt

**Files:**
- Modify: `src/brainlayer/pipeline/enrichment.py` (extend prompt + parse response)
- Modify: `src/brainlayer/vector_store.py` (extend `update_enrichment` with sentiment fields)
- Test: `tests/test_phase6_sentiment.py` (add test)

**Step 1: Write failing test**

```python
def test_enrichment_prompt_includes_sentiment():
    """LLM enrichment prompt asks for sentiment fields."""
    from brainlayer.pipeline.enrichment import ENRICHMENT_PROMPT
    assert "sentiment_label" in ENRICHMENT_PROMPT
    assert "sentiment_score" in ENRICHMENT_PROMPT
```

**Step 2: Run test to verify it fails**

**Step 3: Implement**

- Add `sentiment_label`, `sentiment_score`, `sentiment_signals` to `ENRICHMENT_PROMPT` JSON template
- Add sentiment parsing in `_parse_enrichment_response()`
- Extend `update_enrichment()` to accept `sentiment_label`, `sentiment_score`, `sentiment_signals` params
- LLM enrichment only overwrites sentiment if rule-based didn't already set it (rule-based = confident, LLM = fallback for ambiguous)

**Step 4: Run tests, verify pass**

**Step 5: Commit**

---

### Task 7: Full integration test + run baseline tests

**Files:**
- Test: `tests/test_phase6_sentiment.py` (add integration test)

**Step 1: Write integration test**

```python
def test_full_sentiment_pipeline(tmp_path):
    """End-to-end: insert chunk → rule-based sentiment → search by sentiment."""
    store = VectorStore(tmp_path / "test.db")
    from brainlayer.pipeline.sentiment import analyze_sentiment, batch_analyze_sentiment

    emb = [0.1] * 1024
    store.add_chunks(
        ids=["angry-1", "happy-1", "neutral-1"],
        documents=[
            "what the hell, nothing works anymore",
            "this is perfect, exactly what I wanted!",
            "please update the config file",
        ],
        metadatas=[
            {"source_file": "a.jsonl", "project": "test", "content_type": "user_message"},
            {"source_file": "b.jsonl", "project": "test", "content_type": "user_message"},
            {"source_file": "c.jsonl", "project": "test", "content_type": "user_message"},
        ],
        embeddings=[emb, emb, emb],
    )
    # Set content_type
    for cid, ct in [("angry-1", "user_message"), ("happy-1", "user_message"), ("neutral-1", "user_message")]:
        store.conn.cursor().execute("UPDATE chunks SET content_type = ? WHERE id = ?", [ct, cid])

    processed = batch_analyze_sentiment(store)
    assert processed == 3

    # Search by sentiment
    results = store.search(query_embedding=emb, n_results=10, sentiment_filter="frustration")
    docs = results["documents"][0]
    assert any("hell" in d or "nothing works" in d for d in docs)
```

**Step 2: Run ALL tests (phase 6 + baseline)**

Run: `python3 -m pytest tests/test_phase6_sentiment.py -v && python3 -m pytest tests/ -x -q --ignore=tests/test_vector_store.py --ignore=tests/test_engine.py --ignore=tests/test_think_recall_integration.py`

**Step 3: Commit all Phase 6 work**

---

### Task 8: Update collab.md + create PR

**Step 1:** Update collab.md Messages with commit summary
**Step 2:** Create PR with `gh pr create`
**Step 3:** Poll for CI + reviews
**Step 4:** Address review feedback
**Step 5:** Merge
