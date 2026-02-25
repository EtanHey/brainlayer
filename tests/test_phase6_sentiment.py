"""Phase 6: Sentiment + Conversation Analysis tests."""

import json
from typing import List

from brainlayer.vector_store import VectorStore


def _insert_chunks(store: VectorStore, ids: List[str], documents: List[str],
                   metadatas: List[dict], embeddings: List[list]) -> None:
    """Helper to insert test chunks using upsert_chunks API."""
    chunks = []
    for i, (cid, doc, meta) in enumerate(zip(ids, documents, metadatas)):
        chunks.append({
            "id": cid,
            "content": doc,
            "metadata": meta,
            "source_file": meta.get("source_file", "test.jsonl"),
            "project": meta.get("project"),
            "content_type": meta.get("content_type"),
            "char_count": len(doc),
            "source": meta.get("source", "claude_code"),
        })
    store.upsert_chunks(chunks, embeddings)


# --- Task 1: Schema tests ---


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


# --- Task 2: Rule-based sentiment analyzer ---


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
    assert any(s in result["signals"] for s in ["damn", "frustrating"])


def test_analyze_sentiment_empty_text():
    """Empty text returns neutral."""
    from brainlayer.pipeline.sentiment import analyze_sentiment

    result = analyze_sentiment("")
    assert result["label"] == "neutral"
    assert result["score"] == 0.0
    assert result["signals"] == []


# --- Task 3: VectorStore sentiment methods ---


def test_update_sentiment(tmp_path):
    """Can write and read back sentiment metadata."""
    store = VectorStore(tmp_path / "test.db")
    _insert_chunks(
        store,
        ids=["test-1"],
        documents=["what the fuck is this error"],
        metadatas=[{"source_file": "test.jsonl", "project": "test"}],
        embeddings=[[0.1] * 1024],
    )
    store.update_sentiment("test-1", "frustration", -0.8, ["wtf"])
    cursor = store.conn.cursor()
    row = list(
        cursor.execute(
            "SELECT sentiment_label, sentiment_score, sentiment_signals FROM chunks WHERE id = ?",
            ["test-1"],
        )
    )[0]
    assert row[0] == "frustration"
    assert row[1] == -0.8
    assert json.loads(row[2]) == ["wtf"]


def test_search_sentiment_filter(tmp_path):
    """Search can filter by sentiment_label."""
    store = VectorStore(tmp_path / "test.db")
    emb = [0.1] * 1024
    _insert_chunks(
        store,
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
    results = store.search(query_embedding=emb, n_results=10, sentiment_filter="frustration")
    docs = results["documents"][0]
    assert len(docs) == 1
    assert "broken" in docs[0]


# --- Task 4: MCP schema test ---


def test_brain_search_schema_has_sentiment():
    """brain_search tool schema includes sentiment parameter."""
    import asyncio

    from brainlayer.mcp import list_tools

    tools = asyncio.run(list_tools())
    brain_search = next(t for t in tools if t.name == "brain_search")
    props = brain_search.inputSchema.get("properties", {})
    assert "sentiment" in props
    assert props["sentiment"]["type"] == "string"


# --- Task 5: Batch processing ---


def test_batch_analyze_sentiment(tmp_path):
    """Batch command processes user_message chunks."""
    from brainlayer.pipeline.sentiment import batch_analyze_sentiment

    store = VectorStore(tmp_path / "test.db")
    _insert_chunks(
        store,
        ids=["u1"],
        documents=["this is incredibly frustrating!"],
        metadatas=[{"source_file": "t.jsonl", "project": "test", "content_type": "user_message"}],
        embeddings=[[0.1] * 1024],
    )
    processed = batch_analyze_sentiment(store, batch_size=10, max_chunks=100)
    assert processed >= 1
    row = list(store.conn.cursor().execute("SELECT sentiment_label FROM chunks WHERE id = 'u1'"))[0]
    assert row[0] == "frustration"


# --- Task 6: Enrichment integration ---


def test_enrichment_prompt_includes_sentiment():
    """LLM enrichment prompt asks for sentiment fields."""
    from brainlayer.pipeline.enrichment import ENRICHMENT_PROMPT

    assert "sentiment_label" in ENRICHMENT_PROMPT
    assert "sentiment_score" in ENRICHMENT_PROMPT


# --- Task 7: Full integration test ---


def test_full_sentiment_pipeline(tmp_path):
    """End-to-end: insert chunk -> rule-based sentiment -> search by sentiment."""
    from brainlayer.pipeline.sentiment import batch_analyze_sentiment

    store = VectorStore(tmp_path / "test.db")
    emb = [0.1] * 1024
    _insert_chunks(
        store,
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

    processed = batch_analyze_sentiment(store)
    assert processed == 3

    # Search by sentiment
    results = store.search(query_embedding=emb, n_results=10, sentiment_filter="frustration")
    docs = results["documents"][0]
    assert len(docs) == 1
    assert any("hell" in d or "nothing works" in d for d in docs)

    # Positive filter
    results = store.search(query_embedding=emb, n_results=10, sentiment_filter="positive")
    docs = results["documents"][0]
    assert len(docs) == 1
    assert any("perfect" in d for d in docs)
