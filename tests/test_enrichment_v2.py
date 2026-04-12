"""Regression tests for enrichment prompt v2 fields and persistence."""

import json
import threading
from types import SimpleNamespace

from brainlayer.vector_store import VectorStore


def _insert_chunk(store: VectorStore, chunk_id: str, content: str = "test content") -> None:
    store.conn.cursor().execute(
        """INSERT INTO chunks (id, content, metadata, source_file, project, content_type, char_count)
           VALUES (?, ?, '{}', 'test.jsonl', 'brainlayer', 'assistant_text', ?)""",
        (chunk_id, content, len(content)),
    )


def _sanitizer():
    return SimpleNamespace(
        sanitize=lambda text, metadata=None: SimpleNamespace(
            sanitized=text,
            replacements=[],
            pii_detected=False,
        )
    )


def test_build_prompt_renders_v2_fields_and_60_40_truncation():
    from brainlayer.pipeline.enrichment import build_prompt

    head = "H" * 4800
    middle = "M" * 1200
    tail = "T" * 3200
    prompt = build_prompt(
        {
            "content": head + middle + tail,
            "project": "test",
            "content_type": "user_message",
        }
    )

    assert '"key_facts"' in prompt
    assert '"resolved_queries"' in prompt
    assert '"relation"' in prompt
    assert "[...truncated middle...]" in prompt
    assert head in prompt
    assert tail in prompt
    assert middle not in prompt
    assert "{{" not in prompt


def test_prompt_signature_emits_once_across_threads(monkeypatch):
    from brainlayer.pipeline import enrichment

    writes = []
    monkeypatch.setattr(enrichment.os, "write", lambda fd, data: writes.append((fd, data)) or len(data))
    enrichment._prompt_signature_emitted = False
    enrichment._prompt_signature_lock = threading.Lock()

    threads = [threading.Thread(target=enrichment._emit_prompt_signature_once) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(writes) == 1


def test_prompt_signature_swallow_oserror_and_logs_debug(monkeypatch):
    from brainlayer.pipeline import enrichment

    debug_logs = []
    monkeypatch.setattr(enrichment.os, "write", lambda *_args: (_ for _ in ()).throw(OSError("pipe closed")))
    monkeypatch.setattr(
        enrichment.logger, "debug", lambda msg, *args, **kwargs: debug_logs.append(msg % args if args else msg)
    )
    enrichment._prompt_signature_emitted = False
    enrichment._prompt_signature_lock = threading.Lock()

    enrichment._emit_prompt_signature_once()

    assert enrichment._prompt_signature_emitted is True
    assert any("ENRICHMENT_PROMPT_LOADED" in entry for entry in debug_logs)


def test_build_external_prompt_uses_v2_truncation():
    from brainlayer.pipeline.enrichment import build_external_prompt

    head = "A" * 4800
    middle = "B" * 1200
    tail = "C" * 3200
    prompt, _ = build_external_prompt(
        {
            "content": head + middle + tail,
            "project": "test",
            "content_type": "assistant_text",
        },
        _sanitizer(),
    )

    assert "[...truncated middle...]" in prompt
    assert head in prompt
    assert tail in prompt
    assert middle not in prompt


def test_parse_enrichment_extracts_v2_fields_and_entity_relation():
    from brainlayer.pipeline.enrichment import parse_enrichment

    raw = json.dumps(
        {
            "summary": "BrainLayer chose SQLite on March 12, 2026 for local-first storage.",
            "tags": ["python", "architecture"],
            "importance": 8,
            "intent": "deciding",
            "resolved_query": "legacy singular fallback",
            "key_facts": ["March 12, 2026", "sqlite-vec", "src/brainlayer/vector_store.py"],
            "resolved_queries": [
                "Why did BrainLayer choose SQLite for local-first storage?",
                "brainlayer sqlite sqlite-vec local-first storage decision March 12 2026",
                "BrainLayer chose SQLite + sqlite-vec on March 12, 2026 because local-first storage needed no server.",
                "ignored extra query",
            ],
            "entities": [
                {"name": "BrainLayer", "type": "project", "relation": "uses SQLite for storage"},
                {"name": "SQLite", "type": "technology", "relation": "dependency of BrainLayer"},
            ],
        }
    )

    result = parse_enrichment(raw)

    assert result is not None
    assert result["key_facts"] == ["March 12, 2026", "sqlite-vec", "src/brainlayer/vector_store.py"]
    assert len(result["resolved_queries"]) == 3
    assert result["resolved_queries"][0].startswith("Why did BrainLayer choose SQLite")
    assert result["entities"][0]["relation"] == "uses SQLite for storage"
    assert result["entities"][1]["relation"] == "dependency of BrainLayer"


def test_parse_enrichment_keeps_legacy_resolved_query_when_plural_missing():
    from brainlayer.pipeline.enrichment import parse_enrichment

    raw = json.dumps(
        {
            "summary": "Legacy response still carries one retrievable question.",
            "tags": ["python"],
            "importance": 5,
            "intent": "debugging",
            "resolved_query": "How does the legacy enrichment response stay compatible?",
        }
    )

    result = parse_enrichment(raw)

    assert result is not None
    assert result["resolved_query"] == "How does the legacy enrichment response stay compatible?"
    assert "resolved_queries" not in result


def test_parse_enrichment_extracts_sentiment_fields():
    from brainlayer.pipeline.enrichment import parse_enrichment

    raw = json.dumps(
        {
            "summary": "The CLI is broken and the current output is unusable for the intended workflow.",
            "tags": ["debugging", "cli"],
            "sentiment_label": "frustration",
            "sentiment_score": -0.6,
            "sentiment_signals": ["damn", "broken"],
        }
    )

    result = parse_enrichment(raw)

    assert result is not None
    assert result["sentiment_label"] == "frustration"
    assert result["sentiment_score"] == -0.6
    assert result["sentiment_signals"] == ["damn", "broken"]


def test_parse_enrichment_rejects_invalid_sentiment_label():
    from brainlayer.pipeline.enrichment import parse_enrichment

    raw = json.dumps(
        {
            "summary": "The workflow completed successfully and the user is happy with the outcome.",
            "tags": ["reviewing"],
            "sentiment_label": "happy",
            "sentiment_score": 0.8,
            "sentiment_signals": ["works great"],
        }
    )

    result = parse_enrichment(raw)

    assert result is not None
    assert "sentiment_label" not in result


def test_parse_enrichment_skips_missing_sentiment_fields():
    from brainlayer.pipeline.enrichment import parse_enrichment

    raw = json.dumps(
        {
            "summary": "The parser should still return core enrichment fields when sentiment is omitted.",
            "tags": ["python", "parsing"],
        }
    )

    result = parse_enrichment(raw)

    assert result is not None
    assert "sentiment_label" not in result
    assert "sentiment_score" not in result
    assert "sentiment_signals" not in result


def test_parse_enrichment_rejects_boolean_sentiment_score():
    from brainlayer.pipeline.enrichment import parse_enrichment

    raw = json.dumps(
        {
            "summary": "The parser should reject malformed boolean sentiment scores.",
            "tags": ["python", "parsing"],
            "sentiment_label": "neutral",
            "sentiment_score": True,
        }
    )

    result = parse_enrichment(raw)

    assert result is not None
    assert result["sentiment_label"] == "neutral"
    assert "sentiment_score" not in result


def test_parse_enrichment_rejects_non_finite_sentiment_score():
    from brainlayer.pipeline.enrichment import parse_enrichment

    raw = """
    {
      "summary": "The parser should reject malformed non-finite sentiment scores.",
      "tags": ["python", "parsing"],
      "sentiment_label": "neutral",
      "sentiment_score": NaN
    }
    """

    result = parse_enrichment(raw)

    assert result is not None
    assert result["sentiment_label"] == "neutral"
    assert "sentiment_score" not in result


def test_gemini_schema_supports_v2_fields():
    from brainlayer.enrichment_controller import GEMINI_RESPONSE_SCHEMA

    props = GEMINI_RESPONSE_SCHEMA["properties"]

    assert "key_facts" in props
    assert "resolved_queries" in props
    assert "relation" in props["entities"]["items"]["properties"]


def test_update_enrichment_persists_v2_json_fields_and_fts(tmp_path):
    store = VectorStore(tmp_path / "test.db")
    try:
        _insert_chunk(store, "chunk-v2-1")

        store.update_enrichment(
            "chunk-v2-1",
            summary="Dense summary",
            tags=["python"],
            key_facts=["PR #1722", "src/brainlayer/pipeline/enrichment.py"],
            resolved_queries=[
                "What changed in the enrichment prompt?",
                "enrichment prompt key_facts resolved_queries relation",
                "The enrichment prompt now includes key_facts, resolved_queries, and entity relation.",
            ],
        )

        row = (
            store.conn.cursor()
            .execute(
                "SELECT key_facts, resolved_queries FROM chunks WHERE id = ?",
                ("chunk-v2-1",),
            )
            .fetchone()
        )
        assert json.loads(row[0]) == ["PR #1722", "src/brainlayer/pipeline/enrichment.py"]
        assert len(json.loads(row[1])) == 3

        fts_cols = {r[1] for r in store.conn.cursor().execute("PRAGMA table_info(chunks_fts)")}
        assert "key_facts" in fts_cols
        assert "resolved_queries" in fts_cols

        fts_row = (
            store.conn.cursor()
            .execute(
                "SELECT key_facts, resolved_queries FROM chunks_fts WHERE chunk_id = ?",
                ("chunk-v2-1",),
            )
            .fetchone()
        )
        assert "PR #1722" in fts_row[0]
        assert "What changed in the enrichment prompt?" in fts_row[1]
    finally:
        store.close()


def test_apply_enrichment_persists_entities_and_relation_context(tmp_path):
    import json

    from brainlayer.enrichment_controller import _apply_enrichment

    store = VectorStore(tmp_path / "test.db")
    try:
        _insert_chunk(store, "chunk-v2-entities", content="BrainLayer uses SQLite")

        _apply_enrichment(
            store,
            {"id": "chunk-v2-entities", "content": "BrainLayer uses SQLite"},
            {
                "summary": "BrainLayer uses SQLite for local-first storage.",
                "tags": ["python", "architecture"],
                "key_facts": ["SQLite", "local-first"],
                "resolved_queries": [
                    "What database does BrainLayer use?",
                    "brainlayer sqlite local-first storage",
                    "BrainLayer uses SQLite for local-first storage.",
                ],
                "entities": [
                    {"name": "BrainLayer", "type": "project", "relation": "uses SQLite for local-first storage"},
                    {"name": "SQLite", "type": "technology", "relation": "storage engine for BrainLayer"},
                ],
            },
        )

        chunk_row = (
            store.conn.cursor()
            .execute(
                "SELECT key_facts, resolved_queries, raw_entities_json FROM chunks WHERE id = ?",
                ("chunk-v2-entities",),
            )
            .fetchone()
        )
        assert json.loads(chunk_row[0]) == ["SQLite", "local-first"]
        assert len(json.loads(chunk_row[1])) == 3
        assert json.loads(chunk_row[2]) == [
            {"name": "BrainLayer", "type": "project", "relation": "uses SQLite for local-first storage"},
            {"name": "SQLite", "type": "technology", "relation": "storage engine for BrainLayer"},
        ]
    finally:
        store.close()
