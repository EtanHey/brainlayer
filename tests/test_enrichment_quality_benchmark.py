import sqlite3
from pathlib import Path

from brainlayer.eval.enrichment_quality_benchmark import (
    BenchmarkConfig,
    MemorySnapshot,
    build_selection_query,
    enrichment_from_row,
    is_memory_safe_for_model,
    score_enrichment_pair,
    score_meaningful_chunk,
    select_meaningful_chunks,
)


def test_select_meaningful_chunks_uses_pure_content_not_existing_enrichment(tmp_path: Path):
    db_path = tmp_path / "brainlayer.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            project TEXT,
            content_type TEXT,
            source TEXT,
            source_file TEXT,
            created_at TEXT,
            char_count INTEGER,
            summary TEXT,
            tags TEXT,
            importance REAL,
            intent TEXT,
            key_facts TEXT,
            resolved_queries TEXT,
            raw_entities_json TEXT
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO chunks (
            id, content, project, content_type, source, source_file, created_at,
            char_count, summary, tags, importance, intent, key_facts,
            resolved_queries, raw_entities_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "decision-1",
                "DECISION locked: PR #481 is safe to merge after 13 tests pass. TASK_DONE",
                "brainlayer",
                "assistant_text",
                "claude_code",
                "session.jsonl",
                "2026-06-15T10:00:00Z",
                75,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
            (
                "boring-1",
                "ok thanks",
                "brainlayer",
                "user_text",
                "claude_code",
                "session.jsonl",
                "2026-06-15T10:01:00Z",
                9,
                "Gemini says this is an important correction decision",
                '["decision","correction","important"]',
                10,
                "deciding",
                '["Important baseline-only fact"]',
                '["important baseline only query"]',
                None,
            ),
        ],
    )
    conn.commit()
    conn.close()

    selected = select_meaningful_chunks(db_path, BenchmarkConfig(limit=10, since_days=30, min_score=4))

    assert [row["chunk_id"] for row in selected] == ["decision-1"]
    assert selected[0]["gemini_existing"] == {}


def test_score_meaningful_chunk_prioritizes_corrections_decisions_and_checkpoints():
    correction = score_meaningful_chunk("Etan correction: no, that is wrong; use Flex, not batch.", {})
    checkpoint = score_meaningful_chunk("DONE: pushed branch hc/wi2, commit a99f27f7, tests 13 passed.", {})
    chatter = score_meaningful_chunk("sounds good", {})

    assert correction.score >= 4
    assert "correction" in correction.reasons
    assert checkpoint.score >= 4
    assert "checkpoint" in checkpoint.reasons
    assert chatter.score < 4


def test_enrichment_from_row_keeps_existing_gemini_fields_separate_from_selection():
    row = {
        "summary": "Merged the guard PR.",
        "tags": '["brainlayer","pr"]',
        "importance": 8,
        "intent": "deciding",
        "key_facts": '["PR #481 passed guard tests"]',
        "resolved_queries": '["PR 481 guard tests", "brainlayer provenance guard", "WI2 merge"]',
        "raw_entities_json": '[{"name":"BrainLayer","type":"project"}]',
    }

    enrichment = enrichment_from_row(row)

    assert enrichment["summary"] == "Merged the guard PR."
    assert enrichment["tags"] == ["brainlayer", "pr"]
    assert enrichment["key_facts"] == ["PR #481 passed guard tests"]
    assert enrichment["entities"] == [{"name": "BrainLayer", "type": "project"}]


def test_memory_preflight_blocks_14b_when_available_memory_below_threshold():
    snapshot = MemorySnapshot(
        total_gb=36.0,
        available_gb=5.5,
        used_gb=30.5,
        swap_used_gb=8.0,
        swap_total_gb=13.0,
    )

    allowed, reason = is_memory_safe_for_model(snapshot, model_id="mlx-community/Qwen3-14B-4bit")

    assert allowed is False
    assert "available memory" in reason


def test_score_enrichment_pair_uses_schema_and_semantic_overlap():
    source = "DECISION: use Gemini Batch for the 64K backlog, not local. Tags must be fixed first."
    gemini = {
        "summary": "Use Gemini Batch for backlog after fixing tags.",
        "key_facts": ["64K backlog uses Gemini Batch", "tags fixed before re-enrich"],
        "tags": ["gemini", "batch", "tags"],
        "importance": 8,
        "intent": "deciding",
        "primary_symbols": [],
        "resolved_query": "Gemini Batch backlog tag fix decision",
        "resolved_queries": [
            "Gemini Batch backlog decision",
            "64K backlog enrichment backend",
            "tag fix before re-enrich",
        ],
        "epistemic_level": "validated",
        "version_scope": None,
        "debt_impact": "none",
        "external_deps": ["Gemini Batch"],
        "entities": [{"name": "Gemini Batch", "type": "tool"}],
        "sentiment_label": "neutral",
        "sentiment_score": 0,
        "sentiment_signals": [],
    }
    local = dict(gemini)
    local["summary"] = "Use local MLX for the backlog immediately."
    local["key_facts"] = ["local MLX backlog"]
    local["tags"] = ["local", "mlx", "backlog"]

    score = score_enrichment_pair(source, local, gemini)

    assert 0 <= score.overall <= 1
    assert score.overall < 0.8
    assert score.schema_passed is True


def test_build_selection_query_does_not_reference_enrichment_columns():
    query = build_selection_query(BenchmarkConfig(limit=100, since_days=14))

    lowered = query.lower()
    for forbidden in ["summary", "tags", "importance", "intent", "key_facts", "resolved_queries", "raw_entities_json"]:
        assert forbidden not in lowered
