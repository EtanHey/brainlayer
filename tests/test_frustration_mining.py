"""Tests for frustration-to-benchmark mining."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def frustration_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "frustration.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            tags TEXT,
            sentiment_label TEXT,
            conversation_id TEXT,
            position INTEGER,
            source TEXT
        )
        """
    )
    rows = [
        (
            "ctx-1",
            "How should we handle DB locking in BrainLayer?",
            json.dumps(["question"]),
            "neutral",
            "conv-1",
            1,
            "chat",
        ),
        (
            "fr-1",
            "You should remember this. We discussed using WAL mode and a busy timeout for DB locking.",
            json.dumps(["frustration", "expectation-failure"]),
            "frustration",
            "conv-1",
            2,
            "chat",
        ),
        (
            "target-1",
            "Decision: use WAL mode and a 30 second busy timeout to reduce DB locking issues in BrainLayer.",
            json.dumps(["decision", "database"]),
            "neutral",
            "conv-9",
            1,
            "note",
        ),
        (
            "casual-1",
            "Can you send that again?",
            json.dumps(["question"]),
            "neutral",
            "conv-2",
            1,
            "chat",
        ),
        (
            "ctx-he",
            "מה החלטנו לגבי זיכרון והפריסה?",
            json.dumps(["question"]),
            "neutral",
            "conv-3",
            1,
            "chat",
        ),
        (
            "fr-he",
            "למה אתה לא זוכר? כבר דיברנו על תוכנית הפריסה אתמול.",
            json.dumps(["frustration"]),
            "frustration",
            "conv-3",
            2,
            "chat",
        ),
        (
            "target-he",
            "סיכום: תוכנית הפריסה משתמשת ב-Railway עם בדיקות בריאות לפני החלפה.",
            json.dumps(["deployment"]),
            "neutral",
            "conv-8",
            1,
            "note",
        ),
    ]
    conn.executemany(
        "INSERT INTO chunks(id, content, tags, sentiment_label, conversation_id, position, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db_path


def test_detect_frustration_english():
    from brainlayer.pipeline.frustration_mining import detect_frustration

    assert detect_frustration("You should know this already.") is True


def test_detect_frustration_hebrew():
    from brainlayer.pipeline.frustration_mining import detect_frustration

    assert detect_frustration("למה אתה לא זוכר?") is True


def test_extract_query_from_context(frustration_db: Path):
    from brainlayer.pipeline.frustration_mining import extract_query_from_context, load_conversation_context

    with sqlite3.connect(frustration_db) as conn:
        context = load_conversation_context(conn, "fr-1")

    assert extract_query_from_context(context, frustration_index=1) == "How should we handle DB locking in BrainLayer?"


def test_extract_expected_result(frustration_db: Path):
    from brainlayer.pipeline.frustration_mining import extract_expected_result, load_conversation_context

    with sqlite3.connect(frustration_db) as conn:
        context = load_conversation_context(conn, "fr-1")

    expected = extract_expected_result(context[1]["content"], context)

    assert "WAL mode" in expected
    assert "busy timeout" in expected
    assert "DB locking" in expected


def test_generate_qrels_format():
    from brainlayer.pipeline.frustration_mining import FrustrationPair, generate_qrels

    qrels = generate_qrels(
        [
            FrustrationPair(
                query_id="frustration_001",
                query_text="db locking",
                expected_result="use WAL mode",
                chunk_ids=["target-1", "target-2"],
            )
        ]
    )

    assert qrels == {"frustration_001": {"target-1": 3, "target-2": 3}}


def test_no_false_positives():
    from brainlayer.pipeline.frustration_mining import detect_frustration

    assert detect_frustration("Can you send that again?") is False


def test_pipeline_end_to_end(frustration_db: Path):
    from brainlayer.pipeline.frustration_mining import mine_frustration_pairs

    with sqlite3.connect(frustration_db) as conn:
        pairs = mine_frustration_pairs(conn)

    assert len(pairs) == 2

    english_pair = next(pair for pair in pairs if pair.query_id == "frustration_001")
    hebrew_pair = next(pair for pair in pairs if pair.query_id == "frustration_002")

    assert english_pair.query_text == "How should we handle DB locking in BrainLayer?"
    assert english_pair.chunk_ids == ["target-1"]
    assert "WAL mode" in english_pair.expected_result

    assert hebrew_pair.query_text == "מה החלטנו לגבי זיכרון והפריסה?"
    assert hebrew_pair.chunk_ids == ["target-he"]
    assert "תוכנית הפריסה" in hebrew_pair.expected_result
