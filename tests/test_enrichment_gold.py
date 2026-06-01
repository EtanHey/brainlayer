import json
import sqlite3

import pytest


def _insert_chunk(conn: sqlite3.Connection, chunk_id: str, content: str, **fields: str) -> None:
    conn.execute(
        """
        INSERT INTO chunks (
            id, content, source_file, project, content_type, tags, summary,
            sentiment_label, sender, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            chunk_id,
            content,
            fields.get("source_file", f"/tmp/{chunk_id}.jsonl"),
            fields.get("project", "brainlayer"),
            fields.get("content_type", ""),
            fields.get("tags", "[]"),
            fields.get("summary", ""),
            fields.get("sentiment_label", ""),
            fields.get("sender", "user"),
            fields.get("created_at", "2026-06-01T00:00:00Z"),
        ),
    )


def test_stratified_enrichment_gold_sampler_writes_portable_jsonl(tmp_path):
    from brainlayer.eval.enrichment_gold import sample_enrichment_gold

    db_path = tmp_path / "snapshot.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            source_file TEXT,
            project TEXT,
            content_type TEXT,
            tags TEXT,
            summary TEXT,
            sentiment_label TEXT,
            sender TEXT,
            created_at TEXT
        )
        """
    )
    claude_source = "/Users/etan/.claude/projects/-Users-etan-Gits-brainlayer/session.jsonl"
    _insert_chunk(
        conn,
        "decision-1",
        "Decision: keep the enrichment gold sampler snapshot-only because live DB load is unsafe.",
        source_file=claude_source,
        tags='["decision"]',
    )
    _insert_chunk(conn, "code-1", "def sample_enrichment_gold(snapshot_path):\n    return []", content_type="code")
    _insert_chunk(conn, "conversation-1", "User: can we label this?\nAssistant: yes, using a local HTML labeler.")
    _insert_chunk(conn, "correction-1", "Correction: the prior summary missed PR #413 and must be fixed.")
    _insert_chunk(conn, "entity-1", "Etan Heyman is the owner of BrainLayer and VoiceLayer.", tags='["person","bio"]')
    _insert_chunk(conn, "short-1", "ok", sender="assistant")
    _insert_chunk(conn, "long-1", "A" * 8101)
    _insert_chunk(
        conn, "meta-1", "brain_search('enrichment prompt eval') returned prior research.", tags='["meta-research"]'
    )
    _insert_chunk(
        conn,
        "sentiment-1",
        "This is frustrating and confusing; the DB lock wasted the run.",
        sentiment_label="frustration",
    )
    conn.commit()
    conn.close()

    output_path = tmp_path / "gold.jsonl"
    records = sample_enrichment_gold(
        snapshot_path=db_path,
        output_path=output_path,
        target_size=9,
        seed=123,
    )

    written = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert records == written
    assert len(written) == 9
    assert {row["stratum"] for row in written} == {
        "decision",
        "code",
        "conversation",
        "correction",
        "entity_bio",
        "short_conversational",
        "long_truncation",
        "meta_research",
        "sentiment",
    }
    assert {row["chunk_id"] for row in written} == {
        "decision-1",
        "code-1",
        "conversation-1",
        "correction-1",
        "entity-1",
        "short-1",
        "long-1",
        "meta-1",
        "sentiment-1",
    }
    decision = next(row for row in written if row["chunk_id"] == "decision-1")
    assert decision["eval_doc_id"] == "claude-project:Gits-brainlayer/session.jsonl#decision-1"
    long_record = next(row for row in written if row["stratum"] == "long_truncation")
    assert long_record["content_length"] == 8101
    assert long_record["content"] == "A" * 8101


def test_sampler_rejects_live_production_db_path():
    from brainlayer.eval.enrichment_gold import sample_enrichment_gold

    with pytest.raises(ValueError, match="live BrainLayer DB"):
        sample_enrichment_gold(
            snapshot_path="~/.local/share/brainlayer/brainlayer.db",
            output_path="/tmp/unused.jsonl",
        )
