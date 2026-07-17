import json
import sqlite3
from pathlib import Path

from brainlayer.watcher_bridge import create_flush_callback


def _entry(source_file: Path, text: str) -> dict:
    return {
        "type": "user",
        "message": {"content": [{"type": "text", "text": text}]},
        "timestamp": "2026-07-08T12:00:00Z",
        "_source_file": str(source_file),
        "_line_end_offset": 100,
    }


def _single_inserted_row(db_path: Path) -> tuple[str, str, dict, int, int]:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT content_class, provenance_class, metadata,
                   (SELECT COUNT(*) FROM chunks_fts WHERE chunk_id = chunks.id),
                   (SELECT COUNT(*) FROM chunks_fts_operational WHERE chunk_id = chunks.id)
            FROM chunks
            """
        ).fetchone()
    assert row is not None
    return row[0], row[1], json.loads(row[2]), row[3], row[4]


def test_direct_session_ingest_stays_default_visible_and_never_operational_or_cold(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    source_file = tmp_path / "home" / ".claude" / "projects" / "-Users-anyone-Gits-brainlayer" / "direct-session.jsonl"
    text = "Decision: keep direct control session notes searchable in the normal knowledge index for safety."
    flush = create_flush_callback(db_path=db_path, arbitrated=False)

    result = flush([_entry(source_file, text)])

    assert result.inserted == 1
    content_class, provenance_class, metadata, normal_fts_count, operational_fts_count = _single_inserted_row(db_path)
    assert (content_class, provenance_class, normal_fts_count, operational_fts_count) == (
        "decision",
        "direct-session",
        1,
        0,
    )
    assert metadata["provenance_tag"] == "direct-session"
    assert metadata["provenance_effective_visibility"] == "default"


def test_cursor_agent_transcript_ingest_is_tagged_and_routed_operational(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    source_file = (
        tmp_path / "home" / ".cursor" / "projects" / "brainlayer" / "agent-transcripts" / "cursor-gather.jsonl"
    )
    text = "This gather agent transcript found implementation notes that should not alter knowledge BM25 statistics."
    flush = create_flush_callback(db_path=db_path, arbitrated=False)

    result = flush([_entry(source_file, text)])

    assert result.inserted == 1
    content_class, provenance_class, metadata, normal_fts_count, operational_fts_count = _single_inserted_row(db_path)
    assert (content_class, provenance_class, normal_fts_count, operational_fts_count) == (
        "operational",
        "cursor-gather",
        0,
        1,
    )
    assert metadata["provenance_tag"] == "cursor-gather"
    assert metadata["provenance_effective_visibility"] == "operational"


def test_gemini_session_ingest_is_searchable_by_default(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    source_file = tmp_path / "home" / ".gemini" / "sessions" / "gemini-session.jsonl"
    text = "Gemini session synthesis should remain available to normal BrainLayer search."
    flush = create_flush_callback(db_path=db_path, arbitrated=False)

    result = flush([_entry(source_file, text)])

    assert result.inserted == 1
    content_class, provenance_class, metadata, normal_fts_count, operational_fts_count = _single_inserted_row(db_path)
    assert (content_class, provenance_class, normal_fts_count, operational_fts_count) == (
        "knowledge",
        "gemini-session",
        1,
        0,
    )
    assert metadata["provenance_tag"] == "gemini-session"
    assert metadata["provenance_effective_visibility"] == "default"
