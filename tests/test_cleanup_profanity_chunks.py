import importlib.util
import json
import sqlite3
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cleanup-profanity-chunks.py"


def load_script():
    spec = importlib.util.spec_from_file_location("cleanup_profanity_chunks", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "profanity-cleanup.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            metadata TEXT NOT NULL,
            source_file TEXT NOT NULL,
            project TEXT,
            content_type TEXT,
            value_type TEXT,
            char_count INTEGER,
            tags TEXT,
            summary TEXT,
            importance REAL,
            created_at TEXT,
            conversation_id TEXT,
            resolved_query TEXT,
            resolved_queries TEXT,
            aggregated_into TEXT,
            archived_at TEXT,
            archived INTEGER DEFAULT 0,
            content_hash TEXT
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            content, summary, tags, resolved_query, resolved_queries, chunk_id UNINDEXED
        );
        CREATE TRIGGER chunks_fts_insert AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(content, summary, tags, resolved_query, resolved_queries, chunk_id)
            VALUES (new.content, new.summary, new.tags, new.resolved_query, new.resolved_queries, new.id);
        END;
        CREATE TRIGGER chunks_fts_update
        AFTER UPDATE OF content, summary, tags, resolved_query, resolved_queries ON chunks BEGIN
            DELETE FROM chunks_fts WHERE chunk_id = old.id;
            INSERT INTO chunks_fts(content, summary, tags, resolved_query, resolved_queries, chunk_id)
            VALUES (new.content, new.summary, new.tags, new.resolved_query, new.resolved_queries, new.id);
        END;
        """
    )
    rows = [
        ("rt-a1", "Mehayom lets fucking ship this build tonight", "mehayom", "2026-04-15T20:27:18.000Z", None, ["shipping", "build"], 7.0),
        ("rt-a2", "Mehayom fix the fucking build and ship", "mehayom", "2026-04-15T20:28:18.000Z", "sess-1", ["shipping", "rtl"], 8.0),
        ("rt-a3", "Mehayom we are fucking shipping after build fix", "mehayom", "2026-04-15T20:29:18.000Z", "sess-1", ["shipping", "android"], 6.0),
        ("rt-b1", "Mini app fucking auth bug", "mini", "2026-04-15T10:00:00.000Z", "sess-2", ["auth"], 5.0),
        ("rt-b2", "Mini app fucking auth still broken", "mini", "2026-04-15T10:01:00.000Z", "sess-2", ["auth"], 5.0),
    ]
    for chunk_id, content, project, created_at, session_id, tags, importance in rows:
        metadata_session_id = "sess-1" if chunk_id == "rt-a1" else session_id
        conn.execute(
            """
            INSERT INTO chunks (
                id, content, metadata, source_file, project, content_type, char_count,
                tags, summary, importance, created_at, conversation_id
            ) VALUES (?, ?, ?, ?, ?, 'user_message', ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_id,
                content,
                json.dumps({"session_id": metadata_session_id}),
                f"/tmp/{session_id}.jsonl",
                project,
                len(content),
                json.dumps(tags),
                content[:200],
                importance,
                created_at,
                session_id,
            ),
        )
    conn.execute(
        """
        INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type, char_count,
            tags, summary, importance, created_at, conversation_id
        ) VALUES (?, ?, ?, ?, ?, 'user_message', ?, ?, ?, ?, ?, ?)
        """,
        (
            "rt-bad-metadata",
            "Mini app fucking parser edge case",
            "{bad json",
            "/tmp/bad.jsonl",
            "mini",
            len("Mini app fucking parser edge case"),
            json.dumps(["auth"]),
            "Mini app fucking parser edge case",
            4.0,
            "2026-04-15T11:00:00.000Z",
            None,
        ),
    )
    conn.commit()
    conn.close()
    return db_path


def test_dry_run_reports_changes_without_writing(tmp_path, capsys):
    module = load_script()
    db_path = make_db(tmp_path)

    stats = module.run_cleanup(str(db_path), dry_run=True)

    assert stats == {"clusters": 1, "aggregated": 3, "archived": 3}
    stderr = capsys.readouterr().err
    assert "clusters=1" in stderr
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM chunks WHERE aggregated_into IS NOT NULL").fetchone()[0] == 0
    conn.close()


def test_execute_aggregates_cluster_and_preserves_searchability(tmp_path, capsys):
    module = load_script()
    db_path = make_db(tmp_path)

    stats = module.run_cleanup(str(db_path), dry_run=False)

    assert stats == {"clusters": 1, "aggregated": 3, "archived": 3}
    stderr = capsys.readouterr().err
    assert "archived=3" in stderr
    conn = sqlite3.connect(db_path)
    agg = conn.execute(
        """
        SELECT id, content, tags, importance, created_at, resolved_query, resolved_queries, conversation_id
        FROM chunks
        WHERE id LIKE 'agg-frustration-%'
        """
    ).fetchone()
    assert agg is not None
    assert "[USER FRUSTRATION 3x on 2026-04-15]" in agg[1]
    assert "user-frustration-aggregated" in agg[2]
    assert agg[3] == 8.0
    assert agg[4] == "2026-04-15T20:27:18.000Z"
    assert "fucking" in (agg[5] or "")
    assert "ship" in (agg[6] or "")
    assert agg[7] == "sess-1"
    originals = conn.execute(
        "SELECT value_type FROM chunks WHERE id IN ('rt-a1','rt-a2','rt-a3') AND aggregated_into = ? AND archived = 1 AND archived_at IS NOT NULL",
        (agg[0],),
    ).fetchall()
    assert len(originals) == 3
    assert {row[0] for row in originals} == {"ARCHIVED"}
    untouched = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE id IN ('rt-b1','rt-b2','rt-bad-metadata') AND aggregated_into IS NULL AND archived = 0"
    ).fetchone()[0]
    assert untouched == 3
    match = conn.execute(
        """
        SELECT c.id
        FROM chunks_fts f
        JOIN chunks c ON c.id = f.chunk_id
        WHERE chunks_fts MATCH ? AND c.aggregated_into IS NULL AND c.archived_at IS NULL
        ORDER BY rank
        """,
        ('"fucking" AND "ship" AND "mehayom"',),
    ).fetchall()
    assert [row[0] for row in match] == [agg[0]]
    conn.close()
