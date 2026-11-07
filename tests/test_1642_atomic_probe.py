import apsw
import pytest

from brainlayer.vector_store import VectorStore
from tests.test_runtime_ddl_ledger import SENTINEL, _connect, _current_db, _degrade_to_legacy_fts


def test_fts_repair_remains_atomic_through_trigger_and_backfill_failure(tmp_path):
    db = tmp_path / "copy.db"
    _current_db(db)
    _degrade_to_legacy_fts(db)
    conn = _connect(db)
    for trigger in ("chunk_tags_insert", "chunk_tags_update", "chunk_tags_update_clear", "chunk_tags_delete"):
        conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    conn.execute("DROP TABLE chunk_tags")
    conn.execute("CREATE VIEW chunk_tags AS SELECT id AS chunk_id, tags AS tag FROM chunks")
    conn.close()
    with pytest.raises(apsw.SQLError):
        VectorStore(db)
    conn = _connect(db)
    try:
        actual = conn.execute("SELECT content FROM chunks_fts WHERE chunk_id = 'c1'").fetchall()
        assert actual == [(SENTINEL,)], f"committed FTS repair lost prior searchable row: {actual}"
    finally:
        conn.close()


def test_backfill_failure_preserves_old_searchable_row(tmp_path):
    db = tmp_path / "backfill.db"
    _current_db(db)
    _degrade_to_legacy_fts(db)
    conn = _connect(db)
    conn.execute("UPDATE chunks SET tags = ?", ('["sentinel"]',))
    conn.execute(
        "CREATE TRIGGER reject_backfill BEFORE INSERT ON chunk_fts_rowids BEGIN SELECT RAISE(ABORT, 'backfill failed'); END"
    )
    conn.close()
    with pytest.raises(apsw.ConstraintError, match="backfill failed"):
        VectorStore(db)
    conn = _connect(db)
    try:
        assert conn.execute("SELECT content FROM chunks_fts WHERE chunk_id='c1'").fetchall() == [(SENTINEL,)]
    finally:
        conn.close()
