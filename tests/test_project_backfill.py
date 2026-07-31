import json
import sqlite3

import pytest

from scripts.d2_project_backfill import backfill_numeric_projects


def _create_chunks_db(path, recoverable_source, missing_source, untouched_source):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE chunks (id TEXT PRIMARY KEY, source_file TEXT NOT NULL, project TEXT)"
    )
    conn.execute(
        "INSERT INTO chunks VALUES ('recoverable', ?, '30')",
        (str(recoverable_source),),
    )
    conn.execute(
        "INSERT INTO chunks VALUES ('underivable', ?, '7')",
        (str(missing_source),),
    )
    conn.execute(
        "INSERT INTO chunks VALUES ('untouched', ?, 'already-real')",
        (str(untouched_source),),
    )
    conn.commit()
    conn.close()


def test_backfill_exports_rollback_derives_projects_and_is_idempotent(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    source = tmp_path / "sessions" / "recoverable.jsonl"
    missing_source = tmp_path / "sessions" / "missing.jsonl"
    untouched_source = tmp_path / "sessions" / "untouched.jsonl"
    _create_chunks_db(db_path, source, missing_source, untouched_source)
    source.parent.mkdir()
    source.write_text(
        json.dumps({"type": "session_meta", "payload": {"cwd": "/Users/test/Gits/brainlayer"}}) + "\n",
        encoding="utf-8",
    )
    rollback_path = tmp_path / "rollback.tsv"

    first = backfill_numeric_projects(
        db_path,
        rollback_path=rollback_path,
        batch_size=5_000,
    )

    conn = sqlite3.connect(db_path)
    rows_after_first = dict(conn.execute("SELECT id, project FROM chunks"))
    conn.close()
    assert rows_after_first == {
        "recoverable": "brainlayer",
        "underivable": None,
        "untouched": "already-real",
    }
    assert first.rows_rederived == 1
    assert first.rows_set_null == 1
    assert first.rows_left_untouched == 1
    assert rollback_path.read_text(encoding="utf-8").splitlines() == [
        "id\tproject",
        "recoverable\t30",
        "underivable\t7",
    ]

    second = backfill_numeric_projects(
        db_path,
        batch_size=5_000,
    )

    conn = sqlite3.connect(db_path)
    rows_after_second = dict(conn.execute("SELECT id, project FROM chunks"))
    conn.close()
    assert second.rows_updated == 0
    assert rows_after_second == rows_after_first


@pytest.mark.parametrize("batch_size", (4_999, 10_001))
def test_backfill_requires_production_safe_batch_sizes(tmp_path, batch_size):
    with pytest.raises(ValueError, match="5,000 and 10,000"):
        backfill_numeric_projects(tmp_path / "brainlayer.db", batch_size=batch_size)
