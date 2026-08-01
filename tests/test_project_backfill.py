import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.d2_project_backfill as project_backfill
from brainlayer.vector_store import WriterInUseError
from scripts.d2_project_backfill import backfill_numeric_projects


def _create_chunks_db(path, recoverable_source, missing_source, untouched_source):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, source_file TEXT NOT NULL, project TEXT)")
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
    conn.execute(
        "INSERT INTO chunks VALUES ('already-null', ?, NULL)",
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
        "already-null": None,
        "recoverable": "brainlayer",
        "underivable": None,
        "untouched": "already-real",
    }
    assert first.rows_rederived == 1
    assert first.rows_set_null == 1
    assert first.rows_left_untouched == 2
    assert rollback_path.read_text(encoding="utf-8").splitlines() == [
        "id\tproject",
        "recoverable\t30",
        "underivable\t7",
    ]

    second = backfill_numeric_projects(
        db_path,
        rollback_path=tmp_path / "second-rollback.tsv",
        batch_size=5_000,
    )

    conn = sqlite3.connect(db_path)
    rows_after_second = dict(conn.execute("SELECT id, project FROM chunks"))
    conn.close()
    assert second.rows_updated == 0
    assert rows_after_second == rows_after_first


def test_backfill_requires_rollback_artifact_before_mutating_rows(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    source = tmp_path / "sessions" / "recoverable.jsonl"
    missing_source = tmp_path / "sessions" / "missing.jsonl"
    untouched_source = tmp_path / "sessions" / "untouched.jsonl"
    _create_chunks_db(db_path, source, missing_source, untouched_source)

    with pytest.raises(ValueError, match="rollback_path is required"):
        backfill_numeric_projects(db_path)

    conn = sqlite3.connect(db_path)
    assert dict(conn.execute("SELECT id, project FROM chunks"))["recoverable"] == "30"
    conn.close()


def test_backfill_does_not_mutate_when_rollback_artifact_cannot_be_created(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    source = tmp_path / "sessions" / "recoverable.jsonl"
    missing_source = tmp_path / "sessions" / "missing.jsonl"
    untouched_source = tmp_path / "sessions" / "untouched.jsonl"
    _create_chunks_db(db_path, source, missing_source, untouched_source)
    rollback_path = tmp_path / "rollback.tsv"
    rollback_path.write_text("existing artifact\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="rollback artifact already exists"):
        backfill_numeric_projects(db_path, rollback_path=rollback_path)

    conn = sqlite3.connect(db_path)
    assert dict(conn.execute("SELECT id, project FROM chunks"))["recoverable"] == "30"
    conn.close()


def test_backfill_refuses_while_standard_writer_lock_is_held(tmp_path, monkeypatch):
    db_path = tmp_path / "brainlayer.db"
    source = tmp_path / "sessions" / "recoverable.jsonl"
    missing_source = tmp_path / "sessions" / "missing.jsonl"
    untouched_source = tmp_path / "sessions" / "untouched.jsonl"
    _create_chunks_db(db_path, source, missing_source, untouched_source)
    pidfile_dir = tmp_path / "pidfiles"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))
    repo_root = Path(__file__).resolve().parents[1]
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            """
import sys
from pathlib import Path
from brainlayer.vector_store import VectorStore

store = VectorStore.__new__(VectorStore)
store.db_path = Path(sys.argv[1])
store._writer_pidfile_acquired = False
store._acquire_writer_pidfile()
print("ready", flush=True)
sys.stdin.readline()
store._release_writer_pidfile()
""",
            str(db_path),
        ],
        env={
            **os.environ,
            "PYTHONPATH": f"{repo_root / 'src'}:{os.environ.get('PYTHONPATH', '')}",
        },
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        readiness = holder.stdout.readline().strip()
        if readiness != "ready":
            assert holder.stderr is not None
            stderr = holder.stderr.read() if holder.poll() is not None else "holder did not exit"
            pytest.fail(f"writer lock holder did not become ready: {stderr}")
        with pytest.raises(WriterInUseError, match="another writer is using"):
            backfill_numeric_projects(db_path, rollback_path=tmp_path / "rollback.tsv")
    finally:
        if holder.poll() is None and holder.stdin is not None:
            try:
                holder.stdin.write("release\n")
                holder.stdin.close()
                holder.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                holder.terminate()
                holder.wait(timeout=5)

    conn = sqlite3.connect(db_path)
    assert dict(conn.execute("SELECT id, project FROM chunks"))["recoverable"] == "30"
    conn.close()


def test_backfill_counts_only_rows_changed_by_numeric_guard(tmp_path, monkeypatch):
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
    original_extract = project_backfill._extract_project_from_session_file

    def change_project_before_write(source_file):
        if source_file == str(source):
            conn = sqlite3.connect(db_path)
            conn.execute("UPDATE chunks SET project = 'already-real' WHERE id = 'recoverable'")
            conn.commit()
            conn.close()
        return original_extract(source_file)

    monkeypatch.setattr(project_backfill, "_extract_project_from_session_file", change_project_before_write)
    result = backfill_numeric_projects(db_path, rollback_path=tmp_path / "rollback.tsv")

    assert result.rows_updated == 1
    assert result.rows_rederived == 0
    assert result.rows_set_null == 1


@pytest.mark.parametrize("batch_size", (4_999, 10_001))
def test_backfill_requires_production_safe_batch_sizes(tmp_path, batch_size):
    with pytest.raises(ValueError, match="5,000 and 10,000"):
        backfill_numeric_projects(tmp_path / "brainlayer.db", batch_size=batch_size)
