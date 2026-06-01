import json
import sqlite3
from pathlib import Path

import pytest


def _create_snapshot(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            content_type TEXT,
            content_class TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO chunks (id, content, content_type, content_class) VALUES (?, ?, ?, ?)",
        ("source-1", "Synthetic experiment isolation chunk", "conversation", "knowledge"),
    )
    conn.commit()
    conn.close()


def _create_live_stand_in(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, content TEXT NOT NULL)")
    conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(id, content)")
    rows = [
        ("live-1", "BrainLayer search isolation must remain stable."),
        ("live-2", "Unrelated note about local testing."),
    ]
    conn.executemany("INSERT INTO chunks (id, content) VALUES (?, ?)", rows)
    conn.executemany("INSERT INTO chunks_fts (id, content) VALUES (?, ?)", rows)
    conn.commit()
    conn.close()


def _live_state(path: Path) -> dict[str, object]:
    conn = sqlite3.connect(path)
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name"
        ).fetchall()
    ]
    chunks = conn.execute("SELECT id, content FROM chunks ORDER BY id").fetchall()
    fts_rows = conn.execute("SELECT id, content FROM chunks_fts ORDER BY id").fetchall()
    conn.close()
    return {"tables": tables, "chunks": chunks, "fts_rows": fts_rows}


def _synthetic_brain_search(path: Path, query: str) -> list[tuple[str, str]]:
    conn = sqlite3.connect(path)
    rows = conn.execute(
        """
        SELECT chunks.id, chunks.content
        FROM chunks_fts
        JOIN chunks ON chunks.id = chunks_fts.id
        WHERE chunks_fts MATCH ?
        ORDER BY rank
        """,
        (query,),
    ).fetchall()
    conn.close()
    return rows


def test_experiment_store_rejects_canonical_live_db_path():
    from brainlayer.eval.experiment_store import ExperimentStore
    from brainlayer.paths import get_db_path

    with pytest.raises(ValueError, match="live BrainLayer DB"):
        ExperimentStore(experiment_db_path=get_db_path())


def test_experiment_store_rejects_snapshot_at_canonical_live_db_path():
    from brainlayer.eval.experiment_store import ExperimentStore
    from brainlayer.paths import get_db_path

    with pytest.raises(ValueError, match="live BrainLayer DB"):
        ExperimentStore(snapshot_db_path=get_db_path())


def test_experiment_write_cycle_stays_in_namespace_db_and_leaves_live_stand_in_untouched(tmp_path):
    from brainlayer.eval.experiment_store import ExperimentStore

    live_db = tmp_path / "brainlayer.db"
    snapshot_db = tmp_path / "experiments" / "abcde-snapshot.db"
    experiment_db = tmp_path / "experiments" / "abcde-experiment.db"
    snapshot_db.parent.mkdir()
    _create_live_stand_in(live_db)
    _create_snapshot(snapshot_db)

    before_state = _live_state(live_db)

    with ExperimentStore(experiment_db_path=experiment_db, snapshot_db_path=snapshot_db) as store:
        chunk_id = store.upsert_chunk(
            source_chunk_id="source-1",
            raw_text="Synthetic experiment isolation chunk",
            content_type="conversation",
            content_class="knowledge",
            strata={"content_type": "conversation", "content_class": "knowledge"},
        )
        store.upsert_variant(
            chunk_id=chunk_id,
            variant_id="A",
            enrichment={"summary": "control"},
            model="synthetic",
            prompt_hash="prompt-a",
            grader_scores={"schema": 1.0},
        )
        store.add_judgment(
            chunk_id=chunk_id,
            variant_id="A",
            source="llm",
            scores={"faithfulness": 1.0},
            better_option_flag=False,
            rationale="synthetic regression fixture",
        )

    assert _live_state(live_db) == before_state

    conn = sqlite3.connect(experiment_db)
    exp_tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"exp_chunks", "exp_variants", "exp_judgments", "exp_index_documents"} <= exp_tables
    assert conn.execute("SELECT source_chunk_id, content_class FROM exp_chunks").fetchall() == [
        ("source-1", "knowledge")
    ]
    assert conn.execute("SELECT variant_id, enrichment_json FROM exp_variants").fetchall() == [
        ("A", json.dumps({"summary": "control"}, sort_keys=True))
    ]
    assert conn.execute("SELECT source, better_option_flag FROM exp_judgments").fetchall() == [("llm", 0)]
    conn.close()


def test_synthetic_brain_search_results_are_identical_after_experiment_writes(tmp_path):
    from brainlayer.eval.experiment_store import ExperimentStore

    live_db = tmp_path / "brainlayer.db"
    snapshot_db = tmp_path / "experiments" / "abcde-snapshot.db"
    experiment_db = tmp_path / "experiments" / "abcde-experiment.db"
    snapshot_db.parent.mkdir()
    _create_live_stand_in(live_db)
    _create_snapshot(snapshot_db)

    before = _synthetic_brain_search(live_db, "BrainLayer")
    with ExperimentStore(experiment_db_path=experiment_db, snapshot_db_path=snapshot_db) as store:
        chunk_id = store.upsert_chunk(
            source_chunk_id="source-1",
            raw_text="This experiment output mentions BrainLayer but must not affect live search.",
            content_type="conversation",
            content_class="test",
            strata={"content_type": "conversation", "content_class": "test"},
        )
        store.upsert_variant(
            chunk_id=chunk_id,
            variant_id="B",
            enrichment={"summary": "experiment-only BrainLayer output"},
            model="synthetic",
            prompt_hash="prompt-b",
        )

    assert _synthetic_brain_search(live_db, "BrainLayer") == before


def test_materialize_snapshot_refuses_to_overwrite_live_db_or_run_without_explicit_source(tmp_path):
    from brainlayer.eval.experiment_store import materialize_snapshot
    from brainlayer.paths import get_db_path

    with pytest.raises(ValueError, match="backup_gzip_path is required"):
        materialize_snapshot(backup_gzip_path=None, snapshot_db_path=tmp_path / "abcde-snapshot.db")

    with pytest.raises(ValueError, match="live BrainLayer DB"):
        materialize_snapshot(backup_gzip_path=tmp_path / "backup.gz", snapshot_db_path=get_db_path())
