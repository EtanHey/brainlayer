"""Tests for the repair-(b) legacy model chunk_origin wipe/re-derive."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from brainlayer.chunk_origin import (
    CHUNK_ORIGIN_GEMINI_FLASH_LITE,
    CHUNK_ORIGIN_GROQ,
    CHUNK_ORIGIN_MANUAL,
    CHUNK_ORIGIN_MLX,
    CHUNK_ORIGIN_OLLAMA,
    CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
    CHUNK_ORIGIN_UNKNOWN,
    LEGACY_MODEL_CHUNK_ORIGINS,
    detect_chunk_origin,
)

PRECOMPACT_CONTENT = "[PreCompact checkpoint]\ntimestamp: 2026-05-16\nsession_id: abc"


def _make_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            chunk_origin TEXT
        )
        """
    )
    conn.execute("CREATE TABLE chunks_fts (chunk_id TEXT)")
    conn.execute("CREATE TABLE chunk_vectors_rowids (id TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE chunk_fts_rowids (chunk_id TEXT PRIMARY KEY)")
    return conn


def _seed(conn: sqlite3.Connection) -> None:
    rows = [
        ("gemini-plain", "ordinary assistant memory", CHUNK_ORIGIN_GEMINI_FLASH_LITE),
        ("groq-plain", "another ordinary memory", CHUNK_ORIGIN_GROQ),
        ("ollama-plain", "yet more memory", CHUNK_ORIGIN_OLLAMA),
        ("mlx-precompact", PRECOMPACT_CONTENT, CHUNK_ORIGIN_MLX),
        ("keep-unknown", "untouched unknown row", CHUNK_ORIGIN_UNKNOWN),
        ("keep-manual", "untouched manual row", CHUNK_ORIGIN_MANUAL),
        ("keep-raw", "untouched raw row", "raw"),
    ]
    conn.executemany(
        "INSERT INTO chunks (id, content, chunk_origin) VALUES (?, ?, ?)",
        rows,
    )
    conn.executemany("INSERT INTO chunks_fts (chunk_id) VALUES (?)", [(row[0],) for row in rows])
    conn.executemany("INSERT INTO chunk_vectors_rowids (id) VALUES (?)", [(row[0],) for row in rows])
    conn.executemany("INSERT INTO chunk_fts_rowids (chunk_id) VALUES (?)", [(row[0],) for row in rows])
    conn.commit()


def test_wipe_refuses_live_canonical_path(tmp_path, monkeypatch):
    from brainlayer.chunk_origin_wipe import wipe_legacy_model_chunk_origins

    fake_home = tmp_path / "home"
    live = fake_home / ".local" / "share" / "brainlayer" / "brainlayer.db"
    live.parent.mkdir(parents=True)
    live.write_bytes(b"not-a-real-db")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    with pytest.raises(RuntimeError, match="live BrainLayer DB"):
        wipe_legacy_model_chunk_origins(live, apply=True)


def test_wipe_dry_run_does_not_write(tmp_path):
    from brainlayer.chunk_origin_wipe import wipe_legacy_model_chunk_origins

    db_path = tmp_path / "copy.db"
    conn = _make_db(db_path)
    _seed(conn)
    conn.close()

    result = wipe_legacy_model_chunk_origins(db_path, apply=False, batch_size=2)

    assert result.updated == 0
    assert result.pre_wipe_legacy == 4
    assert result.derived[CHUNK_ORIGIN_UNKNOWN] == 3
    assert result.derived[CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT] == 1
    conn = sqlite3.connect(db_path)
    remaining = dict(conn.execute("SELECT id, chunk_origin FROM chunks"))
    conn.close()
    assert remaining["gemini-plain"] == CHUNK_ORIGIN_GEMINI_FLASH_LITE
    assert remaining["mlx-precompact"] == CHUNK_ORIGIN_MLX


def test_wipe_apply_rederives_legacy_models_and_preserves_others(tmp_path):
    from brainlayer.chunk_origin_wipe import wipe_legacy_model_chunk_origins

    db_path = tmp_path / "copy.db"
    conn = _make_db(db_path)
    _seed(conn)
    conn.close()

    result = wipe_legacy_model_chunk_origins(db_path, apply=True, batch_size=2, checkpoint_every=3)

    assert result.updated == 4
    assert result.pre_wipe_legacy == 4
    assert result.post_wipe_legacy == 0
    conn = sqlite3.connect(db_path)
    remaining = dict(conn.execute("SELECT id, chunk_origin FROM chunks"))
    conn.close()
    assert remaining["gemini-plain"] == CHUNK_ORIGIN_UNKNOWN
    assert remaining["groq-plain"] == CHUNK_ORIGIN_UNKNOWN
    assert remaining["ollama-plain"] == CHUNK_ORIGIN_UNKNOWN
    assert remaining["mlx-precompact"] == CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT
    assert remaining["keep-unknown"] == CHUNK_ORIGIN_UNKNOWN
    assert remaining["keep-manual"] == CHUNK_ORIGIN_MANUAL
    assert remaining["keep-raw"] == "raw"
    assert not LEGACY_MODEL_CHUNK_ORIGINS.intersection(remaining.values())
    assert remaining["mlx-precompact"] == detect_chunk_origin(PRECOMPACT_CONTENT)


def test_wipe_leaves_fts_and_vector_row_counts_unchanged(tmp_path):
    from brainlayer.chunk_origin_wipe import wipe_legacy_model_chunk_origins

    db_path = tmp_path / "copy.db"
    conn = _make_db(db_path)
    _seed(conn)
    conn.close()

    result = wipe_legacy_model_chunk_origins(db_path, apply=True)

    assert result.aux_counts_before == result.aux_counts_after
    assert result.aux_counts_after["chunks_fts"] == 7
    assert result.aux_counts_after["chunk_vectors_rowids"] == 7


def test_wipe_checkpoints_every_three_batches(tmp_path):
    from brainlayer.chunk_origin_wipe import wipe_legacy_model_chunk_origins

    db_path = tmp_path / "copy.db"
    conn = _make_db(db_path)
    _seed(conn)
    conn.close()

    result = wipe_legacy_model_chunk_origins(db_path, apply=True, batch_size=2, checkpoint_every=3)

    assert result.batches == 2
    assert result.checkpoints >= 1


def test_cli_requires_db_and_defaults_to_dry_run(tmp_path, capsys):
    from brainlayer.chunk_origin_wipe import main

    db_path = tmp_path / "copy.db"
    conn = _make_db(db_path)
    _seed(conn)
    conn.close()

    assert main(["--db", str(db_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "dry-run"
    assert payload["updated"] == 0
    conn = sqlite3.connect(db_path)
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE chunk_origin = ?",
            (CHUNK_ORIGIN_GEMINI_FLASH_LITE,),
        ).fetchone()[0]
        == 1
    )
    conn.close()
