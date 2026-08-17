"""Tests for the repair-(b) legacy model chunk_origin wipe/re-derive."""

from __future__ import annotations

import json
import os
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
PREIMAGE_TABLE = "chunk_origin_wipe_preimage"

# Real chunks_fts_update UPDATE OF list from vector_store.py. The fixture trigger must
# use this list so adding chunk_origin to it would make the FTS-untouched test fail.
CHUNKS_FTS_UPDATE_OF = "content, summary, tags, resolved_query, key_facts, resolved_queries, content_class"


def _make_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            summary TEXT,
            tags TEXT,
            resolved_query TEXT,
            key_facts TEXT,
            resolved_queries TEXT,
            content_class TEXT,
            chunk_origin TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE chunks_fts (
            content TEXT,
            summary TEXT,
            tags TEXT,
            resolved_query TEXT,
            key_facts TEXT,
            resolved_queries TEXT,
            chunk_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE chunks_fts_operational (
            content TEXT,
            summary TEXT,
            tags TEXT,
            resolved_query TEXT,
            key_facts TEXT,
            resolved_queries TEXT,
            chunk_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE chunks_fts_trigram (
            content TEXT,
            summary TEXT,
            tags TEXT,
            resolved_query TEXT,
            key_facts TEXT,
            resolved_queries TEXT,
            chunk_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE chunk_fts_rowids (
            chunk_id TEXT PRIMARY KEY,
            fts_rowid INTEGER,
            operational_rowid INTEGER,
            trigram_rowid INTEGER
        )
        """
    )
    conn.execute("CREATE TABLE chunk_vectors_rowids (id TEXT PRIMARY KEY)")
    return conn


def _install_chunks_fts_update_trigger(conn: sqlite3.Connection) -> None:
    """Install the production UPDATE OF list. DELETE+reinsert changes fts_rowid."""
    conn.execute("DROP TRIGGER IF EXISTS chunks_fts_update")
    conn.execute(
        f"""
        CREATE TRIGGER chunks_fts_update
        AFTER UPDATE OF {CHUNKS_FTS_UPDATE_OF} ON chunks BEGIN
            DELETE FROM chunks_fts WHERE chunk_id = old.id;
            DELETE FROM chunks_fts_operational WHERE chunk_id = old.id;
            DELETE FROM chunks_fts_trigram WHERE chunk_id = old.id;
            DELETE FROM chunk_fts_rowids WHERE chunk_id = old.id;
            INSERT INTO chunks_fts(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
            SELECT
                new.content, new.summary, new.tags, new.resolved_query,
                new.key_facts, new.resolved_queries, new.id
            WHERE COALESCE(new.content_class, 'knowledge') NOT IN ('operational', 'test', 'benchmark', 'cold');
            INSERT INTO chunk_fts_rowids(chunk_id, fts_rowid)
            SELECT new.id, last_insert_rowid()
            WHERE COALESCE(new.content_class, 'knowledge') NOT IN ('operational', 'test', 'benchmark', 'cold')
            ON CONFLICT(chunk_id) DO UPDATE SET fts_rowid = excluded.fts_rowid;

            INSERT INTO chunks_fts_operational(
                content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id
            )
            SELECT
                new.content, new.summary, new.tags, new.resolved_query,
                new.key_facts, new.resolved_queries, new.id
            WHERE COALESCE(new.content_class, 'knowledge') = 'operational';
            INSERT INTO chunk_fts_rowids(chunk_id, operational_rowid)
            SELECT new.id, last_insert_rowid()
            WHERE COALESCE(new.content_class, 'knowledge') = 'operational'
            ON CONFLICT(chunk_id) DO UPDATE SET operational_rowid = excluded.operational_rowid;

            INSERT INTO chunks_fts_trigram(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
            SELECT
                new.content, new.summary, new.tags, new.resolved_query,
                new.key_facts, new.resolved_queries, new.id
            WHERE COALESCE(new.content_class, 'knowledge') NOT IN ('operational', 'test', 'benchmark', 'cold');
            INSERT INTO chunk_fts_rowids(chunk_id, trigram_rowid)
            SELECT new.id, last_insert_rowid()
            WHERE COALESCE(new.content_class, 'knowledge') NOT IN ('operational', 'test', 'benchmark', 'cold')
            ON CONFLICT(chunk_id) DO UPDATE SET trigram_rowid = excluded.trigram_rowid;
        END
        """
    )


def _seed(conn: sqlite3.Connection, *, extra_legacy: int = 0) -> list[tuple[str, str, str]]:
    rows = [
        ("gemini-plain", "ordinary assistant memory", CHUNK_ORIGIN_GEMINI_FLASH_LITE),
        ("groq-plain", "another ordinary memory", CHUNK_ORIGIN_GROQ),
        ("ollama-plain", "yet more memory", CHUNK_ORIGIN_OLLAMA),
        ("mlx-precompact", PRECOMPACT_CONTENT, CHUNK_ORIGIN_MLX),
        ("keep-unknown", "untouched unknown row", CHUNK_ORIGIN_UNKNOWN),
        ("keep-manual", "untouched manual row", CHUNK_ORIGIN_MANUAL),
        ("keep-raw", "untouched raw row", "raw"),
    ]
    rows.extend(
        (f"gemini-extra-{index}", f"extra memory {index}", CHUNK_ORIGIN_GEMINI_FLASH_LITE)
        for index in range(extra_legacy)
    )
    conn.executemany(
        "INSERT INTO chunks (id, content, chunk_origin) VALUES (?, ?, ?)",
        rows,
    )
    conn.executemany(
        """
        INSERT INTO chunks_fts(content, chunk_id) VALUES (?, ?)
        """,
        [(row[1], row[0]) for row in rows],
    )
    conn.executemany("INSERT INTO chunk_vectors_rowids (id) VALUES (?)", [(row[0],) for row in rows])
    conn.executemany(
        "INSERT INTO chunk_fts_rowids (chunk_id, fts_rowid) VALUES (?, ?)",
        [(row[0], 1000 + index) for index, row in enumerate(rows)],
    )
    conn.commit()
    return rows


def _fts_rowids(conn: sqlite3.Connection) -> dict[str, int | None]:
    return dict(conn.execute("SELECT chunk_id, fts_rowid FROM chunk_fts_rowids"))


def test_wipe_refuses_live_canonical_path(tmp_path, monkeypatch):
    from brainlayer.chunk_origin_wipe import wipe_legacy_model_chunk_origins

    fake_home = tmp_path / "home"
    live = fake_home / ".local" / "share" / "brainlayer" / "brainlayer.db"
    live.parent.mkdir(parents=True)
    live.write_bytes(b"not-a-real-db")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setattr("brainlayer.chunk_origin_wipe.account_home", lambda: fake_home, raising=False)

    with pytest.raises(RuntimeError, match="live BrainLayer DB"):
        wipe_legacy_model_chunk_origins(live, apply=True)


def test_wipe_dry_run_does_not_write(tmp_path):
    from brainlayer.chunk_origin_wipe import wipe_legacy_model_chunk_origins

    db_path = tmp_path / "copy.db"
    conn = _make_db(db_path)
    _install_chunks_fts_update_trigger(conn)
    _seed(conn)
    conn.close()

    result = wipe_legacy_model_chunk_origins(db_path, apply=False, batch_size=2)

    assert result.updated == 0
    assert result.pre_wipe_legacy == 4
    assert result.derived[CHUNK_ORIGIN_UNKNOWN] == 3
    assert result.derived[CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT] == 1
    conn = sqlite3.connect(db_path)
    remaining = dict(conn.execute("SELECT id, chunk_origin FROM chunks"))
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert remaining["gemini-plain"] == CHUNK_ORIGIN_GEMINI_FLASH_LITE
    assert remaining["mlx-precompact"] == CHUNK_ORIGIN_MLX
    assert PREIMAGE_TABLE not in tables


def test_wipe_apply_rederives_legacy_models_and_preserves_others(tmp_path):
    from brainlayer.chunk_origin_wipe import wipe_legacy_model_chunk_origins

    db_path = tmp_path / "copy.db"
    conn = _make_db(db_path)
    _install_chunks_fts_update_trigger(conn)
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
    _install_chunks_fts_update_trigger(conn)
    _seed(conn)
    before_rowids = _fts_rowids(conn)
    conn.close()

    result = wipe_legacy_model_chunk_origins(db_path, apply=True)

    conn = sqlite3.connect(db_path)
    after_rowids = _fts_rowids(conn)
    conn.close()
    assert result.aux_counts_before == result.aux_counts_after
    assert result.aux_counts_after["chunks_fts"] == 7
    assert result.aux_counts_after["chunk_vectors_rowids"] == 7
    assert after_rowids == before_rowids


def test_wipe_checkpoints_every_three_batches(tmp_path):
    from brainlayer.chunk_origin_wipe import wipe_legacy_model_chunk_origins

    db_path = tmp_path / "copy.db"
    conn = _make_db(db_path)
    _install_chunks_fts_update_trigger(conn)
    _seed(conn)
    conn.close()

    result = wipe_legacy_model_chunk_origins(db_path, apply=True, batch_size=2, checkpoint_every=3)

    assert result.batches == 2
    assert result.checkpoints >= 1


def test_cli_requires_db_and_defaults_to_dry_run(tmp_path, capsys):
    from brainlayer.chunk_origin_wipe import main

    db_path = tmp_path / "copy.db"
    conn = _make_db(db_path)
    _install_chunks_fts_update_trigger(conn)
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


def test_spot_check_rereads_stored_origin_after_commit(tmp_path):
    """F1: a post-UPDATE rewrite to manual must make spot_check fail, not tautology-pass."""
    from brainlayer.chunk_origin_wipe import wipe_legacy_model_chunk_origins

    db_path = tmp_path / "copy.db"
    conn = _make_db(db_path)
    _install_chunks_fts_update_trigger(conn)
    _seed(conn)
    conn.execute(
        """
        CREATE TRIGGER rewrite_origin AFTER UPDATE OF chunk_origin ON chunks
        BEGIN
            UPDATE chunks SET chunk_origin = 'manual' WHERE id = NEW.id;
        END
        """
    )
    conn.commit()
    conn.close()

    result = wipe_legacy_model_chunk_origins(db_path, apply=True, spot_check=4)

    assert result.spot_checks
    assert all("stored" in item for item in result.spot_checks)
    assert not all(item["ok"] for item in result.spot_checks)
    assert any(item["stored"] == CHUNK_ORIGIN_MANUAL for item in result.spot_checks)


def test_spot_check_ok_when_stored_matches_detect(tmp_path):
    """F5: happy-path spot-check re-reads committed chunk_origin."""
    from brainlayer.chunk_origin_wipe import wipe_legacy_model_chunk_origins

    db_path = tmp_path / "copy.db"
    conn = _make_db(db_path)
    _install_chunks_fts_update_trigger(conn)
    _seed(conn)
    conn.close()

    result = wipe_legacy_model_chunk_origins(db_path, apply=True, spot_check=4)

    conn = sqlite3.connect(db_path)
    stored = dict(conn.execute("SELECT id, chunk_origin FROM chunks"))
    contents = dict(conn.execute("SELECT id, content FROM chunks"))
    conn.close()
    assert len(result.spot_checks) == 4
    assert all(item["ok"] for item in result.spot_checks)
    for item in result.spot_checks:
        assert item["stored"] == stored[item["id"]]
        assert item["stored"] == detect_chunk_origin(contents[item["id"]])


def test_fts_update_trigger_fires_on_content_not_origin(tmp_path):
    """F4: fixture trigger must be load-bearing — content UPDATE rewrites fts_rowid."""
    db_path = tmp_path / "copy.db"
    conn = _make_db(db_path)
    _install_chunks_fts_update_trigger(conn)
    _seed(conn)
    before = conn.execute("SELECT fts_rowid FROM chunk_fts_rowids WHERE chunk_id = 'keep-manual'").fetchone()[0]
    conn.execute("UPDATE chunks SET content = content || '!' WHERE id = 'keep-manual'")
    after_content = conn.execute("SELECT fts_rowid FROM chunk_fts_rowids WHERE chunk_id = 'keep-manual'").fetchone()[0]
    conn.execute("UPDATE chunks SET chunk_origin = 'unknown' WHERE id = 'keep-manual'")
    after_origin = conn.execute("SELECT fts_rowid FROM chunk_fts_rowids WHERE chunk_id = 'keep-manual'").fetchone()[0]
    conn.close()
    assert after_content != before
    assert after_origin == after_content


def test_wipe_writes_preimage_before_first_batch(tmp_path):
    """F2: pre-image table captures original legacy origins before any UPDATE."""
    from brainlayer.chunk_origin_wipe import wipe_legacy_model_chunk_origins

    db_path = tmp_path / "copy.db"
    conn = _make_db(db_path)
    _install_chunks_fts_update_trigger(conn)
    _seed(conn)
    conn.close()

    wipe_legacy_model_chunk_origins(db_path, apply=True)

    conn = sqlite3.connect(db_path)
    preimage = dict(conn.execute(f"SELECT id, chunk_origin FROM {PREIMAGE_TABLE}"))
    current = dict(conn.execute("SELECT id, chunk_origin FROM chunks"))
    conn.close()
    assert preimage["gemini-plain"] == CHUNK_ORIGIN_GEMINI_FLASH_LITE
    assert preimage["mlx-precompact"] == CHUNK_ORIGIN_MLX
    assert "keep-unknown" not in preimage
    assert current["gemini-plain"] == CHUNK_ORIGIN_UNKNOWN
    assert current["mlx-precompact"] == CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT


def test_wipe_resumes_after_checkpoint_interrupt(tmp_path, monkeypatch):
    """F5: committed batches survive Ctrl-C; a second apply finishes the remainder."""
    from brainlayer import chunk_origin_wipe

    db_path = tmp_path / "copy.db"
    conn = _make_db(db_path)
    _install_chunks_fts_update_trigger(conn)
    _seed(conn, extra_legacy=8)
    conn.close()

    checkpoints = {"n": 0}
    real_checkpoint = chunk_origin_wipe._checkpoint

    def boom(connection: sqlite3.Connection) -> None:
        checkpoints["n"] += 1
        if checkpoints["n"] >= 2:
            raise KeyboardInterrupt("simulated")
        real_checkpoint(connection)

    monkeypatch.setattr(chunk_origin_wipe, "_checkpoint", boom)
    with pytest.raises(KeyboardInterrupt):
        chunk_origin_wipe.wipe_legacy_model_chunk_origins(db_path, apply=True, batch_size=2, checkpoint_every=1)

    conn = sqlite3.connect(db_path)
    remaining = conn.execute(
        f"SELECT COUNT(*) FROM chunks WHERE chunk_origin IN ({','.join('?' * len(LEGACY_MODEL_CHUNK_ORIGINS))})",
        tuple(sorted(LEGACY_MODEL_CHUNK_ORIGINS)),
    ).fetchone()[0]
    preimage_count = conn.execute(f"SELECT COUNT(*) FROM {PREIMAGE_TABLE}").fetchone()[0]
    conn.close()
    assert 0 < remaining < 12
    assert preimage_count == 12

    monkeypatch.setattr(chunk_origin_wipe, "_checkpoint", real_checkpoint)
    result = chunk_origin_wipe.wipe_legacy_model_chunk_origins(db_path, apply=True, batch_size=2)
    assert result.post_wipe_legacy == 0
    conn = sqlite3.connect(db_path)
    assert conn.execute(f"SELECT COUNT(*) FROM {PREIMAGE_TABLE}").fetchone()[0] == 12
    assert (
        conn.execute("SELECT chunk_origin FROM chunks WHERE id = 'gemini-plain'").fetchone()[0] == CHUNK_ORIGIN_UNKNOWN
    )
    conn.close()


def test_wipe_refuses_resolve_db_path_target(tmp_path):
    """F3: refuse whatever paths.resolve_db_path() currently points at."""
    from brainlayer.chunk_origin_wipe import wipe_legacy_model_chunk_origins
    from brainlayer.paths import resolve_db_path

    live = resolve_db_path()
    live.parent.mkdir(parents=True, exist_ok=True)
    conn = _make_db(live)
    _install_chunks_fts_update_trigger(conn)
    _seed(conn)
    conn.close()

    with pytest.raises(RuntimeError, match="live BrainLayer DB"):
        wipe_legacy_model_chunk_origins(live, apply=True)


def test_wipe_refuses_hardlink_and_account_home_divergence(tmp_path, monkeypatch):
    """F3: inode match and passwd-home canonical catch HOME / alias bypasses."""
    from brainlayer.chunk_origin_wipe import wipe_legacy_model_chunk_origins

    account_home = tmp_path / "account"
    env_home = tmp_path / "envhome"
    live = account_home / ".local" / "share" / "brainlayer" / "brainlayer.db"
    alias = tmp_path / "alias" / "brainlayer.db"
    live.parent.mkdir(parents=True)
    alias.parent.mkdir(parents=True)
    conn = _make_db(live)
    _install_chunks_fts_update_trigger(conn)
    _seed(conn)
    conn.close()
    os.link(live, alias)

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: env_home))
    monkeypatch.setattr("brainlayer.chunk_origin_wipe.account_home", lambda: account_home, raising=False)

    with pytest.raises(RuntimeError, match="live BrainLayer DB"):
        wipe_legacy_model_chunk_origins(alias, apply=True)
    with pytest.raises(RuntimeError, match="live BrainLayer DB"):
        wipe_legacy_model_chunk_origins(live, apply=True)
