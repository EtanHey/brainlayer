from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner


def _seed_rows(db_path: Path) -> list[tuple[str, str, str | None, str]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return conn.execute(
            """
            SELECT id, content, project, created_at
            FROM chunks
            WHERE source_file = 'sandbox-seed:skill-eval-baseline'
            ORDER BY id
            """
        ).fetchall()
    finally:
        conn.close()


def _insert_sentinel(db_path: Path, sentinel: str) -> None:
    from brainlayer.store import store_memory
    from brainlayer.vector_store import VectorStore

    store = VectorStore(db_path)
    try:
        store_memory(
            store,
            embed_fn=None,
            content=sentinel,
            memory_type="note",
            project="sandbox-test",
            chunk_id="sandbox-sentinel",
            created_at="2026-06-24T00:00:00Z",
        )
    finally:
        store.close()


def _search_text(db_path: Path, query: str) -> list[str]:
    from brainlayer.vector_store import VectorStore

    store = VectorStore(db_path, readonly=True)
    try:
        results = store.search(query_text=query, n_results=10)
        return list(results.get("documents", [[]])[0])
    finally:
        store.close()


def _binary_vector_count(db_path: Path) -> int:
    from brainlayer.vector_store import VectorStore

    store = VectorStore(db_path, readonly=True)
    try:
        return int(store.conn.cursor().execute("SELECT COUNT(*) FROM chunk_vectors_binary").fetchone()[0])
    finally:
        store.close()


def test_sandbox_rejects_prod_path_and_accepts_temp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from brainlayer.paths import _CANONICAL_DB_PATH
    from brainlayer.sandbox_db import SandboxDB, SandboxProdLeakError

    monkeypatch.delenv("BRAINLAYER_DB", raising=False)

    with pytest.raises(SandboxProdLeakError):
        SandboxDB(seed="skill-eval-baseline", token="prod", db_path=_CANONICAL_DB_PATH).start()

    temp_db = tmp_path / "brainlayer.db"
    sandbox = SandboxDB(seed="skill-eval-baseline", token="temp", db_path=temp_db).start()
    try:
        assert sandbox.db_path == temp_db
        assert os.environ["BRAINLAYER_DB"] == str(temp_db)
        assert temp_db.exists()
    finally:
        sandbox.stop()


def test_sandbox_rejects_active_noncanonical_db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from brainlayer.sandbox_db import SandboxDB, SandboxProdLeakError

    active_db = tmp_path / "active-live" / "brainlayer.db"
    monkeypatch.setenv("BRAINLAYER_DB", str(active_db))

    with pytest.raises(SandboxProdLeakError):
        SandboxDB(seed="skill-eval-baseline", token="active-live", db_path=active_db).start()


def test_sandbox_seed_is_retrievable_through_real_hybrid_search(monkeypatch: pytest.MonkeyPatch) -> None:
    from brainlayer.isolation_proof import _embed
    from brainlayer.sandbox_db import SandboxDB
    from brainlayer.vector_store import VectorStore

    monkeypatch.delenv("BRAINLAYER_DB", raising=False)
    phrase = "deterministic local embeddings"

    with SandboxDB(seed="skill-eval-baseline", token="seed-search") as sandbox:
        store = VectorStore(sandbox.db_path, readonly=True)
        try:
            results = store.hybrid_search(
                query_embedding=_embed(phrase),
                query_text=phrase,
                n_results=5,
            )
        finally:
            store.close()

        assert "skill-eval-baseline-003" in results["ids"][0]
        assert _binary_vector_count(sandbox.db_path) == len(sandbox.seeded_ids)


def test_sandbox_tokens_do_not_share_stored_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    from brainlayer.sandbox_db import SandboxDB

    monkeypatch.delenv("BRAINLAYER_DB", raising=False)
    sentinel = "sandbox anti cheat sentinel 1782331071313"

    with SandboxDB(seed="skill-eval-baseline", token="anti-cheat-a") as sandbox_a:
        _insert_sentinel(sandbox_a.db_path, sentinel)
        assert _search_text(sandbox_a.db_path, sentinel)

    with SandboxDB(seed="skill-eval-baseline", token="anti-cheat-b") as sandbox_b:
        assert _search_text(sandbox_b.db_path, sentinel) == []


def test_sandbox_cli_stop_removes_db_wal_shm_and_tempdir(monkeypatch: pytest.MonkeyPatch) -> None:
    from brainlayer.cli import app
    from brainlayer.sandbox_db import sandbox_paths_for_token

    monkeypatch.delenv("BRAINLAYER_DB", raising=False)
    token = "teardown-token"
    runner = CliRunner()

    start = runner.invoke(app, ["sandbox", "start", "--seed", "skill-eval-baseline", "--token", token])
    assert start.exit_code == 0, start.stdout
    assert "export BRAINLAYER_DB=" in start.stdout

    paths = sandbox_paths_for_token(token)
    assert paths.db_path.exists()
    paths.wal_path.touch()
    paths.shm_path.touch()

    stop = runner.invoke(app, ["sandbox", "stop", "--token", token])
    assert stop.exit_code == 0, stop.stdout
    assert not paths.db_path.exists()
    assert not paths.wal_path.exists()
    assert not paths.shm_path.exists()
    assert not paths.temp_dir.exists()


def test_sandbox_start_refuses_existing_live_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from brainlayer.sandbox_db import SandboxDB

    monkeypatch.delenv("BRAINLAYER_DB", raising=False)
    sandbox = SandboxDB(seed="skill-eval-baseline", token="duplicate-token").start()
    try:
        with pytest.raises(RuntimeError, match="already exists"):
            SandboxDB(seed="skill-eval-baseline", token="duplicate-token").start()
        assert sandbox.db_path.exists()
    finally:
        sandbox.stop()


def test_sandbox_start_refuses_double_start_and_restores_original_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from brainlayer.sandbox_db import SandboxDB

    original_db = "/tmp/brainlayer-original-env.db"
    monkeypatch.setenv("BRAINLAYER_DB", original_db)

    sandbox = SandboxDB(seed="skill-eval-baseline", token="double-start").start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            sandbox.start()
    finally:
        sandbox.stop()

    assert os.environ["BRAINLAYER_DB"] == original_db


def test_sandbox_seed_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    from brainlayer.sandbox_db import SandboxDB

    monkeypatch.delenv("BRAINLAYER_DB", raising=False)

    with SandboxDB(seed="skill-eval-baseline", token="determinism-a") as sandbox_a:
        rows_a = _seed_rows(sandbox_a.db_path)

    with SandboxDB(seed="skill-eval-baseline", token="determinism-b") as sandbox_b:
        rows_b = _seed_rows(sandbox_b.db_path)

    assert rows_a
    assert rows_a == rows_b
