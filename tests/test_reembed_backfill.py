"""Tests for the one-time unvectored chunk re-embedding backfill."""

from __future__ import annotations

from typer.testing import CliRunner

from brainlayer._helpers import serialize_f32
from brainlayer.vector_store import VectorStore


def _embed(seed: float) -> list[float]:
    return [seed + (i / 10000.0) for i in range(1024)]


def _insert_chunk(
    store: VectorStore,
    *,
    chunk_id: str,
    source: str = "claude_code",
    content: str | None = None,
    created_at: str = "2026-06-14T00:00:00+00:00",
) -> None:
    text = content or f"Backfill content for {chunk_id}"
    store.conn.cursor().execute(
        """INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type,
            value_type, char_count, source, created_at, enriched_at, enrich_status,
            summary, tags, importance, chunk_origin, seen_count, last_seen_at,
            content_class
        ) VALUES (?, ?, '{}', 'test.jsonl', 'test', 'note',
            'HIGH', ?, ?, ?, NULL, NULL,
            NULL, NULL, NULL, 'raw', 1, ?,
            'human-authored')""",
        (chunk_id, text, len(text), source, created_at, created_at),
    )


class FakeBatchModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.batch_sizes: list[int | None] = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        self.batch_sizes.append(kwargs.get("batch_size"))
        return [_embed(float(i + 1)) for i, _ in enumerate(texts)]


def test_count_unvectored_chunks_counts_all_active_sources(tmp_path):
    from brainlayer.reembed_backfill import count_unvectored_chunks

    store = VectorStore(tmp_path / "backfill.db")
    try:
        _insert_chunk(store, chunk_id="manual-missing", source="manual")
        _insert_chunk(store, chunk_id="claude-missing", source="claude_code")
        _insert_chunk(store, chunk_id="watcher-missing", source="realtime_watcher")
        _insert_chunk(store, chunk_id="already-vectored", source="youtube")
        store._upsert_chunk_vector(store.conn.cursor(), "already-vectored", _embed(0.5))

        assert count_unvectored_chunks(store) == 3
    finally:
        store.close()


def test_count_unvectored_chunks_excludes_inactive_by_default(tmp_path):
    from brainlayer.reembed_backfill import count_unvectored_chunks

    store = VectorStore(tmp_path / "backfill.db")
    try:
        _insert_chunk(store, chunk_id="active")
        _insert_chunk(store, chunk_id="archived")
        _insert_chunk(store, chunk_id="superseded")
        _insert_chunk(store, chunk_id="aggregated")
        cursor = store.conn.cursor()
        cursor.execute("UPDATE chunks SET archived_at = '2026-06-14T00:00:00+00:00' WHERE id = 'archived'")
        cursor.execute("UPDATE chunks SET superseded_by = 'replacement' WHERE id = 'superseded'")
        cursor.execute("UPDATE chunks SET aggregated_into = 'aggregate' WHERE id = 'aggregated'")

        assert count_unvectored_chunks(store) == 1
        assert count_unvectored_chunks(store, include_inactive=True) == 4
    finally:
        store.close()


def test_fetch_unvectored_batch_is_resumable_and_ordered(tmp_path):
    from brainlayer.reembed_backfill import fetch_unvectored_batch

    store = VectorStore(tmp_path / "backfill.db")
    try:
        _insert_chunk(store, chunk_id="later", created_at="2026-06-14T00:00:03+00:00")
        _insert_chunk(store, chunk_id="earlier", created_at="2026-06-14T00:00:01+00:00")
        _insert_chunk(store, chunk_id="middle", created_at="2026-06-14T00:00:02+00:00")
        store._upsert_chunk_vector(store.conn.cursor(), "earlier", _embed(0.5))

        rows = fetch_unvectored_batch(store, batch_size=10)

        assert [row.chunk_id for row in rows] == ["middle", "later"]
    finally:
        store.close()


def test_run_reembed_backfill_batches_and_writes_both_vector_tables(tmp_path):
    from brainlayer.reembed_backfill import run_reembed_backfill

    store = VectorStore(tmp_path / "backfill.db")
    try:
        _insert_chunk(store, chunk_id="missing-1", content="first")
        _insert_chunk(store, chunk_id="missing-2", content="second")
    finally:
        store.close()

    model = FakeBatchModel()
    result = run_reembed_backfill(
        db_path=tmp_path / "backfill.db",
        model=model,
        batch_size=64,
        progress_every=1,
    )

    assert result.processed == 2
    assert result.failed == 0
    assert len(model.calls) == 1
    assert model.calls[0] == ["first", "second"]

    check = VectorStore(tmp_path / "backfill.db")
    try:
        cursor = check.conn.cursor()
        assert cursor.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0] == 2
        assert cursor.execute("SELECT COUNT(*) FROM chunk_vectors_binary").fetchone()[0] == 2
    finally:
        check.close()


def test_run_reembed_backfill_reuses_large_candidate_page_for_encode_batches(tmp_path, monkeypatch):
    from brainlayer import reembed_backfill
    from brainlayer.reembed_backfill import run_reembed_backfill

    store = VectorStore(tmp_path / "backfill.db")
    try:
        for index in range(5):
            _insert_chunk(store, chunk_id=f"missing-{index}", content=f"text {index}")
    finally:
        store.close()

    fetch_sizes: list[int] = []
    real_fetch = reembed_backfill.fetch_unvectored_batch

    def tracking_fetch(*args, **kwargs):
        fetch_sizes.append(kwargs["batch_size"])
        return real_fetch(*args, **kwargs)

    monkeypatch.setattr(reembed_backfill, "fetch_unvectored_batch", tracking_fetch)

    model = FakeBatchModel()
    result = run_reembed_backfill(
        db_path=tmp_path / "backfill.db",
        model=model,
        batch_size=2,
        candidate_page_size=5,
    )

    assert result.processed == 5
    assert [len(call) for call in model.calls] == [2, 2, 1]
    assert model.batch_sizes == [2, 2, 1]
    assert fetch_sizes == [5, 5]


def test_write_chunk_embeddings_wraps_batch_in_one_transaction():
    from brainlayer.reembed_backfill import PendingChunk, write_chunk_embeddings

    class FakeCursor:
        def __init__(self) -> None:
            self.statements: list[str] = []

        def execute(self, sql, *_args, **_kwargs):
            self.statements.append(" ".join(str(sql).split()).upper())

    class FakeConn:
        def __init__(self) -> None:
            self.cursor_obj = FakeCursor()

        def cursor(self):
            return self.cursor_obj

    class FakeStore:
        def __init__(self) -> None:
            self.conn = FakeConn()

        def _upsert_chunk_vector(self, cursor, chunk_id, embedding):
            cursor.execute("INSERT INTO chunk_vectors VALUES (?, ?)", (chunk_id, embedding))

    store = FakeStore()
    chunks = [PendingChunk(chunk_id="a", content="first"), PendingChunk(chunk_id="b", content="second")]

    written = write_chunk_embeddings(store, chunks, [_embed(1.0), _embed(2.0)])

    assert written == 2
    assert store.conn.cursor_obj.statements[0] == "BEGIN IMMEDIATE"
    assert store.conn.cursor_obj.statements[-1] == "COMMIT"
    assert store.conn.cursor_obj.statements.count("BEGIN IMMEDIATE") == 1
    assert store.conn.cursor_obj.statements.count("COMMIT") == 1


def test_run_reembed_backfill_is_idempotent(tmp_path):
    from brainlayer.reembed_backfill import run_reembed_backfill

    store = VectorStore(tmp_path / "backfill.db")
    try:
        _insert_chunk(store, chunk_id="already-vectored")
        store.conn.cursor().execute(
            "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
            ("already-vectored", serialize_f32(_embed(0.25))),
        )
    finally:
        store.close()

    model = FakeBatchModel()
    result = run_reembed_backfill(db_path=tmp_path / "backfill.db", model=model, batch_size=64)

    assert result.processed == 0
    assert model.calls == []


def test_reembed_backfill_cli_dry_run_reports_pending_count(tmp_path):
    from brainlayer.cli import app

    db_path = tmp_path / "backfill.db"
    store = VectorStore(db_path)
    try:
        _insert_chunk(store, chunk_id="missing")
    finally:
        store.close()

    result = CliRunner().invoke(app, ["reembed-backfill", "--db", str(db_path), "--dry-run"])

    assert result.exit_code == 0, result.stdout
    assert "Unvectored active chunks: 1" in result.stdout


def test_heavy_ml_mutex_ignores_agent_prompt_text(monkeypatch):
    from brainlayer import reembed_backfill

    class Result:
        stdout = "\n".join(
            [
                "111 /usr/local/bin/claude claude --append-system-prompt mentions mlx ollama enrichment",
                "222 /opt/homebrew/bin/llama-server /opt/homebrew/bin/llama-server --model local.gguf",
                "333 /usr/bin/python3 python -m mlx_audio.tts.generate --model voice",
                "444 /usr/bin/python3 python scripts/embed_backfill.py",
                "555 /opt/homebrew/bi /opt/homebrew/bin/whisper-server -m ggml-large-v3-turbo.bin",
                "666 /Library/Framewo /Library/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python /Library/Frameworks/Python.framework/Versions/3.13/bin/mlx_lm.server --model local",
                "777 /usr/bin/python3 python -m pytest tests/test_deferred_embedding.py -q",
            ]
        )

    monkeypatch.setattr(reembed_backfill.os, "getpid", lambda: 999)
    monkeypatch.setattr(reembed_backfill.subprocess, "run", lambda *_args, **_kwargs: Result())

    matches = reembed_backfill.find_heavy_ml_processes()

    assert len(matches) == 5
    assert not any("claude --append-system-prompt" in match for match in matches)
    assert any("llama-server" in match for match in matches)
    assert any("mlx_audio.tts.generate" in match for match in matches)
    assert any("embed_backfill.py" in match for match in matches)
    assert any("whisper-server" in match for match in matches)
    assert any("mlx_lm.server" in match for match in matches)
