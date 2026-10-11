"""Synthetic at-rest receipts for chunk writers and direct hook writes (#1046)."""

import importlib.util
import io
import json
from pathlib import Path

import pytest

from brainlayer.chunk_write import canonical_content_hash, insert_canonical_chunk
from brainlayer.dedupe import compute_dedupe_fields
from brainlayer.drain import _apply_hook, _apply_store
from brainlayer.hooks.indexer import RealtimeIndexer
from brainlayer.pipeline.secret_scrub import scrub_secrets
from brainlayer.store import store_memory
from brainlayer.vector_store import VectorStore

ROOT = Path(__file__).resolve().parents[1]
TOKENS = [prefix + "SyntheticOnlyAbCdEf0123456789_-xyz" for prefix in ("ya29.", "1//", "GOCSPX-", "sk-ant-")]


def load_hook(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), ROOT / "hooks" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def store(tmp_path):
    db = VectorStore(tmp_path / "synthetic.db")
    yield db
    db.close()


def assert_no_tokens(conn):
    # Check every text cell, including FTS/history, not just the visible content.
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    for table in tables:
        if table.startswith(("chunk_vectors", "vec_", "kg_vec_")):
            continue
        rows = conn.execute('SELECT * FROM "' + table.replace('"', '""') + '"')
        for row in rows:
            for value in row:
                if isinstance(value, str):
                    assert all(token not in value for token in TOKENS), table


@pytest.mark.parametrize("token", TOKENS)
def test_canonical_insert_scrubs_all_text_and_recomputes_derived_fields(store, token):
    content = "Synthetic credential example: " + token
    stale = compute_dedupe_fields(content, "2026-10-01T00:00:00Z")
    values = {
        "id": "direct",
        "content": content,
        "summary": content,
        "preview_text": content,
        "metadata": json.dumps({"nested": [token], token: token}),
        "tags": [token],
        "char_count": len(content),
        "created_at": "2026-10-01T00:00:00Z",
        "dedupe_hash": stale.dedupe_hash,
        "simhash": stale.simhash,
        **{f"simhash_band_{i}": b for i, b in enumerate(stale.bands)},
    }
    row = insert_canonical_chunk(store.conn, values)
    expected = scrub_secrets(content).text
    assert row["content"] == expected
    assert row["content_hash"] == canonical_content_hash(expected)
    assert row["char_count"] == len(expected)
    assert row["dedupe_hash"] == compute_dedupe_fields(expected, row["created_at"]).dedupe_hash
    assert values["content"] == content  # Caller owns its input.
    assert json.loads(row["metadata"])["secret_scrub_redactions"]
    assert_no_tokens(store.conn)


@pytest.mark.parametrize("token", TOKENS)
@pytest.mark.parametrize(
    "writer", ["stop", "prompt", "postcompact", "chapter", "drain-hook", "drain-store", "store", "upsert"]
)
def test_real_writer_paths(store, tmp_path, monkeypatch, token, writer):
    content = "Synthetic credential example: " + token
    if writer == "stop":
        hook = load_hook("brainbar-stop-index")
        monkeypatch.setenv("BRAINLAYER_DB", str(store.db_path))
        monkeypatch.delenv("BRAINLAYER_HOOKS_DISABLED", raising=False)
        monkeypatch.delenv("CLAUDE_NON_INTERACTIVE", raising=False)
        monkeypatch.setattr(hook, "PENDING_DIR", tmp_path / "pending")
        monkeypatch.setattr(hook, "QUEUE_DIR", tmp_path / "queue")
        monkeypatch.setattr(
            hook.sys, "stdin", io.StringIO(json.dumps({"session_id": "synthetic", "last_assistant_message": content}))
        )
        hook.main()
        assert store.conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 1
    elif writer == "prompt":
        store.conn.execute("CREATE TABLE injection_events(session_id, query, chunk_ids, token_count)")
        load_hook("brainlayer-prompt-search").record_injection_event(str(store.db_path), "synthetic", content, [], 0)
        assert store.conn.execute("SELECT count(*) FROM injection_events").fetchone()[0] == 1
    elif writer == "postcompact":
        hook = load_hook("brainbar-postcompact")
        monkeypatch.setenv("BRAINLAYER_DB", str(store.db_path))
        monkeypatch.setattr(
            hook.sys, "stdin", io.StringIO(json.dumps({"session_id": "synthetic", "compact_summary": content}))
        )
        hook.main()
        assert store.conn.execute("SELECT count(*) FROM chapters").fetchone()[0] == 1
    elif writer == "chapter":
        indexer = RealtimeIndexer(db_path=str(store.db_path))
        try:
            assert indexer.record_chapter("synthetic", content, "auto") is not None
        finally:
            indexer._db.close()
    elif writer.startswith("drain-"):
        apply = _apply_hook if writer == "drain-hook" else _apply_store
        apply(store.conn, {"chunk_id": "synthetic", "session_id": "synthetic", "content": content})
        assert store.conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 1
    elif writer == "upsert":
        store.upsert_chunks([{"id": "synthetic", "content": content, "metadata": {"extra": token}}], [[0.0] * 1024])
    else:
        store_memory(store, None, content, "note", tags=[token], files_changed=[token])
    assert_no_tokens(store.conn)


def test_scrubbed_watcher_input_is_stable(store):
    content = scrub_secrets("Watcher fixture " + " ".join(TOKENS)).text
    metadata = {"secret_scrub_redactions": ["google_oauth_access"], "secret_scrub_quarantine_count": 2}
    row = insert_canonical_chunk(store.conn, {"id": "watcher", "content": content, "metadata": metadata})
    assert row["content"] == content
    assert json.loads(row["metadata"]) == metadata
    again = insert_canonical_chunk(store.conn, row)
    assert again == row


def test_store_dedupes_scrubbed_content_before_merge(store):
    first = store_memory(store, None, "Same synthetic note " + TOKENS[0], "note")
    second = store_memory(store, None, "Same synthetic note " + TOKENS[0].replace("xyz", "abc"), "note")
    assert first["id"] == second["id"]
    assert store.conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 1
    assert_no_tokens(store.conn)


def test_prompt_scrubs_before_preview_truncation(store):
    store.conn.execute("CREATE TABLE injection_events(session_id, query, chunk_ids, token_count)")
    content = "x " * 495 + TOKENS[0]
    load_hook("brainlayer-prompt-search").record_injection_event(str(store.db_path), "synthetic", content, [], 0)
    query = store.conn.execute("SELECT query FROM injection_events").fetchone()[0]
    assert query == scrub_secrets(content).text[:1000]
    assert "ya29." not in query


def test_scrub_failure_writes_nothing(store, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic scrub failure")

    monkeypatch.setattr("brainlayer.pipeline.secret_scrub.scrub_secrets", fail)
    with pytest.raises(RuntimeError, match="synthetic scrub failure"):
        insert_canonical_chunk(store.conn, {"id": "failed", "content": "synthetic note"})
    assert store.conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0


def test_metadata_json_escapes_and_nested_strings_are_scrubbed(store):
    escaped = TOKENS[0].replace("y", "\\u0079", 1)
    metadata = '{"nested": "' + escaped + '", "' + escaped + '": "fixture"}'
    row = insert_canonical_chunk(store.conn, {"id": "escaped", "content": "Plain synthetic note", "metadata": metadata})
    decoded = json.loads(row["metadata"])
    assert decoded["nested"] == "[REDACTED:google_oauth_access]"
    assert "[REDACTED:google_oauth_access]" in decoded
    assert_no_tokens(store.conn)


def test_store_embedding_receives_scrubbed_content(store):
    seen = []

    def embed(content):
        seen.append(content)
        return [0.0] * 1024

    store_memory(store, embed, "Synthetic embedding input " + TOKENS[0], "note")
    assert seen == [scrub_secrets("Synthetic embedding input " + TOKENS[0]).text]
    assert_no_tokens(store.conn)


def test_store_same_id_merge_cannot_persist_secrets(store):
    store_memory(store, None, "Existing synthetic note", "note", chunk_id="same-id")
    store_memory(
        store,
        None,
        "Replacement synthetic note " + TOKENS[0],
        "note",
        chunk_id="same-id",
        tags=[TOKENS[1]],
        files_changed=[TOKENS[2]],
    )
    assert_no_tokens(store.conn)


def test_upsert_same_id_merge_cannot_persist_secrets(store):
    store.upsert_chunks([{"id": "same-id", "content": "Existing synthetic note"}], [[0.0] * 1024])
    store.upsert_chunks(
        [{"id": "same-id", "content": "Replacement synthetic note " + TOKENS[0], "metadata": {"nested": TOKENS[1]}}],
        [[0.0] * 1024],
    )
    assert_no_tokens(store.conn)


def test_quarantine_stays_in_content_and_only_counts_in_metadata(store):
    token = "UnlabeledEntropy4vKpZ8bJ6mA3fX0rN9sT"
    content = "Synthetic entropy fixture " + token
    assert scrub_secrets(content).quarantine
    row = insert_canonical_chunk(store.conn, {"id": "entropy", "content": content})
    assert row["content"] == content
    metadata = json.loads(row["metadata"])
    assert metadata["secret_scrub_quarantine_count"] == 1
    assert token not in row["metadata"]
    assert insert_canonical_chunk(store.conn, row) == row


def test_newly_scrubbed_canonical_row_is_idempotent_without_quarantine_count(store):
    row = insert_canonical_chunk(store.conn, {"id": "newly-scrubbed", "content": "All providers " + " ".join(TOKENS)})
    assert "secret_scrub_quarantine_count" not in json.loads(row["metadata"])
    assert insert_canonical_chunk(store.conn, row) == row


@pytest.mark.parametrize("field", ["id", "brick_id", "conversation_id"])
def test_identifiers_requiring_scrub_are_rejected_without_writes(store, field):
    with pytest.raises(ValueError, match="identifier"):
        insert_canonical_chunk(store.conn, {"id": "valid-id", "content": "Synthetic note", field: TOKENS[0]})
    assert store.conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0


def test_upsert_defers_embedding_when_content_is_redacted(store):
    store.upsert_chunks([{"id": "defer", "content": "Synthetic note " + TOKENS[0]}], [[1.0] * 1024])
    assert store.conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 1
    assert store.conn.execute("SELECT count(*) FROM chunk_vectors").fetchone()[0] == 0


def test_upsert_changed_content_clears_existing_embedding(store):
    store.upsert_chunks([{"id": "defer", "content": "Original synthetic note"}], [[1.0] * 1024])
    store.upsert_chunks([{"id": "defer", "content": "Updated synthetic note " + TOKENS[0]}], [[2.0] * 1024])
    assert store.conn.execute("SELECT count(*) FROM chunk_vectors").fetchone()[0] == 0


@pytest.mark.parametrize("metadata", [[TOKENS[0]], json.dumps([TOKENS[0]]), '"fixture"', "42"])
def test_nonobject_metadata_retains_value_with_scrub_findings(store, metadata):
    row = insert_canonical_chunk(
        store.conn, {"id": "nonobject", "content": "Synthetic note " + TOKENS[1], "metadata": metadata}
    )
    stored = json.loads(row["metadata"])
    assert "value" in stored
    assert stored["secret_scrub_redactions"]
    assert_no_tokens(store.conn)


def test_prompt_scrub_failure_is_best_effort_and_writes_nothing(store, monkeypatch):
    store.conn.execute("CREATE TABLE injection_events(session_id, query, chunk_ids, token_count)")

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic scrub failure")

    monkeypatch.setattr("brainlayer.pipeline.secret_scrub.scrub_secrets", fail)
    load_hook("brainlayer-prompt-search").record_injection_event(
        str(store.db_path), "synthetic", "Synthetic note", [], 0
    )
    assert store.conn.execute("SELECT count(*) FROM injection_events").fetchone()[0] == 0
