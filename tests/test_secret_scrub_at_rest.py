"""Secrets are scrubbed at rest on every Python ingest path, not only in the watcher.

Paths: drain store replay, drain realtime hook, ``brainlayer index``, and the
Python digest. Also the two #958 review follow-ups: NER output from
``parse_llm_ner_response`` (N1), and the Ollama/MLX senders, which can point at a
non-local host (N2).

Synthetic tokens only, built from zeros; no network; no production DB.
"""

from __future__ import annotations

import json
import types

import pytest

from brainlayer.drain import _apply_hook, _apply_store
from brainlayer.vector_store import VectorStore

TOKENS = {
    "supabase": "sbp_" + "0" * 40,
    "github": "ghp_" + "0" * 36,
    "google": "AIza" + "0" * 35,
}


def _text() -> str:
    return "deploy note: " + " | ".join(f"{name} {token}" for name, token in TOKENS.items())


def _assert_clean(blob: str, where: str) -> None:
    leaked = [name for name, token in TOKENS.items() if token in blob]
    assert not leaked, f"{where} stored unredacted synthetic token(s): {leaked}"


def _row(store: VectorStore, chunk_id: str) -> dict:
    cursor = store.conn.cursor()
    columns = [info[1] for info in cursor.execute("PRAGMA table_info(chunks)")]
    values = cursor.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
    assert values is not None, chunk_id
    return dict(zip(columns, values))


def _fts_blob(store: VectorStore) -> str:
    cursor = store.conn.cursor()
    rows = []
    for table in ("chunks_fts", "chunks_fts_trigram"):
        try:
            rows.extend(cursor.execute(f"SELECT * FROM {table}"))
        except Exception:
            continue
    return json.dumps([[str(v) for v in row] for row in rows])


@pytest.fixture
def store(tmp_path):
    db = VectorStore(tmp_path / "at-rest.db")
    yield db
    db.close()


def test_drain_store_replay_scrubs_content_summary_tags_and_records_providers(store):
    result = _apply_store(
        store.conn,
        {
            "chunk_id": "manual-at-rest",
            "content": _text(),
            "tags": ["deploy", TOKENS["github"]],
            "memory_type": "note",
            "project": "brainlayer",
            "source": "manual",
            "created_at": "2026-09-25T00:00:00Z",
        },
    )

    row = _row(store, result.chunk_id)
    _assert_clean(json.dumps({k: str(v) for k, v in row.items()}), "drain store row")
    _assert_clean(_fts_blob(store), "drain store FTS")
    assert "[REDACTED:supabase]" in row["content"]
    metadata = json.loads(row["metadata"])
    assert metadata["secret_scrub_redactions"] == ["github", "google", "supabase"]
    import hashlib

    assert row["content_hash"] == hashlib.sha256(row["content"].strip().encode("utf-8")).hexdigest()


def test_drain_hook_scrubs_content_and_records_providers(store):
    result = _apply_hook(
        store.conn,
        {
            "session_id": "session-at-rest",
            "content": _text(),
            "source_file": "realtime-hook",
            "created_at": "2026-09-25T00:00:01Z",
        },
    )

    row = _row(store, result.chunk_id)
    _assert_clean(json.dumps({k: str(v) for k, v in row.items()}), "drain hook row")
    _assert_clean(_fts_blob(store), "drain hook FTS")
    assert json.loads(row["metadata"])["secret_scrub_redactions"] == ["github", "google", "supabase"]


def test_drain_store_dedups_on_the_scrubbed_content(store):
    """Two memories that differ only in the secret value are the same memory once scrubbed."""
    first = _apply_store(
        store.conn,
        {"content": "rotated key ghp_" + "0" * 36 + " today", "source": "manual", "project": "brainlayer"},
    )
    second = _apply_store(
        store.conn,
        {"content": "rotated key ghp_" + "1" * 36 + " today", "source": "manual", "project": "brainlayer"},
    )

    assert first.chunk_id == second.chunk_id
    count = store.conn.cursor().execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert count == 1


def test_index_path_scrubs_before_embedding_and_persisting(monkeypatch, tmp_path):
    from brainlayer import index_new
    from brainlayer.pipeline.chunk import Chunk
    from brainlayer.pipeline.classify import ContentType, ContentValue

    embedded_texts: list[str] = []

    def _fake_embed(chunks, on_progress=None):
        embedded_texts.extend(chunk.content for chunk in chunks)
        return [types.SimpleNamespace(chunk=chunk, embedding=[0.0] * 1024) for chunk in chunks]

    captured: dict = {}

    class _Store:
        def upsert_chunks(self, chunk_data, embeddings, deadline_monotonic=None):
            captured["chunks"] = chunk_data
            return len(chunk_data)

    monkeypatch.setattr(index_new, "embed_chunks", _fake_embed)
    source = tmp_path / "session.jsonl"
    source.write_text('{"timestamp": "2026-09-25T00:00:00Z"}\n', encoding="utf-8")
    chunk = Chunk(
        content=_text(),
        content_type=ContentType.ASSISTANT_TEXT,
        value=ContentValue.HIGH,
        metadata={},
        char_count=len(_text()),
    )

    index_new.index_chunks_to_sqlite([chunk], str(source), project="brainlayer", store=_Store())

    _assert_clean(json.dumps(embedded_texts), "index embedding input")
    stored = captured["chunks"][0]
    _assert_clean(json.dumps(stored, default=str), "index chunk_data")
    assert stored["metadata"]["secret_scrub_redactions"] == ["github", "google", "supabase"]
    assert stored["char_count"] == len(stored["content"])


def test_python_digest_scrubs_before_embedding_and_persisting(monkeypatch):
    from brainlayer.pipeline import digest

    embedded: list[str] = []
    captured: dict = {}

    class _Store:
        def upsert_chunks(self, chunks, embeddings):
            captured["chunks"] = chunks

        def get_entity(self, entity_id):
            return None

    seen_by_extraction: list[str] = []

    def _fake_process_chunk(chunk_dict, seed_entities=None):
        seen_by_extraction.append(chunk_dict["content"])
        return types.SimpleNamespace(entities=[], relations=[])

    monkeypatch.setattr(digest, "process_chunk", _fake_process_chunk)
    monkeypatch.setattr(digest, "store_extraction_result", lambda result, store: {})
    try:
        digest.digest_content(
            _text(),
            _Store(),
            lambda text: embedded.append(text) or [0.0] * 1024,
            faceted_enrich_fn=lambda **kwargs: {"status": "skipped"},
        )
    except Exception:
        pass  # later digest stages are not under test; the persisted chunk is.

    _assert_clean(json.dumps(embedded), "digest embedding input")
    _assert_clean(json.dumps(captured["chunks"], default=str), "digest chunk")
    _assert_clean(json.dumps(seen_by_extraction), "digest entity-extraction input")
    assert captured["chunks"][0]["metadata"]["secret_scrub_redactions"] == ["github", "google", "supabase"]


# ── #958 review follow-ups ───────────────────────────────────────────────


def test_llm_ner_output_is_scrubbed_before_it_can_be_persisted():
    """N1: relation facts and entity names are model-authored and reach the KG tables."""
    from brainlayer.pipeline.entity_extraction import parse_llm_ner_response

    token = TOKENS["supabase"]
    source_text = f"Etan configured Supabase with {token}"
    response = json.dumps(
        {
            "entities": [{"text": "Supabase", "type": "technology"}, {"text": token, "type": "tool"}],
            "relations": [
                {"source": "Etan", "target": "Supabase", "type": "uses", "fact": f"key is {token}"},
            ],
        }
    )

    entities, relations = parse_llm_ner_response(response, source_text)

    blob = json.dumps(
        [vars(e) for e in entities] + [vars(r) for r in relations],
        default=str,
    )
    _assert_clean(blob, "NER output")


def _capture_post(monkeypatch, module):
    sent: list[str] = []

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "{}", "choices": [{"message": {"content": "{}"}}], "usage": {}}

    def _post(url, *, json=None, timeout=None, headers=None, **kwargs):
        sent.append(__import__("json").dumps(json))
        return _Resp()

    monkeypatch.setattr(module.requests, "post", _post)
    return sent


@pytest.mark.parametrize("sender", ["call_glm", "call_mlx"])
def test_local_llm_senders_scrub_because_their_url_can_be_remote(monkeypatch, sender):
    """N2: BRAINLAYER_OLLAMA_URL / BRAINLAYER_MLX_URL are env-overridable to any host."""
    from brainlayer.pipeline import enrichment

    sent = _capture_post(monkeypatch, enrichment)
    monkeypatch.setattr(enrichment, "_log_glm_usage", lambda *args, **kwargs: None)

    getattr(enrichment, sender)(_text())

    assert sent, f"{sender} never reached the fake transport"
    _assert_clean(sent[0], f"{sender} payload")


def test_longitudinal_analyzer_ollama_calls_scrub_their_prompt(monkeypatch):
    """N2: ollama.generate honours OLLAMA_HOST, so it is not necessarily local."""
    from brainlayer.pipeline import longitudinal_analyzer

    prompts: list[str] = []
    fake_ollama = types.SimpleNamespace(
        generate=lambda model=None, prompt=None, options=None, **kwargs: prompts.append(prompt) or {"response": ""}
    )
    monkeypatch.setattr(longitudinal_analyzer, "ollama", fake_ollama, raising=False)

    longitudinal_analyzer._ollama_generate(model="m", prompt=_text(), options={})

    assert prompts
    _assert_clean(prompts[0], "longitudinal ollama prompt")


def test_longitudinal_analyzer_has_no_unscrubbed_ollama_call_site():
    from pathlib import Path

    from brainlayer.pipeline import longitudinal_analyzer

    source = Path(longitudinal_analyzer.__file__).read_text(encoding="utf-8")
    direct_calls = source.count("ollama.generate(")

    assert direct_calls == 1, "every ollama.generate call must go through _ollama_generate"


def test_drain_store_records_providers_found_only_in_tags(store):
    """Same meaning as BrainBar's store path: providers redacted anywhere in the chunk."""
    result = _apply_store(
        store.conn,
        {"content": "an ordinary note", "tags": ["deploy", TOKENS["google"]], "source": "manual"},
    )

    row = _row(store, result.chunk_id)
    _assert_clean(str(row["tags"]), "drain store tags")
    assert json.loads(row["metadata"])["secret_scrub_redactions"] == ["google"]
