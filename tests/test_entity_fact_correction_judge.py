import asyncio

from brainlayer.correction_judge import Verdict
from brainlayer.vector_store import VectorStore


def _dummy_embed(text):  # noqa: ARG001
    return [0.1] * 1024


def _insert_chunk(store: VectorStore, chunk_id: str, content: str, *, key_fact: str) -> None:
    store.upsert_chunks(
        [
            {
                "id": chunk_id,
                "content": content,
                "metadata": {"source_file": "judge-test.jsonl", "project": "brainlayer"},
                "source_file": "judge-test.jsonl",
                "project": "brainlayer",
                "content_type": "user_message",
                "char_count": len(content),
                "source": "claude_code",
            }
        ],
        [_dummy_embed(content)],
    )
    store.update_enrichment(chunk_id, key_facts=[key_fact], summary=key_fact)


class SupersedeJudge:
    def __init__(self) -> None:
        self.calls = []

    def judge(self, entity, new_fact, conflicting_fact, context):
        self.calls.append(
            {
                "entity": entity,
                "new_fact": new_fact,
                "conflicting_fact": conflicting_fact,
                "context": context,
            }
        )
        return Verdict(action="supersede", confidence=0.96, reasoning="The new backend corrects the old one.")


def test_refresh_entity_facts_supersedes_conflicting_active_fact(tmp_path):
    store = VectorStore(tmp_path / "test.db")
    entity_id = store.upsert_entity("project-brainlayer", "project", "BrainLayer", embedding=_dummy_embed("BrainLayer"))

    _insert_chunk(
        store,
        "stale-backend",
        "BrainLayer uses ChromaDB for vector search.",
        key_fact="BrainLayer uses ChromaDB.",
    )
    store.link_entity_chunk(entity_id, "stale-backend", relevance=0.9, context="stale backend")
    store.refresh_entity_facts(entity_id)

    _insert_chunk(
        store,
        "corrected-backend",
        "Correction: BrainLayer uses sqlite-vec for vector search.",
        key_fact="BrainLayer uses sqlite-vec.",
    )
    store.link_entity_chunk(entity_id, "corrected-backend", relevance=0.95, context="corrected backend")

    judge = SupersedeJudge()
    store.refresh_entity_facts(entity_id, correction_judge=judge)

    active_facts = store.get_entity_facts(entity_id)
    assert [fact["fact_text"] for fact in active_facts] == ["BrainLayer uses sqlite-vec."]
    assert judge.calls[0]["new_fact"]["fact_text"] == "BrainLayer uses sqlite-vec."
    assert judge.calls[0]["conflicting_fact"]["fact_text"] == "BrainLayer uses ChromaDB."

    row = (
        store.conn.cursor()
        .execute(
            """
        SELECT status, superseded_by
        FROM entity_facts
        WHERE entity_id = ? AND fact_text = ?
        """,
            (entity_id, "BrainLayer uses ChromaDB."),
        )
        .fetchone()
    )
    assert row == ("superseded", "BrainLayer uses sqlite-vec.")


def test_refresh_entity_facts_uses_factory_judge_by_default(tmp_path, monkeypatch):
    import brainlayer.correction_judge as correction_judge

    store = VectorStore(tmp_path / "test.db")
    entity_id = store.upsert_entity("project-brainlayer", "project", "BrainLayer", embedding=_dummy_embed("BrainLayer"))

    _insert_chunk(
        store,
        "stale-backend",
        "BrainLayer uses ChromaDB for vector search.",
        key_fact="BrainLayer uses ChromaDB.",
    )
    store.link_entity_chunk(entity_id, "stale-backend", relevance=0.9, context="stale backend")
    store.refresh_entity_facts(entity_id, adjudicate_corrections=False)

    _insert_chunk(
        store,
        "corrected-backend",
        "Correction: BrainLayer uses sqlite-vec for vector search.",
        key_fact="BrainLayer uses sqlite-vec.",
    )
    store.link_entity_chunk(entity_id, "corrected-backend", relevance=0.95, context="corrected backend")

    judge = SupersedeJudge()
    monkeypatch.setattr(correction_judge, "get_correction_judge", lambda *, store=None: judge)

    store.refresh_entity_facts(entity_id)

    assert [fact["fact_text"] for fact in store.get_entity_facts(entity_id)] == ["BrainLayer uses sqlite-vec."]
    assert len(judge.calls) == 1


def test_brain_entity_shows_corrected_fact_without_reactivating_stale_fact(tmp_path, monkeypatch):
    import brainlayer.correction_judge as correction_judge
    from brainlayer.mcp.entity_handler import _brain_entity

    store = VectorStore(tmp_path / "test.db")
    entity_id = store.upsert_entity("project-brainlayer", "project", "BrainLayer", embedding=_dummy_embed("BrainLayer"))

    _insert_chunk(
        store,
        "stale-backend",
        "BrainLayer uses ChromaDB for vector search.",
        key_fact="BrainLayer uses ChromaDB.",
    )
    store.link_entity_chunk(entity_id, "stale-backend", relevance=0.9, context="stale backend")
    store.refresh_entity_facts(entity_id)

    _insert_chunk(
        store,
        "corrected-backend",
        "Correction: BrainLayer uses sqlite-vec for vector search.",
        key_fact="BrainLayer uses sqlite-vec.",
    )
    store.link_entity_chunk(entity_id, "corrected-backend", relevance=0.95, context="corrected backend")
    store.refresh_entity_facts(entity_id, correction_judge=SupersedeJudge())

    class DummyModel:
        def embed_query(self, text: str) -> list[float]:
            return _dummy_embed(text)

    monkeypatch.setattr("brainlayer.mcp.entity_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.entity_handler._get_embedding_model", lambda: DummyModel())
    monkeypatch.setattr(
        correction_judge,
        "get_correction_judge",
        lambda *, store=None: (_ for _ in ()).throw(AssertionError("brain_entity must not build a judge")),
    )

    result = asyncio.run(_brain_entity("BrainLayer", entity_type="project"))
    output = result.content[0].text

    assert "BrainLayer uses sqlite-vec." in output
    assert "ChromaDB" not in output
