import importlib.util
from pathlib import Path

import pytest

from brainlayer.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test.db"
    s = VectorStore(db_path)
    yield s
    s.close()


def _load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "kg_p2_crosstype_cleanup.py"
    spec = importlib.util.spec_from_file_location("kg_p2_crosstype_cleanup_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_count_gate_requires_exact_approved_shape():
    cleanup = _load_script()

    with pytest.raises(RuntimeError, match="count gate mismatch"):
        cleanup.validate_counts(
            {
                "merge_sources": 39,
                "launcher_delete_rows": 5,
                "promotions": 3,
            }
        )

    cleanup.validate_counts(
        {
            "merge_sources": 40,
            "launcher_delete_rows": 5,
            "promotions": 3,
        }
    )


def test_promote_entity_type_rejects_existing_agent_name(store):
    cleanup = _load_script()
    store.upsert_entity("tool-codex", "tool", "brainlayerCodex")
    store.upsert_entity("agent-codex", "agent", "brainlayerCodex")

    with pytest.raises(RuntimeError, match="already exists"):
        cleanup.promote_to_agent(store, "tool-codex")


def test_domica_launcher_cleanup_routes_repo_chunks_and_drops_noise(store):
    cleanup = _load_script()
    repo_id = "caa1ccfcb6f24932"
    company_id = "company-1ebe7e404092"
    source_id = "5002583c-d978-593b-8c3f-2fa6213d42e7"
    store.upsert_entity(repo_id, "project", "domica")
    store.upsert_entity(company_id, "company", "domica")
    store.upsert_entity(source_id, "tool", "domicaClaude")
    cursor = store.conn.cursor()
    cursor.execute(
        "INSERT INTO kg_entity_chunks (entity_id, chunk_id, relevance, context, mention_type, relation_tier, weight) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (source_id, "repo-chunk", 0.7, "repo context", "implicit", 4, 0.25),
    )
    cursor.execute(
        "INSERT INTO kg_entity_chunks (entity_id, chunk_id, relevance, context, mention_type, relation_tier, weight) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (source_id, "noise-chunk", 0.9, "launcher context", "explicit", 3, 0.5),
    )

    stats = cleanup.cleanup_domica_launcher_rows(
        store,
        source_ids=[source_id],
        repo_chunk_ids={"repo-chunk"},
        company_chunk_ids=set(),
        drop_chunk_ids={"noise-chunk"},
        repo_id=repo_id,
        company_id=company_id,
    )

    assert stats["domica_launcher_rows_deleted"] == 1
    assert stats["domica_launcher_repo_links"] == 1
    assert stats["domica_launcher_dropped_links"] == 1
    assert store.get_entity(source_id) is None
    assert (
        cursor.execute(
            "SELECT count(*) FROM kg_entity_chunks WHERE entity_id = ? AND chunk_id = ?",
            (repo_id, "repo-chunk"),
        ).fetchone()[0]
        == 1
    )
    assert (
        cursor.execute(
            "SELECT count(*) FROM kg_entity_chunks WHERE chunk_id = ?",
            ("noise-chunk",),
        ).fetchone()[0]
        == 0
    )


def test_move_chunk_link_merges_duplicate_null_metrics(store):
    cleanup = _load_script()
    store.upsert_entity("target", "project", "domica")
    store.upsert_entity("source", "tool", "domicaClaude")
    cursor = store.conn.cursor()
    cursor.execute(
        "INSERT INTO kg_entity_chunks (entity_id, chunk_id, relevance, context, mention_type, relation_tier, weight) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("target", "chunk-1", None, None, None, None, None),
    )
    cursor.execute(
        "INSERT INTO kg_entity_chunks (entity_id, chunk_id, relevance, context, mention_type, relation_tier, weight) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("source", "chunk-1", None, "source context", None, None, None),
    )

    assert cleanup.move_chunk_link(store, "source", "target", "chunk-1") is True

    assert cursor.execute(
        "SELECT relevance, context, mention_type, relation_tier, weight FROM kg_entity_chunks WHERE entity_id = ? AND chunk_id = ?",
        ("target", "chunk-1"),
    ).fetchone() == (None, "source context", None, None, None)
