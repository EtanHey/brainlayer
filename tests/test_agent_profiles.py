import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from typer.testing import CliRunner

from brainlayer import search_repo
from brainlayer._helpers import serialize_f32
from brainlayer.agent_profiles import validate_agent_profile
from brainlayer.cli import app
from brainlayer.search_repo import _hybrid_cache
from brainlayer.vector_store import VectorStore


def _embed(seed: float) -> list[float]:
    return [seed + (i / 10000.0) for i in range(1024)]


def _created_at_days_ago(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds").replace("+00:00", "Z")


def _insert_chunk(
    store: VectorStore,
    *,
    chunk_id: str,
    content: str,
    importance: float,
    embedding: list[float] | None = None,
    created_at: str = "2026-05-01T00:00:00Z",
    chunk_origin: str = "unknown",
) -> None:
    cursor = store.conn.cursor()
    cursor.execute(
        """INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type,
            char_count, source, importance, created_at, chunk_origin
        ) VALUES (?, ?, '{}', 'agent-profile-test.jsonl', 'agent-profile-test',
            'assistant_text', ?, 'claude_code', ?, ?, ?)""",
        (chunk_id, content, len(content), importance, created_at, chunk_origin),
    )
    if embedding is not None:
        cursor.execute(
            "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, serialize_f32(embedding)),
        )


def test_agent_profiles_migration_adds_table(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "brainlayer.db")
    try:
        cols = {row[1]: row[2] for row in store.conn.cursor().execute("PRAGMA table_info(agent_profiles)")}
    finally:
        store.close()

    assert cols == {
        "agent_id": "TEXT",
        "profile_json": "TEXT",
        "updated_at": "REAL",
        "notes": "TEXT",
    }


def test_agent_profile_cli_set_show_roundtrip(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "brainlayer.db"
    profile_path = tmp_path / "orc.json"
    profile_path.write_text(
        json.dumps({"boost_weights": {"fts": 2.0, "vector": 0.5}, "source_weights": {"precompact": 1.3}})
    )
    monkeypatch.setenv("BRAINLAYER_DB", str(db_path))

    runner = CliRunner()
    set_result = runner.invoke(app, ["agent-profile", "set", "orcClaude", str(profile_path)])
    assert set_result.exit_code == 0, set_result.output

    show_result = runner.invoke(app, ["agent-profile", "show", "orcClaude"])
    assert show_result.exit_code == 0, show_result.output
    shown = json.loads(show_result.stdout)
    assert shown["agent_id"] == "orcClaude"
    assert shown["profile"]["boost_weights"]["fts"] == 2.0
    assert shown["profile"]["boost_weights"]["vector"] == 0.5
    assert shown["profile"]["source_weights"]["precompact"] == 1.3


def test_agent_profile_cli_show_uses_readonly_store(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "brainlayer.db"
    profile_path = tmp_path / "orc.json"
    profile_path.write_text(json.dumps({"boost_weights": {"fts": 2.0}}))
    monkeypatch.setenv("BRAINLAYER_DB", str(db_path))

    runner = CliRunner()
    set_result = runner.invoke(app, ["agent-profile", "set", "orcClaude", str(profile_path)])
    assert set_result.exit_code == 0, set_result.output

    writer = VectorStore(db_path)
    try:
        show_result = runner.invoke(app, ["agent-profile", "show", "orcClaude"])
    finally:
        writer.close()

    assert show_result.exit_code == 0, show_result.output
    assert json.loads(show_result.stdout)["agent_id"] == "orcClaude"


def test_agent_profile_rejects_unbounded_weights() -> None:
    try:
        validate_agent_profile({"boost_weights": {"fts": 1000.1}})
    except ValueError as exc:
        assert "positive and <= 1000" in str(exc)
    else:
        raise AssertionError("Expected unbounded profile weight to be rejected")


def test_agent_profile_trims_weight_keys_and_rejects_trimmed_duplicates() -> None:
    normalized = validate_agent_profile({"boost_weights": {" fts ": 2.0}})
    assert normalized["boost_weights"] == {"fts": 2.0}

    try:
        validate_agent_profile({"boost_weights": {"fts": 1.0, " fts ": 2.0}})
    except ValueError as exc:
        assert "duplicated after trimming whitespace" in str(exc)
    else:
        raise AssertionError("Expected duplicate trimmed keys to be rejected")


def test_agent_profile_cli_reports_unreadable_json(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAINLAYER_DB", str(tmp_path / "brainlayer.db"))

    runner = CliRunner()
    result = runner.invoke(app, ["agent-profile", "set", "orcClaude", str(tmp_path / "missing.json")])

    assert result.exit_code == 2
    assert "Could not read profile JSON" in result.output


def test_hybrid_search_agent_profile_applies_orc_weights(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "brainlayer.db")
    query_embedding = _embed(0.1)
    try:
        _insert_chunk(
            store,
            chunk_id="vector-only",
            content="semantic only result without lexical terms",
            importance=10.0,
            embedding=query_embedding,
        )
        _insert_chunk(
            store,
            chunk_id="fts-only",
            content="needle phrase exact lexical result",
            importance=1.0,
            embedding=None,
        )
        store.build_binary_index()
        store._trigram_fts_available = False
        store.set_agent_profile(
            "orcClaude",
            {"boost_weights": {"fts": 3.0, "vector": 0.5}},
            notes="test profile",
        )
        _hybrid_cache.clear()

        default_results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="needle phrase",
            n_results=2,
            agent_id=None,
        )
        _hybrid_cache.clear()
        profiled_results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="needle phrase",
            n_results=2,
            agent_id="orcClaude",
        )
    finally:
        store.close()
        _hybrid_cache.clear()

    assert default_results["ids"][0][0] == "vector-only"
    assert profiled_results["ids"][0][0] == "fts-only"


def test_hybrid_search_agent_profile_scales_recency_intent_neutral_point(monkeypatch, tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "brainlayer.db")
    query_embedding = _embed(0.1)
    multiplier_calls: list[tuple[float, str]] = []
    original_profiled_multiplier = search_repo._profiled_multiplier

    def tracking_profiled_multiplier(base, profile, feature):
        multiplier_calls.append((base, feature))
        return original_profiled_multiplier(base, profile, feature)

    monkeypatch.setattr(search_repo, "_profiled_multiplier", tracking_profiled_multiplier)
    try:
        _insert_chunk(
            store,
            chunk_id="old-vector",
            content="needle phrase archive result",
            importance=1.0,
            embedding=query_embedding,
            created_at=_created_at_days_ago(150),
        )
        _insert_chunk(
            store,
            chunk_id="recent-lexical",
            content="needle phrase recent result",
            importance=1.0,
            embedding=_embed(0.3),
            created_at=_created_at_days_ago(2),
        )
        store.build_binary_index()
        store._trigram_fts_available = False
        store.set_agent_profile(
            "recencyTest",
            {"boost_weights": {"recency_intent": 0.5}},
            notes="test profile",
        )
        _hybrid_cache.clear()

        store.hybrid_search(
            query_embedding=query_embedding,
            query_text="latest needle phrase",
            n_results=2,
            agent_id="recencyTest",
        )
    finally:
        store.close()
        _hybrid_cache.clear()

    assert (2.0, "recency_intent") in multiplier_calls
    assert original_profiled_multiplier(2.0, {"boost_weights": {"recency_intent": 0.5}}, "recency_intent") == 1.5


def test_hybrid_search_agent_profile_uses_chunk_origin_for_semantic_hits(monkeypatch, tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "brainlayer.db")
    query_embedding = _embed(0.1)
    observed_sources: list[str | None] = []
    original_source_weight = search_repo.source_weight

    def tracking_source_weight(profile, source):
        observed_sources.append(source)
        return original_source_weight(profile, source)

    monkeypatch.setattr(search_repo, "source_weight", tracking_source_weight)
    try:
        _insert_chunk(
            store,
            chunk_id="semantic-origin",
            content="semantic only result without lexical terms",
            importance=1.0,
            embedding=query_embedding,
            chunk_origin="profiled_origin",
        )
        store.build_binary_index()
        store._trigram_fts_available = False
        store.set_agent_profile(
            "originTest",
            {"source_weights": {"profiled_origin": 1.5}},
            notes="test profile",
        )
        _hybrid_cache.clear()

        results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="needle phrase",
            n_results=1,
            agent_id="originTest",
        )
    finally:
        store.close()
        _hybrid_cache.clear()

    assert results["metadatas"][0][0]["chunk_origin"] == "profiled_origin"
    assert "profiled_origin" in observed_sources


def test_hybrid_search_missing_profile_matches_default(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "brainlayer.db")
    query_embedding = _embed(0.1)
    try:
        _insert_chunk(
            store,
            chunk_id="vector-only",
            content="semantic only result without lexical terms",
            importance=10.0,
            embedding=query_embedding,
        )
        _insert_chunk(
            store,
            chunk_id="fts-only",
            content="needle phrase exact lexical result",
            importance=1.0,
            embedding=None,
        )
        store.build_binary_index()
        store._trigram_fts_available = False

        default_results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="needle phrase",
            n_results=2,
            agent_id=None,
        )
        _hybrid_cache.clear()
        unknown_results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="needle phrase",
            n_results=2,
            agent_id="unknown-agent",
        )
    finally:
        store.close()
        _hybrid_cache.clear()

    assert unknown_results["ids"][0] == default_results["ids"][0]
