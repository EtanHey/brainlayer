import asyncio
import time

import pytest

import brainlayer.mcp.search_handler as search_handler_module
from brainlayer.mcp import call_tool
from brainlayer.mcp.search_handler import _brain_search, _search
from brainlayer.vector_store import VectorStore


async def _wait_for_embed_release(timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while search_handler_module._EMBED_IN_FLIGHT.locked() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert not search_handler_module._EMBED_IN_FLIGHT.locked()


class FakeEmbeddingModel:
    def embed_query(self, _query: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class SlowEmbeddingModel:
    def embed_query(self, _query: str) -> list[float]:
        time.sleep(0.05)
        return [0.9, 0.9, 0.9]


class CountingSlowEmbeddingModel:
    def __init__(self) -> None:
        self.calls = 0

    def embed_query(self, _query: str) -> list[float]:
        self.calls += 1
        time.sleep(0.2)
        return [0.9, 0.9, 0.9]


class RaisingEmbeddingModel:
    def embed_query(self, _query: str) -> list[float]:
        raise RuntimeError("embedding backend unavailable")


class OriginEmbeddingModel:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding

    def embed_query(self, _query: str) -> list[float]:
        return self.embedding


class RecordingSearchStore:
    def __init__(self) -> None:
        self.hybrid_kwargs = None

    def count(self) -> int:
        return 1

    def hybrid_search(self, **kwargs):
        self.hybrid_kwargs = kwargs
        return {
            "ids": [["chunk-agent-1"]],
            "documents": [["auth implementation result"]],
            "metadatas": [[{"source_file": "test.md", "project": "brainlayer"}]],
            "distances": [[0.25]],
        }

    def enrich_results_with_session_context(self, results):
        return results


class FtsFallbackSearchStore(RecordingSearchStore):
    def hybrid_search(self, **kwargs):
        self.hybrid_kwargs = kwargs
        if kwargs["query_embedding"] is None:
            return {
                "ids": [["fts-fallback-hit"]],
                "documents": [["keyword fallback result from FTS"]],
                "metadatas": [[{"source_file": "fts.md", "project": "brainlayer"}]],
                "distances": [[1.0]],
            }
        return {
            "ids": [["hybrid-hit"]],
            "documents": [["semantic and keyword hybrid result"]],
            "metadatas": [[{"source_file": "hybrid.md", "project": "brainlayer"}]],
            "distances": [[0.25]],
        }


class EmptyFtsFallbackSearchStore(RecordingSearchStore):
    def hybrid_search(self, **kwargs):
        self.hybrid_kwargs = kwargs
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }


class RecordingKgSearchStore(RecordingSearchStore):
    def __init__(self) -> None:
        super().__init__()
        self.kg_hybrid_kwargs = None

    def kg_hybrid_search(self, **kwargs):
        self.kg_hybrid_kwargs = kwargs
        return {
            "chunks": {
                "ids": [["chunk-kg-agent-1"]],
                "documents": [["auth implementation entity result"]],
                "metadatas": [[{"source_file": "kg.md", "project": "brainlayer"}]],
                "distances": [[0.25]],
            },
            "facts": [],
        }


class OriginKgSearchStore(RecordingKgSearchStore):
    def kg_hybrid_search(self, **kwargs):
        self.kg_hybrid_kwargs = kwargs
        return {
            "chunks": {
                "ids": [["kg-new", "kg-old"]],
                "documents": [["newer Alice origin note", "oldest Alice origin note"]],
                "metadatas": [
                    [
                        {
                            "source_file": "kg-new.md",
                            "project": "brainlayer",
                            "created_at": "2026-03-05T09:00:00Z",
                        },
                        {
                            "source_file": "kg-old.md",
                            "project": "brainlayer",
                            "created_at": "2026-01-05T09:00:00Z",
                        },
                    ]
                ],
                "distances": [[0.1, 0.2]],
            },
            "facts": [],
        }


def _origin_embedding(seed: float) -> list[float]:
    return [seed + (i / 100000.0) for i in range(1024)]


def _seed_origin_search_store(db_path) -> tuple[VectorStore, list[float]]:
    store = VectorStore(db_path)
    query_embedding = _origin_embedding(0.01)
    chunks = [
        {
            "id": "origin-old",
            "content": "originmarker retention first decision from the earliest architecture note",
            "metadata": {"role": "assistant"},
            "source_file": "origin-old.jsonl",
            "project": "origin-test",
            "content_type": "assistant_text",
            "char_count": 70,
            "source": "claude_code",
            "created_at": "2026-01-05T09:00:00Z",
        },
        {
            "id": "origin-mid",
            "content": "originmarker retention architecture followup after the first decision",
            "metadata": {"role": "assistant"},
            "source_file": "origin-mid.jsonl",
            "project": "origin-test",
            "content_type": "assistant_text",
            "char_count": 68,
            "source": "claude_code",
            "created_at": "2026-02-05T09:00:00Z",
        },
        {
            "id": "origin-new",
            "content": "originmarker retention architecture final implementation note with exact query terms",
            "metadata": {"role": "assistant"},
            "source_file": "origin-new.jsonl",
            "project": "origin-test",
            "content_type": "assistant_text",
            "char_count": 76,
            "source": "claude_code",
            "created_at": "2026-03-05T09:00:00Z",
        },
    ]
    store.upsert_chunks(
        chunks,
        [
            _origin_embedding(0.90),
            _origin_embedding(0.50),
            query_embedding,
        ],
    )
    return store, query_embedding


@pytest.mark.asyncio
async def test_brain_search_falls_back_to_fts_when_embedding_times_out(monkeypatch):
    store = FtsFallbackSearchStore()

    monkeypatch.setenv("BRAINLAYER_EMBED_TIMEOUT_MS", "1")
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: SlowEmbeddingModel())
    monkeypatch.setattr("brainlayer.mcp.search_handler._normalize_project_name", lambda project: project)

    try:
        started = time.monotonic()
        content, structured = await _search(
            query="keyword fallback",
            source="all",
            num_results=1,
        )
        elapsed = time.monotonic() - started

        assert elapsed < 0.5
        assert store.hybrid_kwargs["query_embedding"] is None
        assert structured["search_mode"] == "fts_only"
        assert structured["fallback_reason"] == "embed_timeout"
        assert [item["chunk_id"] for item in structured["results"]] == ["fts-fallback-hit"]
        assert "FTS-only fallback" in content[0].text
    finally:
        await _wait_for_embed_release()


@pytest.mark.asyncio
async def test_brain_search_falls_back_to_fts_when_embedding_raises(monkeypatch):
    store = FtsFallbackSearchStore()

    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: RaisingEmbeddingModel())
    monkeypatch.setattr("brainlayer.mcp.search_handler._normalize_project_name", lambda project: project)

    _content, structured = await _search(
        query="keyword fallback",
        source="all",
        num_results=1,
    )

    assert store.hybrid_kwargs["query_embedding"] is None
    assert structured["search_mode"] == "fts_only"
    assert structured["fallback_reason"] == "embed_error:RuntimeError"
    assert [item["chunk_id"] for item in structured["results"]] == ["fts-fallback-hit"]


@pytest.mark.asyncio
async def test_brain_search_uses_hybrid_when_embedding_is_fast(monkeypatch):
    store = FtsFallbackSearchStore()

    monkeypatch.setenv("BRAINLAYER_EMBED_TIMEOUT_MS", "1000")
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr("brainlayer.mcp.search_handler._normalize_project_name", lambda project: project)

    _content, structured = await _search(
        query="keyword fallback",
        source="all",
        num_results=1,
    )

    assert store.hybrid_kwargs["query_embedding"] == [0.1, 0.2, 0.3]
    assert structured["search_mode"] == "hybrid"
    assert "fallback_reason" not in structured
    assert [item["chunk_id"] for item in structured["results"]] == ["hybrid-hit"]


@pytest.mark.asyncio
async def test_brain_search_skips_parallel_embedding_after_timeout(monkeypatch):
    store = FtsFallbackSearchStore()
    model = CountingSlowEmbeddingModel()

    monkeypatch.setenv("BRAINLAYER_EMBED_TIMEOUT_MS", "1")
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: model)
    monkeypatch.setattr("brainlayer.mcp.search_handler._normalize_project_name", lambda project: project)

    try:
        _content1, structured1 = await _search(
            query="keyword fallback",
            source="all",
            num_results=1,
        )
        _content2, structured2 = await _search(
            query="keyword fallback",
            source="all",
            num_results=1,
        )

        assert model.calls == 1
        assert structured1["search_mode"] == "fts_only"
        assert structured1["fallback_reason"] == "embed_timeout"
        assert structured2["search_mode"] == "fts_only"
        assert structured2["fallback_reason"] == "embed_busy"
    finally:
        await _wait_for_embed_release()


@pytest.mark.asyncio
async def test_brain_search_releases_embed_lock_when_executor_timeout_cancels_queued_future(monkeypatch):
    store = FtsFallbackSearchStore()
    loop = asyncio.get_running_loop()
    original_run_in_executor = loop.run_in_executor

    def queued_forever(_executor, _func, *_args):
        return loop.create_future()

    monkeypatch.setenv("BRAINLAYER_EMBED_TIMEOUT_MS", "1")
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr("brainlayer.mcp.search_handler._normalize_project_name", lambda project: project)
    monkeypatch.setattr(loop, "run_in_executor", queued_forever)

    _content1, structured1 = await _search(
        query="keyword fallback",
        source="all",
        num_results=1,
    )

    monkeypatch.setattr(loop, "run_in_executor", original_run_in_executor)
    _content2, structured2 = await _search(
        query="keyword fallback",
        source="all",
        num_results=1,
    )

    assert structured1["search_mode"] == "fts_only"
    assert structured1["fallback_reason"] == "embed_timeout"
    assert structured2["search_mode"] == "hybrid"
    assert search_handler_module._EMBED_IN_FLIGHT.locked() is False


@pytest.mark.asyncio
async def test_brain_search_releases_embed_lock_when_cancelled_before_executor_starts(monkeypatch):
    store = FtsFallbackSearchStore()
    loop = asyncio.get_running_loop()
    original_run_in_executor = loop.run_in_executor

    def queued_forever(_executor, _func, *_args):
        return loop.create_future()

    monkeypatch.setenv("BRAINLAYER_EMBED_TIMEOUT_MS", "1000")
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr("brainlayer.mcp.search_handler._normalize_project_name", lambda project: project)
    monkeypatch.setattr(loop, "run_in_executor", queued_forever)

    task = asyncio.create_task(
        _search(
            query="keyword fallback",
            source="all",
            num_results=1,
        )
    )
    try:
        deadline = time.monotonic() + 1.0
        while not search_handler_module._EMBED_IN_FLIGHT.locked() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert search_handler_module._EMBED_IN_FLIGHT.locked()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        monkeypatch.setattr(loop, "run_in_executor", original_run_in_executor)
        _content, structured = await _search(
            query="keyword fallback",
            source="all",
            num_results=1,
        )

        assert structured["search_mode"] == "hybrid"
        assert search_handler_module._EMBED_IN_FLIGHT.locked() is False
    finally:
        if not task.done():
            task.cancel()
        if search_handler_module._EMBED_IN_FLIGHT.locked():
            search_handler_module._EMBED_IN_FLIGHT.release()


def test_embed_timeout_rejects_non_finite_env_values(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_EMBED_TIMEOUT_MS", "nan")
    assert search_handler_module._embed_timeout_ms() == 1000.0

    monkeypatch.setenv("BRAINLAYER_EMBED_TIMEOUT_MS", "inf")
    assert search_handler_module._embed_timeout_ms() == 1000.0


@pytest.mark.asyncio
async def test_brain_search_empty_fts_fallback_includes_fallback_metadata(monkeypatch):
    store = EmptyFtsFallbackSearchStore()

    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: RaisingEmbeddingModel())
    monkeypatch.setattr("brainlayer.mcp.search_handler._normalize_project_name", lambda project: project)

    content, structured = await _search(
        query="keyword fallback",
        source="all",
        num_results=1,
    )

    assert store.hybrid_kwargs["query_embedding"] is None
    assert structured["search_mode"] == "fts_only"
    assert structured["fallback_reason"] == "embed_error:RuntimeError"
    assert structured["total"] == 0
    assert "FTS-only fallback" in content[0].text


@pytest.mark.asyncio
async def test_brain_search_mcp_threads_agent_id_to_hybrid_search(monkeypatch):
    store = RecordingSearchStore()

    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_route_enabled", lambda: False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr("brainlayer.mcp.search_handler._expanded_fts_query", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._exact_chunk_lookup_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._detect_entities", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("brainlayer.mcp.search_handler._normalize_project_name", lambda project: project)

    await call_tool(
        "brain_search",
        {
            "query": "auth implementation",
            "source": "all",
            "agent_id": "codex-test-agent",
        },
    )

    assert store.hybrid_kwargs is not None
    assert store.hybrid_kwargs["agent_id"] == "codex-test-agent"


@pytest.mark.asyncio
async def test_brain_search_origin_order_returns_oldest_matching_chunks_without_changing_default(monkeypatch, tmp_path):
    store, query_embedding = _seed_origin_search_store(tmp_path / "origin.db")

    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_route_enabled", lambda: False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr(
        "brainlayer.mcp.search_handler._get_embedding_model",
        lambda: OriginEmbeddingModel(query_embedding),
    )
    monkeypatch.setattr("brainlayer.mcp.search_handler._detect_entities", lambda *_args, **_kwargs: [])

    try:
        _default_content, default_structured = await _brain_search(
            query="originmarker retention architecture",
            project="origin-test",
            source="all",
            num_results=2,
            allow_helper_route=False,
        )
        default_ids = [item["chunk_id"] for item in default_structured["results"]]
        assert default_ids[0] == "origin-new"

        origin_content, origin_structured = await _brain_search(
            query="originmarker retention architecture",
            project="origin-test",
            source="all",
            num_results=2,
            order="origin",
            allow_helper_route=False,
        )

        origin_ids = [item["chunk_id"] for item in origin_structured["results"]]
        assert origin_ids == ["origin-old", "origin-mid"]
        assert [item["date"] for item in origin_structured["results"]] == ["2026-01-05", "2026-02-05"]
        assert origin_structured["order"] == "origin"
        assert origin_structured["order_scope"] == "expanded_hybrid_candidates"
        assert "- Order: origin" in origin_content[0].text
        assert "expanded hybrid candidates" in origin_content[0].text
    finally:
        store.close()


@pytest.mark.asyncio
async def test_brain_search_origin_order_filters_explicit_file_path(monkeypatch, tmp_path):
    store, query_embedding = _seed_origin_search_store(tmp_path / "origin-file.db")

    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_route_enabled", lambda: False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr(
        "brainlayer.mcp.search_handler._get_embedding_model",
        lambda: OriginEmbeddingModel(query_embedding),
    )
    monkeypatch.setattr("brainlayer.mcp.search_handler._detect_entities", lambda *_args, **_kwargs: [])

    try:
        _content, structured = await _brain_search(
            query="originmarker retention architecture",
            file_path="origin-mid.jsonl",
            project="origin-test",
            source="all",
            num_results=2,
            order="origin",
            allow_helper_route=False,
        )

        assert [item["chunk_id"] for item in structured["results"]] == ["origin-mid"]
        assert structured["order"] == "origin"
        assert structured["order_scope"] == "expanded_hybrid_candidates"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_brain_search_origin_order_sorts_entity_route_chunks(monkeypatch):
    store = OriginKgSearchStore()

    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_route_enabled", lambda: False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr("brainlayer.mcp.search_handler._expanded_fts_query", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._exact_chunk_lookup_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._kg_facts_sql", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        "brainlayer.mcp.search_handler._detect_entities",
        lambda *_args, **_kwargs: [{"name": "Alice"}],
    )
    monkeypatch.setattr("brainlayer.mcp.search_handler._normalize_project_name", lambda project: project)

    _default_content, default_structured = await _brain_search(
        query="Alice originmarker",
        project="brainlayer",
        source="all",
        num_results=2,
        allow_helper_route=False,
    )
    default_ids = [item["chunk_id"] for item in default_structured["results"]]
    assert default_ids == ["kg-new", "kg-old"]

    origin_content, origin_structured = await _brain_search(
        query="Alice originmarker",
        project="brainlayer",
        source="all",
        num_results=2,
        order="origin",
        allow_helper_route=False,
    )

    origin_ids = [item["chunk_id"] for item in origin_structured["results"]]
    assert origin_ids == ["kg-old", "kg-new"]
    assert origin_structured["order"] == "origin"
    assert origin_structured["order_scope"] == "expanded_hybrid_candidates"
    assert store.kg_hybrid_kwargs["n_results"] == 100
    assert "- Order: origin" in origin_content[0].text
    assert "expanded hybrid candidates" in origin_content[0].text


@pytest.mark.parametrize(
    ("signal_name", "handler_name"),
    [
        ("_query_signals_current_context", "_current_context"),
        ("_query_signals_think", "_think"),
        ("_query_signals_recall", "_recall"),
    ],
)
@pytest.mark.asyncio
async def test_brain_search_origin_order_bypasses_smart_routes(monkeypatch, signal_name, handler_name):
    store = RecordingSearchStore()

    async def fail_smart_route(*_args, **_kwargs):
        raise AssertionError("origin order must stay on search route")

    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_route_enabled", lambda: False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr("brainlayer.mcp.search_handler._expanded_fts_query", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._exact_chunk_lookup_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._detect_entities", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("brainlayer.mcp.search_handler._normalize_project_name", lambda project: project)
    monkeypatch.setattr("brainlayer.mcp.search_handler._query_signals_current_context", lambda _query: False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._query_signals_think", lambda _query: False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._query_signals_recall", lambda _query: False)
    monkeypatch.setattr(f"brainlayer.mcp.search_handler.{signal_name}", lambda _query: True)
    monkeypatch.setattr(f"brainlayer.mcp.search_handler.{handler_name}", fail_smart_route)

    _content, structured = await _brain_search(
        query="how did I implement auth originmarker",
        project="brainlayer",
        source="all",
        num_results=1,
        order="origin",
        allow_helper_route=False,
    )

    assert store.hybrid_kwargs is not None
    assert structured["order"] == "origin"
    assert structured["order_scope"] == "expanded_hybrid_candidates"


@pytest.mark.parametrize(
    ("query", "file_path", "regression_signal", "extracted_file"),
    [
        ("auth implementation", "src/auth.py", False, None),
        ("regression history for auth", "src/auth.py", True, None),
        ("src/auth.py originmarker", None, False, "src/auth.py"),
    ],
)
@pytest.mark.asyncio
async def test_brain_search_origin_order_bypasses_file_routes(
    monkeypatch, query, file_path, regression_signal, extracted_file
):
    store = RecordingSearchStore()

    async def fail_file_route(*_args, **_kwargs):
        raise AssertionError("origin order must stay on search route")

    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_route_enabled", lambda: False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr("brainlayer.mcp.search_handler._expanded_fts_query", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._exact_chunk_lookup_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._detect_entities", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("brainlayer.mcp.search_handler._normalize_project_name", lambda project: project)
    monkeypatch.setattr("brainlayer.mcp.search_handler._query_has_regression_signal", lambda _query: regression_signal)
    monkeypatch.setattr("brainlayer.mcp.search_handler._extract_file_path", lambda _query: extracted_file)
    monkeypatch.setattr("brainlayer.mcp.search_handler._file_timeline", fail_file_route)
    monkeypatch.setattr("brainlayer.mcp.search_handler._regression", fail_file_route)
    monkeypatch.setattr("brainlayer.mcp.search_handler._recall", fail_file_route)

    _content, structured = await _brain_search(
        query=query,
        file_path=file_path,
        project="brainlayer",
        source="all",
        num_results=1,
        order="origin",
        allow_helper_route=False,
    )

    expected_file = file_path or extracted_file
    assert store.hybrid_kwargs is not None
    assert store.hybrid_kwargs["source_file_filter_like"] == f"%{expected_file}%"
    assert structured["order"] == "origin"
    assert structured["order_scope"] == "expanded_hybrid_candidates"


@pytest.mark.asyncio
async def test_brain_search_origin_file_path_skips_entity_route(monkeypatch):
    store = RecordingKgSearchStore()

    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_route_enabled", lambda: False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr("brainlayer.mcp.search_handler._expanded_fts_query", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._exact_chunk_lookup_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._kg_facts_sql", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        "brainlayer.mcp.search_handler._detect_entities",
        lambda *_args, **_kwargs: [{"name": "Alice"}],
    )
    monkeypatch.setattr("brainlayer.mcp.search_handler._normalize_project_name", lambda project: project)

    _content, structured = await _brain_search(
        query="Alice auth implementation",
        file_path="src/auth.py",
        project="brainlayer",
        source="all",
        num_results=1,
        order="origin",
        allow_helper_route=False,
    )

    assert store.kg_hybrid_kwargs is None
    assert store.hybrid_kwargs is not None
    assert store.hybrid_kwargs["source_file_filter_like"] == "%src/auth.py%"
    assert structured["order"] == "origin"


@pytest.mark.asyncio
async def test_brain_search_threads_fail_closed_worker_consumer_scope_when_env_unset(monkeypatch):
    store = RecordingSearchStore()

    monkeypatch.delenv("BRAINLAYER_CONSUMER", raising=False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_route_enabled", lambda: False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr("brainlayer.mcp.search_handler._expanded_fts_query", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._exact_chunk_lookup_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._detect_entities", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("brainlayer.mcp.search_handler._normalize_project_name", lambda project: project)
    monkeypatch.setattr("brainlayer.scoping.resolve_project_scope", lambda: "brainlayer")

    await _brain_search(
        query="auth implementation",
        source="all",
        allow_helper_route=False,
    )

    scope = store.hybrid_kwargs["consumer_scope"]
    assert scope.role == "worker"
    assert scope.project_filter == "brainlayer"
    assert scope.allow_null_project is False
    assert store.hybrid_kwargs["project_filter"] == "brainlayer"


@pytest.mark.asyncio
async def test_brain_search_threads_orchestrator_consumer_scope(monkeypatch):
    store = RecordingSearchStore()

    monkeypatch.setenv("BRAINLAYER_CONSUMER", "orchestrator")
    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_route_enabled", lambda: False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr("brainlayer.mcp.search_handler._expanded_fts_query", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._exact_chunk_lookup_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._detect_entities", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("brainlayer.mcp.search_handler._normalize_project_name", lambda project: project)
    monkeypatch.setattr("brainlayer.scoping.resolve_project_scope", lambda: "brainlayer")

    await _brain_search(
        query="auth implementation",
        source="all",
        allow_helper_route=False,
    )

    scope = store.hybrid_kwargs["consumer_scope"]
    assert scope.role == "orchestrator"
    assert scope.project_filter is None
    assert scope.allow_null_project is True
    assert store.hybrid_kwargs["project_filter"] is None


@pytest.mark.asyncio
async def test_brain_search_threads_helper_fast_profile_to_hybrid_search(monkeypatch):
    store = RecordingSearchStore()

    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_route_enabled", lambda: False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr("brainlayer.mcp.search_handler._expanded_fts_query", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._exact_chunk_lookup_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._detect_entities", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("brainlayer.mcp.search_handler._normalize_project_name", lambda project: project)

    await _brain_search(
        query="auth implementation",
        source="all",
        allow_helper_route=False,
        brainbar_helper_fast_profile=True,
    )

    assert store.hybrid_kwargs is not None
    assert store.hybrid_kwargs["brainbar_helper_fast_profile"] is True


@pytest.mark.asyncio
async def test_brain_search_entity_route_threads_agent_id_to_kg_hybrid_search(monkeypatch):
    store = RecordingKgSearchStore()

    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_route_enabled", lambda: False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr("brainlayer.mcp.search_handler._expanded_fts_query", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._exact_chunk_lookup_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._kg_facts_sql", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        "brainlayer.mcp.search_handler._detect_entities",
        lambda *_args, **_kwargs: [{"name": "Alice"}],
    )
    monkeypatch.setattr("brainlayer.mcp.search_handler._normalize_project_name", lambda project: project)

    await call_tool(
        "brain_search",
        {
            "query": "what do we know about Alice",
            "source": "all",
            "agent_id": "codex-test-agent",
        },
    )

    assert store.kg_hybrid_kwargs is not None
    assert store.kg_hybrid_kwargs["agent_id"] == "codex-test-agent"


@pytest.mark.asyncio
async def test_brain_search_threads_helper_fast_profile_to_kg_hybrid_search(monkeypatch):
    store = RecordingKgSearchStore()

    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_route_enabled", lambda: False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr("brainlayer.mcp.search_handler._expanded_fts_query", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._exact_chunk_lookup_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._kg_facts_sql", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        "brainlayer.mcp.search_handler._detect_entities",
        lambda *_args, **_kwargs: [{"name": "Alice"}],
    )
    monkeypatch.setattr("brainlayer.mcp.search_handler._normalize_project_name", lambda project: project)

    await _brain_search(
        query="what do we know about Alice",
        source="all",
        allow_helper_route=False,
        brainbar_helper_fast_profile=True,
    )

    assert store.kg_hybrid_kwargs is not None
    assert store.kg_hybrid_kwargs["brainbar_helper_fast_profile"] is True


@pytest.mark.asyncio
async def test_brain_search_think_route_threads_agent_id_to_hybrid_search(monkeypatch):
    store = RecordingSearchStore()

    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_route_enabled", lambda: False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: FakeEmbeddingModel())

    await call_tool(
        "brain_search",
        {
            "query": "previously auth implementation",
            "source": "all",
            "agent_id": "codex-test-agent",
        },
    )

    assert store.hybrid_kwargs is not None
    assert store.hybrid_kwargs["agent_id"] == "codex-test-agent"


@pytest.mark.asyncio
async def test_brain_search_recall_route_threads_agent_id_to_hybrid_search(monkeypatch):
    store = RecordingSearchStore()

    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_route_enabled", lambda: False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: FakeEmbeddingModel())

    await call_tool(
        "brain_search",
        {
            "query": "history of auth implementation",
            "source": "all",
            "agent_id": "codex-test-agent",
        },
    )

    assert store.hybrid_kwargs is not None
    assert store.hybrid_kwargs["agent_id"] == "codex-test-agent"
