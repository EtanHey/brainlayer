"""Tests for score-based adaptive prompt injection in the BrainLayer hook."""

import importlib.util
import time
from pathlib import Path

import pytest

HOOKS_DIR = Path(__file__).parent.parent / "hooks"


def load_hook_module(filename: str):
    spec = importlib.util.spec_from_file_location(
        filename.replace("-", "_").replace(".py", ""),
        HOOKS_DIR / filename,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def prompt_search():
    return load_hook_module("brainlayer-prompt-search.py")


def _row(
    chunk_id: str,
    score: float,
    *,
    content: str | None = None,
    project: str | None = "brainlayer",
    tags: str | None = '["memory"]',
):
    return {
        "id": chunk_id,
        "content": content or f"content for {chunk_id}",
        "importance": 5.0,
        "project": project,
        "tags": tags,
        "created_at": "2026-04-05T00:00:00Z",
        "rrf_score": score,
    }


class TestAdaptiveInjectionGating:
    def test_score_gating_high_confidence(self, prompt_search):
        rows = [
            _row("best", 0.033),
            _row("second", 0.019),
            _row("third", 0.017),
            _row("tail", 0.011),
        ]

        selected = prompt_search.select_adaptive_injection_rows(rows, entity_count=1)

        assert 2 <= len(selected) <= 3
        assert [row["id"] for row in selected] == ["best", "third", "second"]

    def test_score_gating_moderate(self, prompt_search):
        rows = [
            _row("best", 0.0145),
            _row("second", 0.013),
            _row("third", 0.012),
            _row("fourth", 0.011),
            _row("fifth", 0.0102),
            _row("sixth", 0.008),
        ]

        selected = prompt_search.select_adaptive_injection_rows(rows)

        # MAX_ADAPTIVE_INJECTION=3, so moderate-confidence returns top 3
        assert [row["id"] for row in selected] == ["best", "third", "second"]

    def test_score_gating_light(self, prompt_search):
        rows = [
            _row("best", 0.009),
            _row("second", 0.008),
            _row("third", 0.006),
        ]

        selected = prompt_search.select_adaptive_injection_rows(rows)

        assert [row["id"] for row in selected] == ["best"]

    def test_score_gating_skip(self, prompt_search):
        rows = [
            _row("best", 0.0049),
            _row("second", 0.004),
        ]

        selected = prompt_search.select_adaptive_injection_rows(rows)

        assert selected == []

    def test_strategic_ordering(self, prompt_search):
        rows = [
            _row("best", 0.015),
            _row("second", 0.014),
            _row("third", 0.013),
            _row("fourth", 0.012),
        ]

        ordered = prompt_search.strategic_reorder(rows)

        assert [row["id"] for row in ordered] == ["best", "third", "fourth", "second"]

    def test_max_injection_cap(self, prompt_search):
        rows = [_row(f"chunk-{idx}", 0.012 - (idx * 0.0001)) for idx in range(10)]

        selected = prompt_search.select_adaptive_injection_rows(rows)

        assert len(selected) == 3  # MAX_ADAPTIVE_INJECTION=3


class TestPollutionFiltering:
    def test_eval_pollution_filter_project(self, prompt_search):
        rows = [
            _row("prod", 0.012, project="brainlayer"),
            _row("eval", 0.020, project="eval-sandbox"),
        ]

        filtered = prompt_search.filter_pollution_rows(rows)

        assert [row["id"] for row in filtered] == ["prod"]

    def test_eval_pollution_filter_tag(self, prompt_search):
        rows = [
            _row("prod", 0.012, tags='["memory","decision"]'),
            _row("eval", 0.020, tags='["eval-test","memory"]'),
        ]

        filtered = prompt_search.filter_pollution_rows(rows)

        assert [row["id"] for row in filtered] == ["prod"]


class TestSearchFallback:
    def _wait_for_hybrid_release(self, prompt_search, timeout_s: float = 1.0):
        deadline = time.monotonic() + timeout_s
        while prompt_search.HYBRID_IN_FLIGHT.locked() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not prompt_search.HYBRID_IN_FLIGHT.locked()

    def test_fallback_to_fts_only(self, prompt_search, monkeypatch):
        fts_rows = [_row("fts-best", 0.0)]

        monkeypatch.setattr(
            prompt_search,
            "run_hybrid_search",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("embed failed")),
        )
        monkeypatch.setattr(prompt_search, "run_fts_search", lambda *args, **kwargs: fts_rows)

        rows, used_hybrid = prompt_search.search_prompt_chunks(
            prompt="keyword fallback query",
            db_path="/tmp/test.db",
            keywords=["keyword", "fallback"],
            limit=8,
        )

        assert used_hybrid is False
        assert [row["id"] for row in rows] == ["fts-best"]

    def test_slow_hybrid_search_falls_back_to_fts_only_within_timeout(self, prompt_search, monkeypatch):
        fts_rows = [_row("fts-timeout", 0.0)]

        def slow_hybrid(*args, **kwargs):
            time.sleep(0.05)
            return [_row("late-hybrid", 0.02)]

        monkeypatch.setenv("BRAINLAYER_EMBED_TIMEOUT_MS", "1")
        monkeypatch.setattr(prompt_search, "run_hybrid_search", slow_hybrid)
        monkeypatch.setattr(prompt_search, "run_fts_search", lambda *args, **kwargs: fts_rows)

        try:
            started = time.monotonic()
            rows, used_hybrid = prompt_search.search_prompt_chunks(
                prompt="keyword fallback query",
                db_path="/tmp/test.db",
                keywords=["keyword", "fallback"],
                limit=8,
            )
            elapsed = time.monotonic() - started

            assert elapsed < 0.5
            assert used_hybrid is False
            assert [row["id"] for row in rows] == ["fts-timeout"]
        finally:
            self._wait_for_hybrid_release(prompt_search)

    def test_timed_out_hybrid_search_does_not_spawn_parallel_workers(self, prompt_search, monkeypatch):
        fts_rows = [_row("fts-busy", 0.0)]
        calls = 0

        def slow_hybrid(*args, **kwargs):
            nonlocal calls
            calls += 1
            time.sleep(0.2)
            return [_row("late-hybrid", 0.02)]

        monkeypatch.setenv("BRAINLAYER_EMBED_TIMEOUT_MS", "1")
        monkeypatch.setattr(prompt_search, "run_hybrid_search", slow_hybrid)
        monkeypatch.setattr(prompt_search, "run_fts_search", lambda *args, **kwargs: fts_rows)

        try:
            rows1, used_hybrid1 = prompt_search.search_prompt_chunks(
                prompt="keyword fallback query",
                db_path="/tmp/test.db",
                keywords=["keyword", "fallback"],
                limit=8,
            )
            rows2, used_hybrid2 = prompt_search.search_prompt_chunks(
                prompt="keyword fallback query",
                db_path="/tmp/test.db",
                keywords=["keyword", "fallback"],
                limit=8,
            )

            assert calls == 1
            assert used_hybrid1 is False
            assert used_hybrid2 is False
            assert [row["id"] for row in rows1] == ["fts-busy"]
            assert [row["id"] for row in rows2] == ["fts-busy"]
        finally:
            self._wait_for_hybrid_release(prompt_search)

    def test_embed_timeout_rejects_non_finite_values(self, prompt_search, monkeypatch):
        monkeypatch.setenv("BRAINLAYER_EMBED_TIMEOUT_MS", "nan")
        assert prompt_search.embed_timeout_ms() == 1000.0

        monkeypatch.setenv("BRAINLAYER_EMBED_TIMEOUT_MS", "inf")
        assert prompt_search.embed_timeout_ms() == 1000.0

    def test_fast_hybrid_search_stays_on_hybrid_path(self, prompt_search, monkeypatch):
        hybrid_rows = [_row("hybrid-best", 0.02)]
        fts_rows = [_row("fts-unused", 0.0)]

        monkeypatch.setenv("BRAINLAYER_EMBED_TIMEOUT_MS", "1000")
        monkeypatch.setattr(prompt_search, "run_hybrid_search", lambda *args, **kwargs: hybrid_rows)
        monkeypatch.setattr(prompt_search, "run_fts_search", lambda *args, **kwargs: fts_rows)

        rows, used_hybrid = prompt_search.search_prompt_chunks(
            prompt="keyword fallback query",
            db_path="/tmp/test.db",
            keywords=["keyword", "fallback"],
            limit=8,
        )

        assert used_hybrid is True
        assert [row["id"] for row in rows] == ["hybrid-best"]

    def test_hybrid_search_opens_vector_store_readonly(self, prompt_search, monkeypatch, tmp_path):
        opened = []

        class FakeEmbeddingModel:
            def embed_query(self, _prompt):
                return [0.1, 0.2, 0.3]

        class FakeCursor:
            def execute(self, *_args, **_kwargs):
                return []

        class FakeVectorStore:
            _binary_index_available = False

            def __init__(self, path, readonly=False):
                opened.append((Path(path), readonly))

            def search(self, *, query_embedding, n_results):
                assert query_embedding == [0.1, 0.2, 0.3]
                assert n_results == 9
                return {"ids": [[]], "metadatas": [[]], "documents": [[]]}

            def _read_cursor(self):
                return FakeCursor()

            def close(self):
                pass

        import brainlayer.embeddings
        import brainlayer.vector_store

        monkeypatch.setattr(brainlayer.embeddings, "get_embedding_model", lambda: FakeEmbeddingModel())
        monkeypatch.setattr(brainlayer.vector_store, "VectorStore", FakeVectorStore)

        rows = prompt_search.run_hybrid_search("recent project notes", tmp_path / "brainlayer.db", ["recent"], 3)

        assert rows == []
        assert opened == [(tmp_path / "brainlayer.db", True)]

    def test_no_results_falls_back_to_low_confidence_message(self, prompt_search):
        message = prompt_search.build_low_confidence_fallback([])

        assert message == "No high-confidence memories found. Use brain_search() for deeper retrieval."

    def test_low_relevance_rows_emit_fallback_message(self, prompt_search):
        rows = [_row("low", 0.29)]

        message = prompt_search.build_low_confidence_fallback(rows)

        assert message == "No high-confidence memories found. Use brain_search() for deeper retrieval."

    def test_high_relevance_rows_do_not_emit_fallback_message(self, prompt_search):
        rows = [_row("high", 0.30)]

        message = prompt_search.build_low_confidence_fallback(rows)

        assert message is None
