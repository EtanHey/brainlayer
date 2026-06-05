import importlib
import warnings

import numpy as np


def _scored_candidates():
    return [
        (0.99, "a", "alpha", {}, 0.01),
        (0.98, "b", "beta", {}, 0.02),
        (0.97, "c", "gamma", {}, 0.03),
    ]


class _SpyStore:
    def __init__(self):
        self.embedding_load_calls = 0

    def _load_chunk_embeddings(self, chunk_ids):
        self.embedding_load_calls += 1
        return {}


class _EmbeddingStore:
    def _load_chunk_embeddings(self, chunk_ids):
        return {
            "a": np.array([1.0, 0.0], dtype=np.float32),
            "b": np.array([np.finfo(np.float32).max, np.finfo(np.float32).max], dtype=np.float32),
            "c": np.array([0.0, 1.0], dtype=np.float32),
        }


class _ZeroNormEmbeddingStore:
    def _load_chunk_embeddings(self, chunk_ids):
        return {
            "a": np.array([1.0, 0.0], dtype=np.float32),
            "b": np.array([0.0, 0.0], dtype=np.float32),
            "c": np.array([0.0, 1.0], dtype=np.float32),
        }


class _MixedDimensionEmbeddingStore:
    def _load_chunk_embeddings(self, chunk_ids):
        return {
            "bad": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "similar_a": np.array([1.0, 0.0], dtype=np.float32),
            "similar_b": np.array([1.0, 0.0], dtype=np.float32),
            "diverse": np.array([0.0, 1.0], dtype=np.float32),
        }


def test_mmr_default_enables_diversity_and_loads_embeddings(monkeypatch):
    monkeypatch.delenv("BRAINLAYER_MMR_LAMBDA", raising=False)
    import brainlayer.search_repo as search_repo

    search_repo = importlib.reload(search_repo)
    store = _SpyStore()
    scored = _scored_candidates()

    reranked = search_repo.SearchMixin._mmr_rerank_scored_results(store, scored, n_results=2)

    assert search_repo._MMR_LAMBDA == 0.7
    assert reranked == scored
    assert store.embedding_load_calls == 1


def test_mmr_env_override_reenables_embedding_load(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_MMR_LAMBDA", "0.65")
    import brainlayer.search_repo as search_repo

    search_repo = importlib.reload(search_repo)
    store = _SpyStore()

    search_repo.SearchMixin._mmr_rerank_scored_results(store, _scored_candidates(), n_results=2)

    assert search_repo._MMR_LAMBDA == 0.65
    assert store.embedding_load_calls == 1


def test_invalid_mmr_env_override_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_MMR_LAMBDA", "not-a-float")
    import brainlayer.search_repo as search_repo

    search_repo = importlib.reload(search_repo)

    assert search_repo._MMR_LAMBDA == 0.7


def test_nonfinite_mmr_env_override_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_MMR_LAMBDA", "nan")
    import brainlayer.search_repo as search_repo

    search_repo = importlib.reload(search_repo)

    assert search_repo._MMR_LAMBDA == 0.7


def test_out_of_range_mmr_env_override_is_clamped(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_MMR_LAMBDA", "-0.25")
    import brainlayer.search_repo as search_repo

    search_repo = importlib.reload(search_repo)

    assert search_repo._MMR_LAMBDA == 0.0

    monkeypatch.setenv("BRAINLAYER_MMR_LAMBDA", "1.25")
    search_repo = importlib.reload(search_repo)

    assert search_repo._MMR_LAMBDA == 1.0


def test_mmr_ignores_pathological_embeddings_without_warning(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_MMR_LAMBDA", "0.7")
    import brainlayer.search_repo as search_repo

    search_repo = importlib.reload(search_repo)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        reranked = search_repo.SearchMixin._mmr_rerank_scored_results(
            _EmbeddingStore(),
            _scored_candidates(),
            n_results=2,
        )

    assert len(reranked) == 3
    assert {candidate[1] for candidate in reranked} == {"a", "b", "c"}


def test_mmr_keeps_zero_norm_candidates_without_exhausting_rerank(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_MMR_LAMBDA", "0.7")
    import brainlayer.search_repo as search_repo

    search_repo = importlib.reload(search_repo)

    reranked = search_repo.SearchMixin._mmr_rerank_scored_results(
        _ZeroNormEmbeddingStore(),
        _scored_candidates(),
        n_results=3,
    )

    assert len(reranked) == 3
    assert {candidate[1] for candidate in reranked} == {"a", "b", "c"}


def test_mmr_chooses_common_embedding_shape_not_first_returned(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_MMR_LAMBDA", "0.7")
    import brainlayer.search_repo as search_repo

    search_repo = importlib.reload(search_repo)
    scored = [
        (1.0, "bad", "wrong-dimension", {}, 0.0),
        (0.99, "similar_a", "near duplicate", {}, 0.01),
        (0.98, "similar_b", "near duplicate", {}, 0.02),
        (0.97, "diverse", "distinct", {}, 0.03),
    ]

    reranked = search_repo.SearchMixin._mmr_rerank_scored_results(
        _MixedDimensionEmbeddingStore(),
        scored,
        n_results=2,
    )

    assert [candidate[1] for candidate in reranked] == [
        "bad",
        "similar_a",
        "diverse",
        "similar_b",
    ]
