import importlib


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


def test_mmr_default_is_off_and_does_not_load_embeddings(monkeypatch):
    monkeypatch.delenv("BRAINLAYER_MMR_LAMBDA", raising=False)
    import brainlayer.search_repo as search_repo

    search_repo = importlib.reload(search_repo)
    store = _SpyStore()
    scored = _scored_candidates()

    reranked = search_repo.SearchMixin._mmr_rerank_scored_results(store, scored, n_results=2)

    assert search_repo._MMR_LAMBDA == 1.0
    assert reranked == scored
    assert store.embedding_load_calls == 0


def test_mmr_env_override_reenables_embedding_load(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_MMR_LAMBDA", "0.65")
    import brainlayer.search_repo as search_repo

    search_repo = importlib.reload(search_repo)
    store = _SpyStore()

    search_repo.SearchMixin._mmr_rerank_scored_results(store, _scored_candidates(), n_results=2)

    assert search_repo._MMR_LAMBDA == 0.65
    assert store.embedding_load_calls == 1


def test_invalid_mmr_env_override_falls_back_to_default_off(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_MMR_LAMBDA", "not-a-float")
    import brainlayer.search_repo as search_repo

    search_repo = importlib.reload(search_repo)

    assert search_repo._MMR_LAMBDA == 1.0
