from types import ModuleType, SimpleNamespace

from brainlayer import embeddings


def test_embedding_model_forwards_explicit_device_without_mps_probe(monkeypatch):
    constructor_calls = []

    class FakeSentenceTransformer:
        def __init__(self, model_name, *, device):
            constructor_calls.append((model_name, device))

    fake_sentence_transformers = ModuleType("sentence_transformers")
    fake_sentence_transformers.SentenceTransformer = FakeSentenceTransformer
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: (_ for _ in ()).throw(AssertionError("MPS probe must not run")))
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", fake_sentence_transformers)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    model = embeddings.EmbeddingModel(device="cpu")

    assert model._load_model() is not None
    assert constructor_calls == [(embeddings.DEFAULT_MODEL, "cpu")]


def test_get_embedding_model_cache_distinguishes_explicit_device(monkeypatch):
    monkeypatch.setattr(embeddings, "_embedding_model", None)

    automatic = embeddings.get_embedding_model()
    cpu = embeddings.get_embedding_model(device="cpu")
    same_cpu = embeddings.get_embedding_model(device="cpu")

    assert automatic is not cpu
    assert cpu is same_cpu
