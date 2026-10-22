from types import ModuleType, SimpleNamespace

from brainlayer import embeddings


def test_requested_batch_size_reaches_encoder_without_reordering():
    calls = []

    class Vector:
        def __init__(self, value):
            self.value = value

        def tolist(self):
            return [self.value]

    class Encoder:
        def encode(self, texts, **kwargs):
            calls.append((list(texts), kwargs["batch_size"]))
            return [Vector(int(text)) for text in texts]

    model = embeddings.EmbeddingModel(device="cpu")
    model._model = Encoder()
    chunks = [SimpleNamespace(content=str(i)) for i in range(35)]
    result = model.embed_chunks(chunks, batch_size=16)
    assert calls == [([str(i) for i in range(start, min(start + 16, 35))], 16) for start in (0, 16, 32)]
    assert [item.chunk for item in result] == chunks
    assert [item.embedding for item in result] == [[i] for i in range(35)]


def test_embedding_timings_separate_model_load_from_each_encode(monkeypatch):
    clock, timings, phases = [0.0], {}, []

    class Vector:
        def tolist(self):
            return [0.1]

    class Encoder:
        def encode(self, texts, **kwargs):
            clock[0] += 2
            return [Vector() for _ in texts]

    model = embeddings.EmbeddingModel(device="cpu")

    def load():
        clock[0] += 3
        return Encoder()

    def record(name, milliseconds):
        timings[name] = timings.get(name, 0) + milliseconds

    monkeypatch.setattr(model, "_load_model", load)
    monkeypatch.setattr("brainlayer.embeddings.time.perf_counter", lambda: clock[0], raising=False)
    model.embed_chunks(
        [SimpleNamespace(content=str(i)) for i in range(3)],
        batch_size=2,
        on_timing=record,
        on_phase=phases.append,
    )
    assert timings == {"model_load_ms": 3000, "encode_ms": 4000}
    assert phases == ["model_load", "encode", "encode"]


def test_embedding_model_forwards_explicit_device_without_mps_probe(monkeypatch):
    # This test constructs a FAKE SentenceTransformer, so the suite-hygiene refusal that
    # tests/conftest.py arms for every unmarked test does not apply -- and lifting it explicitly,
    # here, is the point: the exemption is one visible line in the test that earns it, not a hole.
    monkeypatch.delenv("BRAINLAYER_FORBID_EMBEDDING_MODEL", raising=False)
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
    monkeypatch.setattr(embeddings, "_embedding_models", {})

    automatic = embeddings.get_embedding_model()
    cpu = embeddings.get_embedding_model(device="cpu")
    same_cpu = embeddings.get_embedding_model(device="cpu")
    same_automatic = embeddings.get_embedding_model()

    assert automatic is not cpu
    assert cpu is same_cpu
    assert automatic is same_automatic
