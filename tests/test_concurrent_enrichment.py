import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock


def _candidate(chunk_id: str, content: str) -> dict:
    return {
        "id": chunk_id,
        "content": content,
        "project": "brainlayer",
        "content_type": "assistant_text",
        "source": "claude_code",
    }


def test_enrich_concurrency_env_var(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_ENRICH_CONCURRENCY", "17")

    import brainlayer.enrichment_controller as controller

    importlib.reload(controller)

    assert controller.ENRICH_CONCURRENCY == 17

    monkeypatch.delenv("BRAINLAYER_ENRICH_CONCURRENCY", raising=False)
    importlib.reload(controller)


def test_enrich_realtime_processes_chunks(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.return_value = [
        _candidate("c1", "content 1"),
        _candidate("c2", "content 2"),
    ]

    build_calls = []
    apply_calls = []

    monkeypatch.setattr(controller, "ENRICH_CONCURRENCY", 2)
    monkeypatch.setattr(controller, "_ensure_content_hash_column", lambda store: True)
    monkeypatch.setattr(controller, "_is_duplicate_content", lambda store, content: False)
    monkeypatch.setattr(controller, "Sanitizer", SimpleNamespace(from_env=lambda: SimpleNamespace()))
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)
    monkeypatch.setattr(controller, "_emit_enrichment_start", lambda *args, **kwargs: True)
    monkeypatch.setattr(controller, "_emit_enrichment_complete", lambda *args, **kwargs: True)
    monkeypatch.setattr(controller, "_emit_enrichment_error", lambda *args, **kwargs: True)

    def fake_build_external_prompt(chunk, sanitizer):
        build_calls.append(chunk["id"])
        return (f"prompt-for-{chunk['id']}", SimpleNamespace())

    def fake_parse_enrichment(raw):
        return {"summary": f"summary-{raw}", "tags": ["python"]}

    class FakeClient:
        class _Models:
            def generate_content(self, **kwargs):
                return SimpleNamespace(text=kwargs["contents"])

        def __init__(self):
            self.models = self._Models()

    def fake_apply_enrichment(store, chunk, enrichment):
        apply_calls.append((chunk["id"], enrichment["summary"]))

    monkeypatch.setattr(controller, "build_external_prompt", fake_build_external_prompt)
    monkeypatch.setattr(controller, "parse_enrichment", fake_parse_enrichment)
    monkeypatch.setattr(controller, "_get_gemini_client", lambda: FakeClient())
    monkeypatch.setattr(controller, "_apply_enrichment", fake_apply_enrichment)

    result = controller.enrich_realtime(store, limit=2, rate_per_second=0)

    assert result.attempted == 2
    assert result.enriched == 2
    assert result.failed == 0
    assert set(build_calls) == {"c1", "c2"}
    assert apply_calls == [
        ("c1", "summary-prompt-for-c1"),
        ("c2", "summary-prompt-for-c2"),
    ] or apply_calls == [
        ("c2", "summary-prompt-for-c2"),
        ("c1", "summary-prompt-for-c1"),
    ]
