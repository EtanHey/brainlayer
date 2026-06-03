"""Tests for the ABCDE variant enrichment runner. No live API — ChatFn is mocked."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from brainlayer.eval.abcde_enrich_runner import (
    STATUS_ERROR,
    STATUS_OK,
    STATUS_SAFETY_BLOCKED,
    Usage,
    enrich_one,
    judge_row,
    make_http_chat_fn,
    run_batch,
    usage_to_usd,
)
from brainlayer.eval.abcde_variants import ABCDE_VARIANTS_BY_ID
from brainlayer.eval.enrichment_graders import REQUIRED_ENRICHMENT_KEYS, validate_schema_gate
from brainlayer.eval.enrichment_judge import build_judge_request
from scripts import run_abcde_enrich as abcde_driver


class FakeSanitizer:
    """Records what it saw; returns content unchanged with a PII marker on a token."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def sanitize(self, content, metadata=None):
        self.seen.append(content)
        replaced = content.replace("SECRET@EMAIL", "[EMAIL]")
        return SimpleNamespace(
            sanitized=replaced,
            replacements=[("SECRET@EMAIL", "[EMAIL]")] if "SECRET@EMAIL" in content else [],
            pii_detected="SECRET@EMAIL" in content,
        )


def _good_enrichment() -> dict:
    return {
        "summary": "PR #426 merged the ExperimentStore firewall on 2026-06-01.",
        "key_facts": ["PR #426", "2026-06-01", "ExperimentStore"],
        "tags": ["isolation", "ci-gate", "brainlayer"],
        "importance": 8,
        "intent": "implementing",
        "primary_symbols": ["ExperimentStore"],
        "resolved_query": "How was the experiment store isolated?",
        "resolved_queries": [
            "How was the experiment store isolated?",
            "ExperimentStore firewall leak-regression PR 426",
            "PR #426 added an ExperimentStore firewall with a leak-regression CI gate.",
        ],
        "epistemic_level": "validated",
        "version_scope": None,
        "debt_impact": "resolution",
        "external_deps": [],
        "entities": [{"name": "BrainLayer", "type": "project", "relation": "owns the experiment store"}],
        "sentiment_label": "satisfaction",
        "sentiment_score": 0.6,
        "sentiment_signals": ["merged"],
    }


def _chunk(content="ExperimentStore firewall merged in PR #426 on 2026-06-01."):
    return {"id": "chunk-1", "content": content, "project": "brainlayer", "content_type": "assistant_text"}


def _ok_chat_fn(captured=None, *, ticks=12000, ptok=1200, ctok=400):
    def fn(model, prompt, params):
        if captured is not None:
            captured["model"] = model
            captured["prompt"] = prompt
            captured["params"] = params
        return 200, {
            "choices": [{"message": {"content": json.dumps(_good_enrichment())}}],
            "usage": {
                "prompt_tokens": ptok,
                "completion_tokens": ctok,
                "total_tokens": ptok + ctok,
                "cost_in_usd_ticks": ticks,
            },
        }

    return fn


def test_enrich_one_ok_parses_and_passes_schema_gate():
    variant = ABCDE_VARIANTS_BY_ID["C"]
    res = enrich_one(variant, _chunk(), FakeSanitizer(), _ok_chat_fn())
    assert res.status == STATUS_OK
    assert validate_schema_gate(res.enrichment).passed
    assert res.usage.cost_in_usd_ticks == 12000


def test_enrich_one_keeps_raw_json_object_without_schema_parsing():
    raw = {
        **_good_enrichment(),
        "tags": ["MiXeD", "Ci-Gate", "BrainLayer"],
        "importance": 8,
        "version_scope": None,
        "external_deps": [],
        "payload": {"nested": True},
    }

    def raw_fn(model, prompt, params):
        return 200, {
            "choices": [{"message": {"content": f"prefix {json.dumps(raw)} suffix"}}],
            "usage": {"total_tokens": 50},
        }

    res = enrich_one(ABCDE_VARIANTS_BY_ID["A"], _chunk(), FakeSanitizer(), raw_fn)
    assert res.status == STATUS_OK
    assert res.enrichment == raw
    assert set(REQUIRED_ENRICHMENT_KEYS).issubset(res.enrichment)
    assert "version_scope" in res.enrichment
    assert "external_deps" in res.enrichment
    assert isinstance(res.enrichment["importance"], int)
    assert res.enrichment["resolved_queries"] == raw["resolved_queries"]
    assert len(res.enrichment["resolved_queries"]) == 3
    assert validate_schema_gate(res.enrichment).passed
    assert res.error is None


def test_prompt_uses_variant_template_and_sanitized_content():
    captured: dict = {}
    variant = ABCDE_VARIANTS_BY_ID["D"]
    san = FakeSanitizer()
    enrich_one(variant, _chunk("contact SECRET@EMAIL for ExperimentStore"), san, _ok_chat_fn(captured))
    # Sanitizer was invoked on the chunk content...
    assert any("SECRET@EMAIL" in s for s in san.seen)
    # ...and the raw secret never reaches the prompt sent to the backend.
    assert "SECRET@EMAIL" not in captured["prompt"]
    assert "[EMAIL]" in captured["prompt"]
    # Variant D's model + Gemini-style param mapping reached the chat layer.
    assert captured["model"] == variant.model
    assert "max_tokens" in captured["params"]  # mapped from max_output_tokens


def test_safety_block_classified_not_crashed():
    def safety_fn(model, prompt, params):
        return 403, {"error": {"message": "Content violates usage guidelines. Failed check: SAFETY_CHECK_TYPE_BIO"}}

    res = enrich_one(ABCDE_VARIANTS_BY_ID["E"], _chunk(), FakeSanitizer(), safety_fn)
    assert res.status == STATUS_SAFETY_BLOCKED
    assert res.enrichment is None


def test_invalid_json_is_error_not_exception():
    def bad_fn(model, prompt, params):
        return 200, {"choices": [{"message": {"content": "not json at all"}}], "usage": {"total_tokens": 50}}

    res = enrich_one(ABCDE_VARIANTS_BY_ID["A"], _chunk(), FakeSanitizer(), bad_fn)
    assert res.status == STATUS_ERROR
    assert res.error == "invalid_json"


def test_transport_exception_becomes_error_row():
    def boom(model, prompt, params):
        raise ConnectionError("network down")

    res = enrich_one(ABCDE_VARIANTS_BY_ID["A"], _chunk(), FakeSanitizer(), boom)
    assert res.status == STATUS_ERROR
    assert "transport_error" in res.error


def test_usage_to_usd_prefers_ticks_then_falls_back():
    assert usage_to_usd(Usage(cost_in_usd_ticks=1_000_000), tick_usd=1e-9) == pytest.approx(0.001)
    # No ticks → pessimistic per-token fallback.
    assert usage_to_usd(Usage(total_tokens=1_000_000), fallback_usd_per_1m=10.0) == pytest.approx(10.0)


def test_judge_row_shape_is_consumable_by_judge():
    variant = ABCDE_VARIANTS_BY_ID["C"]
    chat_fn = _ok_chat_fn()
    chat_fn.backend_model = "grok-x"
    res = enrich_one(variant, _chunk(), FakeSanitizer(), chat_fn)
    row = judge_row(variant, _chunk(), res)
    # The offline judge must accept the row without raising.
    request = build_judge_request(row)
    assert request["variant_id"] == "C"
    assert request["chunk_id"] == "chunk-1"
    assert request["input"]["enrichment"]["summary"].startswith("PR #426")
    assert row["model"] == variant.model
    assert row["generation"]["backend_model"] == "grok-x"
    assert row["prompt_hash"] == variant.prompt_hash


def test_make_http_chat_fn_model_override_ignores_variant_model(monkeypatch):
    captured: dict = {}

    class FakeResponse:
        status_code = 200
        text = "{}"

        def json(self):
            return {"choices": [{"message": {"content": "{}"}}]}

    def fake_post(url, headers, json, timeout):
        captured["payload"] = json
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)
    variant = ABCDE_VARIANTS_BY_ID["A"]
    chat_fn = make_http_chat_fn(
        base_url="https://example.test/v1",
        api_key="test-key",
        model_override="grok-x",
    )

    status, _body = chat_fn(variant.model, "prompt", {"temperature": 0})

    assert status == 200
    assert captured["payload"]["model"] == "grok-x"


def test_make_http_chat_fn_deepseek_base_url_and_model(monkeypatch):
    captured: dict = {}

    class FakeResponse:
        status_code = 200
        text = "{}"

        def json(self):
            return {"choices": [{"message": {"content": "{}"}}]}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["payload"] = json
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)
    chat_fn = make_http_chat_fn(
        base_url="https://api.deepseek.com",
        api_key="test-key",
        model_override="deepseek-v4-flash",
    )

    status, _body = chat_fn("ignored-variant-model", "prompt", {"temperature": 0})

    assert status == 200
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["payload"]["model"] == "deepseek-v4-flash"


def test_driver_variants_filter_selects_requested_ids_in_registry_order():
    selected = abcde_driver.select_variants("E,A,C")

    assert [variant.id for variant in selected] == ["A", "C", "E"]


def test_driver_resolve_api_key_reads_deepseek_env(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "  deepseek-test-key  ")

    assert abcde_driver.resolve_api_key() == "deepseek-test-key"


def test_run_batch_writes_rows_and_aggregates(tmp_path):
    chunks = [_chunk(), {**_chunk(), "id": "chunk-2"}]
    variants = [ABCDE_VARIANTS_BY_ID["C"], ABCDE_VARIANTS_BY_ID["D"]]
    out = tmp_path / "rows.jsonl"
    stats = run_batch(chunks, variants, FakeSanitizer(), _ok_chat_fn(), output_path=str(out), tick_usd=1e-9)
    assert stats.calls == 4
    assert stats.ok == 4
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 4
    assert all(json.loads(line)["variant_id"] in ("C", "D") for line in lines)
    assert stats.cost_in_usd_ticks == 4 * 12000


def test_run_batch_concurrency_writes_all_rows(tmp_path):
    chunks = [{**_chunk(), "id": f"chunk-{i}"} for i in range(5)]
    variants = [ABCDE_VARIANTS_BY_ID["A"], ABCDE_VARIANTS_BY_ID["C"]]
    out = tmp_path / "rows.jsonl"

    stats = run_batch(
        chunks,
        variants,
        FakeSanitizer(),
        _ok_chat_fn(ticks=0, ptok=100, ctok=100),
        output_path=str(out),
        concurrency=4,
        fallback_usd_per_1m=0.3,
    )

    lines = out.read_text().strip().splitlines()
    assert stats.calls == 10
    assert stats.ok == 10
    assert len(lines) == 10
    assert {json.loads(line)["variant_id"] for line in lines} == {"A", "C"}


def test_run_batch_concurrency_respects_max_calls(tmp_path):
    chunks = [{**_chunk(), "id": f"chunk-{i}"} for i in range(10)]
    variants = [ABCDE_VARIANTS_BY_ID["A"], ABCDE_VARIANTS_BY_ID["C"]]
    out = tmp_path / "rows.jsonl"

    stats = run_batch(
        chunks,
        variants,
        FakeSanitizer(),
        _ok_chat_fn(ticks=0, ptok=100, ctok=100),
        output_path=str(out),
        concurrency=4,
        max_calls=7,
        fallback_usd_per_1m=0.3,
    )

    assert stats.calls == 7
    assert len(out.read_text().strip().splitlines()) == 7


def test_run_batch_hard_budget_stop():
    # Each call costs 0.001 USD (1e6 ticks * 1e-9). Ceiling 0.0025 → stop after ~2 calls.
    chunks = [{**_chunk(), "id": f"c{i}"} for i in range(10)]
    variants = [ABCDE_VARIANTS_BY_ID["C"]]
    stats = run_batch(
        chunks,
        variants,
        FakeSanitizer(),
        _ok_chat_fn(ticks=1_000_000),
        max_usd=0.0025,
        tick_usd=1e-9,
    )
    assert stats.usd_spent <= 0.0025 + 1e-9
    assert stats.calls < 10  # stopped early, did not run all 10


def test_run_batch_concurrency_respects_max_usd_with_fallback_tokens(tmp_path):
    chunks = [{**_chunk(), "id": f"chunk-{i}"} for i in range(10)]
    variants = [ABCDE_VARIANTS_BY_ID["C"]]
    out = tmp_path / "rows.jsonl"

    stats = run_batch(
        chunks,
        variants,
        FakeSanitizer(),
        _ok_chat_fn(ticks=0, ptok=500, ctok=500),
        output_path=str(out),
        concurrency=4,
        max_usd=0.00075,
        fallback_usd_per_1m=0.3,
    )

    assert stats.calls == 2
    assert stats.usd_spent == pytest.approx(0.0006)
    assert len(out.read_text().strip().splitlines()) == 2
