"""ABCDE variant enrichment runner for OpenAI-compatible backends (xAI Grok, DeepSeek).

This is the GENERATION leg of the ABCDE eval: it takes a frozen variant prompt
(from the registry) plus a source chunk, calls an OpenAI-compatible chat API, and
parses the result into the production enrichment JSON schema. The output rows are
consumable by ``enrichment_judge.build_judge_request`` / ``score_jsonl_inline``.

Design constraints:
  * Sanitization is MANDATORY. ``build_external_prompt`` requires a Sanitizer, so
    personal data is scrubbed before any external API call — same guarantee the
    production Gemini path gives.
  * The JUDGE leg stays offline (a free CLI-agent pane). This module only does
    generation.
  * Cost is metered live via ``cost_in_usd_ticks`` (xAI returns it) with a hard
    cumulative spend stop, so a run can never exceed its budget even if the
    per-item estimate is wrong.

The HTTP call is injected (``ChatFn``) so tests never hit a real API.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from brainlayer.eval.abcde_variants import ABCDEVariant
from brainlayer.pipeline.enrichment import build_external_prompt, parse_enrichment

DEFAULT_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-4.20-0309-non-reasoning"
# Only 1e-9 USD/tick is physically plausible: 1e-8 implies ~$76/1M tokens and
# 1e-7 implies ~$765/1M, neither of which exists. See orc gen-7 smoke analysis.
DEFAULT_TICK_USD = 1e-9
# Pessimistic blended fallback price when a backend omits cost_in_usd_ticks
# (e.g. DeepSeek). Used only for the budget meter, never to fabricate telemetry.
DEFAULT_FALLBACK_USD_PER_1M = 10.0

# (model, prompt, params) -> (http_status, response_json)
ChatFn = Callable[[str, str, Mapping[str, Any]], "tuple[int, dict]"]

STATUS_OK = "ok"
STATUS_SAFETY_BLOCKED = "safety_blocked"
STATUS_ERROR = "error"


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_in_usd_ticks: int = 0

    @property
    def has_ticks(self) -> bool:
        return self.cost_in_usd_ticks > 0


@dataclass
class CallResult:
    status: str
    enrichment: Optional[dict]
    raw_text: Optional[str]
    usage: Usage
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


class BudgetExceeded(RuntimeError):
    """Raised by the driver when the cumulative spend ceiling is reached."""


def _to_openai_params(variant_params: Mapping[str, Any]) -> dict[str, Any]:
    """Map registry params (Gemini-style) to OpenAI-compatible chat params."""
    params: dict[str, Any] = {}
    if "temperature" in variant_params:
        params["temperature"] = variant_params["temperature"]
    # Registry uses Gemini's max_output_tokens; OpenAI-compatible uses max_tokens.
    if "max_output_tokens" in variant_params:
        params["max_tokens"] = variant_params["max_output_tokens"]
    elif "max_tokens" in variant_params:
        params["max_tokens"] = variant_params["max_tokens"]
    return params


def _extract_usage(body: Mapping[str, Any]) -> Usage:
    raw = body.get("usage") if isinstance(body, Mapping) else None
    if not isinstance(raw, Mapping):
        return Usage()

    def _int(*names: str) -> int:
        for name in names:
            value = raw.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return max(0, int(value))
        return 0

    return Usage(
        prompt_tokens=_int("prompt_tokens"),
        completion_tokens=_int("completion_tokens"),
        total_tokens=_int("total_tokens"),
        cost_in_usd_ticks=_int("cost_in_usd_ticks"),
    )


def _is_safety_block(status: int, body: Mapping[str, Any]) -> bool:
    if status != 403:
        return False
    text = json.dumps(body).lower()
    return "safety_check" in text or "usage guidelines" in text or "safety" in text


def _error_message(body: Mapping[str, Any]) -> str:
    err = body.get("error") if isinstance(body, Mapping) else None
    if isinstance(err, Mapping):
        return str(err.get("message") or err)
    if isinstance(err, str):
        return err
    return json.dumps(body)[:300]


def enrich_one(
    variant: ABCDEVariant,
    chunk: Mapping[str, Any],
    sanitizer: Any,
    chat_fn: ChatFn,
    *,
    extra_params: Optional[Mapping[str, Any]] = None,
) -> CallResult:
    """Enrich a single chunk with a single variant via an OpenAI-compatible chat API.

    The chunk's content is sanitized inside ``build_external_prompt`` before the
    prompt is sent to ``chat_fn``.
    """
    prompt, _sanitize_result = build_external_prompt(
        dict(chunk), sanitizer, prompt_template=variant.prompt_template
    )
    params = _to_openai_params(variant.params)
    if extra_params:
        params.update(extra_params)

    try:
        status, body = chat_fn(variant.model, prompt, params)
    except Exception as exc:  # noqa: BLE001 — surface transport errors as a result row
        return CallResult(STATUS_ERROR, None, None, Usage(), f"transport_error: {exc}")

    if not isinstance(body, Mapping):
        return CallResult(STATUS_ERROR, None, None, Usage(), f"non_json_response (http {status})")

    usage = _extract_usage(body)

    if status == 200:
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return CallResult(STATUS_ERROR, None, None, usage, "malformed_choices")
        enrichment = parse_enrichment(text)
        if enrichment:
            return CallResult(STATUS_OK, enrichment, text, usage)
        return CallResult(STATUS_ERROR, None, text, usage, "invalid_enrichment")

    if _is_safety_block(status, body):
        return CallResult(STATUS_SAFETY_BLOCKED, None, None, usage, _error_message(body))

    return CallResult(STATUS_ERROR, None, None, usage, f"http_{status}: {_error_message(body)}")


def usage_to_usd(usage: Usage, *, tick_usd: float = DEFAULT_TICK_USD,
                 fallback_usd_per_1m: float = DEFAULT_FALLBACK_USD_PER_1M) -> float:
    """Cost of one call in USD. Prefer authoritative cost_in_usd_ticks; else fall
    back to a pessimistic blended per-token rate so the budget meter never
    under-counts an unknown backend."""
    if usage.has_ticks:
        return usage.cost_in_usd_ticks * tick_usd
    tokens = usage.total_tokens or (usage.prompt_tokens + usage.completion_tokens)
    return tokens * fallback_usd_per_1m / 1_000_000


def judge_row(
    variant: ABCDEVariant,
    chunk: Mapping[str, Any],
    result: CallResult,
) -> dict[str, Any]:
    """Build a JSONL row shaped for enrichment_judge.build_judge_request.

    Includes generation telemetry (status/usage) for cost accounting; the judge
    only reads chunk_id/variant_id/chunk_text/enrichment/model/prompt_hash.
    """
    return {
        "chunk_id": str(chunk["id"]),
        "variant_id": variant.id,
        "chunk_text": chunk.get("content", ""),
        "enrichment": result.enrichment if result.enrichment is not None else {},
        "model": variant.model,
        "prompt_hash": variant.prompt_hash,
        "generation": {
            "status": result.status,
            "error": result.error,
            "usage": {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
                "cost_in_usd_ticks": result.usage.cost_in_usd_ticks,
            },
        },
    }


@dataclass
class RunStats:
    calls: int = 0
    ok: int = 0
    safety_blocked: int = 0
    errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_in_usd_ticks: int = 0
    usd_spent: float = 0.0

    def record(self, result: CallResult, usd: float) -> None:
        self.calls += 1
        if result.status == STATUS_OK:
            self.ok += 1
        elif result.status == STATUS_SAFETY_BLOCKED:
            self.safety_blocked += 1
        else:
            self.errors += 1
        self.prompt_tokens += result.usage.prompt_tokens
        self.completion_tokens += result.usage.completion_tokens
        self.total_tokens += result.usage.total_tokens
        self.cost_in_usd_ticks += result.usage.cost_in_usd_ticks
        self.usd_spent += usd

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "ok": self.ok,
            "safety_blocked": self.safety_blocked,
            "errors": self.errors,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_in_usd_ticks": self.cost_in_usd_ticks,
            "usd_spent": round(self.usd_spent, 6),
            "avg_total_tokens_per_call": (self.total_tokens / self.calls) if self.calls else 0.0,
            "avg_usd_per_call": (self.usd_spent / self.calls) if self.calls else 0.0,
        }


def run_batch(
    chunks: Sequence[Mapping[str, Any]],
    variants: Sequence[ABCDEVariant],
    sanitizer: Any,
    chat_fn: ChatFn,
    *,
    output_path: Optional[str] = None,
    max_usd: Optional[float] = None,
    tick_usd: float = DEFAULT_TICK_USD,
    fallback_usd_per_1m: float = DEFAULT_FALLBACK_USD_PER_1M,
    on_row: Optional[Callable[[dict[str, Any]], None]] = None,
) -> RunStats:
    """Run every (chunk, variant) pair, writing judge-ready rows.

    Hard budget stop: before each call, if the next call could push cumulative
    spend past ``max_usd``, stop cleanly (rows written so far are preserved).
    """
    stats = RunStats()
    sink = None
    if output_path is not None:
        sink = open(output_path, "w", encoding="utf-8")  # noqa: SIM115 — closed in finally
    try:
        for chunk in chunks:
            for variant in variants:
                if max_usd is not None and stats.calls > 0:
                    projected = stats.usd_spent + (stats.usd_spent / stats.calls)
                    if projected > max_usd:
                        return stats
                result = enrich_one(variant, chunk, sanitizer, chat_fn)
                usd = usage_to_usd(result.usage, tick_usd=tick_usd, fallback_usd_per_1m=fallback_usd_per_1m)
                stats.record(result, usd)
                row = judge_row(variant, chunk, result)
                if sink is not None:
                    sink.write(json.dumps(row, sort_keys=True) + "\n")
                    sink.flush()
                if on_row is not None:
                    on_row(row)
                if max_usd is not None and stats.usd_spent >= max_usd:
                    return stats
    finally:
        if sink is not None:
            sink.close()
    return stats


def make_http_chat_fn(*, base_url: str, api_key: str, timeout: int = 60) -> ChatFn:
    """Build a ChatFn backed by a live OpenAI-compatible endpoint.

    No response_format is forced: the variant prompts already instruct "Return
    ONLY the JSON object", and forcing json_object mode tripped xAI's content
    filter on some inputs during smoke testing.
    """
    import requests

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def _chat(model: str, prompt: str, params: Mapping[str, Any]) -> tuple[int, dict]:
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], **params}
        for attempt in range(3):
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            try:
                return resp.status_code, resp.json()
            except ValueError:
                return resp.status_code, {"error": {"message": resp.text[:300]}}
        return resp.status_code, {"error": {"message": "retries_exhausted"}}

    return _chat
