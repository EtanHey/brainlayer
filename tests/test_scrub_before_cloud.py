"""Secret scrub is the gate between BrainLayer text and every remote LLM.

Two directions, both fail closed:
- INPUT: every text sent to a cloud model passes ``scrub_secrets`` first; if the
  scrub raises, nothing is sent.
- OUTPUT: every LLM output field is scrubbed before it is persisted, because a
  cloud model will happily copy a token from its prompt into a summary.

All providers are fakes bound through module attributes. No test here opens a
network socket, the canonical DB, or BrainBar's socket.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Synthetic tokens assembled from obviously fake characters. They match the
# provider shapes the scrubber must catch and are not real credentials.
FAKE_TOKENS = {
    "supabase": "sbp_" + "0" * 40,
    "google": "AIza" + "0" * 35,
    "github": "ghp_" + "0" * 36,
    "openai": "sk-" + "0" * 40,
    "google_oauth_access": "ya29." + "0" * 40,
    "google_oauth_refresh": "1//0" + "0" * 40,
    "google_client_secret": "GOCSPX-" + "0" * 28,
}


def _payload_with_every_token() -> str:
    return "deploy notes: " + " | ".join(f"{name} key {token}" for name, token in FAKE_TOKENS.items())


def _assert_no_token(blob: str, *, where: str) -> None:
    leaked = [name for name, token in FAKE_TOKENS.items() if token in blob]
    assert not leaked, f"{where} carried unredacted synthetic token(s): {leaked}"


# ── Scrubber shape coverage ──────────────────────────────────────────────


def test_scrub_secrets_redacts_every_shape_seen_in_enrichment_leak():
    from brainlayer.pipeline.secret_scrub import scrub_secrets

    result = scrub_secrets(_payload_with_every_token())

    _assert_no_token(result.text, where="scrub_secrets")
    assert {redaction.provider for redaction in result.redactions} == set(FAKE_TOKENS)


@pytest.mark.parametrize("provider", ["google_oauth_access", "google_oauth_refresh", "google_client_secret"])
@pytest.mark.parametrize("leading", ["", "abc"])
def test_oauth_tokens_are_scrubbed_at_rest_and_in_nested_llm_output(provider, leading, tmp_path):
    from brainlayer.drain import _apply_store
    from brainlayer.pipeline.cloud_scrub import normalize_json_strings, scrub_llm_output
    from brainlayer.vector_store import VectorStore

    token = leading + FAKE_TOKENS[provider]
    placeholder = f"{leading}[REDACTED:{provider}]"
    store = VectorStore(tmp_path / "oauth.db")
    try:
        stored = _apply_store(
            store.conn,
            {"content": f"note {token}", "tags": [token], "metadata": {"note": token}, "source": "manual"},
        )
        row = store.conn.cursor().execute("SELECT * FROM chunks WHERE id = ?", (stored.chunk_id,)).fetchone()
        assert token not in str(row)
        assert placeholder in str(row)
        for table in ("chunks_fts", "chunks_fts_trigram"):
            assert token not in str(list(store.conn.cursor().execute(f"SELECT * FROM {table}")))
        store.update_enrichment(stored.chunk_id, summary=token, tags=[token], key_facts=[token])
        row = store.conn.cursor().execute("SELECT * FROM chunks WHERE id = ?", (stored.chunk_id,)).fetchone()
        assert token not in str(row)
        assert placeholder in str(row)
    finally:
        store.close()

    encoded = json.dumps({"nested": token}).replace(token[0], f"\\u{ord(token[0]):04x}")
    output = scrub_llm_output(normalize_json_strings({token: [token, {"encoded": encoded}], "count": 3}))
    assert token not in json.dumps(output)
    assert output["count"] == 3
    assert output[placeholder][0] == placeholder
    assert json.loads(output[placeholder][1]["encoded"])["nested"] == placeholder


@pytest.mark.parametrize("position", ["alone", "before_provider", "after_provider"])
def test_quarantine_is_redacted_before_gemini_send(position, monkeypatch):
    from brainlayer import enrichment_controller as controller
    from brainlayer.pipeline.secret_scrub import scrub_secrets

    # Fixed alphabet filler exercises entropy without a real credential.
    token = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    provider = FAKE_TOKENS["google_oauth_access"]
    prompt = {"alone": token, "before_provider": f"{token} {provider}", "after_provider": f"{provider} {token}"}[
        position
    ]
    result = scrub_secrets(prompt)
    assert [item.value for item in result.quarantine] == [token]
    _neutralize_gemini_cost_accounting(monkeypatch, controller)
    client = _FakeGeminiClient()

    controller._generate_content_with_rate_limit(client, "gemini-test", prompt, {}, None)

    # The long provider placeholder itself is quarantined on the second pass.
    expected = prompt.replace(provider, "[[REDACTED:quarantine]]").replace(token, "[REDACTED:quarantine]")
    assert client.models.sent == [expected]


@pytest.mark.parametrize("sender", ["call_groq", "call_glm", "call_mlx"])
def test_quarantine_is_redacted_in_http_transport(sender, monkeypatch):
    from brainlayer.pipeline import enrichment

    monkeypatch.setattr(enrichment, "GROQ_API_KEY", "test-not-a-key")
    sent = _capture_requests_post(monkeypatch, enrichment.requests)
    token = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    getattr(enrichment, sender)(token)

    assert len(sent) == 1
    assert token not in sent[0]
    assert "[REDACTED:quarantine]" in sent[0]


def test_second_cloud_scrub_pass_failure_prevents_send(monkeypatch):
    from brainlayer import enrichment_controller as controller
    from brainlayer.pipeline import cloud_scrub

    real_scrub = cloud_scrub.scrub_secrets
    calls = []

    def scrub(text):
        calls.append(text)
        if len(calls) == 2:
            raise RuntimeError("synthetic failure")
        return real_scrub(text)

    monkeypatch.setattr(cloud_scrub, "scrub_secrets", scrub)
    _neutralize_gemini_cost_accounting(monkeypatch, controller)
    client = _FakeGeminiClient()
    with pytest.raises(cloud_scrub.CloudScrubError):
        controller._generate_content_with_rate_limit(client, "gemini-test", "ordinary prose", {}, None)
    assert client.models.sent == []


def test_cloud_quarantine_offsets_follow_provider_redaction():
    from brainlayer.pipeline.cloud_scrub import scrub_for_cloud

    token = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    prompt = f"😀 {FAKE_TOKENS['google_oauth_access']} {token}; {token}. done"
    assert scrub_for_cloud(prompt) == "😀 [[REDACTED:quarantine]] [REDACTED:quarantine]; [REDACTED:quarantine]. done"


@pytest.mark.parametrize("dots", [1, 8, 62, 70])
def test_cloud_redacts_quarantine_after_leading_punctuation(dots, monkeypatch):
    from brainlayer import enrichment_controller as controller
    from brainlayer.pipeline.cloud_scrub import scrub_for_cloud

    token = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    prompt = "é" + "." * dots + token
    client = _FakeGeminiClient()
    _neutralize_gemini_cost_accounting(monkeypatch, controller)
    controller._generate_content_with_rate_limit(client, "gemini-test", prompt, {}, None)
    assert client.models.sent == ["é" + "." * dots + "[REDACTED:quarantine]"]
    assert token not in client.models.sent[0]
    assert scrub_for_cloud(client.models.sent[0]) == client.models.sent[0]


@pytest.mark.parametrize("case", ["repeated", "overlapping", "negative", "missing", "empty"])
def test_cloud_quarantine_mismatch_redacts_every_value_or_blocks(case, monkeypatch):
    from brainlayer import enrichment_controller as controller
    from brainlayer.pipeline import cloud_scrub
    from brainlayer.pipeline.secret_scrub import QuarantinedToken, SecretScrubResult

    token = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    missing = case in {"missing", "empty"}
    token = token * 2 if case == "overlapping" else token
    prompt = "ordinary prose" if missing else f"é . {token} and {token}"
    if case == "overlapping":
        prompt = "é . " + token + token[:62]
    value = "" if case == "empty" else token
    real_scrub = cloud_scrub.scrub_secrets
    calls = []

    def scrub(text):
        calls.append(text)
        if len(calls) == 1:
            return real_scrub(text)
        return SecretScrubResult(
            text=text,
            quarantine=[
                QuarantinedToken(value=value, start=4, end=4 + len(value)),
                QuarantinedToken(
                    value=value,
                    start=-len(value) if case == "negative" else 0,
                    end=len(text) if case == "negative" else len(value),
                ),
            ],
        )

    monkeypatch.setattr(cloud_scrub, "scrub_secrets", scrub)
    _neutralize_gemini_cost_accounting(monkeypatch, controller)
    client = _FakeGeminiClient()
    if missing:
        with pytest.raises(cloud_scrub.CloudScrubError) as error:
            controller._generate_content_with_rate_limit(client, "gemini-test", prompt, {}, None)
        assert token not in str(error.value)
        assert client.models.sent == []
    else:
        controller._generate_content_with_rate_limit(client, "gemini-test", prompt, {}, None)
        expected = (
            "é . [REDACTED:quarantine]" if case == "overlapping" else prompt.replace(token, "[REDACTED:quarantine]")
        )
        assert client.models.sent == [expected]
        monkeypatch.setattr(cloud_scrub, "scrub_secrets", real_scrub)
        assert cloud_scrub.scrub_for_cloud(client.models.sent[0]) == client.models.sent[0]


def test_realtime_loop_marks_quarantined_chunk_and_advances(monkeypatch, tmp_path):
    from brainlayer import enrichment_controller as controller
    from brainlayer.vector_store import VectorStore

    token = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    client = _FakeGeminiClient()
    monkeypatch.setenv("BRAINLAYER_ENRICHMENT_QUEUE_WRITES", "0")
    monkeypatch.setenv("BRAINLAYER_ENRICH_COST_DIR", str(tmp_path / "cost"))
    monkeypatch.setattr(controller, "_get_gemini_client", lambda: client)
    monkeypatch.setattr(controller, "Sanitizer", types.SimpleNamespace(from_env=lambda: object()))
    monkeypatch.setattr(controller, "build_external_prompt", lambda chunk, sanitizer: (chunk["content"], None))
    monkeypatch.setattr(controller, "_is_duplicate_content", lambda *args: False)
    monkeypatch.setattr(controller, "_emit_enrichment_start", lambda *args, **kwargs: False)
    monkeypatch.setattr(controller, "_emit_enrichment_complete", lambda *args, **kwargs: False)
    _neutralize_gemini_cost_accounting(monkeypatch, controller)
    store = VectorStore(tmp_path / "loop.db")
    try:
        for chunk_id, content in [
            ("next", "The next ordinary chunk must also be enriched after the first chunk completes."),
            ("quarantined", f"Implement {token} in the deployment helper."),
        ]:
            store.conn.cursor().execute(
                "INSERT INTO chunks (id, content, metadata, source_file, project, content_type, char_count, source) VALUES (?, ?, '{}', 'test.jsonl', 'brainlayer', 'assistant_text', ?, 'claude_code')",
                (chunk_id, content, len(content)),
            )
        assert controller.enrich_realtime(store, limit=1, since_hours=None).enriched == 1
        row = (
            store.conn.cursor()
            .execute("SELECT enriched_at, enrich_status FROM chunks WHERE id = 'quarantined'")
            .fetchone()
        )
        assert row[0] is not None and row[1] == "success"
        assert [chunk["id"] for chunk in store.get_enrichment_candidates()] == ["next"]
        assert controller.enrich_realtime(store, limit=1, since_hours=None).enriched == 1
        assert store.get_enrichment_candidates() == []
        assert len(client.models.sent) == 2
        assert token not in client.models.sent[0]
        assert "[REDACTED:quarantine]" in client.models.sent[0]
    finally:
        store.close()


@pytest.mark.parametrize(
    "text",
    ["ordinary prose about ya29", "https://example.invalid/1//notes", "0" * 64, "12345678-1234-1234-1234-123456789abc"],
)
def test_cloud_scrub_preserves_prose_paths_and_join_keys(text):
    from brainlayer.pipeline.cloud_scrub import scrub_for_cloud

    assert scrub_for_cloud(text) == text


# ── INPUT: every remote send is scrubbed ─────────────────────────────────


class _FakeGeminiModels:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def generate_content(self, *, model, contents, config=None):
        self.sent.append(contents if isinstance(contents, str) else json.dumps(contents))
        return types.SimpleNamespace(text='{"summary": "fine summary", "tags": ["x"]}', usage_metadata=None)


class _FakeGeminiClient:
    def __init__(self) -> None:
        self.models = _FakeGeminiModels()


def _neutralize_gemini_cost_accounting(monkeypatch, controller) -> None:
    monkeypatch.setattr(controller, "_raise_if_enrich_daily_cap_reached", lambda: None)
    monkeypatch.setattr(controller, "_record_enrich_response_usage", lambda response: 0.0)


def test_gemini_realtime_send_scrubs_prompt(monkeypatch):
    from brainlayer import enrichment_controller as controller

    _neutralize_gemini_cost_accounting(monkeypatch, controller)
    client = _FakeGeminiClient()

    controller._generate_content_with_rate_limit(client, "gemini-test", _payload_with_every_token(), {}, None)

    assert len(client.models.sent) == 1
    _assert_no_token(client.models.sent[0], where="Gemini realtime prompt")
    assert "[REDACTED:" in client.models.sent[0]


def test_gemini_send_fails_closed_when_scrub_raises(monkeypatch):
    from brainlayer import enrichment_controller as controller
    from brainlayer.pipeline import cloud_scrub

    _neutralize_gemini_cost_accounting(monkeypatch, controller)

    def _boom(text):
        raise RuntimeError("scrubber exploded")

    monkeypatch.setattr(cloud_scrub, "scrub_secrets", _boom)
    client = _FakeGeminiClient()

    with pytest.raises(cloud_scrub.CloudScrubError):
        controller._generate_content_with_rate_limit(client, "gemini-test", _payload_with_every_token(), {}, None)

    assert client.models.sent == []


def test_retry_wrapper_does_not_retry_a_scrub_failure(monkeypatch):
    from brainlayer import enrichment_controller as controller
    from brainlayer.pipeline.cloud_scrub import CloudScrubError

    calls = []

    def _fn():
        calls.append(1)
        raise CloudScrubError("scrub failed")

    monkeypatch.setattr(controller, "_sleep", lambda seconds: None)
    with pytest.raises(CloudScrubError):
        controller._retry_with_backoff(_fn, max_retries=5)

    assert len(calls) == 1


class _FakeHttpResponse:
    status_code = 200
    headers: dict = {}
    text = ""

    def json(self):
        return {"choices": [{"message": {"content": '{"summary": "fine summary"}'}}], "usage": {}}

    def raise_for_status(self):
        return None


def _capture_requests_post(monkeypatch, target):
    sent: list[str] = []

    def _post(url, *, headers=None, json=None, timeout=None, **kwargs):
        sent.append(__import__("json").dumps(json))
        return _FakeHttpResponse()

    monkeypatch.setattr(target, "post", _post)
    return sent


def test_groq_enrichment_send_scrubs_prompt(monkeypatch):
    from brainlayer.pipeline import enrichment

    monkeypatch.setattr(enrichment, "GROQ_API_KEY", "test-not-a-key")
    monkeypatch.setattr(enrichment, "_groq_last_call", 0.0)
    monkeypatch.setattr(enrichment, "_sleep", lambda seconds: None)
    monkeypatch.setattr(enrichment, "_log_glm_usage", lambda *args, **kwargs: None)
    sent = _capture_requests_post(monkeypatch, enrichment.requests)

    enrichment.call_groq(_payload_with_every_token())

    assert len(sent) == 1
    _assert_no_token(sent[0], where="Groq enrichment payload")


def test_groq_enrichment_send_fails_closed_when_scrub_raises(monkeypatch):
    from brainlayer.pipeline import cloud_scrub, enrichment

    monkeypatch.setattr(enrichment, "GROQ_API_KEY", "test-not-a-key")
    monkeypatch.setattr(enrichment, "_groq_last_call", 0.0)
    monkeypatch.setattr(enrichment, "_sleep", lambda seconds: None)
    sent = _capture_requests_post(monkeypatch, enrichment.requests)
    monkeypatch.setattr(cloud_scrub, "scrub_secrets", lambda text: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(cloud_scrub.CloudScrubError):
        enrichment.call_groq(_payload_with_every_token())

    assert sent == []


def test_groq_ner_sender_is_removed():
    from brainlayer.pipeline import kg_extraction_groq

    assert not hasattr(kg_extraction_groq, "call_groq_ner")


def test_digest_faceted_gemini_send_scrubs_prompt(monkeypatch):
    from brainlayer.pipeline import digest

    client = _FakeGeminiClient()
    fake_genai = types.SimpleNamespace(Client=lambda api_key=None, **kwargs: client)
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-not-a-key")
    monkeypatch.setattr(digest.Sanitizer, "from_env", classmethod(lambda cls: object()))
    monkeypatch.setattr(
        digest,
        "build_external_prompt",
        lambda chunk, sanitizer, prompt_template=None: (
            _payload_with_every_token(),
            types.SimpleNamespace(pii_detected=False),
        ),
    )

    digest._default_faceted_enrich(content="irrelevant", project="p", title=None, participants=None)

    assert client.models.sent, "digest faceted enrichment never reached the fake client"
    for prompt in client.models.sent:
        _assert_no_token(prompt, where="digest faceted Gemini prompt")


# ── OUTPUT: every LLM output field is scrubbed before persistence ────────


def _llm_response_echoing_tokens() -> str:
    t = FAKE_TOKENS
    return json.dumps(
        {
            "summary": _payload_with_every_token(),
            "tags": ["deploy", t["github"]],
            "importance": 6,
            "intent": "implementing",
            "primary_symbols": [t["openai"]],
            "resolved_query": f"which key is {t['github']} used for",
            "key_facts": [f"supabase token is {t['supabase']}", f"openai key {t['openai']}"],
            "resolved_queries": [f"where is {t['google']} configured", "how was the deploy wired"],
            "version_scope": f"v1 with {t['openai']}",
            "external_deps": [t["google"]],
            "sentiment_signals": [f"relieved about {t['github']}"],
            "entities": [{"name": t["supabase"], "type": "tool", "relation": f"uses {t['openai']}"}],
        }
    )


def test_parse_enrichment_scrubs_every_output_field():
    from brainlayer.pipeline.enrichment import parse_enrichment

    enrichment = parse_enrichment(_llm_response_echoing_tokens())

    assert enrichment is not None
    assert enrichment.get("summary")
    _assert_no_token(json.dumps(enrichment), where="parsed enrichment")


def test_parse_enrichment_fails_closed_when_output_scrub_raises(monkeypatch):
    from brainlayer.pipeline import cloud_scrub
    from brainlayer.pipeline.enrichment import parse_enrichment

    monkeypatch.setattr(cloud_scrub, "scrub_secrets", lambda text: (_ for _ in ()).throw(RuntimeError("boom")))

    assert parse_enrichment(_llm_response_echoing_tokens()) is None


def test_update_enrichment_persists_redacted_fields(tmp_path):
    from brainlayer.vector_store import VectorStore

    t = FAKE_TOKENS
    store = VectorStore(tmp_path / "update.db")
    try:
        store.conn.cursor().execute(
            """
            INSERT INTO chunks (id, content, metadata, source_file, project, content_type, char_count, source)
            VALUES ('chunk-1', 'content', '{}', 'test.jsonl', 'brainlayer', 'assistant_text', 7, 'claude_code')
            """
        )
        store.update_enrichment(
            "chunk-1",
            summary=f"summary with {t['supabase']}",
            tags=["deploy", t["github"]],
            resolved_query=f"query {t['google']}",
            key_facts=[f"fact {t['openai']}"],
            resolved_queries=[f"rq {t['google']}"],
            version_scope=f"scope {t['github']}",
            external_deps=[t["openai"]],
            primary_symbols=[t["supabase"]],
            sentiment_signals=[f"signal {t['github']}"],
        )
        store.update_reenrichment_preview("chunk-1", summary_v2=f"preview {t['supabase']}")
        row = store.conn.cursor().execute("SELECT * FROM chunks WHERE id = 'chunk-1'").fetchone()
    finally:
        store.close()

    _assert_no_token(json.dumps([str(value) for value in row]), where="persisted chunk row")


def test_drain_apply_enrichment_scrubs_queued_output():
    """A queue file written before this fix still holds raw LLM output; the drain must scrub it."""
    from brainlayer.drain import _apply_enrichment

    t = FAKE_TOKENS
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY, content TEXT NOT NULL, metadata TEXT NOT NULL, source_file TEXT NOT NULL,
            summary TEXT, tags TEXT, key_facts TEXT, resolved_queries TEXT, resolved_query TEXT,
            version_scope TEXT, primary_symbols TEXT, external_deps TEXT, sentiment_signals TEXT,
            raw_entities_json TEXT, enriched_at TEXT, enrich_status TEXT
        )
        """
    )
    conn.execute("INSERT INTO chunks (id, content, metadata, source_file) VALUES ('chunk-1', 'c', '{}', 't.jsonl')")

    _apply_enrichment(
        conn,
        {
            "chunk_id": "chunk-1",
            "enrichment": {
                "summary": f"summary {t['supabase']}",
                "tags": [t["github"]],
                "key_facts": [f"fact {t['openai']}"],
                "resolved_queries": [f"rq {t['google']}"],
                "version_scope": f"scope {t['github']}",
                "primary_symbols": [t["supabase"]],
                "external_deps": [t["openai"]],
                "sentiment_signals": [f"signal {t['google']}"],
            },
            "entities": [{"name": t["supabase"], "type": "tool"}],
        },
    )

    row = conn.execute("SELECT * FROM chunks WHERE id = 'chunk-1'").fetchone()
    _assert_no_token(json.dumps([str(value) for value in row]), where="drained chunk row")
    assert row[4], "summary was dropped instead of redacted"


def test_session_enrichment_persists_redacted_fields(tmp_path):
    from brainlayer.vector_store import VectorStore

    t = FAKE_TOKENS
    store = VectorStore(tmp_path / "session.db")
    try:
        store.upsert_session_enrichment(
            {
                "session_id": "session-1",
                "session_summary": f"summary {t['supabase']}",
                "primary_intent": f"intent {t['github']}",
                "outcome": "success",
                "decisions_made": [f"decided {t['google']}"],
                "corrections": [{"text": f"corrected {t['openai']}"}],
                "learnings": [f"learned {t['github']}"],
                "mistakes": [f"mistake {t['supabase']}"],
                "patterns": [f"pattern {t['google']}"],
                "topic_tags": [t["github"]],
                "what_worked": f"worked {t['google']}",
                "what_failed": f"failed {t['openai']}",
            }
        )
        record = store.get_session_enrichment("session-1")
        fts_rows = list(store.conn.cursor().execute("SELECT * FROM session_enrichments_fts"))
    finally:
        store.close()

    _assert_no_token(json.dumps(record, default=str), where="session enrichment record")
    _assert_no_token(json.dumps([list(map(str, r)) for r in fts_rows]), where="session enrichment FTS")


def test_digest_faceted_output_is_scrubbed():
    from brainlayer.pipeline.digest import _parse_faceted_enrichment

    t = FAKE_TOKENS
    parsed = _parse_faceted_enrichment(
        json.dumps({"topics": [t["github"]], "activity": "act:build", "domains": [f"dom:{t['google']}"]})
    )

    assert parsed is not None
    _assert_no_token(json.dumps(parsed), where="digest faceted output")


def test_groq_ner_output_is_scrubbed():
    from brainlayer.pipeline.kg_extraction_groq import parse_multi_chunk_response

    t = FAKE_TOKENS
    parsed = parse_multi_chunk_response(
        json.dumps(
            {
                "chunks": [
                    {
                        "chunk_id": "chunk-1",
                        "entities": [{"text": t["supabase"], "type": "tool"}],
                        "relations": [{"source": t["openai"], "target": "x", "type": "uses"}],
                    }
                ]
            }
        )
    )

    assert parsed and parsed[0]["chunk_id"] == "chunk-1"
    _assert_no_token(json.dumps(parsed), where="Groq NER output")
