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
import os
import sqlite3
import subprocess
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
    assert {redaction.provider for redaction in result.redactions} >= {"supabase", "google", "github", "openai"}


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


def test_groq_ner_send_scrubs_prompt(monkeypatch):
    import requests

    from brainlayer.pipeline import kg_extraction_groq

    monkeypatch.setenv("GROQ_API_KEY", "test-not-a-key")
    sent = _capture_requests_post(monkeypatch, requests)

    kg_extraction_groq.call_groq_ner(_payload_with_every_token(), max_retries=1)

    assert len(sent) == 1
    _assert_no_token(sent[0], where="Groq NER payload")


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


def test_cloud_backfill_batch_request_line_scrubs_prompt():
    from brainlayer.cloud_backfill import build_batch_request_line

    line = build_batch_request_line("chunk-1", _payload_with_every_token())

    _assert_no_token(json.dumps(line), where="Gemini batch request line")
    assert line["key"] == "chunk-1"


def test_cloud_backfill_submit_scrubs_a_pre_fix_export_before_upload(monkeypatch, tmp_path):
    from brainlayer import cloud_backfill

    export = tmp_path / "batch_000.jsonl"
    raw_line = {
        "key": "chunk-1",
        "request": {"contents": [{"role": "user", "parts": [{"text": _payload_with_every_token()}]}]},
    }
    export.write_text(json.dumps(raw_line) + "\n", encoding="utf-8")

    uploaded: list[str] = []

    class _Files:
        def upload(self, *, file, config=None):
            uploaded.append(Path(file).read_text(encoding="utf-8"))
            return types.SimpleNamespace(name="files/fake")

    class _Batches:
        def create(self, *, model, src, config=None):
            return types.SimpleNamespace(name="batches/fake", state="JOB_STATE_PENDING")

    monkeypatch.setattr(cloud_backfill, "_raise_if_enrich_daily_cap_reached", lambda: None)
    monkeypatch.setattr(
        cloud_backfill, "_get_genai_client", lambda: types.SimpleNamespace(files=_Files(), batches=_Batches())
    )

    assert cloud_backfill.submit_gemini_batch(export, store=None) == "batches/fake"

    assert len(uploaded) == 1
    _assert_no_token(uploaded[0], where="uploaded Gemini batch file")
    assert json.loads(uploaded[0].splitlines()[0])["key"] == "chunk-1"


def test_abcde_http_chat_fn_scrubs_prompt(monkeypatch):
    import requests

    from brainlayer.eval.abcde_enrich_runner import make_http_chat_fn

    sent = _capture_requests_post(monkeypatch, requests)
    chat = make_http_chat_fn(base_url="https://llm.invalid/v1", api_key="test-not-a-key")

    chat("model-x", _payload_with_every_token(), {})

    assert len(sent) == 1
    _assert_no_token(sent[0], where="ABCDE chat payload")


def test_eval_llm_judge_send_scrubs_prompt(monkeypatch, tmp_path):
    from brainlayer import enrichment_controller as controller
    from brainlayer.eval import enrichment_llm_judge as judge

    client = _FakeGeminiClient()
    monkeypatch.setattr(controller, "_get_gemini_client", lambda: client)
    monkeypatch.setattr(judge, "_read_jsonl", lambda path: [])
    monkeypatch.setattr(
        judge,
        "build_pair_requests",
        lambda *args, **kwargs: [{"chunk_id": "chunk-1", "label_to_system": {}}],
    )
    monkeypatch.setattr(judge, "_build_prompt", lambda request: _payload_with_every_token())
    monkeypatch.setattr(judge, "parse_judge_response", lambda text: None)

    try:
        judge.run_judge(
            tmp_path / "s.jsonl",
            tmp_path / "l.jsonl",
            tmp_path / "f.jsonl",
            tmp_path / "out.jsonl",
            tmp_path / "summary.json",
        )
    except Exception:
        pass  # summary shaping of the fake row is not under test; the send is.

    assert len(client.models.sent) == 1
    _assert_no_token(client.models.sent[0], where="eval judge prompt")


@pytest.mark.parametrize(
    "script",
    [
        "enrichment_backfill.py",
        "enrich_recent.py",
        "enrichment_pilot.py",
        "batch_submit_paced.py",
        "cloud_stream.py",
    ],
)
def test_legacy_unsanitized_cloud_scripts_are_gated_off(script, tmp_path):
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "BRAINLAYER_DB": str(tmp_path / "never-opened.db"),
    }
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert proc.returncode != 0
    assert "GATED OFF" in proc.stderr


# ── OUTPUT: every LLM output field is scrubbed before persistence ────────


def _llm_response_echoing_tokens() -> str:
    t = FAKE_TOKENS
    return json.dumps(
        {
            "summary": f"Configured the service with {t['supabase']} and {t['google']}",
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
