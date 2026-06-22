import json
import logging
import time

import pytest

from brainlayer import search_profile
from brainlayer.mcp.search_handler import _brain_search, _search


class FakeEmbeddingModel:
    def embed_query(self, _query):
        return [0.1, 0.2, 0.3]


class FakeSearchStore:
    def count(self):
        return 1

    def hybrid_search(self, **_kwargs):
        return {
            "ids": [["chunk-profile-1"]],
            "documents": [["auth refactor profile result"]],
            "metadatas": [[{"source_file": "test.md", "project": "brainlayer"}]],
            "distances": [[0.25]],
        }

    def enrich_results_with_session_context(self, results):
        return results


class FakeFailingSearchStore(FakeSearchStore):
    def hybrid_search(self, **_kwargs):
        raise RuntimeError("profile failure")


class SlowEmbeddingModel:
    def embed_query(self, _query):
        time.sleep(0.05)
        return [0.1, 0.2, 0.3]


def _profile_events(caplog):
    events = []
    for record in caplog.records:
        try:
            event = json.loads(record.getMessage())
        except json.JSONDecodeError:
            continue
        if str(event.get("scope", "")).startswith("search."):
            events.append(event)
    return events


@pytest.mark.asyncio
async def test_brain_search_profile_flag_emits_timing_json(monkeypatch, caplog):
    monkeypatch.setenv("BRAINLAYER_SEARCH_PROFILE", "1")
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: FakeSearchStore())
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr("brainlayer.mcp.search_handler._expanded_fts_query", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._detect_entities", lambda *_args, **_kwargs: [])

    caplog.set_level(logging.INFO)

    await _brain_search(query="auth refactor", project="brainlayer", source="all", detail="compact")

    profile_events = _profile_events(caplog)

    assert len(profile_events) >= 3
    assert {event["step"] for event in profile_events} >= {"brain_search", "embed", "hybrid_search"}
    assert all("query_id" in event for event in profile_events)
    assert all("ts" in event for event in profile_events)


@pytest.mark.asyncio
async def test_brain_search_profile_flag_emits_failed_hybrid_timing(monkeypatch, caplog):
    monkeypatch.setenv("BRAINLAYER_SEARCH_PROFILE", "1")
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: FakeFailingSearchStore())
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr("brainlayer.mcp.search_handler._expanded_fts_query", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._detect_entities", lambda *_args, **_kwargs: [])

    caplog.set_level(logging.INFO)

    await _brain_search(query="auth refactor", project="brainlayer", source="all", detail="compact")

    hybrid_events = [event for event in _profile_events(caplog) if event.get("step") == "hybrid_search"]
    assert len(hybrid_events) == 1
    assert hybrid_events[0]["error"] == "RuntimeError"


@pytest.mark.asyncio
async def test_brain_search_embed_timeout_returns_fts_fallback(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_EMBED_TIMEOUT_MS", "1")
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: FakeSearchStore())
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: SlowEmbeddingModel())
    monkeypatch.setattr("brainlayer.mcp.search_handler._expanded_fts_query", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._detect_entities", lambda *_args, **_kwargs: [])

    texts, structured = await _search(query="auth refactor", project="brainlayer", source="all", detail="compact")

    assert structured["total"] == 1
    assert structured["search_mode"] == "fts_fallback"
    assert structured["fallback_reason"] == "embed_timeout"
    assert "auth refactor profile result" in texts[0].text


@pytest.mark.asyncio
async def test_brain_search_profile_flag_emits_for_file_path_return(monkeypatch, caplog):
    async def fake_file_timeline(**_kwargs):
        return []

    async def fake_recall(**_kwargs):
        return ([], {})

    monkeypatch.setenv("BRAINLAYER_SEARCH_PROFILE", "1")
    monkeypatch.setattr("brainlayer.mcp.search_handler._file_timeline", fake_file_timeline)
    monkeypatch.setattr("brainlayer.mcp.search_handler._recall", fake_recall)

    caplog.set_level(logging.INFO)

    await _brain_search(query="notes for auth refactor", file_path="auth.md", project="brainlayer")

    brain_search_events = [event for event in _profile_events(caplog) if event.get("step") == "brain_search"]
    assert len(brain_search_events) == 1
    assert brain_search_events[0]["scope"] == "search.mcp"


@pytest.mark.asyncio
async def test_brain_search_profile_keeps_query_id_across_file_path_recursion(monkeypatch, caplog):
    async def fake_file_timeline(**_kwargs):
        return []

    async def fake_recall(**_kwargs):
        return ([], {})

    monkeypatch.setenv("BRAINLAYER_SEARCH_PROFILE", "1")
    monkeypatch.setattr("brainlayer.mcp.search_handler._extract_file_path", lambda _query: "auth.md")
    monkeypatch.setattr("brainlayer.mcp.search_handler._file_timeline", fake_file_timeline)
    monkeypatch.setattr("brainlayer.mcp.search_handler._recall", fake_recall)

    caplog.set_level(logging.INFO)

    await _brain_search(query="show auth.md", project="brainlayer")

    brain_search_events = [event for event in _profile_events(caplog) if event.get("step") == "brain_search"]
    assert len(brain_search_events) == 2
    assert len({event["query_id"] for event in brain_search_events}) == 1


def test_search_profile_emit_preserves_reserved_keys(monkeypatch, caplog):
    monkeypatch.setenv("BRAINLAYER_SEARCH_PROFILE", "1")
    caplog.set_level(logging.INFO, logger="brainlayer.search_profile")

    search_profile.emit(
        "search.mcp",
        "brain_search",
        "q-good",
        12.3,
        ts="bad",
        rows=1,
    )

    events = _profile_events(caplog)
    assert len(events) == 1
    assert events[0]["scope"] == "search.mcp"
    assert events[0]["step"] == "brain_search"
    assert events[0]["query_id"] == "q-good"
    assert events[0]["dur_ms"] == 12.3
    assert events[0]["rows"] == 1


def test_search_profile_emit_stringifies_non_json_fields(monkeypatch, caplog):
    monkeypatch.setenv("BRAINLAYER_SEARCH_PROFILE", "1")
    caplog.set_level(logging.INFO, logger="brainlayer.search_profile")

    search_profile.emit("search.mcp", "brain_search", details={"values": {1, 2}})

    events = _profile_events(caplog)
    assert len(events) == 1
    assert events[0]["details"] == "{'values': {1, 2}}"
