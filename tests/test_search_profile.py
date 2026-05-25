import json
import logging

import pytest

from brainlayer.mcp.search_handler import _brain_search


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


@pytest.mark.asyncio
async def test_brain_search_profile_flag_emits_timing_json(monkeypatch, caplog):
    monkeypatch.setenv("BRAINLAYER_SEARCH_PROFILE", "1")
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: FakeSearchStore())
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr("brainlayer.mcp.search_handler._expanded_fts_query", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._detect_entities", lambda *_args, **_kwargs: [])

    caplog.set_level(logging.INFO)

    await _brain_search(query="auth refactor", project="brainlayer", source="all", detail="compact")

    profile_events = []
    for record in caplog.records:
        try:
            event = json.loads(record.getMessage())
        except json.JSONDecodeError:
            continue
        if str(event.get("scope", "")).startswith("search."):
            profile_events.append(event)

    assert len(profile_events) >= 3
    assert {event["step"] for event in profile_events} >= {"brain_search", "embed", "hybrid_search"}
    assert all("query_id" in event for event in profile_events)
    assert all("ts" in event for event in profile_events)
