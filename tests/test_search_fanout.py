from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import CallToolResult, TextContent

from brainlayer.mcp import call_tool, list_tools
from brainlayer.mcp._format import format_search_results
from brainlayer.mcp.search_handler import _brain_search
from brainlayer.scoping import ConsumerScope
from brainlayer.search_fanout import plan_fan_out_scopes, run_fan_out
from brainlayer.search_repo import _effective_project_filters, _metadata_matches_project_scope

NOW = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)
KNOWN_TAGS = {"pm/decision", "tech/architecture", "tech/testing"}


def _structured(*chunk_ids: str, search_mode: str = "hybrid", **extra):
    return (
        [TextContent(type="text", text="fake search")],
        {
            "query": "architecture decision",
            "total": len(chunk_ids),
            "results": [
                {
                    "chunk_id": chunk_id,
                    "score": 0.9 - (index / 10),
                    "content": f"content for {chunk_id}",
                }
                for index, chunk_id in enumerate(chunk_ids)
            ],
            "search_mode": search_mode,
            **extra,
        },
    )


def test_plan_fan_out_scopes_is_bounded_and_deterministic():
    scopes = plan_fan_out_scopes(
        query="architecture decision",
        project="brainlayer",
        date_from=None,
        tag=None,
        now=NOW,
        known_tags=KNOWN_TAGS,
    )

    assert [scope.name for scope in scopes] == [
        "raw",
        "project",
        "recent",
        "tag:tech/architecture",
    ]
    assert len(scopes) == 4
    assert scopes[0].overrides == {"project": None}
    assert scopes[1].overrides == {"project": "brainlayer"}
    assert scopes[2].overrides == {"project": "brainlayer", "date_from": "2026-06-10"}
    assert scopes[3].overrides == {"project": "brainlayer", "tag": "tech/architecture"}


def test_plan_fan_out_scopes_respects_stricter_date_and_explicit_tag():
    scopes = plan_fan_out_scopes(
        query="architecture decision",
        project=None,
        date_from="2026-07-01",
        tag="pm/decision",
        now=NOW,
        known_tags=KNOWN_TAGS,
    )

    assert [scope.name for scope in scopes] == ["raw", "recent"]
    assert scopes[1].overrides == {"project": None, "date_from": "2026-07-01"}


def test_full_detail_fan_out_format_preserves_complete_content():
    content = "x" * 220 + "END-OF-FULL-CONTENT"

    output = format_search_results(
        "full fan-out",
        [{"chunk_id": "full-1", "content": content, "fan_out_provenance": ["raw"]}],
        1,
        detail="full",
    )

    assert "END-OF-FULL-CONTENT" in output


@pytest.mark.asyncio
async def test_run_fan_out_merges_dedupes_and_records_scope_provenance():
    responses = {
        None: _structured("a", "b"),
        "brainlayer": _structured("b", "c"),
        "2026-06-10": _structured("c", "d"),
        "tech/architecture": _structured("d", "a"),
    }

    async def fake_search(**kwargs):
        key = kwargs.get("tag") or kwargs.get("date_from") or kwargs.get("project")
        return responses[key]

    result = await run_fan_out(
        search=fake_search,
        query="architecture decision",
        num_results=3,
        base_kwargs={},
        project="brainlayer",
        date_from=None,
        tag=None,
        now=NOW,
        known_tags=KNOWN_TAGS,
    )

    assert [item["chunk_id"] for item in result["results"]] == ["a", "b", "c"]
    assert result["results"][0]["fan_out_provenance"] == ["raw", "tag:tech/architecture"]
    assert result["results"][1]["fan_out_provenance"] == ["raw", "project"]
    assert result["candidate_count"] == 8
    assert result["total"] == 3
    assert result["ranking"] == "rrf(k=60), tie-break=first-seen,chunk_id"
    assert result["degraded"] is False


@pytest.mark.asyncio
async def test_run_fan_out_does_not_double_count_duplicate_ids_within_one_scope():
    async def fake_search(**kwargs):
        return _structured("duplicate", "duplicate") if not kwargs.get("date_from") else _structured()

    result = await run_fan_out(
        search=fake_search,
        query="unmatched surface",
        num_results=2,
        base_kwargs={"tag": "already-scoped"},
        project=None,
        date_from=None,
        tag="already-scoped",
        now=NOW,
        known_tags=KNOWN_TAGS,
    )

    assert [item["chunk_id"] for item in result["results"]] == ["duplicate"]
    assert result["results"][0]["fan_out_provenance"] == ["raw"]
    assert result["results"][0]["fan_out_score"] == round(1 / 61, 8)


@pytest.mark.asyncio
async def test_run_fan_out_project_legs_narrow_orchestrator_consumer_scope():
    effective_projects = {}

    async def fake_search(**kwargs):
        scope_key = kwargs.get("tag") or kwargs.get("date_from") or kwargs.get("project") or "raw"
        effective_projects[scope_key] = _effective_project_filters(kwargs.get("project"), kwargs["consumer_scope"])
        return _structured()

    await run_fan_out(
        search=fake_search,
        query="architecture decision",
        num_results=3,
        base_kwargs={"consumer_scope": ConsumerScope.for_orchestrator()},
        project="golems",
        date_from=None,
        tag=None,
        now=NOW,
        known_tags=KNOWN_TAGS,
    )

    assert effective_projects == {
        "raw": (),
        "golems": ("golems",),
        "2026-06-10": ("golems",),
        "tech/architecture": ("golems",),
    }


@pytest.mark.asyncio
async def test_run_fan_out_project_provenance_excludes_cross_project_candidates():
    candidates = [
        {"chunk_id": "golems-hit", "project": "golems", "content": "golems"},
        {"chunk_id": "brainlayer-hit", "project": "brainlayer", "content": "brainlayer"},
    ]

    async def fake_search(**kwargs):
        visible = [
            item
            for item in candidates
            if _metadata_matches_project_scope(item, kwargs.get("project"), kwargs["consumer_scope"])
        ]
        return ([TextContent(type="text", text="fake")], {"results": visible, "search_mode": "hybrid"})

    result = await run_fan_out(
        search=fake_search,
        query="architecture decision",
        num_results=2,
        base_kwargs={"consumer_scope": ConsumerScope.for_orchestrator()},
        project="golems",
        date_from=None,
        tag=None,
        now=NOW,
        known_tags=KNOWN_TAGS,
    )

    by_id = {item["chunk_id"]: item for item in result["results"]}
    assert by_id["golems-hit"]["fan_out_provenance"] == [
        "raw",
        "project",
        "recent",
        "tag:tech/architecture",
    ]
    assert by_id["brainlayer-hit"]["fan_out_provenance"] == ["raw"]


@pytest.mark.asyncio
async def test_run_fan_out_enforces_four_search_and_forty_candidate_bounds():
    calls = []

    async def fake_search(**kwargs):
        calls.append(kwargs)
        scope_key = kwargs.get("tag") or kwargs.get("date_from") or kwargs.get("project") or "raw"
        return _structured(*(f"{scope_key}-{index}" for index in range(20)))

    result = await run_fan_out(
        search=fake_search,
        query="architecture decision",
        num_results=100,
        base_kwargs={},
        project="brainlayer",
        date_from=None,
        tag=None,
        now=NOW,
        known_tags=KNOWN_TAGS,
    )

    assert len(calls) == 4
    assert all(call["num_results"] == 10 for call in calls)
    assert result["candidate_count"] == 40
    assert len(result["results"]) == 40
    assert result["candidate_limit"] == 40


@pytest.mark.asyncio
async def test_run_fan_out_recent_scope_ranks_newest_candidate_first():
    async def fake_search(**kwargs):
        if kwargs.get("date_from"):
            content, structured = _structured("older", "newer")
            structured["results"][0]["date"] = "2026-06-15"
            structured["results"][1]["date"] = "2026-07-09"
            return content, structured
        return _structured()

    result = await run_fan_out(
        search=fake_search,
        query="unmatched surface",
        num_results=2,
        base_kwargs={"tag": "already-scoped"},
        project=None,
        date_from=None,
        tag="already-scoped",
        now=NOW,
        known_tags=KNOWN_TAGS,
    )

    assert [item["chunk_id"] for item in result["results"]] == ["newer", "older"]


@pytest.mark.asyncio
async def test_run_fan_out_propagates_degraded_scopes_without_dropping_good_results():
    async def fake_search(**kwargs):
        if kwargs.get("project") == "brainlayer" and not kwargs.get("date_from") and not kwargs.get("tag"):
            return _structured("project-hit", search_mode="fts_fallback", fallback_reason="embed_busy")
        if kwargs.get("date_from"):
            return CallToolResult(
                content=[TextContent(type="text", text="Search error: database busy")],
                isError=True,
            )
        return _structured("good-hit")

    result = await run_fan_out(
        search=fake_search,
        query="architecture decision",
        num_results=5,
        base_kwargs={},
        project="brainlayer",
        date_from=None,
        tag=None,
        now=NOW,
        known_tags=KNOWN_TAGS,
    )

    assert result["degraded"] is True
    assert result["degraded_scopes"] == [
        {"scope": "project", "reason": "embed_busy"},
        {"scope": "recent", "reason": "Search error: database busy"},
    ]
    assert {item["chunk_id"] for item in result["results"]} == {"good-hit", "project-hit"}


@pytest.mark.asyncio
async def test_brain_search_schema_exposes_opt_in_fan_out():
    search_tool = next(tool for tool in await list_tools() if tool.name == "brain_search")

    fan_out = search_tool.inputSchema["properties"]["fan_out"]
    assert fan_out == {
        "type": "boolean",
        "default": False,
        "description": "Run bounded deterministic scoped fan-out (max 4 searches / 40 candidates). Generic relevance queries only; incompatible with order='origin', file_path, chunk_id, and entity_id.",
    }


@pytest.mark.asyncio
async def test_call_tool_forwards_fan_out_to_recall():
    with patch(
        "brainlayer.mcp._brain_recall", new_callable=AsyncMock, return_value=CallToolResult(content=[])
    ) as recall:
        await call_tool("brain_search", {"query": "architecture decision", "fan_out": True})

    assert recall.await_args.kwargs["fan_out"] is True


@pytest.mark.asyncio
async def test_brain_search_fan_out_bypasses_helper_and_uses_generic_fan_out(monkeypatch):
    sentinel = ([TextContent(type="text", text="fan-out")], {"fan_out": True})

    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_route_enabled", lambda: True)
    monkeypatch.setattr(
        "brainlayer.mcp.search_handler._warm_helper_socket_candidates",
        lambda: pytest.fail("fan-out must not use the helper socket"),
    )
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: object())
    monkeypatch.setattr("brainlayer.mcp.search_handler._exact_chunk_lookup_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._expanded_fts_query", lambda *_args, **_kwargs: None)
    fan_out_search = AsyncMock(return_value=sentinel)
    monkeypatch.setattr("brainlayer.mcp.search_handler._fan_out_search", fan_out_search)

    result = await _brain_search(
        query="architecture decision",
        project="brainlayer",
        consumer="worker",
        source="all",
        fan_out=True,
    )

    assert result == sentinel
    assert fan_out_search.await_count == 1
    assert fan_out_search.await_args.kwargs["project"] == "brainlayer"


@pytest.mark.asyncio
async def test_brain_search_fan_out_keeps_explicit_project_leg_for_orchestrator(monkeypatch):
    sentinel = ([TextContent(type="text", text="fan-out")], {"fan_out": True})

    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_route_enabled", lambda: False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: object())
    monkeypatch.setattr("brainlayer.mcp.search_handler._exact_chunk_lookup_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._expanded_fts_query", lambda *_args, **_kwargs: None)
    fan_out_search = AsyncMock(return_value=sentinel)
    monkeypatch.setattr("brainlayer.mcp.search_handler._fan_out_search", fan_out_search)

    result = await _brain_search(
        query="architecture decision",
        project="brainlayer",
        consumer="orchestrator",
        source="all",
        fan_out=True,
    )

    assert result == sentinel
    assert fan_out_search.await_args.kwargs["project"] == "brainlayer"


@pytest.mark.asyncio
async def test_brain_search_fan_out_normalizes_worktree_before_resolving_worker_scope(monkeypatch):
    sentinel = ([TextContent(type="text", text="fan-out")], {"fan_out": True})

    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_route_enabled", lambda: False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: object())
    monkeypatch.setattr("brainlayer.mcp.search_handler._exact_chunk_lookup_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._expanded_fts_query", lambda *_args, **_kwargs: None)
    fan_out_search = AsyncMock(return_value=sentinel)
    monkeypatch.setattr("brainlayer.mcp.search_handler._fan_out_search", fan_out_search)

    result = await _brain_search(
        query="architecture decision",
        project="brainlayer-nightshift-1770775282043",
        consumer="worker",
        source="all",
        fan_out=True,
    )

    assert result == sentinel
    call = fan_out_search.await_args.kwargs
    assert call["project"] == "brainlayer"
    assert call["consumer_scope"].project_filter == "brainlayer"
    assert call["consumer_scope"].project_filters == ("brainlayer",)


@pytest.mark.asyncio
async def test_call_tool_fan_out_preserves_orchestrator_requested_project(monkeypatch):
    sentinel = ([TextContent(type="text", text="fan-out")], {"fan_out": True})

    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_route_enabled", lambda: False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: object())
    monkeypatch.setattr("brainlayer.mcp.search_handler._exact_chunk_lookup_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._expanded_fts_query", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.scoping.resolve_project_scope", lambda: "brainlayer")
    fan_out_search = AsyncMock(return_value=sentinel)
    monkeypatch.setattr("brainlayer.mcp.search_handler._fan_out_search", fan_out_search)

    result = await call_tool(
        "brain_search",
        {
            "query": "architecture decision",
            "project": "golems",
            "consumer": "orchestrator",
            "source": "all",
            "fan_out": True,
        },
    )

    assert result == sentinel
    assert fan_out_search.await_args.kwargs["project"] == "golems"


@pytest.mark.asyncio
async def test_brain_search_fan_out_bypasses_smart_think_routing(monkeypatch):
    sentinel = ([TextContent(type="text", text="fan-out")], {"fan_out": True})

    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_route_enabled", lambda: False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: object())
    monkeypatch.setattr("brainlayer.mcp.search_handler._exact_chunk_lookup_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._expanded_fts_query", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "brainlayer.mcp.search_handler._think",
        AsyncMock(side_effect=AssertionError("fan-out must bypass smart think routing")),
    )
    fan_out_search = AsyncMock(return_value=sentinel)
    monkeypatch.setattr("brainlayer.mcp.search_handler._fan_out_search", fan_out_search)

    result = await _brain_search(
        query="how did I implement architecture",
        project="brainlayer",
        consumer="worker",
        source="all",
        fan_out=True,
    )

    assert result == sentinel
    assert fan_out_search.await_count == 1
