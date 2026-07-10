"""Deterministic, bounded fan-out primitives for memory search."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Iterable

from mcp.types import CallToolResult

from .scoping import ConsumerScope
from .tag_normalization import valid_taxonomy_tags

_MAX_SCOPES = 4
_CANDIDATES_PER_SCOPE = 10
_RRF_K = 60


@dataclass(frozen=True)
class FanOutScope:
    name: str
    overrides: dict[str, Any]


def _matching_tag(query: str, known_tags: Iterable[str]) -> str | None:
    normalized_query = re.sub(r"[^a-z0-9]+", " ", query.casefold()).strip()
    matches: list[tuple[int, str]] = []
    for tag in known_tags:
        leaf = re.sub(r"[^a-z0-9]+", " ", tag.rsplit("/", 1)[-1].casefold()).strip()
        if leaf and re.search(rf"(?:^| ){re.escape(leaf)}(?: |$)", normalized_query):
            matches.append((len(leaf), tag))
    return min(matches, key=lambda item: (-item[0], item[1]))[1] if matches else None


def plan_fan_out_scopes(
    *,
    query: str,
    project: str | None,
    date_from: str | None,
    tag: str | None,
    now: datetime | None = None,
    known_tags: Iterable[str] | None = None,
) -> list[FanOutScope]:
    """Build a stable scope list without exceeding the hard four-search bound."""
    now = now or datetime.now(timezone.utc)
    recent_from = (now - timedelta(days=30)).date().isoformat()
    scopes = [FanOutScope("raw", {"project": None})]
    if project:
        scopes.append(FanOutScope("project", {"project": project}))
    scopes.append(FanOutScope("recent", {"project": project, "date_from": max(date_from or "", recent_from)}))
    if tag is None and (matched_tag := _matching_tag(query, known_tags or valid_taxonomy_tags())):
        scopes.append(FanOutScope(f"tag:{matched_tag}", {"project": project, "tag": matched_tag}))
    return scopes[:_MAX_SCOPES]


def _error_text(result: CallToolResult) -> str:
    for item in result.content:
        text = getattr(item, "text", None)
        if text:
            return str(text)
    return "search error"


def _narrow_consumer_scope(scope: ConsumerScope, project: str) -> ConsumerScope:
    allowed = scope.project_filters or ((scope.project_filter,) if scope.project_filter else ())
    if allowed and project not in allowed:
        return replace(scope, project_filter=None, project_filters=(), allow_null_project=False, deny_all=True)
    return replace(scope, project_filter=project, project_filters=(project,), allow_null_project=False)


async def run_fan_out(
    *,
    search: Callable[..., Awaitable[Any]],
    query: str,
    num_results: int,
    base_kwargs: dict[str, Any],
    project: str | None,
    date_from: str | None,
    tag: str | None,
    now: datetime | None = None,
    known_tags: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Run bounded scopes sequentially and merge results by stable RRF."""
    scopes = plan_fan_out_scopes(
        query=query,
        project=project,
        date_from=date_from,
        tag=tag,
        now=now,
        known_tags=known_tags,
    )
    merged: dict[str, dict[str, Any]] = {}
    candidate_count = 0
    first_seen = 0
    degraded_scopes: list[dict[str, str]] = []

    for scope in scopes:
        search_kwargs = {
            **base_kwargs,
            **scope.overrides,
            "query": query,
            "num_results": _CANDIDATES_PER_SCOPE,
            "order": "relevance",
        }
        if scope.overrides.get("project") and isinstance(search_kwargs.get("consumer_scope"), ConsumerScope):
            search_kwargs["consumer_scope"] = _narrow_consumer_scope(
                search_kwargs["consumer_scope"], scope.overrides["project"]
            )
        response = await search(**search_kwargs)
        if isinstance(response, CallToolResult):
            degraded_scopes.append({"scope": scope.name, "reason": _error_text(response)})
            continue
        _content, structured = response
        reason = structured.get("degrade_reason") or structured.get("fallback_reason")
        if structured.get("degraded") or structured.get("search_mode") not in (None, "hybrid"):
            degraded_scopes.append({"scope": scope.name, "reason": reason or "degraded search"})

        candidates = list(structured.get("results") or [])[:_CANDIDATES_PER_SCOPE]
        if scope.name == "recent":
            candidates.sort(key=lambda item: str(item.get("date") or item.get("created_at") or ""), reverse=True)
        scope_chunk_ids: set[str] = set()
        for rank, item in enumerate(candidates, start=1):
            candidate_count += 1
            chunk_id = item.get("chunk_id")
            if not chunk_id:
                if not any(entry["scope"] == scope.name for entry in degraded_scopes):
                    degraded_scopes.append({"scope": scope.name, "reason": "candidate missing chunk_id"})
                continue
            if chunk_id in scope_chunk_ids:
                continue
            scope_chunk_ids.add(chunk_id)
            if chunk_id not in merged:
                first_seen += 1
                merged[chunk_id] = {
                    "item": dict(item),
                    "score": 0.0,
                    "first_seen": first_seen,
                    "provenance": [],
                }
            entry = merged[chunk_id]
            entry["score"] += 1 / (_RRF_K + rank)
            entry["provenance"].append(scope.name)

    ranked = sorted(
        merged.values(), key=lambda entry: (-entry["score"], entry["first_seen"], entry["item"]["chunk_id"])
    )
    results = []
    for entry in ranked[:num_results]:
        item = entry["item"]
        item["fan_out_score"] = round(entry["score"], 8)
        item["fan_out_provenance"] = entry["provenance"]
        results.append(item)

    return {
        "query": query,
        "total": len(results),
        "results": results,
        "fan_out": True,
        "fan_out_scopes": [scope.name for scope in scopes],
        "fan_out_manifest": [{"scope": scope.name, **scope.overrides} for scope in scopes],
        "candidate_count": candidate_count,
        "candidate_limit": _MAX_SCOPES * _CANDIDATES_PER_SCOPE,
        "ranking": "rrf(k=60), tie-break=first-seen,chunk_id",
        "degraded": bool(degraded_scopes),
        "degraded_scopes": degraded_scopes,
    }
