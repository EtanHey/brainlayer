"""Search and recall MCP handlers."""

import asyncio
import json
import re
from typing import Any

import apsw
from mcp.types import TextContent

from ..lexical_defense import _normalize_surface, load_lexical_defense_dictionary

# Retry settings for DB lock resilience on reads
_RETRY_MAX_ATTEMPTS = 3
_retry_delay = 0.1  # base delay in seconds (exposed for test patching)
_VALID_SEARCH_DETAILS = frozenset({"compact", "full"})
_MAX_PUBLIC_NUM_RESULTS = 100
_MIN_PUBLIC_NUM_RESULTS = 1

from ._format import format_kg_search, format_search_results, format_stats
from ._shared import (
    _build_compact_result,
    _error_result,
    _extract_file_path,
    _get_embedding_model,
    _get_vector_store,
    _memory_to_dict,
    _normalize_project_name,
    _query_has_regression_signal,
    _query_signals_current_context,
    _query_signals_recall,
    _query_signals_think,
    logger,
)

_CHUNK_ID_QUERY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(?:-[A-Za-z0-9_]+)+$")


def _quote_fts_phrase(value: str) -> str:
    return f'"{value.replace(chr(34), "")}"'


def _lexical_defense_variants(query: str) -> list[str]:
    dictionary = load_lexical_defense_dictionary()
    variants: list[str] = [query]
    seen = {query.casefold().strip()}

    for candidate in {query, *query.split()}:
        entry = dictionary.lookup(candidate)
        if not entry:
            continue
        for surface in entry.all_surfaces:
            dedupe_key = surface.casefold().strip()
            if dedupe_key and dedupe_key not in seen:
                seen.add(dedupe_key)
                variants.append(surface)

    return variants


def _kg_alias_variants(query: str, store: Any) -> list[str]:
    normalized_candidates = {_normalize_surface(query)}
    normalized_candidates.update(_normalize_surface(token) for token in query.split())
    normalized_candidates.discard("")
    if not normalized_candidates:
        return []

    cursor = store._read_cursor()
    placeholders = ", ".join("?" for _ in normalized_candidates)
    normalizer = "LOWER(REPLACE(REPLACE(REPLACE(REPLACE({col}, '-', ''), '_', ''), '.', ''), ' ', ''))"
    params = [*normalized_candidates, *normalized_candidates]
    entity_rows = list(
        cursor.execute(
            f"""
            SELECT DISTINCT e.id, e.name
            FROM kg_entities e
            LEFT JOIN kg_entity_aliases a ON a.entity_id = e.id
            WHERE {normalizer.format(col='e.name')} IN ({placeholders})
               OR {normalizer.format(col='a.alias')} IN ({placeholders})
            """,
            params,
        )
    )
    if not entity_rows:
        return []

    variants: list[str] = []
    seen = set()
    entity_ids = []
    for entity_id, entity_name in entity_rows:
        entity_ids.append(entity_id)
        dedupe_key = entity_name.casefold().strip()
        if dedupe_key and dedupe_key not in seen:
            seen.add(dedupe_key)
            variants.append(entity_name)

    alias_placeholders = ", ".join("?" for _ in entity_ids)
    alias_rows = list(
        cursor.execute(
            f"SELECT alias FROM kg_entity_aliases WHERE entity_id IN ({alias_placeholders})",
            entity_ids,
        )
    )
    for (alias,) in alias_rows:
        dedupe_key = alias.casefold().strip()
        if dedupe_key and dedupe_key not in seen:
            seen.add(dedupe_key)
            variants.append(alias)

    return variants


def _expanded_fts_query(query: str, store: Any) -> str | None:
    variants = _lexical_defense_variants(query)
    seen = {value.casefold().strip() for value in variants if value.strip()}
    for variant in _kg_alias_variants(query, store):
        dedupe_key = variant.casefold().strip()
        if dedupe_key and dedupe_key not in seen:
            seen.add(dedupe_key)
            variants.append(variant)

    if len(variants) <= 1:
        return None
    return " OR ".join(_quote_fts_phrase(variant) for variant in variants)


def _exact_chunk_lookup_result(query: str, store: Any, detail: str) -> tuple[list[TextContent], dict] | None:
    """Return an exact chunk hit for chunk-id shaped queries, or None on miss."""
    candidate = query.strip()
    if not candidate or " " in candidate or not _CHUNK_ID_QUERY_RE.fullmatch(candidate):
        return None

    chunk = store.get_chunk(candidate)
    if not chunk:
        return None

    tags = chunk.get("tags")
    parsed_tags = None
    if tags:
        try:
            parsed_tags = json.loads(tags) if isinstance(tags, str) else tags
        except (json.JSONDecodeError, TypeError):
            parsed_tags = None

    item = {
        "score": 1.0,
        "chunk_id": chunk["id"],
        "project": _normalize_project_name(chunk.get("project")) or chunk.get("project", "unknown"),
        "content": chunk.get("content", ""),
        "source_file": chunk.get("source_file", "unknown"),
        "date": chunk.get("created_at", "")[:10] if chunk.get("created_at") else None,
        "importance": chunk.get("importance"),
        "summary": chunk.get("summary"),
    }
    if parsed_tags:
        item["tags"] = parsed_tags

    if detail == "compact":
        structured_results = [_build_compact_result(item)]
    else:
        structured_results = [item]

    structured = {"query": query, "total": 1, "results": structured_results}
    formatted_text = format_search_results(query, structured_results, 1)
    return ([TextContent(type="text", text=formatted_text)], structured)


def _detect_entities(query: str, store: Any) -> list[dict]:
    """Detect known KG entity names in a query string.

    Checks bigrams and individual capitalized words against kg_entities_fts.
    Returns list of matched entities with id, name, entity_type.
    """
    if not query or len(query) < 2:
        return []

    words = query.split()
    candidates: list[str] = []

    # Generate bigrams (most entity names are 2 words: "Avi Simon", "Michal Cohen")
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i + 1]}"
        # Only check bigrams where at least one word is capitalized or all-caps
        if words[i][0].isupper() or words[i + 1][0].isupper():
            candidates.append(bigram)

    # Single capitalized words (3+ chars to avoid false positives on "I", "A", etc.)
    for w in words:
        if len(w) >= 3 and w[0].isupper() and not w.isupper():
            candidates.append(w)

    # Also check all lowercase words ≥4 chars — handles "anthropic", "cantaloupe"
    for w in words:
        cleaned = w.strip("?.,!;:'\"")
        if (
            len(cleaned) >= 4
            and cleaned.islower()
            and cleaned
            not in ("what", "who", "how", "when", "where", "which", "does", "that", "this", "from", "with", "about")
        ):
            candidates.append(cleaned)

    if not candidates:
        return []

    try:
        cursor = store._read_cursor()
        matched = []
        seen_ids: set[str] = set()

        for candidate in candidates:
            # Exact name match first (case-insensitive)
            rows = list(
                cursor.execute(
                    """SELECT id, name, entity_type FROM kg_entities
                   WHERE LOWER(name) = LOWER(?) LIMIT 1""",
                    (candidate,),
                )
            )
            if rows:
                eid, name, etype = rows[0]
                if eid not in seen_ids:
                    seen_ids.add(eid)
                    matched.append({"id": eid, "name": name, "entity_type": etype})
                continue

            # FTS fallback — fuzzy match on entity names
            from .._helpers import _escape_fts5_query

            fts_q = _escape_fts5_query(candidate)
            if not fts_q.strip():
                continue
            rows = list(
                cursor.execute(
                    """SELECT e.id, e.name, e.entity_type
                   FROM kg_entities_fts f
                   JOIN kg_entities e ON f.entity_id = e.id
                   WHERE kg_entities_fts MATCH ?
                   ORDER BY f.rank LIMIT 1""",
                    (fts_q,),
                )
            )
            if rows:
                eid, name, etype = rows[0]
                if eid not in seen_ids:
                    seen_ids.add(eid)
                    matched.append({"id": eid, "name": name, "entity_type": etype})

        return matched
    except Exception as e:
        logger.debug("Entity detection failed: %s", e)
        return []


def _kg_facts_sql(store: Any, entity_name: str) -> list[dict]:
    """Pure SQL KG fact lookup — no embeddings, no vector search.

    Returns typed relations for an entity, excluding co_occurs_with noise.
    This always works even when the embedding model isn't loaded.
    """
    try:
        cursor = store._read_cursor()
        # Find entity ID
        row = list(
            cursor.execute(
                "SELECT id FROM kg_entities WHERE LOWER(name) = LOWER(?) LIMIT 1",
                (entity_name,),
            )
        )
        if not row:
            return []

        entity_id = row[0][0]

        # Get all semantic relations (exclude co_occurs_with)
        facts_raw = list(
            cursor.execute(
                """SELECT r.relation_type, r.properties, r.confidence,
                          se.name as source_name, se.entity_type as source_type,
                          te.name as target_name, te.entity_type as target_type
                   FROM kg_relations r
                   JOIN kg_entities se ON r.source_id = se.id
                   JOIN kg_entities te ON r.target_id = te.id
                   WHERE (r.source_id = ? OR r.target_id = ?)
                     AND r.relation_type != 'co_occurs_with'
                   ORDER BY r.confidence DESC
                   LIMIT 20""",
                (entity_id, entity_id),
            )
        )

        import json as _json

        result = []
        for rel_type, props_str, confidence, src_name, src_type, tgt_name, tgt_type in facts_raw:
            props = {}
            if props_str:
                try:
                    props = _json.loads(props_str)
                except (ValueError, TypeError):
                    pass
            desc = props.get("description") or props.get("fact") or ""
            result.append(
                {
                    "relation": rel_type,
                    "source": src_name,
                    "target": tgt_name,
                    "description": desc,
                    "score": confidence or 0.5,
                }
            )
        return result
    except Exception as e:
        logger.warning("SQL KG lookup failed for %s: %s", entity_name, e)
        return []


async def _brain_search(
    query: str,
    project: str | None = None,
    file_path: str | None = None,
    chunk_id: str | None = None,
    content_type: str | None = None,
    source: str | None = None,
    tag: str | None = None,
    intent: str | None = None,
    importance_min: float | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    sentiment: str | None = None,
    entity_id: str | None = None,
    num_results: int = 5,
    before: int = 3,
    after: int = 3,
    max_results: int = 10,
    detail: str = "compact",
    source_filter: str | None = None,
    correction_category: str | None = None,
):
    """Unified search dispatcher -- routes to the right internal handler."""

    if detail not in _VALID_SEARCH_DETAILS:
        return _error_result(f"Invalid detail='{detail}'. Must be one of: {sorted(_VALID_SEARCH_DETAILS)}")
    if num_results < _MIN_PUBLIC_NUM_RESULTS or num_results > _MAX_PUBLIC_NUM_RESULTS:
        return _error_result(
            f"num_results={num_results} must be between {_MIN_PUBLIC_NUM_RESULTS} and {_MAX_PUBLIC_NUM_RESULTS}"
        )

    if project is None and entity_id is None and source not in ("youtube", "whatsapp", "telegram", "all"):
        try:
            from ..scoping import resolve_project_scope

            project = resolve_project_scope()
        except Exception:
            logger.debug("Project auto-scope failed, proceeding without scope")

    if entity_id is not None:
        return await _search(
            query=query,
            project=project,
            content_type=content_type,
            num_results=num_results,
            source=source,
            tag=tag,
            intent=intent,
            importance_min=importance_min,
            date_from=date_from,
            date_to=date_to,
            sentiment=sentiment,
            entity_id=entity_id,
            detail=detail,
            source_filter_like=source_filter,
            correction_category=correction_category,
        )

    store = _get_vector_store()
    exact_chunk_hit = _exact_chunk_lookup_result(query, store, detail)
    if exact_chunk_hit is not None:
        return exact_chunk_hit
    fts_query_override = _expanded_fts_query(query, store)

    if chunk_id is not None:
        return await _context(chunk_id=chunk_id, before=before, after=after)

    if file_path is not None and _query_has_regression_signal(query):
        regression_result = await _regression(file_path=file_path, project=project)
        recall_result = await _recall(file_path=file_path, project=project, max_results=max_results)
        merged_text = []
        if isinstance(regression_result, list):
            merged_text.extend(regression_result)
        if isinstance(recall_result, tuple):
            merged_text.extend(recall_result[0])
        else:
            merged_text.extend(recall_result)
        return merged_text

    if file_path is not None:
        timeline = await _file_timeline(file_path=file_path, project=project, limit=50)
        recall_result = await _recall(file_path=file_path, project=project, max_results=max_results)
        merged_text = []
        if isinstance(timeline, list):
            merged_text.extend(timeline)
        if isinstance(recall_result, tuple):
            merged_text.extend(recall_result[0])
        else:
            merged_text.extend(recall_result)
        return merged_text

    extracted_file = _extract_file_path(query)
    if extracted_file:
        return await _brain_search(
            query=query,
            project=project,
            file_path=extracted_file,
            content_type=content_type,
            source=source,
            tag=tag,
            intent=intent,
            importance_min=importance_min,
            date_from=date_from,
            date_to=date_to,
            sentiment=sentiment,
            num_results=num_results,
            max_results=max_results,
            detail=detail,
            source_filter=source_filter,
            correction_category=correction_category,
        )

    if _query_signals_current_context(query):
        ctx = await _current_context(hours=24)
        think_result = await _think(context=query, project=project, max_results=max_results)
        merged_text = []
        if isinstance(ctx, tuple):
            merged_text.extend(ctx[0])
        else:
            merged_text.extend(ctx)
        if isinstance(think_result, tuple):
            merged_text.extend(think_result[0])
        else:
            merged_text.extend(think_result)
        return merged_text

    if _query_signals_think(query):
        return await _think(context=query, project=project, max_results=max_results)

    if _query_signals_recall(query):
        return await _recall(topic=query, project=project, max_results=max_results)

    # Entity-aware routing: detect known entity names in query.
    # Path 1: Pure SQL KG lookup (no embeddings, always works).
    # Path 2: Full kg_hybrid_search with vector similarity (optional, needs embedding model).
    # Skip entity routing when additional filters are active.
    has_active_filters = any(
        [
            content_type,
            source,
            tag,
            intent,
            importance_min,
            date_from,
            date_to,
            sentiment,
            source_filter,
            correction_category,
        ]
    )
    detected_entities = _detect_entities(query, store) if not has_active_filters else []
    if detected_entities:
        entity_name = detected_entities[0]["name"]

        # Path 1: Pure SQL KG facts (no embedding model needed, always runs)
        fact_items = _kg_facts_sql(store, entity_name)

        # Path 2: Try full hybrid search (embedding + vector + KG)
        structured_results = []
        kg_degraded = False
        try:
            normalized_project = _normalize_project_name(project)
            loop = asyncio.get_running_loop()
            model = _get_embedding_model()
            query_embedding = await loop.run_in_executor(None, model.embed_query, query)

            kg_results = store.kg_hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                n_results=num_results,
                entity_name=entity_name,
                project_filter=normalized_project,
            )
            chunk_results = kg_results.get("chunks", {})

            if chunk_results.get("ids") and chunk_results["ids"][0]:
                for cid, doc, meta, dist in zip(
                    chunk_results["ids"][0],
                    chunk_results["documents"][0],
                    chunk_results["metadatas"][0],
                    chunk_results["distances"][0],
                ):
                    score = 1 - dist if dist is not None else 0
                    if detail == "compact":
                        item = _build_compact_result(
                            {
                                "score": round(score, 4),
                                "chunk_id": cid,
                                "project": _normalize_project_name(meta.get("project")) or "unknown",
                                "content": doc,
                                "source_file": meta.get("source_file", "unknown"),
                                "date": meta.get("created_at", "")[:10] if meta.get("created_at") else None,
                                "importance": meta.get("importance"),
                                "summary": meta.get("summary"),
                            }
                        )
                    else:
                        item = {
                            "score": round(score, 4),
                            "chunk_id": cid,
                            "content": doc,
                            "entity": entity_name,
                        }
                    structured_results.append(item)
        except (RuntimeError, OSError, MemoryError) as e:
            logger.warning("KG hybrid search failed (embedding/model issue), using SQL-only: %s", e)
            kg_degraded = True
        except Exception as e:
            logger.warning("KG hybrid search failed unexpectedly: %s", e, exc_info=True)
            kg_degraded = True

        # If we have KG facts OR chunk results, return them
        if fact_items or structured_results:
            structured = {
                "query": query,
                "entity": entity_name,
                "total": len(structured_results),
                "results": structured_results,
                "facts": fact_items,
            }
            if kg_degraded:
                structured["kg_degraded"] = True
            formatted_text = format_kg_search(entity_name, structured_results, fact_items, query)
            if kg_degraded:
                formatted_text += "\n⚠ KG search degraded — showing SQL-only results"
            return ([TextContent(type="text", text=formatted_text)], structured)

    return await _search(
        query=query,
        project=project,
        content_type=content_type,
        num_results=num_results,
        source=source,
        tag=tag,
        intent=intent,
        importance_min=importance_min,
        date_from=date_from,
        date_to=date_to,
        sentiment=sentiment,
        detail=detail,
        fts_query_override=fts_query_override,
        source_filter_like=source_filter,
        correction_category=correction_category,
    )


def _infer_recall_mode(arguments: dict) -> str:
    """Auto-infer recall mode from provided arguments."""
    if arguments.get("session_id") and not arguments.get("plan_name"):
        return "summary"
    if arguments.get("plan_name"):
        return "plan"
    if arguments.get("days") or arguments.get("limit"):
        return "sessions"
    return "context"


def _smart_detect_mode(query: str | None, mode: str | None) -> str:
    """Detect the best recall mode from query text when mode is not explicit.

    Heuristics:
    - "stats" / "how many" / "count" → stats
    - "context" / "right now" / "working on" → context
    - Capitalized proper-noun pattern (e.g. "BrainLayer", "Etan Heyman") → entity
    - Default → search
    """
    if mode is not None:
        return mode
    if not query:
        return "context"

    q_lower = query.lower().strip()

    # Stats signals
    if any(sig in q_lower for sig in ("stats", "how many", "count", "statistics", "total chunks")):
        return "stats"

    # Context signals
    if any(sig in q_lower for sig in ("what am i working on", "right now", "current context", "what's active")):
        return "context"

    # Entity signals: query is a short capitalized proper-noun phrase (1-3 words, all start uppercase)
    words = query.strip().split()
    if 1 <= len(words) <= 3 and all(w[0].isupper() for w in words if len(w) > 0):
        # Single-word common English words that aren't entities
        _COMMON_WORDS = frozenset(
            {
                "The",
                "How",
                "What",
                "When",
                "Where",
                "Why",
                "Who",
                "Which",
                "Find",
                "Search",
                "Get",
                "Show",
                "List",
                "Tell",
                "Give",
            }
        )
        if len(words) == 1 and words[0] in _COMMON_WORDS:
            return "search"
        return "entity"

    return "search"


async def _brain_recall(
    mode: str | None = None,
    project: str | None = None,
    hours: int = 24,
    days: int = 7,
    limit: int = 20,
    session_id: str | None = None,
    plan_name: str | None = None,
    # --- Phase 1 additions: search + entity mode params ---
    query: str | None = None,
    file_path: str | None = None,
    chunk_id: str | None = None,
    content_type: str | None = None,
    source: str | None = None,
    tag: str | None = None,
    intent: str | None = None,
    importance_min: float | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    sentiment: str | None = None,
    entity_id: str | None = None,
    num_results: int = 5,
    before: int = 3,
    after: int = 3,
    max_results: int = 10,
    detail: str = "compact",
    entity_type: str | None = None,
    # --- T3 filter additions ---
    source_filter: str | None = None,
    correction_category: str | None = None,
):
    """Unified recall dispatcher -- routes to session/context/search/entity handlers.

    Phase 1 additions:
    - mode=search: delegates to _brain_search (hybrid semantic + keyword search)
    - mode=entity: delegates to _brain_entity (knowledge graph entity lookup)
    - Smart routing: auto-detects mode from query when mode is not explicit
    """
    # If mode is explicitly one of the new modes, use it directly.
    # If mode is explicitly one of the old modes, use old inference.
    # If mode is None, use smart detection when query is present,
    # otherwise fall back to old _infer_recall_mode.
    if mode in ("search", "entity"):
        resolved_mode = mode
    elif mode is not None:
        # Explicit old mode (context, sessions, operations, plan, summary, stats)
        resolved_mode = mode
    elif query is not None:
        # Smart detection from query text
        resolved_mode = _smart_detect_mode(query, None)
    else:
        # No query, no mode — old inference from other params
        resolved_mode = _infer_recall_mode(
            {
                "session_id": session_id,
                "plan_name": plan_name,
                "days": days if days != 7 else None,
                "limit": limit if limit != 20 else None,
            }
        )

    # --- New modes: search + entity ---

    if resolved_mode == "search":
        if not query:
            return _error_result("query is required for mode=search")
        return await _brain_search(
            query=query,
            project=project,
            file_path=file_path,
            chunk_id=chunk_id,
            content_type=content_type,
            source=source,
            tag=tag,
            intent=intent,
            importance_min=importance_min,
            date_from=date_from,
            date_to=date_to,
            sentiment=sentiment,
            entity_id=entity_id,
            num_results=num_results,
            before=before,
            after=after,
            max_results=max_results,
            detail=detail,
            source_filter=source_filter,
            correction_category=correction_category,
        )

    if resolved_mode == "entity":
        if not query:
            return _error_result("query is required for mode=entity")
        from .entity_handler import _brain_entity as _entity_handler

        return await _entity_handler(query=query, entity_type=entity_type)

    # --- Original modes ---

    if resolved_mode == "context":
        return await _current_context(hours=hours)
    elif resolved_mode == "sessions":
        return await _sessions(project=project, days=max(1, min(days, 365)), limit=max(1, min(limit, 100)))
    elif resolved_mode == "operations":
        if not session_id:
            return _error_result("session_id required for mode=operations")
        return await _operations(session_id=session_id)
    elif resolved_mode == "plan":
        return await _plan_links(plan_name=plan_name, session_id=session_id, project=project)
    elif resolved_mode == "summary":
        if not session_id:
            return _error_result("session_id required for mode=summary")
        return await _session_summary(session_id=session_id)
    elif resolved_mode == "stats":
        stats_result = await _stats()
        projects_result = await _list_projects()
        merged_text = []
        if isinstance(stats_result, tuple):
            merged_text.extend(stats_result[0])
        else:
            merged_text.extend(stats_result)
        if isinstance(projects_result, list):
            merged_text.extend(projects_result)
        return merged_text
    else:
        return _error_result(f"Unknown recall mode: {resolved_mode}")


async def _search(
    query: str,
    project: str | None = None,
    content_type: str | None = None,
    num_results: int = 5,
    source: str | None = None,
    tag: str | None = None,
    intent: str | None = None,
    importance_min: float | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    sentiment: str | None = None,
    entity_id: str | None = None,
    detail: str = "compact",
    fts_query_override: str | None = None,
    # Backward compat: accept old 'format' kwarg
    output_format: str | None = None,
    # --- T3 filter additions ---
    source_filter_like: str | None = None,
    correction_category: str | None = None,
):
    """Execute a hybrid search query (semantic + keyword via RRF). Retries on BusyError."""
    try:
        # Backward compat: old 'format' kwarg overrides 'detail'
        if output_format is not None:
            detail = output_format

        if num_results < 1:
            num_results = 5
        elif num_results > 100:
            num_results = 100

        store = _get_vector_store()

        if store.count() == 0:
            empty = {"query": query, "total": 0, "results": []}
            return (
                [TextContent(type="text", text="Knowledge base is empty. Run 'brainlayer index' to populate it.")],
                empty,
            )

        normalized_project = _normalize_project_name(project)
        loop = asyncio.get_running_loop()
        model = _get_embedding_model()
        query_embedding = await loop.run_in_executor(None, model.embed_query, query)

        if source == "all":
            source_filter = None
        elif source:
            source_filter = source
        else:
            source_filter = (
                None  # AIDEV-NOTE: was "claude_code" — excluded brain_store ("manual") chunks. Default to all sources.
            )

        if entity_id and not source:
            source_filter = None

        # Retry hybrid_search on BusyError — WAL reads shouldn't block but
        # they can during checkpoint or when enrichment holds exclusive lock.
        results = None
        for attempt in range(_RETRY_MAX_ATTEMPTS):
            try:
                results = store.hybrid_search(
                    query_embedding=query_embedding,
                    query_text=query,
                    fts_query_override=fts_query_override,
                    n_results=num_results,
                    project_filter=normalized_project,
                    content_type_filter=content_type,
                    source_filter=source_filter,
                    tag_filter=tag,
                    intent_filter=intent,
                    importance_min=importance_min,
                    date_from=date_from,
                    date_to=date_to,
                    sentiment_filter=sentiment,
                    entity_id=entity_id,
                    source_filter_like=source_filter_like,
                    correction_category=correction_category,
                )
                break
            except Exception as e:
                is_lock = isinstance(e, apsw.BusyError) or "locked" in str(e).lower() or "busy" in str(e).lower()
                if is_lock and attempt < _RETRY_MAX_ATTEMPTS - 1:
                    delay = _retry_delay * (2**attempt)
                    logger.warning(
                        "Search BusyError (attempt %d/%d), retrying in %.2fs", attempt + 1, _RETRY_MAX_ATTEMPTS, delay
                    )
                    await asyncio.sleep(delay)
                    continue
                raise  # Non-lock error or retries exhausted

        if not results["documents"][0]:
            empty = {"query": query, "total": 0, "results": []}
            return ([TextContent(type="text", text="No results found.")], empty)

        results = store.enrich_results_with_session_context(results)

        if detail == "compact":
            structured_results = []
            for cid, doc, meta, dist in zip(
                results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
            ):
                score = 1 - dist if dist is not None else 0
                item = _build_compact_result(
                    {
                        "score": round(score, 4),
                        "chunk_id": cid,
                        "project": _normalize_project_name(meta.get("project")) or meta.get("project", "unknown"),
                        "content": doc,
                        "source_file": meta.get("source_file", "unknown"),
                        "date": meta.get("created_at", "")[:10] if meta.get("created_at") else None,
                        "importance": meta.get("importance"),
                        "summary": meta.get("summary"),
                        "tags": [str(t) for t in meta["tags"][:5]]
                        if meta.get("tags") and isinstance(meta["tags"], list)
                        else None,
                    }
                )
                structured_results.append(item)
            structured = {"query": query, "total": len(structured_results), "results": structured_results}
            formatted_text = format_search_results(query, structured_results, len(structured_results))
            return ([TextContent(type="text", text=formatted_text)], structured)

        output_parts = [f"## Search Results for: {query}\n"]
        structured_results = []

        for i, (cid, doc, meta, dist) in enumerate(
            zip(results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0])
        ):
            score = 1 - dist if dist is not None else 0
            item = {
                "score": round(score, 4),
                "chunk_id": cid,
                "project": _normalize_project_name(meta.get("project")) or meta.get("project", "unknown"),
                "content_type": meta.get("content_type", "unknown"),
                "content": doc,
                "source_file": meta.get("source_file", "unknown"),
            }
            if meta.get("created_at"):
                item["date"] = meta["created_at"][:10] if len(meta.get("created_at", "")) >= 10 else meta["created_at"]
            if meta.get("source") and meta["source"] != "claude_code":
                item["source"] = meta["source"]
            if meta.get("summary"):
                item["summary"] = meta["summary"]
            if meta.get("tags") and isinstance(meta["tags"], list):
                item["tags"] = [str(t) for t in meta["tags"][:5]]
            if meta.get("intent"):
                item["intent"] = meta["intent"]
            if meta.get("importance") is not None:
                item["importance"] = meta["importance"]
            if meta.get("session_summary"):
                item["session_summary"] = meta["session_summary"]
            if meta.get("session_outcome"):
                item["session_outcome"] = meta["session_outcome"]
            if meta.get("session_quality") is not None:
                item["session_quality"] = meta["session_quality"]
            structured_results.append(item)

            output_parts.append(f"\n### Result {i + 1} (score: {score:.3f})")
            enrichment_parts = []
            if meta.get("intent"):
                enrichment_parts.append(f"Intent: {meta['intent']}")
            if meta.get("importance") is not None:
                enrichment_parts.append(f"Importance: {meta['importance']:.0f}/10")
            if meta.get("tags") and isinstance(meta["tags"], list):
                enrichment_parts.append(f"Tags: {', '.join(str(t) for t in meta['tags'][:5])}")
            project_display = item["project"]
            if project_display == "unknown" and meta.get("contact_name"):
                project_display = meta["contact_name"]
            header = f"**Project:** {project_display} | **Type:** {meta.get('content_type', 'unknown')}"
            if item.get("date"):
                header += f" | **Date:** {item['date']}"
            if item.get("source"):
                header += f" | **Source:** {item['source']}"
            output_parts.append(header)
            if enrichment_parts:
                output_parts.append(f"**{' | '.join(enrichment_parts)}**")
            if meta.get("summary"):
                output_parts.append(f"> {meta['summary']}")
            if meta.get("session_summary"):
                output_parts.append(f"**Session:** {meta['session_summary'][:200]}")
            output_parts.append(f"**File:** `{meta.get('source_file', 'unknown')}`\n")
            output_parts.append(doc)
            output_parts.append("\n---")

        structured = {"query": query, "total": len(structured_results), "results": structured_results}
        return ([TextContent(type="text", text="\n".join(output_parts))], structured)

    except Exception as e:
        return _error_result(f"Search error (query='{query[:50]}...'): {str(e)}")


async def _stats():
    """Get knowledge base statistics."""
    try:
        store = _get_vector_store()
        stats = store.get_stats()
        fts5_health = store.check_fts5_health()
        structured = {
            "total_chunks": stats["total_chunks"],
            "projects": stats["projects"],
            "content_types": stats["content_types"],
            "fts5_health": fts5_health,
        }
        output = format_stats(structured)
        output += (
            "\n"
            f"FTS5: {fts5_health['severity']} | chunks={fts5_health['chunk_count']:,} "
            f"| fts={fts5_health['fts_count']:,} | desync={fts5_health['desync_pct']:.2f}%"
        )
        return ([TextContent(type="text", text=output)], structured)
    except Exception as e:
        return _error_result(f"Stats error: {str(e)}")


async def _list_projects() -> list[TextContent]:
    """List all projects."""
    try:
        store = _get_vector_store()
        stats = store.get_stats()
        if not stats["projects"]:
            return [TextContent(type="text", text="No projects indexed yet.")]
        output = "## Indexed Projects\n\n"
        for proj in sorted(stats["projects"]):
            output += f"- {proj}\n"
        return [TextContent(type="text", text=output)]
    except Exception as e:
        return _error_result(f"Error listing projects: {str(e)}")


async def _context(chunk_id: str, before: int = 3, after: int = 3) -> list[TextContent]:
    """Get surrounding conversation context for a chunk."""
    try:
        store = _get_vector_store()
        result = store.get_context(chunk_id, before=before, after=after)
        if result.get("error"):
            return _error_result(f"Unknown chunk_id '{chunk_id[:20]}...'. Use chunk_id from brainlayer_search results.")
        if not result.get("context"):
            return [TextContent(type="text", text="No context available for this chunk.")]
        output_parts = ["## Conversation Context\n"]
        for chunk in result["context"]:
            marker = " **[TARGET]**" if chunk.get("is_target") else ""
            ctype = chunk.get("content_type", "unknown")
            pos = chunk.get("position", "?")
            output_parts.append(f"\n### Position {pos} ({ctype}){marker}\n")
            content = chunk.get("content", "")
            output_parts.append(content[:1500] + ("..." if len(content) > 1500 else ""))
            output_parts.append("\n---")
        return [TextContent(type="text", text="\n".join(output_parts))]
    except Exception as e:
        return _error_result(f"Context error: {str(e)}")


async def _file_timeline(file_path: str, project: str | None = None, limit: int = 50) -> list[TextContent]:
    """Get interaction timeline for a file."""
    try:
        store = _get_vector_store()
        interactions = store.get_file_timeline(file_path, project=project, limit=limit)
        if not interactions:
            return [TextContent(type="text", text=f"No interactions found for '{file_path}'.")]
        output_parts = [f"## File Timeline: {file_path}\n", f"Found {len(interactions)} interactions:\n"]
        for i, row in enumerate(interactions):
            ts = row.get("timestamp", "?")
            action = row.get("action", "?")
            session = row.get("session_id", "?")[:8]
            proj = row.get("project", "?")
            fp = row.get("file_path", file_path)
            output_parts.append(f"{i + 1}. **{action}** `{fp}` at {ts} (session: {session}, project: {proj})")
        return [TextContent(type="text", text="\n".join(output_parts))]
    except Exception as e:
        return _error_result(f"File timeline error: {str(e)}")


async def _operations(session_id: str) -> list[TextContent]:
    """Get operations for a session."""
    try:
        store = _get_vector_store()
        ops = store.get_session_operations(session_id)
        if not ops:
            return [TextContent(type="text", text=f"No operations for session '{session_id[:8]}...'.")]
        output_parts = [f"## Operations: {session_id[:8]}...\n", f"Found {len(ops)} operations:\n"]
        for i, op in enumerate(ops):
            outcome = op.get("outcome", "unknown")
            ts = (op.get("started_at") or "?")[:19]
            output_parts.append(
                f"{i + 1}. **{op.get('operation_type', '?')}** -- {op.get('summary') or '?'} [{outcome}] at {ts}"
            )
        return [TextContent(type="text", text="\n".join(output_parts))]
    except Exception as e:
        return _error_result(f"Operations error: {str(e)}")


async def _regression(file_path: str, project: str | None = None) -> list[TextContent]:
    """Analyze a file for regressions."""
    try:
        store = _get_vector_store()
        result = store.get_file_regression(file_path, project=project)
        if not result["timeline"]:
            return [TextContent(type="text", text=f"No interactions found for '{file_path}'.")]
        parts = [f"## Regression Analysis: {file_path}\n", f"Timeline: {len(result['timeline'])} interactions\n"]
        if result["last_success"]:
            ls = result["last_success"]
            parts.append(
                f"**Last success:** {ls['timestamp']} (session {ls['session_id'][:8]}, branch {ls.get('branch', '?')})\n"
            )
        else:
            parts.append("**No successful operations found**\n")
        if result["changes_after"]:
            parts.append(f"**Changes after last success:** {len(result['changes_after'])}\n")
            for i, c in enumerate(result["changes_after"][:15]):
                ts = (c["timestamp"] or "?")[:19]
                branch = c.get("branch") or "?"
                parts.append(f"{i + 1}. {c['action']} at {ts} (branch: {branch})")
        return [TextContent(type="text", text="\n".join(parts))]
    except Exception as e:
        return _error_result(f"Regression error: {str(e)}")


async def _plan_links(
    plan_name: str | None = None, session_id: str | None = None, project: str | None = None
) -> list[TextContent]:
    """Query plan-linked sessions."""
    try:
        store = _get_vector_store()
        if session_id:
            ctx = store.get_session_context(session_id)
            if not ctx:
                return [TextContent(type="text", text=f"No context for session '{session_id[:8]}'.")]
            parts = [
                f"## Session {ctx['session_id'][:8]}\n",
                f"- Branch: {ctx.get('branch') or '?'}",
                f"- PR: #{ctx.get('pr_number') or '?'}",
                f"- Plan: {ctx.get('plan_name') or '(none)'}",
                f"- Phase: {ctx.get('plan_phase') or '(none)'}",
                f"- Story: {ctx.get('story_id') or '(none)'}",
            ]
            return [TextContent(type="text", text="\n".join(parts))]

        sessions = store.get_sessions_by_plan(plan_name=plan_name, project=project)
        if not sessions:
            msg = f"No sessions linked to plan '{plan_name}'." if plan_name else "No plan-linked sessions found."
            return [TextContent(type="text", text=msg)]

        title = plan_name or "All Plans"
        parts = [f"## Sessions: {title}\n"]
        for s in sessions[:30]:
            sid = (s["session_id"] or "")[:8]
            branch = s.get("branch") or "?"
            pr = f"#{s['pr_number']}" if s.get("pr_number") else ""
            phase = s.get("plan_phase") or ""
            plan = s.get("plan_name") or ""
            started = (s.get("started_at") or "")[:19]
            parts.append(f"- {sid} | {plan}/{phase} | {branch} {pr} | {started}")
        stats = store.get_plan_linking_stats()
        parts.append(f"\nTotal: {stats['linked_sessions']}/{stats['total_sessions']} linked")
        return [TextContent(type="text", text="\n".join(parts))]
    except Exception as e:
        return _error_result(f"Plan links error: {str(e)}")


async def _think(context: str, project: str | None = None, max_results: int = 10):
    """Execute think -- retrieve relevant memories for current task."""
    try:
        from ..engine import think

        store = _get_vector_store()
        model = _get_embedding_model()
        loop = asyncio.get_running_loop()

        def _embed(text: str) -> list[float]:
            return model.embed_query(text)

        normalized_project = _normalize_project_name(project)
        result = await loop.run_in_executor(
            None,
            lambda: think(
                context=context, store=store, embed_fn=_embed, project=normalized_project, max_results=max_results
            ),
        )
        structured = {
            "query": result.query,
            "total": result.total,
            "decisions": [_memory_to_dict(i) for i in result.decisions],
            "patterns": [_memory_to_dict(i) for i in result.patterns],
            "bugs": [_memory_to_dict(i) for i in result.bugs],
            "context": [_memory_to_dict(i) for i in result.context],
        }
        return ([TextContent(type="text", text=result.format())], structured)
    except Exception as e:
        return _error_result(f"Think error: {str(e)}")


async def _recall(
    file_path: str | None = None, topic: str | None = None, project: str | None = None, max_results: int = 10
):
    """Execute recall -- proactive context retrieval."""
    try:
        from ..engine import recall

        store = _get_vector_store()
        model = _get_embedding_model()
        normalized_project = _normalize_project_name(project)
        loop = asyncio.get_running_loop()

        def _embed(text: str) -> list[float]:
            return model.embed_query(text)

        result = await loop.run_in_executor(
            None,
            lambda: recall(
                store=store,
                embed_fn=_embed,
                file_path=file_path,
                topic=topic,
                project=normalized_project,
                max_results=max_results,
            ),
        )
        structured = {
            "target": result.target,
            "file_history": [
                {
                    "timestamp": (h.get("timestamp") or "")[:19],
                    "action": h.get("action", ""),
                    "session_id": h.get("session_id", ""),
                    "file_path": h.get("file_path", ""),
                }
                for h in result.file_history
            ],
            "related_chunks": [_memory_to_dict(c) for c in result.related_chunks],
            "session_summaries": [
                {
                    "session_id": s.get("session_id", ""),
                    "branch": s.get("branch", ""),
                    "plan_name": s.get("plan_name", ""),
                    "started_at": (s.get("started_at") or "")[:19],
                }
                for s in result.session_summaries
            ],
        }
        return ([TextContent(type="text", text=result.format())], structured)
    except Exception as e:
        return _error_result(f"Recall error: {str(e)}")


async def _sessions(project: str | None = None, days: int = 7, limit: int = 20) -> list[TextContent]:
    """List recent sessions."""
    try:
        from ..engine import format_sessions, sessions

        store = _get_vector_store()
        normalized_project = _normalize_project_name(project)
        result = sessions(store=store, project=normalized_project, days=days, limit=limit)
        return [TextContent(type="text", text=format_sessions(result, days=days))]
    except Exception as e:
        return _error_result(f"Sessions error: {str(e)}")


async def _session_summary(session_id: str):
    """Get enriched session summary."""
    try:
        store = _get_vector_store()
        enrichment = store.get_session_enrichment(session_id)
        if not enrichment:
            return [
                TextContent(
                    type="text",
                    text=f"No enrichment data for session '{session_id[:8]}...'. Run 'brainlayer enrich-sessions' first.",
                )
            ]
        parts = [f"## Session Summary: {session_id[:8]}...\n"]
        if enrichment.get("session_summary"):
            parts.append(f"**Summary:** {enrichment['session_summary']}\n")
        if enrichment.get("primary_intent"):
            parts.append(f"**Intent:** {enrichment['primary_intent']}")
        if enrichment.get("outcome"):
            parts.append(f"**Outcome:** {enrichment['outcome']}")
        if enrichment.get("session_quality_score"):
            parts.append(f"**Quality:** {enrichment['session_quality_score']}/10")
        if enrichment.get("complexity_score"):
            parts.append(f"**Complexity:** {enrichment['complexity_score']}/10")
        if enrichment.get("duration_seconds"):
            mins = enrichment["duration_seconds"] // 60
            parts.append(f"**Duration:** {mins} min")
        parts.append(
            f"**Messages:** {enrichment.get('message_count', 0)} (user: {enrichment.get('user_message_count', 0)}, assistant: {enrichment.get('assistant_message_count', 0)})\n"
        )
        if enrichment.get("decisions_made"):
            parts.append("### Decisions")
            for d in enrichment["decisions_made"]:
                if isinstance(d, dict):
                    parts.append(f"- {d.get('decision', '?')} -- *{d.get('rationale', '')}*")
                else:
                    parts.append(f"- {d}")
        if enrichment.get("corrections"):
            parts.append("\n### Corrections")
            for c in enrichment["corrections"]:
                if isinstance(c, dict):
                    parts.append(f"- Wrong: {c.get('what_was_wrong', '?')} -> Wanted: {c.get('what_user_wanted', '?')}")
                else:
                    parts.append(f"- {c}")
        if enrichment.get("learnings"):
            parts.append("\n### Learnings")
            for l in enrichment["learnings"]:
                parts.append(f"- {l}")
        if enrichment.get("mistakes"):
            parts.append("\n### Mistakes")
            for m in enrichment["mistakes"]:
                parts.append(f"- {m}")
        if enrichment.get("what_worked"):
            parts.append(f"\n**What worked:** {enrichment['what_worked']}")
        if enrichment.get("what_failed"):
            parts.append(f"**What failed:** {enrichment['what_failed']}")
        if enrichment.get("topic_tags"):
            parts.append(f"\n**Tags:** {', '.join(enrichment['topic_tags'][:10])}")
        return [TextContent(type="text", text="\n".join(parts))]
    except Exception as e:
        return _error_result(f"Session summary error: {str(e)}")


async def _current_context(hours: int = 24):
    """Lightweight session awareness."""
    try:
        from ..engine import current_context

        store = _get_vector_store()
        result = current_context(store=store, hours=hours)
        structured = {
            "active_projects": result.active_projects,
            "active_branches": result.active_branches,
            "active_plan": result.active_plan,
            "recent_files": result.recent_files,
            "recent_sessions": [
                {
                    "session_id": s.session_id,
                    "project": s.project,
                    "branch": s.branch,
                    "started_at": s.started_at[:19] if s.started_at else "",
                    "plan_name": s.plan_name,
                }
                for s in result.recent_sessions
            ],
        }
        return ([TextContent(type="text", text=result.format())], structured)
    except Exception as e:
        return _error_result(f"Current context error: {str(e)}")
