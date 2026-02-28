"""Search and recall MCP handlers."""

import asyncio

from mcp.types import TextContent

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
):
    """Unified search dispatcher -- routes to the right internal handler."""

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
        )

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
        format=format,
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


async def _brain_recall(
    mode: str | None = None,
    project: str | None = None,
    hours: int = 24,
    days: int = 7,
    limit: int = 20,
    session_id: str | None = None,
    plan_name: str | None = None,
):
    """Unified recall dispatcher -- routes to session/context handlers."""
    resolved_mode = mode or _infer_recall_mode(
        {
            "session_id": session_id,
            "plan_name": plan_name,
            "days": days if days != 7 else None,
            "limit": limit if limit != 20 else None,
        }
    )

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
    # Backward compat: accept old 'format' kwarg
    format: str | None = None,
):
    """Execute a hybrid search query (semantic + keyword via RRF)."""
    try:
        # Backward compat: old 'format' kwarg overrides 'detail'
        if format is not None:
            detail = format

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
            source_filter = None  # AIDEV-NOTE: was "claude_code" — excluded brain_store ("manual") chunks. Default to all sources.

        if entity_id and not source:
            source_filter = None

        results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text=query,
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
        )

        if not results["documents"][0]:
            empty = {"query": query, "total": 0, "results": []}
            return ([TextContent(type="text", text="No results found.")], empty)

        results = store.enrich_results_with_session_context(results)

        if detail == "compact":
            structured_results = []
            for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                score = 1 - dist if dist is not None else 0
                item = _build_compact_result(
                    {
                        "score": round(score, 4),
                        "chunk_id": meta.get("chunk_id"),
                        "project": _normalize_project_name(meta.get("project")) or meta.get("project", "unknown"),
                        "content": doc,
                        "source_file": meta.get("source_file", "unknown"),
                        "date": meta.get("created_at", "")[:10] if meta.get("created_at") else None,
                        "importance": meta.get("importance"),
                        "summary": meta.get("summary"),
                    }
                )
                structured_results.append(item)
            structured = {"query": query, "total": len(structured_results), "results": structured_results}
            return ([], structured)

        output_parts = [f"## Search Results for: {query}\n"]
        structured_results = []

        for i, (doc, meta, dist) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
        ):
            score = 1 - dist if dist is not None else 0
            item = {
                "score": round(score, 4),
                "project": _normalize_project_name(meta.get("project")) or meta.get("project", "unknown"),
                "content_type": meta.get("content_type", "unknown"),
                "content": doc[:1000],
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
            if meta.get("chunk_id"):
                item["chunk_id"] = meta["chunk_id"]
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
            output_parts.append(doc[:1000] + ("..." if len(doc) > 1000 else ""))
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
        output = f"""## BrainLayer Knowledge Base Stats\n\n- **Total Chunks:** {stats["total_chunks"]}\n- **Projects:** {", ".join(stats["projects"][:15])}{"..." if len(stats["projects"]) > 15 else ""}\n- **Content Types:** {", ".join(stats["content_types"])}\n"""
        structured = {
            "total_chunks": stats["total_chunks"],
            "projects": stats["projects"],
            "content_types": stats["content_types"],
        }
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
