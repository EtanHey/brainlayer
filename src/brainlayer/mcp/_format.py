"""Beautiful terminal output formatting for MCP tool responses.

Uses Unicode box-drawing characters for clean, professional display.
No ANSI color codes (MCP tool output doesn't support them in Claude Code).
"""

import os


def _truncate(text: str, max_len: int = 80) -> str:
    """Truncate text with ellipsis."""
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "\u2026"


def _pad(text: str, width: int, align: str = "left") -> str:
    """Pad text to width with alignment."""
    text = str(text) if text is not None else ""
    if len(text) > width:
        text = text[: width - 1] + "\u2026"
    if align == "right":
        return text.rjust(width)
    elif align == "center":
        return text.center(width)
    return text.ljust(width)


def _basename(value: str | None) -> str:
    if not value:
        return "unknown"
    return os.path.basename(str(value).strip()) or str(value).strip() or "unknown"


def _date_only(value: str | None) -> str:
    if not value:
        return "unknown"
    return str(value)[:10]


def _relation_target(rel: dict) -> str:
    target = rel.get("target")
    if isinstance(target, dict):
        return target.get("name", "")
    return rel.get("target_name") or rel.get("name") or target or ""


def _expired_date(rel: dict) -> str | None:
    raw = rel.get("expired_at") or rel.get("expiredAt")
    return str(raw)[:10] if raw else None


def format_search_results(query: str, results: list[dict], total: int) -> str:
    """Format search results as labeled-field markdown.

    Each result dict should have: chunk_id, score, project, date, snippet, summary, importance.
    """
    if total == 0:
        return f'## Search results for "{_truncate(query, 50)}" - 0 of 0 shown\n\nNo results found.'

    lines = []
    lines.append(f'## Search results for "{_truncate(query, 50)}" - {len(results)} of {total} shown')

    for i, r in enumerate(results):
        summary = r.get("summary") or ""
        snippet = r.get("snippet") or r.get("content") or ""
        title = _truncate(summary or snippet or "Untitled result", 100)
        source = _basename(r.get("source_file") or r.get("project"))
        date = _date_only(r.get("date") or r.get("created_at"))
        preview = _truncate(snippet or summary, 200)

        lines.append("")
        lines.append(f"### {i + 1}. {title}")
        lines.append(f"- Source: {source}")
        lines.append(f"- Date: {date}")
        lines.append(f"- Preview: {preview}")

    return "\n".join(lines)


def format_recalled_context(query: str, chunks: list[dict]) -> str:
    """Format recalled chunk context as labeled markdown."""
    lines = [f'## Recalled context for "{_truncate(query, 80)}"']
    if not chunks:
        lines.extend(["", "No context available."])
        return "\n".join(lines)

    for index, chunk in enumerate(chunks, start=1):
        source = _basename(chunk.get("source_file") or chunk.get("project"))
        content = (chunk.get("content") or chunk.get("snippet") or chunk.get("summary") or "").strip()
        lines.append("")
        lines.append(f"### Chunk {index} - {source}")
        if len(content) <= 1500:
            lines.append(content)
        else:
            lines.append(content[:1500] + "...")
            if chunk.get("chunk_id"):
                lines.append("")
                lines.append(f"Reference: {chunk['chunk_id']}")
    return "\n".join(lines)


def format_store_result(chunk_id: str, superseded: str | None = None, queued: bool = False) -> str:
    """Format store confirmation as a clean one-liner."""
    if queued:
        return "\u2502 \u23f3 Memory queued (DB busy) \u2500 will flush on next successful store."

    parts = [f"\u2714 Stored \u2192 {chunk_id}"]
    if superseded:
        parts.append(f" (superseded {superseded})")
    return "".join(parts)


def format_entity_card(entity: dict) -> str:
    """Format entity lookup as a structured card.

    entity dict may have: entity_id, name, profile, relations, memories, etc.
    """
    name = entity.get("name", "Unknown")
    lines = [f"## Entity: {name}"]
    if entity.get("description"):
        lines.extend(["", _truncate(entity.get("description"), 200)])

    lines.extend(["", "### KG Facts"])
    relations = entity.get("relations", [])
    if relations:
        for rel in relations[:8]:
            if isinstance(rel, dict):
                rtype = rel.get("relation_type", "")
                line = f"- {rtype}: {_relation_target(rel)}"
                if expired := _expired_date(rel):
                    line += f" (expired {expired})"
                lines.append(line)
            else:
                lines.append(f"- {rel}")
    else:
        lines.append("- None")

    lines.extend(["", "### Recent context"])
    memories = entity.get("memories", [])
    if memories:
        for mem in memories[:5]:
            mcontent = _truncate(mem.get("content") or mem.get("summary") or "", 60)
            lines.append(f"- {mcontent}")
    else:
        lines.append("- None")

    lines.extend(["", "### Likely follow-ups"])
    followups = [_relation_target(rel) for rel in relations if isinstance(rel, dict) and _relation_target(rel)]
    lines.extend(f"- {target}" for target in followups[:5])
    if not followups:
        lines.append("- None")
    return "\n".join(lines)


def format_entity_simple(entity: dict) -> str:
    """Format a simple entity lookup result (from _brain_entity)."""
    if not entity:
        return ""

    name = entity.get("name", "Unknown")
    lines = [f"## Entity: {name}"]

    # Description from metadata or entity
    desc = entity.get("description") or ""
    if not desc:
        metadata = entity.get("metadata", {})
        if isinstance(metadata, dict):
            desc = metadata.get("description", "")
    if desc:
        lines.extend(["", _truncate(desc, 200)])

    # Relations from entity_lookup result (filter co_occurs_with)
    lines.extend(["", "### KG Facts"])
    relations = entity.get("relations", [])
    semantic_rels = [r for r in relations if isinstance(r, dict) and r.get("relation_type") != "co_occurs_with"]
    if semantic_rels:
        for rel in semantic_rels[:10]:
            rtype = rel.get("relation_type", "related_to")
            line = f"- {rtype}: {_relation_target(rel)}"
            if expired := _expired_date(rel):
                line += f" (expired {expired})"
            lines.append(line)
    else:
        lines.append("- None")

    # Chunks / memories
    lines.extend(["", "### Recent context"])
    chunks = entity.get("chunks", [])
    if chunks:
        for c in chunks[:5]:
            snippet = _truncate(c.get("content", ""), 150)
            lines.append(f"- {snippet}")
    else:
        lines.append("- None")

    lines.extend(["", "### Likely follow-ups"])
    followups = [_relation_target(rel) for rel in semantic_rels if _relation_target(rel)]
    lines.extend(f"- {target}" for target in followups[:5])
    if not followups:
        lines.append("- None")
    return "\n".join(lines)


def format_stats(stats: dict) -> str:
    """Format knowledge base stats."""
    total = stats.get("total_chunks", 0)
    projects = stats.get("projects", [])
    types = stats.get("content_types", [])

    lines = []
    lines.append("\u250c\u2500 BrainLayer Stats")
    lines.append(f"\u2502 Chunks: {total:,}")
    lines.append(f"\u2502 Projects: {', '.join(projects[:12])}{'...' if len(projects) > 12 else ''}")
    lines.append(f"\u2502 Types: {', '.join(types)}")
    lines.append("\u2514\u2500")
    return "\n".join(lines)


def format_digest_result(result: dict) -> str:
    """Format digest/enrich result."""
    mode = result.get("mode", "digest")

    if "attempted" in result:
        attempted = result.get("attempted", 0)
        enriched = result.get("enriched", 0)
        skipped = result.get("skipped", 0)
        failed = result.get("failed", 0)
        lines = [
            "\u250c\u2500 brain_digest (enrich)",
            f"\u2502 Attempted: {attempted}  Enriched: {enriched}  Skipped: {skipped}  Failed: {failed}",
            "\u2514\u2500",
        ]
        return "\n".join(lines)

    # digest / connect mode
    # For connect mode, stats are nested inside result["stats"]
    stats = result.get("stats", {})
    chunks = result.get("chunks_created", stats.get("chunks_created", result.get("chunks", 0)))
    entities = result.get("entities_created", stats.get("entities_found", result.get("entities", 0)))
    relations = result.get("relations_created", stats.get("relations_created", result.get("relations", 0)))

    lines = [
        f"\u250c\u2500 brain_digest ({mode})",
        f"\u2502 Chunks: {chunks}  Entities: {entities}  Relations: {relations}",
    ]

    # Action items - may be nested in "extracted" for connect mode
    extracted = result.get("extracted", {})
    actions = result.get("action_items", extracted.get("action_items", []))
    if actions:
        lines.append(f"\u251c\u2500 Action items ({len(actions)})")
        for a in actions[:5]:
            if isinstance(a, dict):
                lines.append(f"\u2502   \u2022 {_truncate(a.get('description', str(a)), 60)}")
            else:
                lines.append(f"\u2502   \u2022 {_truncate(str(a), 60)}")

    lines.append("\u2514\u2500")
    return "\n".join(lines)


def format_kg_search(entity_name: str, results: list[dict], facts: list[dict], query: str) -> str:
    """Format entity-aware KG hybrid search results."""
    total = len(results)
    lines = [f'## Search results for "{_truncate(query, 50)}" - {total} of {total} shown']

    if facts:
        lines.append("")
        lines.append(f"### KG Facts for {entity_name}")
        for f in facts[:8]:
            src = f.get("source", "")
            rel = f.get("relation", "")
            tgt = f.get("target", "")
            desc = f.get("description", "")
            lines.append(f"- {src} {rel} {tgt}".strip())
            if desc:
                lines.append(f"  - {_truncate(desc, 120)}")

    for i, r in enumerate(results):
        snippet = r.get("snippet") or r.get("content", "")
        summary = r.get("summary") or snippet
        title = _truncate(summary.splitlines()[0] if summary else "Untitled result", 100)
        source = _basename(r.get("source_file") or r.get("project") or "unknown") or "unknown"
        date = _date_only(r.get("date") or r.get("created_at") or "unknown") or "unknown"
        lines.append("")
        lines.append(f"### {i + 1}. {title}")
        lines.append(f"- Source: {source}")
        lines.append(f"- Date: {date}")
        lines.append(f"- Preview: {_truncate(snippet or summary, 200)}")
    return "\n".join(lines)
