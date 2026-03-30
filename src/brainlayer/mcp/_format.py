"""Beautiful terminal output formatting for MCP tool responses.

Uses Unicode box-drawing characters for clean, professional display.
No ANSI color codes (MCP tool output doesn't support them in Claude Code).
"""


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


def format_search_results(query: str, results: list[dict], total: int) -> str:
    """Format search results as a clean table.

    Each result dict should have: chunk_id, score, project, date, snippet, summary, importance.
    """
    if total == 0:
        return f'\u250c\u2500 brain_search: "{_truncate(query, 50)}"\n\u2502 No results found.\n\u2514\u2500'

    lines = []
    lines.append(
        f'\u250c\u2500 brain_search: "{_truncate(query, 50)}" \u2500 {total} result{"s" if total != 1 else ""}'
    )
    lines.append("\u2502")

    for i, r in enumerate(results):
        score = r.get("score", 0)
        chunk_id = (r.get("chunk_id") or "")[:12]
        project = _truncate(r.get("project") or "", 16)
        date = (r.get("date") or "")[:10]
        importance = r.get("importance")
        summary = r.get("summary") or ""
        snippet = r.get("snippet") or ""

        # Use summary if available, fall back to snippet
        display_text = _truncate(summary or snippet, 72)

        imp_str = f"{int(importance):2d}" if importance is not None else " \u2500"
        score_str = f"{score:.2f}" if score else "0.00"

        lines.append(f"\u251c\u2500 [{i + 1}] {chunk_id}  score:{score_str}  imp:{imp_str}  {date}")
        lines.append(f"\u2502  {_pad(project, 16)} \u2502 {display_text}")
        if r.get("tags") and isinstance(r["tags"], list):
            tag_str = ", ".join(str(t) for t in r["tags"][:4])
            lines.append(f"\u2502  tags: {tag_str}")
        lines.append("\u2502")

    lines.append("\u2514\u2500")
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
    eid = entity.get("entity_id") or entity.get("id", "")
    entity_type = entity.get("entity_type") or (entity.get("profile", {}).get("entity_type", ""))

    lines = []
    lines.append(f"\u250c\u2500 Entity: {name}")
    lines.append(f"\u2502 id: {eid}  type: {entity_type or 'unknown'}")

    # Profile / metadata
    profile = entity.get("profile", {})
    if profile:
        # Show select profile fields
        for key in ("role", "company", "location", "email", "phone"):
            if profile.get(key):
                lines.append(f"\u2502 {key}: {profile[key]}")

    # Hard constraints
    constraints = entity.get("hard_constraints", {})
    if constraints:
        lines.append("\u251c\u2500 Constraints")
        for k, v in list(constraints.items())[:5]:
            lines.append(f"\u2502   {k}: {v}")

    # Preferences
    prefs = entity.get("preferences", {})
    if prefs:
        lines.append("\u251c\u2500 Preferences")
        for k, v in list(prefs.items())[:5]:
            lines.append(f"\u2502   {k}: {v}")

    # Contact info
    contact = entity.get("contact_info", {})
    if contact:
        lines.append("\u251c\u2500 Contact")
        for k, v in list(contact.items())[:5]:
            lines.append(f"\u2502   {k}: {v}")

    # Relations
    relations = entity.get("relations", [])
    if relations:
        lines.append(f"\u251c\u2500 Relations ({len(relations)})")
        for rel in relations[:8]:
            if isinstance(rel, dict):
                rtype = rel.get("relation_type", "")
                target = (
                    rel.get("target", {}).get("name", "")
                    if isinstance(rel.get("target"), dict)
                    else str(rel.get("target", ""))
                )
                lines.append(f"\u2502   \u2192 {rtype}: {target}")
            else:
                lines.append(f"\u2502   \u2192 {rel}")

    # Memories
    memories = entity.get("memories", [])
    mem_count = entity.get("memory_count", len(memories))
    if memories:
        lines.append(f"\u251c\u2500 Memories ({mem_count})")
        for mem in memories[:5]:
            mtype = mem.get("type") or ""
            mdate = (mem.get("date") or "")[:10]
            mcontent = _truncate(mem.get("content") or mem.get("summary") or "", 60)
            lines.append(f"\u2502   [{mtype:8s}] {mdate} {mcontent}")

    lines.append("\u2514\u2500")
    return "\n".join(lines)


def format_entity_simple(entity: dict) -> str:
    """Format a simple entity lookup result (from _brain_entity)."""
    if not entity:
        return ""

    name = entity.get("name", "Unknown")
    eid = entity.get("id", "")
    etype = entity.get("entity_type", "")

    lines = []
    lines.append(f"\u250c\u2500 Entity: {name}")
    lines.append(f"\u2502 id: {eid}  type: {etype or 'unknown'}")

    # Relations from entity_lookup result
    relations = entity.get("relations", [])
    if relations:
        lines.append(f"\u251c\u2500 Relations ({len(relations)})")
        for rel in relations[:8]:
            if isinstance(rel, dict):
                rtype = rel.get("relation_type", "related_to")
                target = rel.get("target_name", rel.get("name", ""))
                lines.append(f"\u2502   \u2192 {rtype}: {target}")

    # Chunks / memories
    chunks = entity.get("chunks", [])
    if chunks:
        lines.append(f"\u251c\u2500 Associated memories ({len(chunks)})")
        for c in chunks[:5]:
            snippet = _truncate(c.get("content", ""), 60)
            lines.append(f"\u2502   {snippet}")

    # Metadata
    metadata = entity.get("metadata", {})
    if metadata and isinstance(metadata, dict):
        interesting = {k: v for k, v in metadata.items() if v and k not in ("id", "name", "entity_type")}
        if interesting:
            lines.append("\u251c\u2500 Metadata")
            for k, v in list(interesting.items())[:5]:
                val = _truncate(str(v), 50) if isinstance(v, str) else str(v)
                lines.append(f"\u2502   {k}: {val}")

    lines.append("\u2514\u2500")
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
    lines = []
    total = len(results)
    lines.append(
        f'\u250c\u2500 Entity search: "{entity_name}" (query: "{_truncate(query, 40)}") \u2500 {total} result{"s" if total != 1 else ""}'
    )

    if facts:
        lines.append(f"\u251c\u2500 Knowledge Graph ({len(facts)} fact{'s' if len(facts) != 1 else ''})")
        for f in facts[:5]:
            src = f.get("source", "")
            rel = f.get("relation", "")
            tgt = f.get("target", "")
            lines.append(f"\u2502   {src} \u2500[{rel}]\u2192 {tgt}")
        lines.append("\u2502")

    if results:
        lines.append(f"\u251c\u2500 Memories ({total})")
        for i, r in enumerate(results):
            score = r.get("score", 0)
            chunk_id = (r.get("chunk_id") or "")[:12]
            snippet = _truncate(r.get("snippet") or r.get("content", ""), 60)
            score_str = f"{score:.2f}" if score else "0.00"
            lines.append(f"\u2502 [{i + 1}] {chunk_id}  score:{score_str}")
            lines.append(f"\u2502     {snippet}")

    lines.append("\u2514\u2500")
    return "\n".join(lines)
