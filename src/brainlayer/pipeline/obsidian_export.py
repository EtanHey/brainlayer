"""Obsidian Export Pipeline — Generate Obsidian vault from BrainLayer data.

Phase 9: Export enriched knowledge graph as Obsidian-compatible
markdown notes with YAML frontmatter and wikilinks.

Usage:
    from brainlayer.pipeline.obsidian_export import export_obsidian
    export_obsidian(vector_store, vault_path="~/.brainlayer-brain")
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_VAULT = Path.home() / ".brainlayer-brain" / "BrainLayer"


def _sanitize_filename(name: str) -> str:
    """Make a string safe for use as a filename."""
    # Strip common extensions to avoid double .md.md
    for ext in (".md", ".ts", ".tsx", ".js", ".py", ".json",
                ".css", ".svg", ".txt", ".yaml", ".toml"):
        if name.endswith(ext):
            name = name[:-len(ext)]
            break
    bad = '<>:"/\\|?*'
    for c in bad:
        name = name.replace(c, "-")
    name = name.strip(". ")
    # Clean up ugly directory names
    if name == "__tests__":
        name = "tests"
    return name or "unnamed"


def _format_date(ts: Optional[str]) -> str:
    """Extract YYYY-MM-DD from ISO timestamp."""
    if not ts:
        return "unknown"
    return ts[:10]


def _session_title(ctx: Dict[str, Any]) -> str:
    """Generate a human-readable session title."""
    date = _format_date(ctx.get("started_at"))
    plan = ctx.get("plan_name") or ""
    phase = ctx.get("plan_phase") or ""
    sid = (ctx.get("session_id") or "")[:8]

    # Use plan+phase if available, otherwise branch
    if plan and phase:
        label = f"{plan}-{phase}"
    elif plan:
        label = plan
    else:
        branch = ctx.get("branch") or ""
        label = branch.replace("feature/", "").replace(
            "llm-", ""
        ).replace("componentize-", "")

    if label:
        return f"{date} {_sanitize_filename(label)} ({sid})"
    return f"{date} session ({sid})"


def generate_session_note(
    ctx: Dict[str, Any],
    operations: List[Dict[str, Any]],
    files: List[str],
) -> str:
    """Generate a session note with YAML frontmatter."""
    date = _format_date(ctx.get("started_at"))
    sid = ctx.get("session_id", "")[:8]
    branch = ctx.get("branch") or ""
    pr = ctx.get("pr_number")
    plan = ctx.get("plan_name") or ""
    phase = ctx.get("plan_phase") or ""

    # Build tags from operations
    op_types = list({
        o.get("operation_type", "unknown")
        for o in operations
    })

    # YAML frontmatter
    lines = [
        "---",
        f"date: {date}",
        f"session_id: {sid}",
    ]
    if branch:
        lines.append(f"branch: {branch}")
    if pr:
        lines.append(f"pr: {pr}")
    if plan:
        lines.append(f"plan: {plan}")
    if phase:
        lines.append(f"phase: {phase}")
    if files:
        file_list = ", ".join(
            f.split("/")[-1] for f in files[:20]
        )
        lines.append(f"files: [{file_list}]")
    if op_types:
        tags = ", ".join(op_types)
        lines.append(f"tags: [{tags}]")
    lines.append(f"operations: {len(operations)}")
    lines.append("---")
    lines.append("")

    # Title
    title_parts = []
    if plan:
        title_parts.append(plan)
    if phase:
        title_parts.append(phase)
    if branch:
        title_parts.append(
            branch.replace("feature/", "")
        )
    title = " / ".join(title_parts) if title_parts else f"Session {sid}"
    lines.append(f"# {title}")
    lines.append("")

    # Operations (limit to 50 most significant)
    if operations:
        # Prioritize non-research ops, then by chunk count
        sorted_ops = sorted(
            operations,
            key=lambda o: (
                o.get("operation_type") == "research",
                -len(o.get("chunk_ids") or []),
            ),
        )
        display_ops = sorted_ops[:50]
        lines.append(
            f"## Operations ({len(operations)} total,"
            f" showing {len(display_ops)})"
        )
        lines.append("")
        for i, op in enumerate(display_ops, 1):
            otype = op.get("operation_type", "?")
            summary = op.get("summary", "")
            outcome = op.get("outcome", "")
            icon = {
                "research": "🔍",
                "edit-cycle": "✏️",
                "feature-cycle": "🚀",
                "debug": "🐛",
                "review": "👀",
            }.get(otype, "📋")
            line = f"{i}. {icon} **{otype}**"
            if summary:
                line += f" — {summary}"
            if outcome:
                line += f" [{outcome}]"
            lines.append(line)
        lines.append("")

    # Related files
    if files:
        lines.append("## Files Touched")
        lines.append("")
        for f in files[:30]:
            fname = f.split("/")[-1]
            lines.append(f"- [[{fname}]]")
        lines.append("")

    # Related links
    lines.append("## Related")
    lines.append("")
    if pr:
        lines.append(f"- [[PR-{pr}]]")
    if plan:
        lines.append(f"- [[{plan}]]")
    lines.append("")

    return "\n".join(lines)


def generate_file_note(
    file_path: str,
    interactions: List[Dict[str, Any]],
) -> str:
    """Generate a file note with interaction timeline."""
    fname = file_path.split("/")[-1]
    last_ts = ""
    if interactions:
        last_ts = _format_date(interactions[-1].get("timestamp"))

    lines = [
        "---",
        f"file: {file_path}",
        f"interactions: {len(interactions)}",
        f"last_modified: {last_ts}",
        "---",
        "",
        f"# {fname}",
        "",
        f"Full path: `{file_path}`",
        "",
        "## Interaction Timeline",
        "",
        "| Date | Action | Branch | Session |",
        "|------|--------|--------|---------|",
    ]

    seen: set = set()
    for i in interactions[:50]:
        ts = _format_date(i.get("timestamp"))
        action = i.get("action", "?")
        branch = i.get("branch") or ""
        sid = (i.get("session_id") or "")[:8]
        key = (ts, action, sid)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"| {ts} | {action} | {branch} | {sid} |")

    lines.append("")

    # Dataview query for sessions that touched this file
    lines.append("## Sessions (Dataview)")
    lines.append("")
    lines.append("```dataview")
    lines.append("TABLE date, tags, plan, phase")
    lines.append('FROM "Sessions"')
    lines.append(f'WHERE contains(files, "{fname}")')
    lines.append("SORT date DESC")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def generate_plan_note(
    plan_name: str,
    sessions: List[Dict[str, Any]],
) -> str:
    """Generate a plan note."""
    lines = [
        "---",
        f"plan: {plan_name}",
        f"sessions: {len(sessions)}",
        "---",
        "",
        f"# {plan_name}",
        "",
        "## Sessions",
        "",
        "| Date | Phase | Branch | Session |",
        "|------|-------|--------|---------|",
    ]

    for s in sessions:
        date = _format_date(s.get("started_at"))
        phase = s.get("plan_phase") or ""
        branch = s.get("branch") or ""
        title = _session_title(s)
        lines.append(
            f"| {date} | {phase} | {branch}"
            f" | [[{title}]] |"
        )

    lines.append("")

    # Dataview query
    lines.append("## All Sessions (Dataview)")
    lines.append("")
    lines.append("```dataview")
    lines.append("TABLE date, phase, branch, tags")
    lines.append('FROM "Sessions"')
    lines.append(f'WHERE plan = "{plan_name}"')
    lines.append("SORT date ASC")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def generate_dashboard(dashboard_type: str) -> str:
    """Generate a Dataview dashboard note."""
    if dashboard_type == "recent":
        return "\n".join([
            "---",
            "dashboard: recent-sessions",
            "---",
            "",
            "# Recent Sessions",
            "",
            "```dataview",
            "TABLE date, tags, plan, phase, operations",
            'FROM "Sessions"',
            "WHERE date >= date(today) - dur(7 days)",
            "SORT date DESC",
            "```",
            "",
        ])
    elif dashboard_type == "files":
        return "\n".join([
            "---",
            "dashboard: most-modified",
            "---",
            "",
            "# Most Modified Files",
            "",
            "```dataview",
            "TABLE interactions, last_modified",
            'FROM "Files"',
            "SORT interactions DESC",
            "LIMIT 20",
            "```",
            "",
        ])
    elif dashboard_type == "plans":
        return "\n".join([
            "---",
            "dashboard: plans",
            "---",
            "",
            "# Plans Overview",
            "",
            "```dataview",
            "TABLE sessions, plan",
            'FROM "Plans"',
            "SORT sessions DESC",
            "```",
            "",
        ])
    return ""


def export_obsidian(
    vector_store: Any,
    vault_path: Optional[str] = None,
    project: Optional[str] = None,
    force: bool = False,
) -> Dict[str, int]:
    """Export BrainLayer data to Obsidian vault.

    Args:
        vector_store: VectorStore instance
        vault_path: Path to Obsidian vault root
        project: Filter to specific project
        force: Overwrite existing notes

    Returns:
        Dict with counts of generated notes
    """
    vault = Path(vault_path) if vault_path else DEFAULT_VAULT
    counts = {
        "sessions": 0,
        "files": 0,
        "plans": 0,
        "dashboards": 0,
    }

    # Create directory structure
    dirs = ["Sessions", "Files", "Plans", "Dashboards"]
    for d in dirs:
        (vault / d).mkdir(parents=True, exist_ok=True)

    # 1. Export session notes
    cursor = vector_store.conn.cursor()
    query = "SELECT session_id FROM session_context"
    params: list = []
    if project:
        query += " WHERE project = ?"
        params.append(project)
    query += " ORDER BY started_at ASC"

    session_ids = [
        r[0] for r in cursor.execute(query, params)
    ]
    logger.info(
        "Exporting %d sessions", len(session_ids)
    )

    for sid in session_ids:
        ctx = vector_store.get_session_context(sid)
        if not ctx:
            continue

        title = _session_title(ctx)
        note_path = vault / "Sessions" / f"{title}.md"
        if note_path.exists() and not force:
            continue

        # Get operations for this session
        ops = vector_store.get_session_operations(sid)

        # Get files touched
        file_interactions = list(cursor.execute(
            "SELECT DISTINCT file_path FROM file_interactions"
            " WHERE session_id = ?",
            (sid,),
        ))
        files = [r[0] for r in file_interactions]

        content = generate_session_note(ctx, ops, files)
        note_path.write_text(content)
        counts["sessions"] += 1

    # 2. Export file notes (top 100 most-interacted files)
    file_query = (
        "SELECT file_path, COUNT(*) as cnt"
        " FROM file_interactions"
    )
    file_params: list = []
    if project:
        file_query += " WHERE project = ?"
        file_params.append(project)
    file_query += " GROUP BY file_path ORDER BY cnt DESC LIMIT 100"

    top_files = list(
        cursor.execute(file_query, file_params)
    )
    logger.info(
        "Exporting %d file notes", len(top_files)
    )

    for row in top_files:
        fp = row[0]
        fname = _sanitize_filename(fp.split("/")[-1])
        note_path = vault / "Files" / f"{fname}.md"
        if note_path.exists() and not force:
            continue

        timeline = vector_store.get_file_timeline(
            fp, project=project, limit=50
        )
        content = generate_file_note(fp, timeline)
        note_path.write_text(content)
        counts["files"] += 1

    # 3. Export plan notes
    plan_stats = vector_store.get_plan_linking_stats()
    for plan_name in plan_stats.get("plans", {}):
        note_path = vault / "Plans" / f"{plan_name}.md"
        if note_path.exists() and not force:
            continue

        sessions = vector_store.get_sessions_by_plan(
            plan_name=plan_name, project=project
        )
        content = generate_plan_note(plan_name, sessions)
        note_path.write_text(content)
        counts["plans"] += 1

    # 4. Generate dashboards
    dashboards = {
        "Recent Sessions": "recent",
        "Most Modified Files": "files",
        "Plans Overview": "plans",
    }
    for name, dtype in dashboards.items():
        note_path = vault / "Dashboards" / f"{name}.md"
        if note_path.exists() and not force:
            continue
        content = generate_dashboard(dtype)
        if content:
            note_path.write_text(content)
            counts["dashboards"] += 1

    return counts
