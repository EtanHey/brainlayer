#!/usr/bin/env python3
"""
BrainLayer SessionStart Hook — auto-injects project context at session start.

Queries BrainLayer's SQLite DB (FTS5) for recent decisions and milestones
relevant to the current project.

Output: plain text to stdout (injected as Claude context).
Target: <200 tokens, <500ms total.
"""

import json
import os
import re
import sqlite3
import sys
import time

# Budget: abort if we're taking too long
DEADLINE_MS = 450


def should_activate(hook_input):
    """Conditional activation gate — skip hook when not needed.

    Checks (in order, cheapest first):
    1. BRAINLAYER_HOOKS_DISABLED=1 env var → skip all BrainLayer hooks
    2. Non-interactive mode (--print) → skip context injection
    3. BRAINLAYER_HOOKS_LIGHT=1 → reduce to 2 results (overnight workers)

    Returns (activate: bool, light_mode: bool).
    """
    if os.environ.get("BRAINLAYER_HOOKS_DISABLED") == "1":
        return False, False

    if os.environ.get("CLAUDE_NON_INTERACTIVE") == "1":
        return False, False

    session_id = hook_input.get("session_id", "")
    if not session_id:
        # Sessions without IDs are rare/manual — give full context
        return True, False

    light = os.environ.get("BRAINLAYER_HOOKS_LIGHT") == "1"

    return True, light

# Map cwd basenames to BrainLayer project names where they differ
PROJECT_ALIASES = {
    "etanheyman.com": "etanheyman-com",
    "golems-dashboard": "golems-dashboard",
}

# FTS5 doesn't index the 'project' column — we search by a short name
# that appears in content/summary/tags. For monorepo packages, the
# full project name (golems-packages-coach) rarely appears in content,
# but the short name (coach) does.
SEARCH_TERM_OVERRIDES = {
    "etanheyman.com": "etanheyman",
    "etanheyman-com": "etanheyman",
    "golems-dashboard": "dashboard",
    "6pm-mini": "6pm",
    "Mehayom-app": "mehayom",
}

# Map config profile names to their actual directory basenames
# (profile name in config may differ from filesystem dir name)
_PROFILE_TO_DIR = {
    "mehayom": "Mehayom-app",
    "golems": "golems",
    "orchestrator": "orchestrator",
    "brainlayer": "brainlayer",
    "voicelayer": "voicelayer",
}

# Hardcoded fallback if config parsing fails
_SCOPED_PROJECTS_FALLBACK = {
    "Mehayom-app": "mehayom",
}

_scoped_projects_cache = None


def load_scoped_projects():
    """Load scoped projects from ~/.golems/config.yaml.

    Returns dict mapping dir basename → BrainLayer project name,
    for profiles with brainlayer.scope = project.
    Falls back to hardcoded dict if config is missing or parse fails.
    """
    global _scoped_projects_cache
    if _scoped_projects_cache is not None:
        return _scoped_projects_cache

    config_path = os.path.expanduser("~/.golems/config.yaml")
    if not os.path.exists(config_path):
        _scoped_projects_cache = _SCOPED_PROJECTS_FALLBACK
        return _scoped_projects_cache

    try:
        with open(config_path) as f:
            lines = f.readlines()

        result = {}
        in_context_profiles = False
        current_profile = None
        in_brainlayer = False

        for line in lines:
            stripped = line.rstrip()
            if not stripped or stripped.lstrip().startswith("#"):
                continue

            indent = len(line) - len(line.lstrip())
            content = stripped.strip()

            if indent == 0 and content == "contextProfiles:":
                in_context_profiles = True
                current_profile = None
                in_brainlayer = False
            elif indent == 0 and in_context_profiles:
                in_context_profiles = False
            elif not in_context_profiles:
                pass
            elif indent == 2 and content.endswith(":") and not content.startswith("-"):
                current_profile = content[:-1]
                in_brainlayer = False
            elif indent == 4 and content == "brainlayer:" and current_profile:
                in_brainlayer = True
            elif indent == 4 and in_brainlayer:
                in_brainlayer = False
            elif indent == 6 and in_brainlayer and current_profile:
                if content.startswith("scope:"):
                    val = content.split(":", 1)[1].strip().split("#")[0].strip()
                    if val == "project":
                        dir_name = _PROFILE_TO_DIR.get(current_profile, current_profile)
                        result[dir_name] = current_profile

        _scoped_projects_cache = result if result else _SCOPED_PROJECTS_FALLBACK
        return _scoped_projects_cache
    except Exception:
        _scoped_projects_cache = _SCOPED_PROJECTS_FALLBACK
        return _scoped_projects_cache

DB_PATHS = [
    os.path.expanduser("~/.local/share/zikaron/zikaron.db"),
    os.path.expanduser("~/.local/share/brainlayer/brainlayer.db"),
]


def get_db_path():
    env = os.environ.get("BRAINLAYER_DB")
    if env and os.path.exists(env):
        return env
    for p in DB_PATHS:
        if os.path.exists(p):
            return p
    return None


def get_project_info(cwd):
    """Returns (project_name, fts_search_term) from cwd.

    project_name: for display (e.g. golems-packages-coach)
    fts_search_term: for FTS5 queries (e.g. coach)
    """
    basename = os.path.basename(cwd)

    # Check if inside golems monorepo package
    parts = cwd.split("/")
    project_name = basename
    search_term = basename
    try:
        gi = parts.index("golems")
        if gi + 2 < len(parts) and parts[gi + 1] == "packages":
            pkg = parts[gi + 2]
            project_name = f"golems-packages-{pkg}"
            search_term = pkg
    except ValueError:
        pass

    if basename in PROJECT_ALIASES:
        project_name = PROJECT_ALIASES[basename]

    if project_name in SEARCH_TERM_OVERRIDES:
        search_term = SEARCH_TERM_OVERRIDES[project_name]
    elif basename in SEARCH_TERM_OVERRIDES:
        search_term = SEARCH_TERM_OVERRIDES[basename]

    return project_name, search_term


def truncate(text, max_chars=150):
    # Collapse multi-line to single line for compact output
    text = re.sub(r"\n+", " | ", text.strip())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def elapsed_ms(start):
    return (time.monotonic() - start) * 1000


def main():
    start = time.monotonic()

    try:
        hook_input = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError):
        hook_input = {}

    activate, light_mode = should_activate(hook_input)
    if not activate:
        sys.exit(0)

    cwd = hook_input.get("cwd", os.getcwd())
    project, search_term = get_project_info(cwd)

    db_path = get_db_path()
    if not db_path:
        sys.exit(0)

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=1)
        # AIDEV-NOTE: WAL already set by Python writer. Don't set on readonly — needs write lock.
        conn.execute("PRAGMA busy_timeout=1000")
        conn.execute("PRAGMA query_only=true")
    except sqlite3.Error:
        sys.exit(0)

    lines = []
    injected_chunk_ids = []
    injected_briefs = []

    # Check if this project should be scoped (containerized BrainLayer)
    basename = os.path.basename(cwd)
    scope_project = load_scoped_projects().get(basename)

    result_limit = 2 if light_mode else 5

    try:
        # Single combined query: decisions + milestones for this project
        # Uses both tags: prefix (catches brain_store'd tagged chunks)
        # and plain content search (catches mentions in conversation)
        if elapsed_ms(start) < DEADLINE_MS:
            fts_query = (
                f'(tags:"{search_term}" OR "{search_term}") '
                f'AND ("decision" OR "decided" OR "milestone" OR "shipped" OR "merged")'
            )
            if scope_project:
                # Containerized: only show this project's memories
                rows = conn.execute(
                    """
                    SELECT c.id, c.content, c.importance, c.created_at, c.content_type
                    FROM chunks_fts f
                    JOIN chunks c ON c.id = f.chunk_id
                    WHERE chunks_fts MATCH ? AND c.project = ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, scope_project, result_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT c.id, c.content, c.importance, c.created_at, c.content_type
                    FROM chunks_fts f
                    JOIN chunks c ON c.id = f.chunk_id
                    WHERE chunks_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, result_limit),
                ).fetchall()

            if rows:
                lines.append(f"[BrainLayer] Recent context for {project}:")
                for chunk_id, content, importance, created_at, content_type in rows:
                    date = created_at[:10] if created_at else "?"
                    imp = f" imp:{importance:.0f}" if importance else ""
                    lines.append(f"- [{date}{imp}] {truncate(content)}")
                    injected_chunk_ids.append(chunk_id)
                    injected_briefs.append(truncate(content, max_chars=80))

    except sqlite3.Error:
        pass
    finally:
        conn.close()

    # Hebrew voice calibration for projects with Hebrew content
    # Source: Phase 2+3 session mining — persistent Hebrew style violations
    HEBREW_PROJECTS = {"coach", "mehayom", "Mehayom-app", "golems-packages-coach"}
    if project in HEBREW_PROJECTS or search_term in {"coach", "mehayom"}:
        lines.append(
            "[Hebrew Style] When writing Hebrew text: no em dashes (use hyphen-minus), "
            "verify contact gender before using gendered forms, "
            "use text-right alignment, keep sentences short and direct."
        )

    if lines:
        print("\n".join(lines))

    # Write coordination file for dedup with UserPromptSubmit hook
    session_id = hook_input.get("session_id", "")
    if session_id and injected_chunk_ids:
        try:
            from dedup_coordination import register_chunks

            token_estimate = sum(len(c) // 4 for c in injected_briefs)
            register_chunks(
                session_id=session_id,
                chunk_ids=injected_chunk_ids,
                source_hook="SessionStart",
                briefs=injected_briefs,
                token_estimate=token_estimate,
            )
        except Exception:
            pass  # Coordination is best-effort — never block session start

    sys.exit(0)


if __name__ == "__main__":
    main()
