"""Plan Linking Pipeline — Link sessions to active plans.

Phase 8c: Match sessions to plans by branch name and PR number.
Scans docs/plan/*/README.md progress tables to build mappings.

Usage:
    from brainlayer.pipeline.plan_linking import run_plan_linking
    run_plan_linking(vector_store, repo_root="/path/to/your-project")
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def parse_plan_readme(
    plan_dir: Path,
) -> List[Dict[str, Any]]:
    """Parse a plan README.md to extract branch/PR mappings.

    Looks for:
    - Markdown table rows with PR numbers (#NNN)
    - Branch names in backticks or table cells
    - Phase headers with branch names

    Returns list of mapping dicts with:
        plan_name, plan_phase, branch, pr_number
    """
    readme = plan_dir / "README.md"
    if not readme.exists():
        return []

    plan_name = plan_dir.name
    text = readme.read_text()
    mappings: List[Dict[str, Any]] = []

    # Pattern 1: Phase headers with branch names
    # "### Phase N: Name — `feature/branch-name`"
    header_pattern = re.compile(
        r"###\s+Phase\s+(\d+)[^—\n]*—\s*`([^`]+)`"
    )
    for match in header_pattern.finditer(text):
        phase_num = match.group(1)
        branch = match.group(2)
        mappings.append({
            "plan_name": plan_name,
            "plan_phase": f"phase-{phase_num}",
            "branch": branch,
            "pr_number": None,
        })

    # Pattern 2: Table rows with PR numbers
    # "| N | Description | ... | #NNN |" or "| ... | `branch` | ..."
    # Match PR numbers like #121, #134, "#131, #132"
    table_row_pattern = re.compile(
        r"^\|([^|]+)\|([^|]+)\|(.+)\|$", re.MULTILINE
    )
    for match in table_row_pattern.finditer(text):
        full_row = match.group(0)

        # Skip header/separator rows
        if "---" in full_row or "Step" in full_row or "Phase" in full_row:
            if "Phase" in full_row and "|" in full_row:
                # Could be a data row like "| 1 | Phase name |..."
                cells = [c.strip() for c in full_row.split("|")]
                cells = [c for c in cells if c]
                if not cells or not cells[0].strip().isdigit():
                    continue
            else:
                continue

        cells = [c.strip() for c in full_row.split("|")]
        cells = [c for c in cells if c]  # Remove empty from leading/trailing |

        if not cells:
            continue

        # Extract PR numbers from any cell
        pr_matches = re.findall(r"#(\d+)", full_row)
        pr_numbers = [int(p) for p in pr_matches]

        # Extract branch from backtick-quoted text
        branch_match = re.search(r"`([^`]*feature/[^`]+)`", full_row)
        branch = branch_match.group(1) if branch_match else None

        # Extract phase/step info from first cell or description
        phase_str = None
        step_num = cells[0].strip() if cells else None

        # Try to get phase from folder reference [phase-N]
        folder_match = re.search(
            r"\[phase-([^\]]+)\]", full_row
        )
        if folder_match:
            phase_str = f"phase-{folder_match.group(1)}"
        elif step_num and step_num.isdigit():
            phase_str = f"step-{step_num}"

        # Description for context
        desc = cells[1].strip() if len(cells) > 1 else ""

        # Create a mapping for each PR number
        if pr_numbers:
            for pr_num in pr_numbers:
                mappings.append({
                    "plan_name": plan_name,
                    "plan_phase": phase_str or desc[:50],
                    "branch": branch,
                    "pr_number": pr_num,
                })
        elif branch:
            mappings.append({
                "plan_name": plan_name,
                "plan_phase": phase_str or desc[:50],
                "branch": branch,
                "pr_number": None,
            })

    # Deduplicate by (plan_name, branch, pr_number)
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    for m in mappings:
        key = (m["plan_name"], m.get("branch"), m.get("pr_number"))
        if key not in seen:
            seen.add(key)
            unique.append(m)

    return unique


def scan_all_plans(
    repo_root: Path,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """Scan all plan READMEs and build lookup indexes.

    Returns:
        (branch_index, pr_index) where:
        - branch_index maps branch_name → plan mapping
        - pr_index maps pr_number → plan mapping
    """
    plan_dir = repo_root / "docs" / "plan"
    if not plan_dir.exists():
        logger.warning("Plan directory not found: %s", plan_dir)
        return {}, {}

    branch_index: Dict[str, Dict[str, Any]] = {}
    pr_index: Dict[int, Dict[str, Any]] = {}

    for subdir in sorted(plan_dir.iterdir()):
        if not subdir.is_dir():
            continue
        mappings = parse_plan_readme(subdir)
        for m in mappings:
            if m.get("branch"):
                branch_index[m["branch"]] = m
            if m.get("pr_number"):
                pr_index[m["pr_number"]] = m
        if mappings:
            logger.info(
                "Plan '%s': %d mappings (%d branches, %d PRs)",
                subdir.name,
                len(mappings),
                sum(1 for m in mappings if m.get("branch")),
                sum(1 for m in mappings if m.get("pr_number")),
            )

    return branch_index, pr_index


def match_session_to_plan(
    session: Dict[str, Any],
    branch_index: Dict[str, Dict[str, Any]],
    pr_index: Dict[int, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Match a session to a plan using branch and PR number.

    Priority:
    1. Exact branch match
    2. PR number match
    3. Branch prefix match (e.g., feature/llm- → local-llm-integration)
    """
    branch = session.get("branch")
    pr_number = session.get("pr_number")

    # 1. Exact branch match
    if branch and branch in branch_index:
        return branch_index[branch]

    # 2. PR number match
    if pr_number and pr_number in pr_index:
        return pr_index[pr_number]

    # 3. Branch prefix matching for known patterns
    if branch:
        prefix_patterns = {
            "feature/componentize-": "componentize-app",
            "feature/llm-": "local-llm-integration",
            "feature/backend-": "backend-overhaul",
            "feature/phase1-": "phase-1-ship",
            "feature/phase2-": "phase-2-cloud",
            "feature/phase3-": "phase-3-teller",
            "feature/phase4-": "phase-4-tooling",
            "feature/job-search-": "job-search-command-center",
            "feature/style-card": "local-llm-integration",
        }
        for prefix, plan_name in prefix_patterns.items():
            if branch.startswith(prefix):
                # Extract phase or step from branch
                phase_match = re.search(
                    r"(?:phase|step)[- ]?(\d+|[a-z]+)",
                    branch,
                )
                phase = None
                if phase_match:
                    phase = f"step-{phase_match.group(1)}"
                return {
                    "plan_name": plan_name,
                    "plan_phase": phase,
                    "branch": branch,
                    "pr_number": pr_number,
                }

    return None


def run_plan_linking(
    vector_store: Any,
    repo_root: Optional[str] = None,
    project: Optional[str] = None,
    force: bool = False,
) -> Dict[str, int]:
    """Run plan linking for all sessions with git context.

    Args:
        vector_store: VectorStore instance
        repo_root: Path to project repo root (for plan READMEs)
        project: Filter to specific project
        force: Clear existing plan links first

    Returns:
        Dict with counts: sessions_checked, sessions_linked
    """
    if repo_root:
        root = Path(repo_root)
    else:
        root = Path(__file__).parents[5]  # pipeline → brainlayer → src → brainlayer → repo-root/

    if force:
        cleared = vector_store.clear_plan_links(project)
        if cleared:
            logger.info("Cleared %d existing plan links", cleared)

    # Scan plan READMEs (no early-exit guard — the query
    # filters for plan_name IS NULL when not forcing)
    branch_index, pr_index = scan_all_plans(root)
    logger.info(
        "Built plan index: %d branches, %d PRs",
        len(branch_index),
        len(pr_index),
    )

    if not branch_index and not pr_index:
        logger.warning("No plan mappings found")
        return {"sessions_checked": 0, "sessions_linked": 0}

    # Get all sessions with git context
    cursor = vector_store.conn.cursor()
    query = (
        "SELECT session_id, project, branch, pr_number"
        " FROM session_context"
    )
    params: list = []
    if project:
        query += " WHERE project = ?"
        params.append(project)

    if not force:
        if params:
            query += " AND plan_name IS NULL"
        else:
            query += " WHERE plan_name IS NULL"

    sessions = list(cursor.execute(query, params))
    logger.info("Checking %d sessions for plan matches", len(sessions))

    linked = 0
    for row in sessions:
        session = {
            "session_id": row[0],
            "project": row[1],
            "branch": row[2],
            "pr_number": row[3],
        }
        plan = match_session_to_plan(
            session, branch_index, pr_index
        )
        if plan:
            vector_store.update_session_plan(
                session_id=session["session_id"],
                plan_name=plan["plan_name"],
                plan_phase=plan.get("plan_phase"),
                story_id=plan.get("story_id"),
            )
            linked += 1
            logger.debug(
                "Linked session %s → %s/%s",
                session["session_id"][:8],
                plan["plan_name"],
                plan.get("plan_phase", "?"),
            )

    return {
        "sessions_checked": len(sessions),
        "sessions_linked": linked,
    }
