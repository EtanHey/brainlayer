"""Temporal Chains Pipeline — Link sessions by shared files.

Phase 8d: Build topic chains across sessions and enable
regression detection queries.

Usage:
    from brainlayer.pipeline.temporal_chains import run_temporal_chains
    run_temporal_chains(vector_store, project="my-project")
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _parse_timestamp(ts: Optional[str]) -> Optional[float]:
    """Parse ISO timestamp to epoch seconds."""
    if not ts:
        return None
    try:
        from datetime import datetime

        ts_clean = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts_clean)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def build_file_session_map(
    vector_store: Any,
    project: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build a map of file → sessions that touched it.

    Returns:
        Dict mapping file_path to list of session info dicts
        (session_id, actions, first_timestamp, last_timestamp)
    """
    cursor = vector_store.conn.cursor()
    query = """
        SELECT fi.file_path, fi.session_id,
               fi.action, fi.timestamp
        FROM file_interactions fi
    """
    params: list = []
    if project:
        query += " WHERE fi.project = ?"
        params.append(project)
    query += " ORDER BY fi.file_path, fi.timestamp"

    rows = list(cursor.execute(query, params))

    # Group by file, then by session
    file_sessions: Dict[
        str, Dict[str, Dict[str, Any]]
    ] = defaultdict(lambda: defaultdict(lambda: {
        "actions": [],
        "first_ts": None,
        "last_ts": None,
    }))

    for row in rows:
        fp, sid, action, ts = row[0], row[1], row[2], row[3]
        entry = file_sessions[fp][sid]
        if action:
            entry["actions"].append(action)
        if ts:
            if not entry["first_ts"] or ts < entry["first_ts"]:
                entry["first_ts"] = ts
            if not entry["last_ts"] or ts > entry["last_ts"]:
                entry["last_ts"] = ts

    # Convert to list format
    result: Dict[str, List[Dict[str, Any]]] = {}
    for fp, sessions in file_sessions.items():
        session_list = []
        for sid, info in sessions.items():
            session_list.append({
                "session_id": sid,
                "action_count": len(info["actions"]),
                "actions": list(set(info["actions"])),
                "first_ts": info["first_ts"],
                "last_ts": info["last_ts"],
            })
        # Sort by first timestamp
        session_list.sort(
            key=lambda x: x.get("first_ts") or ""
        )
        result[fp] = session_list

    return result


def build_topic_chains(
    file_session_map: Dict[str, List[Dict[str, Any]]],
    project: Optional[str] = None,
    min_sessions: int = 2,
) -> List[Dict[str, Any]]:
    """Build topic chains from file→session map.

    A topic chain links consecutive sessions that touched
    the same file. This reveals how work on a file evolves
    across sessions.

    Args:
        file_session_map: Output from build_file_session_map
        project: Project name for metadata
        min_sessions: Min sessions per file to create chains

    Returns:
        List of chain dicts for store_topic_chains()
    """
    chains = []

    for fp, sessions in file_session_map.items():
        if len(sessions) < min_sessions:
            continue

        # Link consecutive sessions
        for i in range(len(sessions) - 1):
            sa = sessions[i]
            sb = sessions[i + 1]

            # Calculate time delta
            ta = _parse_timestamp(sa.get("last_ts"))
            tb = _parse_timestamp(sb.get("first_ts"))
            delta_hours = None
            if ta and tb:
                delta_hours = round((tb - ta) / 3600, 2)

            # Count shared actions
            shared = len(
                set(sa["actions"]) & set(sb["actions"])
            )

            chains.append({
                "file_path": fp,
                "session_a": sa["session_id"],
                "session_b": sb["session_id"],
                "shared_actions": shared,
                "time_delta_hours": delta_hours,
                "project": project,
            })

    return chains


def run_temporal_chains(
    vector_store: Any,
    project: Optional[str] = None,
    force: bool = False,
) -> Dict[str, int]:
    """Run temporal chain building.

    Args:
        vector_store: VectorStore instance
        project: Filter to specific project
        force: Clear existing chains first

    Returns:
        Dict with counts: files_analyzed, chains_created
    """
    if force:
        vector_store.clear_topic_chains(project)

    # Check if chains already exist
    stats = vector_store.get_topic_chain_stats()
    if stats["total_chains"] > 0 and not force:
        logger.info(
            "Topic chains already exist (%d chains)."
            " Use --force to rebuild.",
            stats["total_chains"],
        )
        return {
            "files_analyzed": 0,
            "chains_created": 0,
        }

    # Build file→session map
    file_map = build_file_session_map(
        vector_store, project=project
    )
    logger.info(
        "Built file map: %d files across sessions",
        len(file_map),
    )

    # Build chains
    chains = build_topic_chains(
        file_map, project=project
    )
    logger.info(
        "Found %d topic chains", len(chains)
    )

    # Store
    if chains:
        count = vector_store.store_topic_chains(chains)
    else:
        count = 0

    return {
        "files_analyzed": len(file_map),
        "chains_created": count,
    }
