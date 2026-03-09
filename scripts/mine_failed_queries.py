#!/usr/bin/env python3
"""Mine Claude Code session logs for brain_search queries with 0/few results.

Scans JSONL session files for brain_search tool calls and their results.
Identifies queries that returned no results or very few results.
Outputs candidates for eval suite expansion.

Usage:
    python3 scripts/mine_failed_queries.py                  # all projects
    python3 scripts/mine_failed_queries.py --project brainlayer
    python3 scripts/mine_failed_queries.py --min-count 3    # queries seen 3+ times
    python3 scripts/mine_failed_queries.py --json            # JSON output
"""

import argparse
import collections
import glob
import json
import os
import re
import sys

SESSIONS_BASE = os.path.expanduser("~/.claude/projects/")


def find_session_files(project=None):
    """Find all JSONL session files, optionally filtered by project."""
    if project:
        patterns = [
            SESSIONS_BASE + f"*{project}*/*.jsonl",
            SESSIONS_BASE + f"*{project}*/**/*.jsonl",
        ]
    else:
        patterns = [SESSIONS_BASE + "**/*.jsonl"]

    files = set()
    for pat in patterns:
        files.update(glob.glob(pat, recursive=True))
    return sorted(files, key=os.path.getmtime, reverse=True)


def extract_queries_and_results(filepath):
    """Parse a JSONL file for brain_search tool_use + tool_result pairs.

    Returns list of (query, result_count, project_filter) tuples.
    """
    results = []
    tool_calls = {}  # tool_id -> query info

    try:
        with open(filepath) as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

                msg = obj.get("message", obj)
                content = msg.get("content", [])

                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue

                        # Capture tool_use (brain_search call)
                        if block.get("type") == "tool_use":
                            name = block.get("name", "")
                            if "brain_search" in name:
                                inp = block.get("input", {})
                                query = inp.get("query", "")
                                if query and len(query) > 5:
                                    tool_calls[block.get("id", "")] = {
                                        "query": query,
                                        "project": inp.get("project", ""),
                                    }

                        # Capture tool_result (response to brain_search)
                        if block.get("type") == "tool_result":
                            tool_id = block.get("tool_use_id", "")
                            if tool_id in tool_calls:
                                result_text = ""
                                result_content = block.get("content", "")
                                if isinstance(result_content, list):
                                    for rc in result_content:
                                        if isinstance(rc, dict) and rc.get("type") == "text":
                                            result_text += rc.get("text", "")
                                elif isinstance(result_content, str):
                                    result_text = result_content

                                count = _extract_result_count(result_text)
                                info = tool_calls.pop(tool_id)
                                results.append((info["query"], count, info["project"]))

                # Also check for inline tool_result (not in content list)
                if isinstance(content, dict) and content.get("type") == "tool_result":
                    tool_id = content.get("tool_use_id", "")
                    if tool_id in tool_calls:
                        result_text = str(content.get("content", ""))
                        count = _extract_result_count(result_text)
                        info = tool_calls.pop(tool_id)
                        results.append((info["query"], count, info["project"]))

    except (OSError, PermissionError):
        pass

    return results


def _extract_result_count(text):
    """Extract result count from brain_search response text."""
    if not text:
        return -1  # unknown

    # Look for "total": N in JSON response
    m = re.search(r'"total":\s*(\d+)', text)
    if m:
        return int(m.group(1))

    # Look for "N results" pattern
    m = re.search(r"(\d+)\s+results?", text)
    if m:
        return int(m.group(1))

    # Look for "No results" or "No matches"
    if re.search(r"no\s+results|no\s+matches|nothing\s+found", text, re.I):
        return 0

    # Count markdown list items (common in compact results)
    items = re.findall(r"^- \[", text, re.MULTILINE)
    if items:
        return len(items)

    return -1  # couldn't determine


def analyze_queries(all_results, min_count=1):
    """Analyze queries for patterns. Returns categorized results."""
    # Group by normalized query
    query_stats = collections.defaultdict(lambda: {"count": 0, "results": [], "projects": set()})

    for query, count, project in all_results:
        # Normalize: lowercase, strip extra whitespace
        normalized = " ".join(query.lower().split())
        stats = query_stats[normalized]
        stats["count"] += 1
        stats["results"].append(count)
        stats["original"] = query  # keep one original
        if project:
            stats["projects"].add(project)

    # Categorize
    zero_results = []
    low_results = []
    unknown_results = []

    for norm_query, stats in sorted(query_stats.items(), key=lambda x: x[1]["count"], reverse=True):
        if stats["count"] < min_count:
            continue

        avg_results = [r for r in stats["results"] if r >= 0]
        avg = sum(avg_results) / len(avg_results) if avg_results else -1

        entry = {
            "query": stats["original"],
            "normalized": norm_query,
            "times_searched": stats["count"],
            "avg_results": round(avg, 1) if avg >= 0 else "unknown",
            "result_counts": stats["results"],
            "projects": list(stats["projects"]),
        }

        if avg == 0:
            zero_results.append(entry)
        elif 0 < avg <= 2:
            low_results.append(entry)
        elif avg < 0:
            unknown_results.append(entry)

    return zero_results, low_results, unknown_results


def main():
    parser = argparse.ArgumentParser(description="Mine failed search queries from session logs")
    parser.add_argument("--project", help="Filter by project name")
    parser.add_argument("--min-count", type=int, default=1, help="Minimum times query was searched")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--limit", type=int, default=50, help="Max files to scan (newest first)")
    args = parser.parse_args()

    files = find_session_files(args.project)
    if args.limit:
        files = files[: args.limit]

    if not files:
        print("No session files found.", file=sys.stderr)
        sys.exit(1)

    all_results = []
    for f in files:
        results = extract_queries_and_results(f)
        all_results.extend(results)

    zero, low, unknown = analyze_queries(all_results, args.min_count)

    if args.json:
        print(
            json.dumps(
                {
                    "files_scanned": len(files),
                    "total_queries": len(all_results),
                    "zero_results": zero,
                    "low_results": low,
                    "unknown_results_count": len(unknown),
                },
                indent=2,
            )
        )
        return

    print(f"Scanned {len(files)} session files, found {len(all_results)} brain_search calls\n")

    if zero:
        print(f"=== ZERO RESULTS ({len(zero)} queries) ===")
        for q in zero[:20]:
            times = f" (x{q['times_searched']})" if q["times_searched"] > 1 else ""
            proj = f" [{','.join(q['projects'])}]" if q["projects"] else ""
            print(f"  {q['query'][:100]}{times}{proj}")
        print()

    if low:
        print(f"=== LOW RESULTS (1-2 avg, {len(low)} queries) ===")
        for q in low[:20]:
            times = f" (x{q['times_searched']})" if q["times_searched"] > 1 else ""
            print(f"  [{q['avg_results']} avg] {q['query'][:100]}{times}")
        print()

    total_gaps = len(zero) + len(low)
    print(f"Summary: {total_gaps} gap queries found ({len(zero)} zero, {len(low)} low)")
    if total_gaps:
        print("These are candidates for eval suite expansion (C6).")


if __name__ == "__main__":
    main()
