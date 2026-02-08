#!/usr/bin/env python3
"""Pre-label 200 stratified sample chunks via heuristics.

Outputs data/labeled-samples.json with pre-labels + confidence.
Human reviews/corrects using label-chunks.py.
"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zikaron.vector_store import VectorStore

DB_PATH = Path.home() / ".local" / "share" / "zikaron" / "zikaron.db"
OUTPUT = Path(__file__).parent.parent / "data" / "labeled-samples.json"
TAXONOMY = Path(__file__).parent.parent / "src" / "zikaron" / "taxonomy.json"
SAMPLE_SIZE = 200


def load_taxonomy():
    with open(TAXONOMY) as f:
        data = json.load(f)
    labels = {}
    for cat, info in data["categories"].items():
        for label, desc in info["labels"].items():
            labels[label] = desc
    return labels


def heuristic_labels(content: str, metadata: dict, project: str) -> list[tuple[str, float]]:
    """Return list of (label, confidence) tuples based on heuristics."""
    tags = []
    ct = metadata.get("content_type", "")
    src = metadata.get("source_file", "")
    text = content.lower()

    # Content type heuristics
    if ct == "stack_trace":
        tags.append(("tech/debug/error-trace", 0.95))
    if ct == "ai_code":
        tags.append(("tech/code-snippet", 0.85))
    if ct == "git_diff":
        tags.append(("tech/refactor", 0.6))

    # Keyword heuristics
    patterns = {
        "tech/testing": r"\b(test|pytest|jest|expect\(|describe\(|it\()\b",
        "tech/database": r"\b(sql|migration|supabase|postgres|select\s|insert\s|create table)\b",
        "tech/api": r"\b(endpoint|api|fetch|axios|http|rest|graphql)\b",
        "tech/frontend": r"\b(react|component|css|tailwind|next\.js|jsx|tsx)\b",
        "tech/mobile": r"\b(expo|react.native|ios|android|mobile)\b",
        "tech/ai-ml": r"\b(embedding|llm|rag|model|anthropic|openai|claude|ollama|haiku|opus)\b",
        "tech/devops": r"\b(docker|railway|deploy|ci\/cd|github.actions|launchd)\b",
        "tech/security": r"\b(auth|jwt|otp|secret|credential|password|1password)\b",
        "tech/config": r"\b(\.env|config|settings|pyproject|package\.json)\b",
        "tech/performance": r"\b(performance|optimize|profil|latency|cache|slow)\b",
        "pm/decision": r"\b(decided|chose|recommend|trade.off|approach|vs\b|compare)\b",
        "pm/planning": r"\b(roadmap|sprint|plan|phase|milestone|backlog)\b",
        "pm/review": r"\b(review|pr\s|pull.request|coderabbit|feedback)\b",
        "personal/job-search": r"\b(job|interview|resume|cv|recruiter|application)\b",
        "personal/writing": r"\b(blog|post|article|draft|content|soltome)\b",
        "personal/communication": r"\b(email|message|whatsapp|telegram|outreach)\b",
        "tech/debug/investigation": r"\b(debug|investigate|root.cause|why.is|broken|fix)\b",
    }

    for label, pattern in patterns.items():
        if re.search(pattern, text):
            # Don't duplicate if already tagged
            if not any(t[0] == label for t in tags):
                tags.append((label, 0.7))

    # Platform heuristics
    platform_patterns = {
        "platform/claude": r"\b(claude|anthropic|mcp|claude.code)\b",
        "platform/github": r"\b(github|gh\s|pull.request|issue|actions)\b",
        "platform/supabase": r"\b(supabase|rls|edge.function)\b",
        "platform/telegram": r"\b(telegram|grammy|bot\.api)\b",
        "platform/convex": r"\b(convex|mutation|query\()\b",
        "platform/railway": r"\b(railway)\b",
        "platform/obsidian": r"\b(obsidian|vault|\.md)\b",
    }

    for label, pattern in platform_patterns.items():
        if re.search(pattern, text):
            if not any(t[0] == label for t in tags):
                tags.append((label, 0.65))

    # Project heuristics (from metadata)
    project_map = {
        "golems": "project/golems",
        "zikaron": "project/zikaron",
        "songscript": "project/songscript",
        "domica": "project/domica",
        "etanheyman-com": "project/etanheyman-com",
        "union": "project/union",
    }
    if project:
        for key, label in project_map.items():
            if key in project.lower():
                tags.append((label, 0.9))
                break

    return tags


def sample_chunks(store: VectorStore, n: int = SAMPLE_SIZE) -> list[dict]:
    """Stratified sample: spread across projects + content types."""
    cursor = store.conn.cursor()

    # Get distribution
    groups = list(cursor.execute("""
        SELECT project, content_type, COUNT(*) as cnt
        FROM chunks
        WHERE project IS NOT NULL AND content_type IS NOT NULL
        GROUP BY project, content_type
        ORDER BY cnt DESC
    """))

    # Allocate samples proportionally (minimum 1 per group, cap at 20)
    total = sum(g[2] for g in groups)
    allocations = []
    for proj, ct, cnt in groups:
        alloc = max(1, min(20, round(n * cnt / total)))
        allocations.append((proj, ct, alloc))

    # Adjust to hit target
    allocated = sum(a[2] for a in allocations)
    if allocated > n:
        # Trim from largest allocations
        allocations.sort(key=lambda x: x[2], reverse=True)
        for i in range(len(allocations)):
            if allocated <= n:
                break
            if allocations[i][2] > 1:
                allocations[i] = (allocations[i][0], allocations[i][1], allocations[i][2] - 1)
                allocated -= 1

    # Sample from each group
    samples = []
    for proj, ct, alloc in allocations:
        rows = list(cursor.execute("""
            SELECT id, content, metadata, source_file, project, content_type
            FROM chunks
            WHERE project = ? AND content_type = ?
            ORDER BY RANDOM()
            LIMIT ?
        """, (proj, ct, alloc)))

        for row in rows:
            samples.append({
                "id": row[0],
                "content": row[1][:2000],  # Truncate for labeling
                "metadata": json.loads(row[2]) if row[2] else {},
                "source_file": row[3],
                "project": row[4],
                "content_type": row[5],
            })

    random.shuffle(samples)
    return samples[:n]


def main():
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        sys.exit(1)

    taxonomy = load_taxonomy()
    print(f"Taxonomy: {len(taxonomy)} labels across 5 categories")

    store = VectorStore(DB_PATH)
    print(f"Database: {store.count():,} chunks")

    # Sample
    print(f"\nSampling {SAMPLE_SIZE} chunks (stratified by project + content_type)...")
    samples = sample_chunks(store)
    print(f"Sampled {len(samples)} chunks")

    # Pre-label
    labeled = 0
    for sample in samples:
        tags = heuristic_labels(
            sample["content"],
            sample.get("metadata", {}),
            sample.get("project", "")
        )
        # Merge content_type as metadata
        meta = sample.get("metadata", {})
        meta["content_type"] = sample.get("content_type", "")

        sample["pre_labels"] = [{"label": t[0], "confidence": t[1]} for t in tags]
        sample["human_labels"] = []  # To be filled by human
        sample["reviewed"] = False

        if tags:
            labeled += 1

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"\nPre-labeled: {labeled}/{len(samples)} chunks ({labeled*100//len(samples)}%)")
    print(f"Output: {OUTPUT}")
    print(f"\nNext: python scripts/label-chunks.py")

    store.close()


if __name__ == "__main__":
    main()
