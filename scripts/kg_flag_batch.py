#!/usr/bin/env python3
"""KG Cleanup Phase-1 — FLAG batch generator (read-only).

Re-derives the post-phase-1 flag set from the live DB (active rows only;
everything pruned/merged in phase-1 is archived and drops out) and emits a
category-sorted markdown review file + JSON for Etan's batch approval.

Categories (safest first):
  identical-name   — every member name is the same string (pure type dup)
  case-only        — names differ only by casing
  sep-variants     — names differ by hyphen/underscore/space
  prefix-variants  — at least one member has a leading / or @ (command-vs-
                     concept risk; the /mcp-vs-MCP trap lives here)
  diagnosis-flag   — stems the diagnosis explicitly flagged for human review
"""

import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

DB = Path.home() / ".local/share/brainlayer/brainlayer.db"

DIAGNOSIS_FLAG_STEMS = {
    "whatsapp business",
    "agent c",
    "mcp",
    "claude web",
    "easysend",
    "cantaloupe",
    "cantaloupe cto",
    "launcher",
    "ios qa codex",
    "tailwind css v4",
    "easysend tech star program",
}


def normalize_stem(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"^[@/✳\s]+", "", s)
    s = re.sub(r"[-_\s]+", " ", s).strip()
    return s


def categorize(stem: str, names: list[str]) -> str:
    if stem in DIAGNOSIS_FLAG_STEMS:
        return "diagnosis-flag"
    if any(re.match(r"^[@/✳]", n.strip()) for n in names):
        return "prefix-variants"
    if len({n.strip() for n in names}) == 1:
        return "identical-name"
    if len({n.strip().lower() for n in names}) == 1:
        return "case-only"
    return "sep-variants"


def main() -> None:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT e.id, e.entity_type, e.name,
                  (SELECT COUNT(*) FROM kg_entity_chunks c
                    WHERE c.entity_id = e.id) AS n_chunks
           FROM kg_entities e WHERE e.status='active'"""
    ).fetchall()

    by_stem = defaultdict(list)
    for r in rows:
        by_stem[normalize_stem(r["name"])].append(r)

    cats = defaultdict(list)
    for stem, members in sorted(by_stem.items()):
        if len(members) < 2 or not stem:
            continue
        names = [m["name"] for m in members]
        cluster = {
            "stem": stem,
            "size": len(members),
            "members": [
                {"id": m["id"], "name": m["name"], "type": m["entity_type"], "chunks": m["n_chunks"]}
                for m in sorted(members, key=lambda m: -m["n_chunks"])
            ],
        }
        cats[categorize(stem, names)].append(cluster)

    out_json = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval_results/kg-phase1-flag-batch-2026-06-05.json")
    out_json.write_text(json.dumps(dict(cats), indent=2))

    order = ["identical-name", "case-only", "sep-variants", "prefix-variants", "diagnosis-flag"]
    lines = [
        "# KG Phase-1 FLAG Batch — Etan review (2026-06-05)",
        "",
        "Post-phase-1 leftover name-match clusters, **no auto-action taken**.",
        "Approve per category (or strike individual clusters).",
        "Machine copy: `" + str(out_json) + "`",
        "",
    ]
    for cat in order:
        clusters = cats.get(cat, [])
        n_rows = sum(c["size"] for c in clusters)
        lines.append(f"## {cat} — {len(clusters)} clusters / {n_rows} rows")
        lines.append("")
        for c in sorted(clusters, key=lambda c: -c["size"]):
            members = ", ".join(f"{m['name']}({m['type']},{m['chunks']}ch)" for m in c["members"])
            lines.append(f"- [ ] **{c['stem']}** ×{c['size']}: {members}")
        lines.append("")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
