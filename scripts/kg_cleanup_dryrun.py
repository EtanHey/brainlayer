#!/usr/bin/env python3
"""KG Cleanup Phase-1 — DRY RUN (read-only).

Computes the exact destructive scope per the 2026-06-05 KG diagnosis:
  - AUTO-PRUNE bucket: deterministic junk regex families (+ orphan/one-off gate)
  - AUTO-MERGE bucket: pure-variant clusters (normalized name stem) with
    relationship/chunk evidence
  - FLAG bucket: everything borderline (left for human review)

NO WRITES. Opens the DB in read-only mode. Output: JSON scope + samples.
"""

import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brainlayer.kg_cleanup import classify_junk, select_prune  # noqa: E402

DB = Path.home() / ".local/share/brainlayer/brainlayer.db"

# Canonical stems named MERGE@high in the 2026-06-05 diagnosis (§2).
# 'mcp' and 'launcher' deliberately excluded — their facets carry FLAG verdicts.
DIAGNOSIS_NAMED_STEMS = {
    "codex",
    "cursor",
    "claude",
    "claude code",
    "claude opus 4.6",
    "gemini",
    "coachclaude",
    "mehayomclaude",
    "groq",
    "llama 3.3 70b",
    "llama 3.2 3b",
    "qwen2.5 coder 14b instruct 4bit",
    "crucial x9 pro 1tb ssd",
    "careergateway.io",
    "mailinh ho",
    "cantaloupe ai",
    "entity grill",
    "skill creator",
    "eas prebuild check",
    "agent routing",
    "cmux agents",
    "cmux mcp",
    "voicelayer mcp",
    "claude web research",
    "using git worktrees",
    "orc agent",
    "golems cli",
    "golems glm",
    "tailwind css",
    "convex",
    "expo",
    "linear",
    "nextdns",
    "coderabbitai[bot]",
    "kiro cli",
    "f5 tts",
    "f5 tts mlx",
    "baai/bge large en v1.5",
    "enrichment launchagent",
}


# (stem, entity_type) facets the diagnosis FLAGGED inside otherwise-mergeable
# clusters — pulled out of tier-1 and routed to the human FLAG batch.
FLAGGED_FACETS = {
    ("cursor", "company"),  # vendor-vs-product facet
    ("expo", "concept"),  # 1-chunk concept node, flagged
    ("convex", "tool"),  # straddles skill + BaaS
}


def load_dictionary() -> set:
    try:
        with open("/usr/share/dict/words") as f:
            return {w.strip().lower() for w in f}
    except OSError:
        return set()


def normalize_stem(name: str) -> str:
    """Casing / hyphen / slash-prefix / @-prefix variants → one stem."""
    s = name.strip().lower()
    s = re.sub(r"^[@/✳\s]+", "", s)
    s = re.sub(r"[-_\s]+", " ", s).strip()
    return s


def main() -> None:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    rows = con.execute(
        """SELECT e.id, e.entity_type, e.name, e.user_verified, e.importance,
                  e.status, e.canonical_name,
                  (SELECT COUNT(*) FROM kg_entity_chunks c WHERE c.entity_id = e.id) AS n_chunks,
                  (SELECT COUNT(*) FROM kg_relations r
                    WHERE r.source_id = e.id OR r.target_id = e.id) AS n_rels
           FROM kg_entities e WHERE e.status = 'active'"""
    ).fetchall()

    def protected(r):
        return r["user_verified"] == 1 or (r["importance"] or 0) >= 0.7

    # ---------- AUTO-PRUNE (selection logic lives in brainlayer.kg_cleanup) --
    prune: dict[str, list] = defaultdict(list)
    for sel in select_prune(con):
        prune[sel["family"]].append(sel)
    prune_ids = {sel["id"] for fam in prune.values() for sel in fam}
    protected_skips = [
        (classify_junk(r["name"]), r["name"], r["importance"])
        for r in rows
        if protected(r) and classify_junk(r["name"])
    ]

    # Structural orphans (0 chunks AND 0 relations) — FLAG ONLY, never auto.
    # Dry-run sampling showed real entities here (Gemini, Andy Galpin):
    # structure without a junk-family name is not a prune signal.
    orphans = [
        r for r in rows if r["n_chunks"] == 0 and r["n_rels"] == 0 and r["id"] not in prune_ids and not protected(r)
    ]

    # ---------- AUTO-MERGE (pure-variant clusters w/ evidence) ----------
    by_stem: dict[str, list] = defaultdict(list)
    for r in rows:
        if r["id"] in prune_ids or protected(r):
            # protected rows may still anchor a cluster as canonical
            pass
        if r["id"] not in prune_ids:
            by_stem[normalize_stem(r["name"])].append(r)

    dictionary = load_dictionary()
    tier1, tier2, merge_flagged = [], [], []
    flagged_facet_rows = []
    for stem, members in by_stem.items():
        if len(members) < 2 or not stem:
            continue
        kept = [m for m in members if (stem, m["entity_type"]) not in FLAGGED_FACETS]
        flagged_facet_rows += [
            {"stem": stem, "name": m["name"], "type": m["entity_type"]}
            for m in members
            if (stem, m["entity_type"]) in FLAGGED_FACETS
        ]
        members = kept
        if len(members) < 2:
            continue
        ids = [m["id"] for m in members]
        ph = ",".join("?" * len(ids))
        shared_chunks = con.execute(
            f"""SELECT COUNT(*) FROM (
                  SELECT chunk_id FROM kg_entity_chunks WHERE entity_id IN ({ph})
                  GROUP BY chunk_id HAVING COUNT(DISTINCT entity_id) >= 2)""",
            ids,
        ).fetchone()[0]
        intra_rels = con.execute(
            f"""SELECT COUNT(*) FROM kg_relations
                 WHERE source_id IN ({ph}) AND target_id IN ({ph})""",
            ids + ids,
        ).fetchone()[0]
        canonical = max(members, key=lambda m: (m["n_chunks"], m["n_rels"]))
        cluster = {
            "stem": stem,
            "canonical": {
                "id": canonical["id"],
                "type": canonical["entity_type"],
                "name": canonical["name"],
                "chunks": canonical["n_chunks"],
            },
            "members": [
                {"id": m["id"], "type": m["entity_type"], "name": m["name"], "chunks": m["n_chunks"]}
                for m in members
                if m["id"] != canonical["id"]
            ],
            "shared_chunks": shared_chunks,
            "intra_relations": intra_rels,
            "size": len(members),
        }
        has_evidence = shared_chunks >= 1 or intra_rels >= 1
        # Generic single dictionary words (git, research, jobs...) are merge
        # traps even with shared chunks — never auto, always flag.
        is_generic = " " not in stem and stem in dictionary
        if stem in DIAGNOSIS_NAMED_STEMS:
            # Diagnosis verdict MERGE@high IS the evidence (relationship-
            # grounded disambiguation already ran the cmux/cmuxlayer test).
            cluster["tier"] = 1
            tier1.append(cluster)
        elif has_evidence and not is_generic:
            cluster["tier"] = 2  # pure variants + evidence, not in diagnosis
            tier2.append(cluster)
        else:
            cluster["tier"] = "flag"
            merge_flagged.append(cluster)

    for lst in (tier1, tier2, merge_flagged):
        lst.sort(key=lambda c: -c["size"])

    # ---------- Summary ----------
    n_prune = sum(len(v) for v in prune.values())
    scope = {
        "db": str(DB),
        "total_active": len(rows),
        "auto_prune": {
            "total": n_prune,
            "by_family": {k: len(v) for k, v in prune.items()},
            "protected_rows_skipped": len(protected_skips),
            "samples": {k: [r["name"] for r in v[:8]] for k, v in prune.items()},
            "protected_skip_samples": protected_skips[:10],
        },
        "auto_merge": {
            "tier1_diagnosis_named": {
                "clusters": len(tier1),
                "rows_merged_away": sum(c["size"] - 1 for c in tier1),
                "all_clusters": tier1,
            },
            "tier2_evidence_not_named": {
                "clusters": len(tier2),
                "rows_merged_away": sum(c["size"] - 1 for c in tier2),
                "all_clusters": tier2,
            },
        },
        "flag_for_review": {
            "diagnosis_flagged_facets": flagged_facet_rows,
            "structural_orphans": [r["name"] for r in orphans],
            "name_match_no_evidence_clusters": len(merge_flagged),
            "rows_in_those_clusters": sum(c["size"] for c in merge_flagged),
            "samples": [
                {
                    "stem": c["stem"],
                    "size": c["size"],
                    "names": [c["canonical"]["name"]] + [m["name"] for m in c["members"]][:5],
                }
                for c in merge_flagged[:20]
            ],
        },
    }
    json.dump(scope, sys.stdout, indent=2, default=str)


if __name__ == "__main__":
    main()
