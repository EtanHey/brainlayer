"""KG Cleanup Phase-1 executor — prune/merge/rollback (2026-06-05, Etan GO).

Everything reversible:
  - prunes:  ``status='archived'`` + log row (prior state in payload)
  - merges:  losers archived with ``parent_id=canonical``, variant names
    written to ``kg_entity_aliases``, chunk links and relations re-pointed
    to the canonical with the original mapping logged
  - rollback: reverse-replay of ``kg_cleanup_log`` for a run_id

No ``DELETE FROM kg_entities`` anywhere.
"""

import hashlib
import json
import re
import sqlite3
import time

# --- Deterministic junk families (2026-06-05 diagnosis §3/§4) -----------
# Canonical home of the selection logic; scripts/kg_cleanup_dryrun.py and
# the apply CLI must both use THESE patterns.
JUNK_FAMILIES = {
    "anonymizer_placeholder": re.compile(r"^\[?(PERSON|EMAIL|IMAGE|ADDRESS|PHONE|URL|NAME|IP)_[A-Za-z0-9]+\]?$"),
    "surface_artifact": re.compile(r"^surface:\d+$", re.I),
    "env_var": re.compile(r"^[A-Z][A-Z0-9_]{2,}=\S*$"),
    "transient_ref_id": re.compile(
        r"^((PR|issue|Task|element|Error)\s*#?\d+|R\d+|[A-Z]{2,}-\d+|DECISION-\d+)$",
        re.I,
    ),
    "bare_hex_or_uuid": re.compile(
        r"^([0-9a-f]{8,}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$",
        re.I,
    ),
    "acompact": re.compile(r"^acompact-[0-9a-f]+", re.I),
    "filename": re.compile(
        r"^[\w./\\-]+\.(md|ts|tsx|py|sh|json|csv|html|jsonl|plist|log|yml|yaml|toml|lock|sql|swift)$",
        re.I,
    ),
}

SAFE_NAMES = {"etanhey", "etanheyman", "etanface"}


class DeviationError(RuntimeError):
    """Actual selection deviates from the approved dry-run scope."""


def _protected(row) -> bool:
    return row["user_verified"] == 1 or (row["importance"] or 0) >= 0.7


def classify_junk(name: str) -> str | None:
    """Return the junk family for a name, or None if it is not junk."""
    stripped = name.strip()
    if stripped.lower() in SAFE_NAMES:
        return None
    for family, rx in JUNK_FAMILIES.items():
        if rx.match(stripped):
            return family
    return None


def select_prune(con: sqlite3.Connection) -> list[dict]:
    """Active, unprotected entities whose name matches a junk family."""
    out = []
    for row in con.execute(
        "SELECT id, entity_type, name, user_verified, importance, status FROM kg_entities WHERE status='active'"
    ):
        if _protected(row):
            continue
        family = classify_junk(row["name"])
        if family:
            out.append({"id": row["id"], "name": row["name"], "entity_type": row["entity_type"], "family": family})
    return out


def ensure_log_table(con: sqlite3.Connection) -> None:
    con.execute(
        """CREATE TABLE IF NOT EXISTS kg_cleanup_log (
               run_id TEXT NOT NULL,
               action TEXT NOT NULL,
               entity_id TEXT NOT NULL,
               canonical_id TEXT,
               mechanism TEXT,
               evidence TEXT,
               payload_json TEXT,
               ts TEXT NOT NULL
           )"""
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_kg_cleanup_log_run ON kg_cleanup_log(run_id)")
    con.commit()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def apply_prune(
    con: sqlite3.Connection,
    rows: list[dict],
    run_id: str,
    expected: int,
    batch_size: int = 5000,
) -> int:
    """Archive junk rows. Aborts (no writes) if count deviates >1% from
    the approved ``expected`` count."""
    if expected and abs(len(rows) - expected) / expected > 0.01:
        raise DeviationError(f"prune selection {len(rows)} deviates >1% from approved {expected}")
    ts = _now()
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        for r in batch:
            con.execute(
                "UPDATE kg_entities SET status='archived' WHERE id=?",
                (r["id"],),
            )
            con.execute(
                "INSERT INTO kg_cleanup_log (run_id, action, entity_id,"
                " mechanism, evidence, payload_json, ts)"
                " VALUES (?,?,?,?,?,?,?)",
                (
                    run_id,
                    "prune",
                    r["id"],
                    "status=archived",
                    r["family"],
                    json.dumps({"prior_status": "active", "name": r["name"]}),
                    ts,
                ),
            )
        con.commit()
    return len(rows)


def apply_merges(con: sqlite3.Connection, clusters: list[dict], run_id: str) -> dict:
    """Merge each cluster into its canonical. Losers archived w/ parent_id,
    names aliased, chunk links + relations re-pointed (originals logged)."""
    ts = _now()
    merged_away = 0
    for cluster in clusters:
        canon = cluster["canonical"]["id"]
        for member in cluster["members"]:
            mid = member["id"]
            prior = con.execute("SELECT status, parent_id FROM kg_entities WHERE id=?", (mid,)).fetchone()
            chunk_ids = [
                r["chunk_id"]
                for r in con.execute(
                    "SELECT chunk_id FROM kg_entity_chunks WHERE entity_id=?",
                    (mid,),
                )
            ]
            rel_rows = con.execute(
                "SELECT id, source_id, target_id FROM kg_relations WHERE source_id=? OR target_id=?",
                (mid, mid),
            ).fetchall()

            # archive loser under canonical
            con.execute(
                "UPDATE kg_entities SET status='archived', parent_id=? WHERE id=?",
                (canon, mid),
            )
            # variant name resolvable via alias (INSERT OR IGNORE: several
            # losers may share the same surface name)
            con.execute(
                "INSERT OR IGNORE INTO kg_entity_aliases (alias, entity_id, alias_type, created_at) VALUES (?,?,?,?)",
                (member["name"], canon, "merge_variant", ts),
            )
            # re-point chunk links; canonical may already link the chunk.
            # Track which links we CREATED so rollback never deletes a
            # link the canonical owned before the merge.
            created_links = []
            for chunk_id in chunk_ids:
                cur = con.execute(
                    "INSERT OR IGNORE INTO kg_entity_chunks (entity_id, chunk_id) VALUES (?,?)",
                    (canon, chunk_id),
                )
                if cur.rowcount == 1:
                    created_links.append(chunk_id)
                con.execute(
                    "DELETE FROM kg_entity_chunks WHERE entity_id=? AND chunk_id=?",
                    (mid, chunk_id),
                )
            # re-point relations (skip would-be self-loops / unique clashes)
            moved_rels = []
            for rel in rel_rows:
                new_src = canon if rel["source_id"] == mid else rel["source_id"]
                new_tgt = canon if rel["target_id"] == mid else rel["target_id"]
                if new_src == new_tgt:
                    continue
                try:
                    con.execute(
                        "UPDATE kg_relations SET source_id=?, target_id=? WHERE id=?",
                        (new_src, new_tgt, rel["id"]),
                    )
                    moved_rels.append({"id": rel["id"], "source_id": rel["source_id"], "target_id": rel["target_id"]})
                except sqlite3.IntegrityError:
                    pass  # canonical already has this typed edge

            con.execute(
                "INSERT INTO kg_cleanup_log (run_id, action, entity_id,"
                " canonical_id, mechanism, evidence, payload_json, ts)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    "merge",
                    mid,
                    canon,
                    "archive+parent_id+alias+repoint",
                    cluster["stem"],
                    json.dumps(
                        {
                            "prior_status": prior["status"],
                            "prior_parent_id": prior["parent_id"],
                            "alias": member["name"],
                            "chunk_ids": chunk_ids,
                            "created_links": created_links,
                            "relations": moved_rels,
                        }
                    ),
                    ts,
                ),
            )
            merged_away += 1
        con.commit()
    return {"clusters": len(clusters), "rows_merged_away": merged_away}


def _keep_separate_entity_id(decision: dict) -> str:
    stem = str(decision.get("stem") or "").strip()
    category = str(decision.get("category") or "").strip()
    if stem and category:
        return f"cat:{category}:stem:{stem}"
    if stem:
        return f"stem:{stem}"
    if decision.get("id"):
        return f"id:{decision['id']}"
    digest = hashlib.sha256(json.dumps(decision, sort_keys=True).encode("utf-8")).hexdigest()
    return f"decision:{digest[:16]}"


def apply_keep_separate(
    con: sqlite3.Connection,
    decisions: list[dict],
    run_id: str,
    batch_size: int = 5000,
) -> int:
    """Persist human keep-separate decisions as log-only rows."""
    ts = _now()
    logged = 0
    for i in range(0, len(decisions), batch_size):
        batch = decisions[i : i + batch_size]
        for decision in batch:
            con.execute(
                "INSERT INTO kg_cleanup_log (run_id, action, entity_id,"
                " canonical_id, mechanism, evidence, payload_json, ts)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    "keep_separate",
                    _keep_separate_entity_id(decision),
                    None,
                    "flag_decision_keep_separate",
                    str(decision.get("category") or decision.get("source") or ""),
                    json.dumps({"decision": decision}, sort_keys=True),
                    ts,
                ),
            )
            logged += 1
        con.commit()
    return logged


def rollback(con: sqlite3.Connection, run_id: str) -> dict:
    """Reverse-replay the log for ``run_id`` (newest first)."""
    entries = con.execute(
        "SELECT rowid, action, entity_id, canonical_id, payload_json"
        " FROM kg_cleanup_log WHERE run_id=? ORDER BY rowid DESC",
        (run_id,),
    ).fetchall()
    counts = {"prune": 0, "merge": 0}
    for e in entries:
        payload = json.loads(e["payload_json"])
        if e["action"] == "prune":
            con.execute(
                "UPDATE kg_entities SET status=? WHERE id=?",
                (payload["prior_status"], e["entity_id"]),
            )
        elif e["action"] == "merge":
            mid, canon = e["entity_id"], e["canonical_id"]
            con.execute(
                "UPDATE kg_entities SET status=?, parent_id=? WHERE id=?",
                (payload["prior_status"], payload["prior_parent_id"], mid),
            )
            con.execute(
                "DELETE FROM kg_entity_aliases WHERE alias=? AND entity_id=? AND alias_type='merge_variant'",
                (payload["alias"], canon),
            )
            created = set(payload.get("created_links", payload["chunk_ids"]))
            for chunk_id in payload["chunk_ids"]:
                con.execute(
                    "INSERT OR IGNORE INTO kg_entity_chunks (entity_id, chunk_id) VALUES (?,?)",
                    (mid, chunk_id),
                )
                if chunk_id in created:
                    con.execute(
                        "DELETE FROM kg_entity_chunks WHERE entity_id=? AND chunk_id=?",
                        (canon, chunk_id),
                    )
            for rel in payload["relations"]:
                con.execute(
                    "UPDATE kg_relations SET source_id=?, target_id=? WHERE id=?",
                    (rel["source_id"], rel["target_id"], rel["id"]),
                )
        elif e["action"] == "keep_separate":
            pass
        else:
            raise RuntimeError(f"unknown kg cleanup action: {e['action']}")
        counts[e["action"]] = counts.get(e["action"], 0) + 1
        con.execute("DELETE FROM kg_cleanup_log WHERE rowid=?", (e["rowid"],))
    con.commit()
    return counts
