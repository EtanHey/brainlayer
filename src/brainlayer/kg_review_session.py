"""KG flag-batch review session driver — shared by voice and visual review surfaces.

Pure-file module: reads the category-sorted flag-batch JSON emitted by phase-1
(`eval_results/kg-phase1-flag-batch-*.json`) and maintains a decisions JSON that
`kg_cleanup_apply.py` (or a follow-up applier) can consume. No DB access here.

Decisions file contract (v1 stub — coordinate changes via the kg-cleanup collab):

    {
      "version": 1,
      "decisions": {
        "<category>:<stem>": {
          "action": "merge_all" | "keep_all" | "mixed" | "skip",
          "canonical_id": "...",            # merge_all only
          "members": {"<id>": "merge|keep|prune"},  # mixed only
          "note": "verbatim reviewer reasoning",
          "source": "voice" | "visual" | "rule",
          "decided_at": "ISO-8601"
        }
      },
      "rules": [ {"match": {...}, "action": ..., "canonical": ..., "applied": N,
                  "source": ..., "recorded_at": "ISO-8601"} ]
    }

Merge-safety: every write is read-modify-write under an exclusive `fcntl.flock`
on a sidecar lock file, then an atomic `os.replace`. Two writers (dashboard +
voice session) interleave safely; last write per CLUSTER wins, never per file.

AIDEV-NOTE: engine-agnostic on purpose (Etan 2026-06-05: no lock-in to the old
voice architecture) — the voice loop, a browser dashboard, and any future WS
bridge all drive this same module.
"""

from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VALID_ACTIONS = {"merge_all", "keep_all", "mixed", "skip"}
VALID_MEMBER_ACTIONS = {"merge", "keep", "prune"}
VALID_SOURCES = {"voice", "visual", "rule"}


def cluster_id(category: str, stem: str) -> str:
    return f"{category}:{stem}"


def load_flag_batch(batch_path: str | Path) -> list[dict[str, Any]]:
    """Flatten {category: [cluster, ...]} into a stable ordered list with ids."""
    raw = json.loads(Path(batch_path).read_text())
    clusters: list[dict[str, Any]] = []
    for category in raw:
        for cluster in raw[category]:
            clusters.append(
                {
                    "cluster_id": cluster_id(category, cluster["stem"]),
                    "category": category,
                    "stem": cluster["stem"],
                    "size": cluster.get("size", len(cluster.get("members", []))),
                    "members": cluster.get("members", []),
                }
            )
    return clusters


def _load_decisions(decisions_path: Path) -> dict[str, Any]:
    if decisions_path.exists():
        return json.loads(decisions_path.read_text())
    return {"version": 1, "decisions": {}, "rules": []}


def _write_locked(decisions_path: Path, mutate) -> Any:
    """Read-modify-write under flock + atomic replace. Returns mutate's result."""
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = decisions_path.with_suffix(decisions_path.suffix + ".lock")
    with open(lock_path, "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        data = _load_decisions(decisions_path)
        result = mutate(data)
        tmp_path = decisions_path.with_suffix(decisions_path.suffix + ".tmp")
        with open(tmp_path, "w") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, decisions_path)
    return result


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_decision(decision: dict[str, Any]) -> None:
    action = decision.get("action")
    if action not in VALID_ACTIONS:
        raise ValueError(f"invalid action {action!r}; expected one of {sorted(VALID_ACTIONS)}")
    if decision.get("source") not in VALID_SOURCES:
        raise ValueError(f"invalid source {decision.get('source')!r}")
    if action == "mixed":
        members = decision.get("members")
        if not isinstance(members, dict) or not members:
            raise ValueError("mixed decision requires a non-empty members map")
        bad = set(members.values()) - VALID_MEMBER_ACTIONS
        if bad:
            raise ValueError(f"invalid member actions: {sorted(bad)}")
    if action == "merge_all" and not decision.get("canonical_id"):
        raise ValueError("merge_all decision requires canonical_id")


def record_decision(
    decisions_path: str | Path,
    cluster_id: str,
    decision: dict[str, Any],
) -> dict[str, Any]:
    """Validate + persist one cluster decision. Stamps decided_at."""
    _validate_decision(decision)
    stamped = {**decision, "decided_at": _now()}

    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        data["decisions"][cluster_id] = stamped
        return stamped

    return _write_locked(Path(decisions_path), mutate)


def next_undecided(
    batch_path: str | Path,
    decisions_path: str | Path,
    category: str | None = None,
) -> dict[str, Any] | None:
    """Next cluster (batch order) without a recorded decision."""
    decided = set(_load_decisions(Path(decisions_path))["decisions"])
    for cluster in load_flag_batch(batch_path):
        if category and cluster["category"] != category:
            continue
        if cluster["cluster_id"] not in decided:
            return cluster
    return None


def _pick_canonical(members: list[dict[str, Any]], strategy: str) -> str:
    if strategy != "most_chunks":
        raise ValueError(f"unknown canonical strategy {strategy!r}")
    return max(members, key=lambda m: m.get("chunks", 0))["id"]


def apply_rule(
    batch_path: str | Path,
    decisions_path: str | Path,
    rule: dict[str, Any],
) -> int:
    """Bulk-decide all UNDECIDED clusters matching the rule. Never overwrites.

    rule = {match: {category?, stem_contains?}, action, canonical?, note?, source}
    Returns number of clusters decided.
    """
    if rule.get("action") not in VALID_ACTIONS:
        raise ValueError(f"invalid rule action {rule.get('action')!r}")
    match = rule.get("match", {})
    clusters = load_flag_batch(batch_path)

    def mutate(data: dict[str, Any]) -> int:
        applied = 0
        for cluster in clusters:
            if cluster["cluster_id"] in data["decisions"]:
                continue  # manual decisions always win; rules never overwrite
            if match.get("category") and cluster["category"] != match["category"]:
                continue
            if match.get("stem_contains") and match["stem_contains"] not in cluster["stem"]:
                continue
            decision: dict[str, Any] = {
                "action": rule["action"],
                "source": "rule",
                "decided_at": _now(),
            }
            if rule.get("note"):
                decision["note"] = rule["note"]
            if rule["action"] == "merge_all":
                decision["canonical_id"] = _pick_canonical(cluster["members"], rule.get("canonical", "most_chunks"))
            data["decisions"][cluster["cluster_id"]] = decision
            applied += 1
        data["rules"].append({**rule, "applied": applied, "recorded_at": _now()})
        return applied

    return _write_locked(Path(decisions_path), mutate)


def speak_text(cluster: dict[str, Any]) -> str:
    """Spoken summary of a cluster, optimized for TTS (short, concrete)."""
    members = cluster["members"]
    names = sorted({m["name"] for m in members})
    parts = [f"Cluster {cluster['stem']!r}, category {cluster['category']}, {len(members)} entries."]
    if len(names) == 1:
        parts.append(f"All named {names[0]}.")
    else:
        parts.append("Name variants: " + ", ".join(names) + ".")
    by_member = ", ".join(f"{m['name']} as {m['type']} with {m.get('chunks', 0)} chunks" for m in members)
    parts.append(by_member + ".")
    parts.append("Merge, keep separate, mixed, or skip?")
    return " ".join(parts)


def stats(batch_path: str | Path, decisions_path: str | Path) -> dict[str, Any]:
    decided = set(_load_decisions(Path(decisions_path))["decisions"])
    out: dict[str, Any] = {}
    for cluster in load_flag_batch(batch_path):
        bucket = out.setdefault(cluster["category"], {"total": 0, "decided": 0})
        bucket["total"] += 1
        if cluster["cluster_id"] in decided:
            bucket["decided"] += 1
    return out
