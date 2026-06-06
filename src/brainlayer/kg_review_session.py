"""KG flag-batch review session driver shared by voice and visual review surfaces.

Pure-file module: reads the category-sorted flag-batch JSON emitted by phase-1
(`eval_results/kg-phase1-flag-batch-*.json`) and maintains the shared dashboard
decisions JSON that `kg_cleanup_apply.py` (or a follow-up applier) can consume.
No DB access here.

Decisions file contract: `kg-flag-decisions-v1`, matching the dashboard export:

    {
      "schema": "kg-flag-decisions-v1",
      "source": "kg-phase1-flag-batch-2026-06-05",
      "rules": {"<category>": "merge" | "keep"},
      "per_category": {"<cat>": {total, explicit, by_rule, undecided, rule}},
      "counts": {merge_clusters, rows_merged_away, keep, explicit, by_rule},
      "merge": [
        {
          "stem": "...",
          "category": "...",
          "source": "explicit" | "rule" | "voice" | "voice-rule",
          "canonical": {"id": "...", "name": "...", "type": "..."},
          "members": [{"id": "...", "name": "...", "type": "..."}],
          "note": "optional verbatim reasoning",
          "decided_at": "optional ISO-8601 timestamp"
        }
      ],
      "keep": [{"stem": "...", "category": "...", "source": "..."}]
    }

Contract gaps handled additively:

* `skip` is not exported as a decision, so it remains undecided in v1 counts.
  The driver records optional top-level `skipped` rows so a voice session does
  not re-ask skipped clusters.
* `mixed` cannot be expressed in v1. The driver exports it as `keep` with a
  `MIXED: <member map json>` note and also records optional top-level
  `needs_v1_1` rows for the schema revision discussion.

Merge-safety: every write is read-modify-write under an exclusive `fcntl.flock`
on a sidecar lock file, then an atomic `os.replace`. Two writers (dashboard +
voice session) interleave safely; last write per cluster wins, never per file.

AIDEV-NOTE: engine-agnostic on purpose (Etan 2026-06-05: no lock-in to the old
voice architecture) -- the voice loop, a browser dashboard, and any future WS
bridge all drive this same module.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DECISIONS_SCHEMA = "kg-flag-decisions-v1"
DEFAULT_SOURCE = "kg-phase1-flag-batch-2026-06-05"

VALID_RECORD_ACTIONS = {"merge", "keep", "merge_all", "keep_all", "mixed", "skip"}
VALID_RULE_ACTIONS = {"merge", "keep", "merge_all", "keep_all"}
VALID_MEMBER_ACTIONS = {"merge", "keep", "prune"}
VALID_ITEM_SOURCES = {"explicit", "rule", "voice", "voice-rule"}
RULE_SOURCES = {"rule", "voice-rule"}
CONTRACT_ITEM_FIELDS = {"stem", "category", "source", "canonical", "members", "note", "decided_at"}


def cluster_id(category: str, stem: str) -> str:
    return f"{category}:{stem}"


def load_flag_batch(batch_path: str | Path) -> list[dict[str, Any]]:
    """Flatten {category: [cluster, ...]} into a stable ordered list with ids."""
    raw = json.loads(Path(batch_path).read_text())
    clusters: list[dict[str, Any]] = []
    for category in raw:
        for cluster in raw[category]:
            members = cluster.get("members", [])
            clusters.append(
                {
                    "cluster_id": cluster_id(category, cluster["stem"]),
                    "category": category,
                    "stem": cluster["stem"],
                    "size": cluster.get("size", len(members)),
                    "members": members,
                    "item_kind": cluster.get("item_kind") or _item_kind_from_members(members),
                }
            )
    return clusters


def _item_kind_from_members(members: list[dict[str, Any]]) -> str:
    return "question" if any(member.get("type") == "question" for member in members) else "cluster"


def _source_from_batch(batch_path: str | Path) -> str:
    stem = Path(batch_path).stem
    return stem or DEFAULT_SOURCE


def _empty_decisions(source: str = DEFAULT_SOURCE) -> dict[str, Any]:
    return {
        "schema": DECISIONS_SCHEMA,
        "source": source,
        "rules": {},
        "per_category": {},
        "counts": {"merge_clusters": 0, "rows_merged_away": 0, "keep": 0, "explicit": 0, "by_rule": 0},
        "merge": [],
        "keep": [],
    }


def _ensure_v1_shape(data: dict[str, Any], source: str) -> dict[str, Any]:
    if "schema" not in data:
        data["schema"] = DECISIONS_SCHEMA
    if data.get("schema") != DECISIONS_SCHEMA:
        raise ValueError(f"decisions file must use schema {DECISIONS_SCHEMA!r}; got {data.get('schema')!r}")
    data.setdefault("source", source)
    data.setdefault("rules", {})
    data.setdefault("per_category", {})
    data.setdefault("counts", _empty_decisions(source)["counts"])
    data.setdefault("merge", [])
    data.setdefault("keep", [])
    if not isinstance(data["rules"], dict):
        raise ValueError("decisions rules must be an object")
    if not isinstance(data["merge"], list) or not isinstance(data["keep"], list):
        raise ValueError("decisions merge and keep must be arrays")
    return data


def _load_decisions(decisions_path: Path, source: str = DEFAULT_SOURCE) -> dict[str, Any]:
    if decisions_path.exists():
        data = json.loads(decisions_path.read_text())
    else:
        data = _empty_decisions(source)
    return _ensure_v1_shape(data, source)


def _write_locked(
    decisions_path: Path,
    source: str,
    clusters: list[dict[str, Any]],
    mutate: Callable[[dict[str, Any]], Any],
) -> Any:
    """Read-modify-write under flock + atomic replace. Returns mutate's result."""
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = decisions_path.with_suffix(decisions_path.suffix + ".lock")
    with open(lock_path, "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        data = _load_decisions(decisions_path, source)
        result = mutate(data)
        _recompute_rollups(data, clusters)
        _drop_empty_optional_lists(data)
        tmp_path = decisions_path.with_suffix(decisions_path.suffix + ".tmp")
        with open(tmp_path, "w") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, decisions_path)
    return result


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_action(action: Any) -> str:
    if action not in VALID_RECORD_ACTIONS:
        raise ValueError(f"invalid action {action!r}; expected one of {sorted(VALID_RECORD_ACTIONS)}")
    if action == "merge_all":
        return "merge"
    if action == "keep_all":
        return "keep"
    return action


def _normalize_rule_action(action: Any) -> str:
    if action not in VALID_RULE_ACTIONS:
        raise ValueError(f"invalid rule action {action!r}; rules accept merge or keep")
    if action == "merge_all":
        return "merge"
    if action == "keep_all":
        return "keep"
    return action


def _normalize_record_source(source: Any) -> str:
    if source is None:
        return "voice"
    if source == "visual":
        return "explicit"
    if source not in VALID_ITEM_SOURCES:
        raise ValueError(f"invalid source {source!r}; expected one of {sorted(VALID_ITEM_SOURCES)}")
    return source


def _normalize_rule_source(source: Any) -> str:
    if source in (None, "voice", "voice-rule"):
        return "voice-rule"
    if source in ("visual", "explicit", "rule"):
        return "rule"
    raise ValueError(f"invalid rule source {source!r}")


def _validate_member_map(members: Any) -> dict[str, str]:
    if not isinstance(members, dict) or not members:
        raise ValueError("mixed decision requires a non-empty members map")
    bad = set(members.values()) - VALID_MEMBER_ACTIONS
    if bad:
        raise ValueError(f"invalid member actions: {sorted(bad)}")
    return members


def _validate_record_decision(decision: dict[str, Any]) -> tuple[str, str]:
    action = _normalize_action(decision.get("action"))
    source = _normalize_record_source(decision.get("source"))
    if action == "mixed":
        _validate_member_map(decision.get("members"))
    return action, source


def _key(category: str, stem: str) -> tuple[str, str]:
    return category, stem


def _cluster_key(cluster: dict[str, Any]) -> tuple[str, str]:
    return _key(cluster["category"], cluster["stem"])


def _item_key(item: dict[str, Any]) -> tuple[str, str]:
    return _key(item["category"], item["stem"])


def _find_cluster(clusters: list[dict[str, Any]], cid: str) -> dict[str, Any]:
    for cluster in clusters:
        if cluster["cluster_id"] == cid:
            return cluster
    raise ValueError(f"unknown cluster_id {cid!r}")


def _member_ref(member: dict[str, Any]) -> dict[str, Any]:
    return {"id": member["id"], "name": member["name"], "type": member["type"]}


def _canonical_override(decision: dict[str, Any]) -> tuple[str | None, str | None]:
    canonical = decision.get("canonical")
    canonical_id = decision.get("canonical_id")
    canonical_name = decision.get("canonical_name")
    if isinstance(canonical, dict):
        canonical_id = canonical_id or canonical.get("id")
        canonical_name = canonical_name or canonical.get("name")
    elif isinstance(canonical, str):
        canonical_id = canonical_id or canonical
        canonical_name = canonical_name or canonical
    return canonical_id, canonical_name


def _select_canonical(cluster: dict[str, Any], decision: dict[str, Any] | None = None) -> dict[str, Any]:
    members = cluster["members"]
    if not members:
        raise ValueError(f"cluster {cluster['cluster_id']!r} has no members")
    if not decision:
        return members[0]
    canonical_id, canonical_name = _canonical_override(decision)
    if not canonical_id and not canonical_name:
        return members[0]
    for member in members:
        if canonical_id and member["id"] == canonical_id:
            return member
        if canonical_name and member["name"] == canonical_name:
            return member
    raise ValueError(
        f"canonical override {canonical_id or canonical_name!r} is not in cluster {cluster['cluster_id']!r}"
    )


def _merge_item(
    cluster: dict[str, Any],
    source: str,
    decided_at: str,
    note: str | None = None,
    decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    canonical = _select_canonical(cluster, decision)
    item: dict[str, Any] = {
        "stem": cluster["stem"],
        "category": cluster["category"],
        "source": source,
        "canonical": _member_ref(canonical),
        "members": [_member_ref(member) for member in cluster["members"] if member["id"] != canonical["id"]],
        "decided_at": decided_at,
    }
    if note:
        item["note"] = note
    return item


def _keep_item(cluster: dict[str, Any], source: str, decided_at: str, note: str | None = None) -> dict[str, Any]:
    item = {"stem": cluster["stem"], "category": cluster["category"], "source": source, "decided_at": decided_at}
    if note:
        item["note"] = note
    return item


def _skip_item(cluster: dict[str, Any], source: str, decided_at: str, note: str | None = None) -> dict[str, Any]:
    item = {"stem": cluster["stem"], "category": cluster["category"], "source": source, "decided_at": decided_at}
    if note:
        item["note"] = note
    return item


def _remove_keyed(data: dict[str, Any], key: tuple[str, str], fields: tuple[str, ...] = ("merge", "keep")) -> None:
    for field in fields:
        data[field] = [item for item in data.get(field, []) if _item_key(item) != key]


def _upsert_keyed(data: dict[str, Any], field: str, item: dict[str, Any]) -> None:
    key = _item_key(item)
    data[field] = [old for old in data.get(field, []) if _item_key(old) != key]
    data[field].append(item)


def _upsert_optional(data: dict[str, Any], field: str, item: dict[str, Any]) -> None:
    key = _item_key(item)
    data[field] = [old for old in data.get(field, []) if _item_key(old) != key]
    data[field].append(item)


def _remove_optional(data: dict[str, Any], key: tuple[str, str], field: str) -> None:
    if field in data:
        data[field] = [item for item in data[field] if _item_key(item) != key]


def _foreign_fields(data: dict[str, Any], key: tuple[str, str], fields: tuple[str, ...]) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    for field in fields:
        for item in data.get(field, []):
            if _item_key(item) == key:
                extras.update({k: v for k, v in item.items() if k not in CONTRACT_ITEM_FIELDS})
    return extras


def _with_foreign_fields(item: dict[str, Any], foreign: dict[str, Any]) -> dict[str, Any]:
    return {**foreign, **item}


def _drop_empty_optional_lists(data: dict[str, Any]) -> None:
    for field in ("skipped", "needs_v1_1"):
        if field in data and not data[field]:
            del data[field]


def _decided_keys(data: dict[str, Any]) -> set[tuple[str, str]]:
    return {_item_key(item) for item in data.get("merge", []) + data.get("keep", [])}


def _skipped_keys(data: dict[str, Any]) -> set[tuple[str, str]]:
    return {_item_key(item) for item in data.get("skipped", [])}


def _recompute_rollups(data: dict[str, Any], clusters: list[dict[str, Any]]) -> None:
    cluster_keys = {_cluster_key(cluster) for cluster in clusters}
    per_category: dict[str, dict[str, Any]] = {}
    for cluster in clusters:
        row = per_category.setdefault(
            cluster["category"],
            {"total": 0, "explicit": 0, "by_rule": 0, "undecided": 0, "rule": data["rules"].get(cluster["category"])},
        )
        row["total"] += 1

    merge = data.get("merge", [])
    keep = data.get("keep", [])
    exported = merge + keep
    seen: set[tuple[str, str]] = set()
    for item in exported:
        item_key = _item_key(item)
        if item_key in seen:
            raise ValueError(f"duplicate decision for {item['category']}:{item['stem']}")
        seen.add(item_key)
        if item_key not in cluster_keys:
            raise ValueError(f"decision references unknown cluster {item['category']}:{item['stem']}")
        if item["source"] not in VALID_ITEM_SOURCES:
            raise ValueError(f"invalid decision source {item['source']!r}")
        row = per_category[item["category"]]
        if item["source"] in RULE_SOURCES:
            row["by_rule"] += 1
        else:
            row["explicit"] += 1

    for category, rule in data["rules"].items():
        if rule not in {"merge", "keep"}:
            raise ValueError(f"invalid rule for {category!r}: {rule!r}")
        per_category.setdefault(category, {"total": 0, "explicit": 0, "by_rule": 0, "undecided": 0, "rule": rule})
        per_category[category]["rule"] = rule

    for row in per_category.values():
        row["undecided"] = row["total"] - row["explicit"] - row["by_rule"]
        if row["undecided"] < 0:
            raise ValueError("decision counts exceed category total")

    data["per_category"] = per_category
    data["counts"] = {
        "merge_clusters": len(merge),
        "rows_merged_away": sum(len(item["members"]) for item in merge),
        "keep": len(keep),
        "explicit": sum(1 for item in exported if item["source"] not in RULE_SOURCES),
        "by_rule": sum(1 for item in exported if item["source"] in RULE_SOURCES),
    }


def record_decision(
    batch_path: str | Path,
    decisions_path: str | Path,
    cluster_id: str,
    decision: dict[str, Any],
) -> dict[str, Any]:
    """Validate and persist one cluster decision in kg-flag-decisions-v1 shape."""
    action, source = _validate_record_decision(decision)
    clusters = load_flag_batch(batch_path)
    cluster = _find_cluster(clusters, cluster_id)
    source_name = _source_from_batch(batch_path)
    decided_at = _now()
    key = _cluster_key(cluster)
    note = decision.get("note")

    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        foreign = _foreign_fields(data, key, ("merge", "keep", "skipped"))
        needs_foreign = _foreign_fields(data, key, ("needs_v1_1",))
        _remove_keyed(data, key, fields=("merge", "keep"))
        _remove_optional(data, key, "skipped")
        _remove_optional(data, key, "needs_v1_1")

        if action == "skip":
            item = _with_foreign_fields(_skip_item(cluster, source, decided_at, note), foreign)
            _upsert_optional(data, "skipped", item)
            return item

        if action == "mixed":
            member_map = _validate_member_map(decision.get("members"))
            mixed_json = json.dumps(member_map, sort_keys=True, ensure_ascii=False)
            mixed_note = f"MIXED: {mixed_json}"
            if note:
                mixed_note = f"{mixed_note}; {note}"
            item = _with_foreign_fields(_keep_item(cluster, source, decided_at, mixed_note), foreign)
            _upsert_keyed(data, "keep", item)
            needs_item: dict[str, Any] = {
                "stem": cluster["stem"],
                "category": cluster["category"],
                "source": source,
                "members": member_map,
                "decided_at": decided_at,
            }
            if note:
                needs_item["note"] = note
            needs_item = _with_foreign_fields(needs_item, needs_foreign)
            _upsert_optional(data, "needs_v1_1", needs_item)
            return item

        if action == "merge":
            item = _with_foreign_fields(_merge_item(cluster, source, decided_at, note, decision), foreign)
            _upsert_keyed(data, "merge", item)
            return item

        item = _with_foreign_fields(_keep_item(cluster, source, decided_at, note), foreign)
        _upsert_keyed(data, "keep", item)
        return item

    return _write_locked(Path(decisions_path), source_name, clusters, mutate)


def next_undecided(
    batch_path: str | Path,
    decisions_path: str | Path,
    category: str | None = None,
) -> dict[str, Any] | None:
    """Next cluster (batch order) without an exported decision or session skip."""
    clusters = load_flag_batch(batch_path)
    data = _load_decisions(Path(decisions_path), _source_from_batch(batch_path))
    decided = _decided_keys(data)
    skipped = _skipped_keys(data)
    for cluster in clusters:
        if category and cluster["category"] != category:
            continue
        key = _cluster_key(cluster)
        if key not in decided and key not in skipped:
            return cluster
    return None


def apply_rule(
    batch_path: str | Path,
    decisions_path: str | Path,
    rule: dict[str, Any],
) -> int:
    """Bulk-decide all undecided clusters in one category. Never overwrites.

    rule = {match: {category}, action: "merge"|"keep", note?, source?}
    Returns number of clusters newly decided.

    `mixed` is rejected for rules: it needs a per-member map, which a bulk rule
    cannot supply for arbitrary clusters. `skip` is also rejected because v1
    rules are only per-category merge/keep decisions.
    """
    action = _normalize_rule_action(rule.get("action"))
    source = _normalize_rule_source(rule.get("source"))
    match = rule.get("match", {})
    category = match.get("category") or rule.get("category")
    if not category:
        raise ValueError("rule requires match.category")
    if set(match) - {"category"}:
        raise ValueError("kg-flag-decisions-v1 rules are per-category only")

    clusters = load_flag_batch(batch_path)
    source_name = _source_from_batch(batch_path)
    decided_at = _now()
    note = rule.get("note")

    def mutate(data: dict[str, Any]) -> int:
        applied = 0
        for cluster in clusters:
            if cluster["category"] != category:
                continue
            key = _cluster_key(cluster)
            if key in _decided_keys(data):
                continue
            foreign = _foreign_fields(data, key, ("skipped",))
            _remove_optional(data, key, "skipped")
            if action == "merge":
                _upsert_keyed(
                    data, "merge", _with_foreign_fields(_merge_item(cluster, source, decided_at, note), foreign)
                )
            else:
                _upsert_keyed(
                    data, "keep", _with_foreign_fields(_keep_item(cluster, source, decided_at, note), foreign)
                )
            applied += 1
        if applied:
            data["rules"][category] = action
        return applied

    return _write_locked(Path(decisions_path), source_name, clusters, mutate)


def speak_text(cluster: dict[str, Any]) -> str:
    """Spoken summary of a cluster, optimized for TTS (short, concrete)."""
    members = cluster["members"]
    if any(m.get("type") == "question" for m in members):
        # Dictionary-question pseudo-cluster: the member name IS the question.
        question = next(m["name"] for m in members if m.get("type") == "question")
        return f"{question} Answer in your own words; I will capture it verbatim and mark this item handled."
    names = sorted({m["name"] for m in members})
    parts = [f"Cluster {cluster['stem']!r}, category {cluster['category']}, {len(members)} entries."]
    if len(names) == 1:
        # Single shared name: say it once, with the type spread only.
        types = ", ".join(sorted({m["type"] for m in members}))
        parts.append(f"All named {names[0]}, typed {types}.")
    else:
        parts.append("Name variants: " + ", ".join(names) + ".")
        by_member = ", ".join(f"{m['name']} as {m['type']}" for m in members)
        parts.append(by_member + ".")
    parts.append("Merge, keep separate, mixed, or skip?")
    return " ".join(parts)


def stats(batch_path: str | Path, decisions_path: str | Path) -> dict[str, Any]:
    clusters = load_flag_batch(batch_path)
    data = _load_decisions(Path(decisions_path), _source_from_batch(batch_path))
    _recompute_rollups(data, clusters)
    return {"counts": data["counts"], "per_category": data["per_category"]}
