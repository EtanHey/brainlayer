"""Harvest Etan voice-session answers and apply-safe KG decisions.

The live review batch includes six dictionary questions followed by contested
KG clusters whose first member is synthetic context (`ctx-*`). This module
splits question notes into dictionary answers and emits a clean decisions file
that `scripts/kg_cleanup_apply.py --decisions` can validate without ever
touching the BrainLayer DB.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

DECISIONS_SCHEMA = "kg-flag-decisions-v1"
QUESTION_STEM_RE = re.compile(r"^Q([1-6]) of 6\b")
RULE_SOURCES = {"rule", "voice-rule"}


def harvest_session(
    batch_path: str | Path,
    session_decisions_path: str | Path,
    *,
    answers_path: str | Path,
    decisions_path: str | Path,
) -> dict[str, Any]:
    """Write dictionary answers and ctx-free cleanup decisions.

    The output decisions file is validated by importing
    `scripts/kg_cleanup_apply.py::load_decisions`; this function never opens the
    BrainLayer SQLite database.
    """
    clusters = _load_batch(batch_path)
    decisions = _load_decisions(session_decisions_path)
    _validate_decision_stems(decisions, clusters)

    questions = _question_clusters(clusters)
    answers = _build_answers(questions, decisions)
    clean = _build_clean_decisions(clusters, questions, decisions)

    answers_outputs = _write_answers(Path(answers_path), answers)
    decisions_out = Path(decisions_path)
    decisions_out.parent.mkdir(parents=True, exist_ok=True)
    decisions_out.write_text(json.dumps(clean, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _load_apply_decisions(decisions_out)

    return {
        "answers_count": len(answers),
        "answers_json": str(answers_outputs["json"]),
        "answers_markdown": str(answers_outputs["markdown"]),
        "decisions": str(decisions_out),
        "decision_counts": clean["counts"],
    }


def _load_batch(path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("session batch must be a category object")

    clusters: list[dict[str, Any]] = []
    for category, rows in raw.items():
        if not isinstance(rows, list):
            raise ValueError(f"session batch category {category!r} must be a list")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("stem"), str):
                raise ValueError(f"invalid cluster in category {category!r}")
            members = row.get("members", [])
            if not isinstance(members, list):
                raise ValueError(f"cluster {row.get('stem')!r} members must be a list")
            _validate_members(row["stem"], members)
            clusters.append({**row, "category": category, "members": members})
    return clusters


def _validate_members(stem: str, members: list[Any]) -> None:
    for member in members:
        if not isinstance(member, dict) or not all(isinstance(member.get(key), str) for key in ("id", "name", "type")):
            raise ValueError(f"invalid member in cluster {stem!r}: {member!r}")


def _load_decisions(path: str | Path) -> dict[str, Any]:
    decisions = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(decisions, dict):
        raise ValueError(f"session decisions must be a JSON object, got {type(decisions).__name__}")
    if decisions.get("schema") != DECISIONS_SCHEMA:
        raise ValueError(f"session decisions schema must be {DECISIONS_SCHEMA!r}; got {decisions.get('schema')!r}")
    for field in ("merge", "keep", "skipped", "needs_v1_1"):
        if not isinstance(decisions.get(field, []), list):
            raise ValueError(f"session decisions field {field!r} must be a list")
        for item in decisions.get(field, []):
            if not isinstance(item, dict):
                raise ValueError(f"decision item in {field!r} must be an object, got {type(item).__name__}")
    return decisions


def _question_clusters(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dictionary_category = [cluster for cluster in clusters if cluster["category"] == "dictionary-questions"]
    questions = dictionary_category or [cluster for cluster in clusters if QUESTION_STEM_RE.match(cluster["stem"])]
    if len(questions) != 6:
        raise ValueError(f"expected exactly 6 dictionary question stems, found {len(questions)}")

    by_number: dict[int, dict[str, Any]] = {}
    for cluster in questions:
        match = QUESTION_STEM_RE.match(cluster["stem"])
        if match is None:
            raise ValueError(f"question stem must start with Qn of 6: {cluster['stem']!r}")
        number = int(match.group(1))
        if number in by_number:
            raise ValueError(f"duplicate question stem number Q{number}")
        by_number[number] = cluster

    expected = set(range(1, 7))
    if set(by_number) != expected:
        raise ValueError(f"question stems must cover Q1-Q6, got {sorted(by_number)}")
    return [by_number[number] for number in range(1, 7)]


def _cluster_key(cluster: dict[str, Any]) -> tuple[str, str]:
    return cluster["category"], cluster["stem"]


def _decision_key(item: dict[str, Any], field: str) -> tuple[str, str]:
    stem = item.get("stem")
    if not isinstance(stem, str):
        raise ValueError(f"decision in {field} is missing a string stem")
    category = item.get("category")
    if not isinstance(category, str):
        raise ValueError(f"decision in {field} is missing a string category")
    return category, stem


def _validate_decision_stems(decisions: dict[str, Any], clusters: list[dict[str, Any]]) -> None:
    cluster_keys = {_cluster_key(cluster) for cluster in clusters}
    for field in ("merge", "keep", "skipped", "needs_v1_1"):
        for item in decisions.get(field, []):
            category, stem = _decision_key(item, field)
            if (category, stem) not in cluster_keys:
                raise ValueError(f"unknown decision stem {stem!r} in category {category!r} for {field}")


def _build_answers(questions: list[dict[str, Any]], decisions: dict[str, Any]) -> list[dict[str, Any]]:
    decisions_by_key = _decision_items_by_key(decisions)
    answers = []
    for question in questions:
        stem = question["stem"]
        decision = decisions_by_key.get(_cluster_key(question))
        if decision is None:
            raise ValueError(f"missing dictionary answer decision for {stem!r}")
        note = decision.get("note")
        if not isinstance(note, str) or not note.strip():
            raise ValueError(f"dictionary answer for {stem!r} must be a non-empty note")
        answers.append(
            {
                "stem": stem,
                "question": _question_text(question),
                "answer": note,
                "decided_at": decision.get("decided_at"),
            }
        )
    return answers


def _decision_items_by_key(decisions: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for field in ("merge", "keep", "skipped"):
        for item in decisions.get(field, []):
            key = _decision_key(item, field)
            if key in by_key:
                category, stem = key
                raise ValueError(f"duplicate exported decision for stem {stem!r} in category {category!r}")
            by_key[key] = item
    return by_key


def _question_text(cluster: dict[str, Any]) -> str:
    members = cluster.get("members", [])
    if len(members) != 1:
        raise ValueError(f"question cluster {cluster['stem']!r} must have exactly one member")
    member = members[0]
    if member.get("type") != "question":
        raise ValueError(f"question cluster {cluster['stem']!r} first member must have type 'question'")
    name = member.get("name")
    if not isinstance(name, str):
        raise ValueError(f"question cluster {cluster['stem']!r} member name must be a string")
    return name


def _build_clean_decisions(
    clusters: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    decisions: dict[str, Any],
) -> dict[str, Any]:
    question_keys = {_cluster_key(cluster) for cluster in questions}
    clusters_by_key = _clusters_by_key(clusters)

    merge = []
    for item in decisions.get("merge", []):
        key = _decision_key(item, "merge")
        if key not in question_keys:
            merge.append(_clean_merge_item(item, clusters_by_key[key]))

    keep = []
    for item in decisions.get("keep", []):
        key = _decision_key(item, "keep")
        if key not in question_keys:
            keep.append(_clean_keep_item(item, clusters_by_key[key]))

    needs = []
    for item in decisions.get("needs_v1_1", []):
        key = _decision_key(item, "needs_v1_1")
        if key not in question_keys:
            needs.append(_clean_needs_item(item, clusters_by_key[key]))
    needs = [item for item in needs if item is not None]

    rules: dict[str, Any] = {}
    clean: dict[str, Any] = {
        "schema": DECISIONS_SCHEMA,
        "source": decisions.get("source"),
        "rules": rules,
        "per_category": _per_category(clusters, question_keys, merge, keep, rules),
        "counts": _counts(merge, keep),
        "merge": merge,
        "keep": keep,
    }
    if needs:
        clean["needs_v1_1"] = needs
    if _contains_ctx(clean):
        raise ValueError("ctx- member leaked into clean decisions")
    return clean


def _clusters_by_key(clusters: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for cluster in clusters:
        key = _cluster_key(cluster)
        if key in by_key:
            category, stem = key
            raise ValueError(f"duplicate cluster stem {stem!r} in category {category!r}")
        by_key[key] = cluster
    return by_key


def _clean_merge_item(item: dict[str, Any], cluster: dict[str, Any]) -> dict[str, Any]:
    real_members = _real_members(cluster)
    members_by_id = {member["id"]: member for member in real_members}
    if not real_members:
        raise ValueError(f"cluster {cluster['stem']!r} has no real members after ctx-strip")

    canonical = item.get("canonical")
    canonical_id = canonical.get("id") if isinstance(canonical, dict) else None
    if canonical_id in members_by_id:
        clean_canonical = _member_ref(members_by_id[canonical_id])
    elif canonical_id is None or _is_ctx_id(canonical_id):
        clean_canonical = _member_ref(_canonical_by_chunks(real_members))
    else:
        raise ValueError(f"canonical {canonical_id!r} is not a real member of {cluster['stem']!r}")

    selected_members = item.get("members")
    if not isinstance(selected_members, list):
        raise ValueError(f"merge decision for {cluster['stem']!r} must include a members list")
    original_loser_ids = []
    for member in selected_members:
        if not isinstance(member, dict) or not isinstance(member.get("id"), str):
            raise ValueError(f"merge member for {cluster['stem']!r} must be an object with a string id")
        original_loser_ids.append(member["id"])
    unknown_loser_ids = [
        member_id for member_id in original_loser_ids if member_id not in members_by_id and not _is_ctx_id(member_id)
    ]
    if unknown_loser_ids:
        raise ValueError(f"merge members {unknown_loser_ids!r} are not real members of {cluster['stem']!r}")
    clean_member_ids = _dedupe([member_id for member_id in original_loser_ids if member_id in members_by_id])
    if not clean_member_ids:
        raise ValueError(f"merge decision for {cluster['stem']!r} selects no real members after ctx-strip")
    members = [
        _member_ref(members_by_id[member_id]) for member_id in clean_member_ids if member_id != clean_canonical["id"]
    ]
    if not members:
        raise ValueError(f"merge decision for {cluster['stem']!r} selects no real loser members")

    clean: dict[str, Any] = {
        "stem": cluster["stem"],
        "category": cluster["category"],
        "source": item.get("source", "voice"),
        "canonical": clean_canonical,
        "members": members,
    }
    _copy_optional_item_fields(item, clean)
    if isinstance(clean.get("note"), str):
        clean["note"] = _clean_note(clean["note"], cluster)
    return clean


def _clean_keep_item(item: dict[str, Any], cluster: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {
        "stem": cluster["stem"],
        "category": cluster["category"],
        "source": item.get("source", "voice"),
    }
    _copy_optional_item_fields(item, clean)
    if isinstance(clean.get("note"), str):
        clean["note"] = _clean_note(clean["note"], cluster)
    return clean


def _clean_needs_item(item: dict[str, Any], cluster: dict[str, Any]) -> dict[str, Any] | None:
    members = item.get("members")
    if not isinstance(members, dict):
        raise ValueError(f"needs_v1_1 item for {cluster['stem']!r} must include a member map")
    real_member_ids = {member["id"] for member in _real_members(cluster)}
    unknown_member_ids = [
        member_id for member_id in members if member_id not in real_member_ids and not _is_ctx_id(member_id)
    ]
    if unknown_member_ids:
        raise ValueError(f"mixed members {unknown_member_ids!r} are not real members of {cluster['stem']!r}")
    filtered_members = {member_id: action for member_id, action in members.items() if not _is_ctx_id(member_id)}
    if not filtered_members:
        return None
    clean: dict[str, Any] = {
        "stem": cluster["stem"],
        "category": cluster["category"],
        "source": item.get("source", "voice"),
        "members": filtered_members,
    }
    _copy_optional_item_fields(item, clean)
    if isinstance(clean.get("note"), str):
        clean["note"] = _clean_note(clean["note"], cluster)
    return clean


def _copy_optional_item_fields(source: dict[str, Any], target: dict[str, Any]) -> None:
    for field in ("note", "decided_at"):
        if field in source:
            target[field] = source[field]


def _real_members(cluster: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        member
        for member in cluster.get("members", [])
        if isinstance(member, dict) and not _is_ctx_id(member.get("id")) and member.get("type") != "context"
    ]


def _member_ref(member: dict[str, Any]) -> dict[str, Any]:
    return {"id": member["id"], "name": member["name"], "type": member["type"]}


def _canonical_by_chunks(members: list[dict[str, Any]]) -> dict[str, Any]:
    return max(members, key=lambda member: int(member.get("chunks") or 0))


def _is_ctx_id(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("ctx-")


def _dedupe(values: list[Any]) -> list[Any]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _clean_mixed_note(note: str) -> str:
    prefix = "MIXED: "
    if not note.startswith(prefix):
        return note
    payload, separator, suffix = note[len(prefix) :].partition("; ")
    try:
        member_map = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("MIXED note payload must be valid JSON object") from exc
    if not isinstance(member_map, dict):
        raise ValueError("MIXED note payload must be a JSON object")
    filtered = {member_id: action for member_id, action in member_map.items() if not _is_ctx_id(member_id)}
    clean = f"{prefix}{json.dumps(filtered, sort_keys=True, ensure_ascii=False)}"
    if separator:
        clean = f"{clean}; {suffix}"
    return clean


def _clean_note(note: str, cluster: dict[str, Any]) -> str:
    clean = _clean_mixed_note(note)
    for ctx_id in sorted(_ctx_member_ids(cluster), key=len, reverse=True):
        clean = re.sub(rf"(?<![\w-]){re.escape(ctx_id)}(?![\w-])", "synthetic context", clean)
    return clean


def _ctx_member_ids(cluster: dict[str, Any]) -> list[str]:
    return [
        member["id"]
        for member in cluster.get("members", [])
        if isinstance(member, dict) and _is_ctx_id(member.get("id"))
    ]


def _counts(merge: list[dict[str, Any]], keep: list[dict[str, Any]]) -> dict[str, int]:
    exported = merge + keep
    return {
        "merge_clusters": len(merge),
        "rows_merged_away": sum(len(item["members"]) for item in merge),
        "keep": len(keep),
        "explicit": sum(1 for item in exported if item.get("source") not in RULE_SOURCES),
        "by_rule": sum(1 for item in exported if item.get("source") in RULE_SOURCES),
    }


def _per_category(
    clusters: list[dict[str, Any]],
    question_keys: set[tuple[str, str]],
    merge: list[dict[str, Any]],
    keep: list[dict[str, Any]],
    rules: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    per_category: dict[str, dict[str, Any]] = {}
    cluster_keys = set()
    for cluster in clusters:
        key = _cluster_key(cluster)
        if key in question_keys:
            continue
        cluster_keys.add(key)
        category = cluster["category"]
        row = per_category.setdefault(
            category,
            {"total": 0, "explicit": 0, "by_rule": 0, "undecided": 0, "rule": rules.get(category)},
        )
        row["total"] += 1

    seen: set[tuple[str, str]] = set()
    for item in merge + keep:
        key = _decision_key(item, "exported")
        if key in seen:
            category, stem = key
            raise ValueError(f"duplicate decision for {category}:{stem}")
        seen.add(key)
        if key not in cluster_keys:
            category, stem = key
            raise ValueError(f"decision references unknown non-question cluster {category}:{stem}")
        category = item["category"]
        row = per_category[category]
        if item.get("source") in RULE_SOURCES:
            row["by_rule"] += 1
        else:
            row["explicit"] += 1

    for category, rule in rules.items():
        per_category.setdefault(
            category,
            {"total": 0, "explicit": 0, "by_rule": 0, "undecided": 0, "rule": rule},
        )
        per_category[category]["rule"] = rule

    for row in per_category.values():
        row["undecided"] = row["total"] - row["explicit"] - row["by_rule"]
        if row["undecided"] < 0:
            raise ValueError("decision counts exceed category total")
    return per_category


def _write_answers(path: Path, answers: list[dict[str, Any]]) -> dict[str, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        json_path = path
        markdown_path = path.with_suffix(".md")
    else:
        markdown_path = path
        json_path = path.with_suffix(".json")

    json_path.write_text(json.dumps(answers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_answers_markdown(answers), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def _render_answers_markdown(answers: list[dict[str, Any]]) -> str:
    lines = ["# Dictionary Answers", ""]
    for answer in answers:
        lines.extend(
            [
                f"## {answer['stem']}",
                "",
                f"Question: {answer['question']}",
                "",
                f"Answer: {answer['answer']}",
                "",
                f"Decided at: {answer.get('decided_at') or ''}",
                "",
            ]
        )
    return "\n".join(lines)


def _load_apply_decisions(path: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    validator_path = root / "scripts" / "kg_cleanup_apply.py"
    spec = importlib.util.spec_from_file_location("brainlayer_kg_cleanup_apply_validator", validator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import kg cleanup validator from {validator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_decisions(path)


def _contains_ctx(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_ctx(key) or _contains_ctx(val) for key, val in value.items())
    if isinstance(value, list):
        return any(_contains_ctx(item) for item in value)
    return _is_ctx_id(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Harvest Etan KG voice-session answers and clean decisions.")
    parser.add_argument("--batch", required=True, help="Session batch JSON path")
    parser.add_argument(
        "--session-decisions",
        "--input-decisions",
        dest="session_decisions",
        required=True,
        help="Raw session decisions JSON path",
    )
    parser.add_argument("--answers", required=True, help="Answers output path; JSON/Markdown sidecar is also written")
    parser.add_argument("--decisions", required=True, help="Clean kg-flag-decisions-v1 output path")
    args = parser.parse_args(argv)

    result = harvest_session(
        args.batch,
        args.session_decisions,
        answers_path=args.answers,
        decisions_path=args.decisions,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = ["harvest_session", "main"]
