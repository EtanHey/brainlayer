#!/usr/bin/env python3
"""Quarantine noisy F-infra and PreCompact chunks.

Usage:
    python3 scripts/quarantine_noise.py
    python3 scripts/quarantine_noise.py --limit 100
    python3 scripts/quarantine_noise.py --apply
    python3 scripts/quarantine_noise.py --apply --limit 100 --json
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from brainlayer.chunk_origin import is_precompact_checkpoint_content
from brainlayer.paths import get_db_path

NOISE_TAG = "quarantined/noise"
CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT = "precompact_checkpoint"
PRECOMPACT_ID_PREFIXES = ("c89", "0cb", "d3f", "493", "303", "362", "8f1", "037", "3c8", "16b", "dec", "ce2")
F_INFRA_NOISE_PREFIXES = (
    "brainlayer mcp not connected",
    "brainlayer mcp is down",
    "done. summary:  brainlayer mcp not connected",
    "done. summary: brainlayer mcp not connected",
)


def _to_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _normalize_headline(content: str | None) -> str:
    """Normalize a chunk to a best-effort leading-line comparable form."""
    if not content:
        return ""

    text = content.strip()
    if not text:
        return ""

    text = re.sub(r"(?i)^assistant:\s*", "", text)

    normalized = []
    for line in text.replace("\r", "\n").splitlines():
        normalized.append(line.strip())
    text = " ".join(part for part in normalized if part)

    # Strip leading wrapper chars used by diagnostic formatting.
    text = re.sub(r"^\W+", "", text)
    text = text.lstrip(" \t*`>-•_")
    text = text.replace("**", "")
    text = text.lower()
    return text


def is_f_infra_root_content(content: str | None) -> bool:
    """Return True for F-infra diagnostic chatter candidates."""
    normalized = _normalize_headline(content)
    if not normalized:
        return False
    # Root diagnostics tend to start near the beginning.
    return any(normalized.startswith(prefix) for prefix in F_INFRA_NOISE_PREFIXES)


def _chunk_id_has_prefix(chunk_id: str | None, prefixes: tuple[str, ...]) -> bool:
    if not chunk_id:
        return False
    return any(chunk_id.startswith(f"brainbar-{prefix}") for prefix in prefixes)


def _parse_tags(raw: str | None) -> list[str] | None:
    if raw is None:
        return []
    if isinstance(raw, list):
        parsed = raw
    else:
        if not raw:
            return []
        if not isinstance(raw, str):
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, list):
        return None
    tags: list[str] = []
    for item in parsed:
        if isinstance(item, str):
            tag = item.strip()
            if tag:
                tags.append(tag)
    return tags


def _tags_indicate_precompact(raw: str | None) -> bool:
    tags = _parse_tags(raw)
    if tags is None:
        return False
    checkpoint_tags = {
        "pre-compact",
        "precompact",
        "precompact-checkpoint",
        "precompact_checkpoint",
        "pre-compact-checkpoint",
    }
    return any(tag.strip().lower() in checkpoint_tags for tag in tags)


def _normalize_and_append_tag(existing: list[str], tag: str) -> tuple[list[str], bool]:
    """Append tag if missing (case-insensitive), preserving existing order."""
    lowered = {t.lower() for t in existing}
    if tag.lower() in lowered:
        return existing, False
    return existing + [tag], True


def is_precompact_candidate(
    content: str | None,
    chunk_origin: str | None,
    tags: str | None,
    chunk_id: str | None = None,
) -> bool:
    if is_precompact_checkpoint_content(content):
        return True
    if (chunk_origin or "").strip().lower() == CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT:
        return True
    if _chunk_id_has_prefix(chunk_id, PRECOMPACT_ID_PREFIXES):
        return True
    return _tags_indicate_precompact(tags)


def _connect(db_path: Path, apply: bool) -> sqlite3.Connection:
    if apply:
        conn = sqlite3.connect(db_path)
    else:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    if apply:
        conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _table_columns(conn: sqlite3.Connection) -> set[str]:
    return {row["name"] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}


def _fetch_candidates(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    columns = _table_columns(conn)
    has_chunk_origin = "chunk_origin" in columns

    where_parts = [
        "content IS NOT NULL",
        "(LOWER(content) LIKE :infra_phrase)",
    ]
    params: dict[str, str] = {"infra_phrase": "%brainlayer mcp not connected%"}

    if has_chunk_origin:
        where_parts.append("chunk_origin = :chunk_origin_precompact")
        params["chunk_origin_precompact"] = CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT

    where_parts.extend(
        [
            "(LOWER(content) LIKE :precompact_content)",
            "(COALESCE(tags, '') LIKE :precompact_tag)",
            "(LOWER(content) LIKE :precompact_wrapped)",
        ]
    )
    params.update(
        {
            "precompact_content": "%[precompact checkpoint]%",
            "precompact_tag": "%precompact%",
            "precompact_wrapped": "%content=\"[precompact checkpoint]%",
        }
    )
    precompact_id_like_clauses = [
        "id LIKE :precompact_id_like_" + re.sub(r"[^a-z0-9]", "", prefix)
        for prefix in PRECOMPACT_ID_PREFIXES
    ]
    params.update({
        f"precompact_id_like_{re.sub(r'[^a-z0-9]', '', prefix)}": f"brainbar-{prefix}%"
        for prefix in PRECOMPACT_ID_PREFIXES
    })
    where_parts.append(f"({' OR '.join(precompact_id_like_clauses)})")

    select_parts = [
        "id",
        "content",
        "tags",
        "project",
        "source",
        "source_file",
        "created_at",
        "importance",
        "archived",
        "status",
        "archived_at",
    ]
    if has_chunk_origin:
        select_parts.append("chunk_origin")
    else:
        select_parts.append("NULL AS chunk_origin")

    sql = f"""
        SELECT {", ".join(select_parts)}
        FROM chunks
        WHERE (
            {" OR ".join(where_parts)}
        )
        ORDER BY COALESCE(created_at, ''), id
    """

    return list(conn.execute(sql, params))


def _candidate_map(rows: list[sqlite3.Row]) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}

    for row in rows:
        row_content = (row["content"] or "").strip()
        content_preview = row_content.replace("\n", " ")
        if len(content_preview) > 140:
            content_preview = content_preview[:140] + "…"

        reasons = []
        if is_f_infra_root_content(row["content"]):
            reasons.append("f_infra_root")
        if is_precompact_candidate(row["content"], row["chunk_origin"], row["tags"], row["id"]):
            reasons.append("precompact")
        if not reasons:
            continue

        existing = candidates.setdefault(row["id"], {})
        existing.setdefault("id", row["id"])
        existing.setdefault("project", row["project"])
        existing.setdefault("source", row["source"])
        existing.setdefault("source_file", row["source_file"])
        existing.setdefault("created_at", row["created_at"])
        existing.setdefault("tags", row["tags"])
        existing.setdefault("chunk_origin", row["chunk_origin"])
        existing.setdefault("importance", row["importance"])
        existing.setdefault("archived", _to_bool(row["archived"]))
        existing.setdefault("status", row["status"])
        existing.setdefault("archived_at", row["archived_at"])
        existing.setdefault("content_preview", content_preview)
        reasons_set = set(existing.get("reasons", []))
        reasons_set.update(reasons)
        existing["reasons"] = sorted(reasons_set)
        candidates[row["id"]] = existing

    return candidates


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    return {
        "total": len(rows),
        "f_infra_root": sum("f_infra_root" in row["reasons"] for row in rows),
        "precompact": sum("precompact" in row["reasons"] for row in rows),
        "already_archived": sum(bool(_to_bool(row["archived"])) for row in rows),
        "taggable": sum(_parse_tags(row["tags"]) is not None for row in rows),
    }


def _human_preview(rows: list[dict[str, Any]]) -> str:
    lines = []
    for row in rows[:50]:
        lines.append(
            "  - {id} [{project}] source={source} created={created_at} reasons={reasons} tags={tags_added}".format(
                id=row["id"],
                project=row.get("project"),
                source=row.get("source"),
                created_at=row.get("created_at"),
                reasons=",".join(row["reasons"]),
                tags_added="yes" if row.get("tag_added") else "no",
            )
        )
        preview = row.get("content_preview")
        if preview:
            lines.append(f"    {preview}")
    if not lines:
        return "  (none)"
    return "\n".join(lines)


def _apply_quarantine(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, int | bool]:
    if not rows:
        return {"updated": 0, "tags_added": 0}

    now = datetime.now(timezone.utc).isoformat()
    base_stmt = """
        UPDATE chunks
           SET importance = ?,
               archived = ?,
               archived_at = ?,
               status = ?
         WHERE id = ?
    """
    tag_stmt = "UPDATE chunks SET tags = ? WHERE id = ?"
    updated = 0
    tags_added = 0

    try:
        cursor = conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        for row in rows:
            cursor.execute(base_stmt, (0, 1, now, "archived", row["id"]))
            updated += 1
            parsed = _parse_tags(row["tags"])
            if parsed is None:
                row["tag_added"] = False
                row["tag_error"] = "malformed_tags"
                continue
            merged, changed = _normalize_and_append_tag(parsed, NOISE_TAG)
            if changed:
                cursor.execute(tag_stmt, (json.dumps(merged, ensure_ascii=False), row["id"]))
                row["tag_added"] = True
                tags_added += 1
            else:
                row["tag_added"] = False
            row["tags"] = json.dumps(merged, ensure_ascii=False)
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return {"updated": updated, "tags_added": tags_added}


def _build_output(candidates: list[dict[str, Any]], dry_run: bool, stats: dict[str, int | float], result: dict[str, int | bool]) -> dict[str, Any]:
    return {
        "mode": "dry_run" if dry_run else "apply",
        "candidates": candidates,
        "summary": stats,
        "applied": result["updated"],
        "tags_added": result["tags_added"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run and quarantine noise chunks in the BrainLayer DB.")
    parser.add_argument(
        "--db",
        type=Path,
        default=get_db_path(),
        help="Path to BrainLayer sqlite DB (default: ~/.local/share/brainlayer/brainlayer.db)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply quarantines. Without this flag, the script only reports matches.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of rows to process.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON summary for automation.",
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")

    conn = _connect(args.db, apply=args.apply)
    try:
        prefetch = _fetch_candidates(conn)
        candidates = list(_candidate_map(prefetch).values())
        candidate_rows = candidates[: args.limit] if args.limit is not None else candidates
        summary = _summarize_rows(candidate_rows)

        if not candidate_rows:
            if args.json:
                print(json.dumps(
                    _build_output(
                        candidates=[],
                        dry_run=not args.apply,
                        stats=summary,
                        result={"updated": 0, "tags_added": 0},
                    ),
                    indent=2,
                ))
            else:
                print("No candidates found.")
            return 0

        if args.apply:
            result = _apply_quarantine(conn, candidate_rows)
            if args.json:
                print(json.dumps(_build_output(candidate_rows, dry_run=False, stats=summary, result=result), indent=2))
            else:
                print(f"Quarantined {result['updated']} chunks.")
                print(f"Tag updates applied: {result['tags_added']}")
                print(_human_preview(candidate_rows))
            return 0

        result = {"updated": 0, "tags_added": 0}
        if args.json:
            print(json.dumps(_build_output(candidate_rows, dry_run=True, stats=summary, result=result), indent=2))
            return 0

        print(f"Candidates: {summary['total']}")
        print(f"  - f-infra root: {summary['f_infra_root']}")
        print(f"  - precompact: {summary['precompact']}")
        print(f"  - already_archived: {summary['already_archived']}")
        print("Dry-run mode: add --apply to write changes.\n")
        print(_human_preview(candidate_rows))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(run())
