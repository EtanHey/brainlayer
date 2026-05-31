#!/usr/bin/env python3
"""Dry-run and apply content_class backfill for BrainLayer chunks."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from brainlayer.content_class import (
    CONTENT_CLASS_VALUES,
    classify_content_class,
    classify_content_class_raw,
    has_operational_marker,
    keep_visible_signals,
    normalize_content_class,
    strip_operational_markers,
)
from brainlayer.paths import get_db_path
from brainlayer.vector_store import VectorStore

_CLASS_ORDER = ("decision", "knowledge", "operational", "test")
def _preview(content: str | None, limit: int = 260) -> str:
    return re.sub(r"\s+", " ", content or "").strip()[:limit]


def _sample_row(
    row: dict[str, Any],
    proposed_class: str,
    *,
    risk_signals: list[str] | None = None,
    proposed_raw: str | None = None,
) -> dict[str, Any]:
    sample = {
        "chunk_id": row["id"],
        "current_class": normalize_content_class(row.get("content_class")),
        "proposed_class": proposed_class,
        "content_type": row.get("content_type"),
        "project": row.get("project"),
        "source": row.get("source"),
        "source_file": row.get("source_file"),
        "preview": _preview(row.get("content")),
    }
    if proposed_raw is not None:
        sample["proposed_raw"] = proposed_raw
    if risk_signals:
        sample["risk_signals"] = risk_signals
    return sample


def iter_classifications(store: VectorStore):
    cursor = store.conn.cursor()
    rows = cursor.execute(
        """
        SELECT id, content, content_type, tags, source, source_file, project,
               COALESCE(content_class, 'knowledge') AS content_class
        FROM chunks
        ORDER BY id
        """
    )
    for chunk_id, content, content_type, tags, source, source_file, project, content_class in rows:
        row = {
            "id": chunk_id,
            "content": content,
            "content_type": content_type,
            "tags": tags,
            "source": source,
            "source_file": source_file,
            "project": project,
            "content_class": content_class,
        }
        proposed_raw = classify_content_class_raw(
            content,
            content_type=content_type,
            tags=tags,
            source=source,
            source_file=source_file,
            project=project,
        )
        proposed = classify_content_class(
            content,
            content_type=content_type,
            tags=tags,
            source=source,
            source_file=source_file,
            project=project,
        )
        signals = keep_visible_signals(content, content_type=content_type)
        yield row, proposed_raw, proposed, signals


def build_backfill_report(store: VectorStore, *, sample_limit: int = 30) -> dict[str, Any]:
    sample_limit = max(1, int(sample_limit))
    counts = {key: 0 for key in _CLASS_ORDER}
    samples = {key: [] for key in _CLASS_ORDER}
    hidden_risk_rows: list[dict[str, Any]] = []
    keep_visible_override_samples: list[dict[str, Any]] = []
    operational_marker_kept_samples: list[dict[str, Any]] = []
    keep_visible_override_total = 0
    operational_marker_kept_total = 0
    updates_needed = 0

    for row, proposed_raw, proposed, signals in iter_classifications(store):
        if proposed not in CONTENT_CLASS_VALUES:
            proposed = "knowledge"
        counts[proposed] += 1
        if len(samples[proposed]) < sample_limit:
            samples[proposed].append(_sample_row(row, proposed))
        if normalize_content_class(row.get("content_class")) != proposed:
            updates_needed += 1
        if proposed_raw in {"operational", "test"} and proposed == "knowledge" and signals:
            keep_visible_override_total += 1
            if len(keep_visible_override_samples) < 10:
                keep_visible_override_samples.append(
                    _sample_row(row, proposed, proposed_raw=proposed_raw, risk_signals=signals)
                )
        if proposed not in {"operational", "test"} and has_operational_marker(row.get("content")):
            operational_marker_kept_total += 1
            if len(operational_marker_kept_samples) < 8:
                sample = _sample_row(row, proposed, proposed_raw=proposed_raw, risk_signals=signals)
                sample["residual_after_marker_strip"] = _preview(strip_operational_markers(row.get("content")), 180)
                operational_marker_kept_samples.append(sample)
        if proposed in {"operational", "test"} and signals:
            hidden_risk_rows.append(_sample_row(row, proposed, proposed_raw=proposed_raw, risk_signals=signals))

    hidden_total = counts["operational"] + counts["test"]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "dry_run",
        "counts": counts,
        "updates_needed": updates_needed,
        "hidden_total": hidden_total,
        "keep_visible_override_total": keep_visible_override_total,
        "keep_visible_override_samples": keep_visible_override_samples,
        "operational_marker_kept_total": operational_marker_kept_total,
        "operational_marker_kept_samples": operational_marker_kept_samples,
        "personal_hidden": len(hidden_risk_rows),
        "hidden_decision_or_personal_risk_total": len(hidden_risk_rows),
        "samples": samples,
        "personal_hidden_rows": hidden_risk_rows,
        "hidden_decision_or_personal_risk_rows": hidden_risk_rows,
        # Backward-compatible alias for reviewers looking specifically at over-exclusion.
        "operational_test_over_exclusion_check": hidden_risk_rows,
    }


def apply_backfill(store: VectorStore, *, limit: int | None = None) -> dict[str, Any]:
    counts = {key: 0 for key in _CLASS_ORDER}
    updated = 0
    cursor = store.conn.cursor()
    cursor.execute("BEGIN IMMEDIATE")
    try:
        for row, _proposed_raw, proposed, _signals in iter_classifications(store):
            if limit is not None and updated >= limit:
                break
            proposed = normalize_content_class(proposed)
            if normalize_content_class(row.get("content_class")) == proposed:
                continue
            cursor.execute("UPDATE chunks SET content_class = ? WHERE id = ?", (proposed, row["id"]))
            counts[proposed] += 1
            updated += 1
        cursor.execute("COMMIT")
    except Exception:
        cursor.execute("ROLLBACK")
        raise
    try:
        from brainlayer.search_repo import clear_hybrid_search_cache

        clear_hybrid_search_cache(getattr(store, "db_path", None))
    except Exception:
        pass
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "apply",
        "updated": updated,
        "updated_counts": counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=get_db_path())
    parser.add_argument("--sample-limit", type=int, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--apply", action="store_true", help="Apply class-only updates. Do not use before review gate.")
    parser.add_argument("--confirm-apply", action="store_true", help="Required with --apply.")
    parser.add_argument("--limit", type=int, help="Optional apply limit for staged rollout.")
    args = parser.parse_args()

    if args.apply and not args.confirm_apply:
        parser.error("--apply requires --confirm-apply")

    store = VectorStore(args.db_path)
    try:
        report = apply_backfill(store, limit=args.limit) if args.apply else build_backfill_report(
            store,
            sample_limit=args.sample_limit,
        )
    finally:
        store.close()

    payload = json.dumps(report, indent=2, sort_keys=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
