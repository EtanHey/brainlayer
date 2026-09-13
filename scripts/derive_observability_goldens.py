#!/usr/bin/env python3
"""Derive observability DB sections directly from built fixture SQLite files.

This is deliberately independent from the observability producer.  It is a
plain-SQL oracle for reviewing the committed fixture artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

DERIVATION_SQL = Path(__file__).resolve().parents[1] / "tests/fixtures/observability/derivation.sql"


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _input(case: dict[str, Any], root: Path) -> dict[str, Any]:
    path = case["inputs"]["db"]
    db = root / path
    with sqlite3.connect(db) as connection:
        count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    return {
        "mtime": case["input_mtimes"][path],
        "path": path,
        "rows_or_bytes": count,
        "sha256_first_64kb": None,
        "skipped_lines": 0,
        "status": "read",
    }


def _columns(connection: sqlite3.Connection) -> set[str]:
    return {row[1] for row in connection.execute("PRAGMA table_info(chunks)")}


def _preview(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()[:80]


def _emitter(row: sqlite3.Row) -> tuple[str, str]:
    if row["source"]:
        return row["source"], "source"
    if row["sender"]:
        return row["sender"], "sender"
    return row["source_file"].rsplit("/", 1)[-1], "source_file"


def _db_sections(case: dict[str, Any], root: Path, template: dict[str, Any]) -> dict[str, Any]:
    path = root / case["inputs"]["db"]
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        inputs = [_input(case, root)]
        if "source_class" not in _columns(connection):
            reason = "required column missing: chunks.source_class"
            return {
                name: {"inputs": inputs, "reason": reason, "state": "unmeasurable"}
                for name in ("stores", "emitters", "author_unknown")
            }
        generated = datetime.fromisoformat(case["generated_at"].replace("Z", "+00:00"))
        window_start = generated - timedelta(hours=case["window_hours"])
        rows = connection.execute(
            "SELECT id, content, source_file, source, sender, created_at, provenance_class, "
            "source_class, content_class, archived_at, superseded_by FROM chunks"
        ).fetchall()
        window = [row for row in rows if window_start <= datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")) <= generated]
        by_class = connection.execute(
            "SELECT COALESCE(content_class, 'knowledge') AS content_class, COUNT(*) AS count "
            "FROM chunks GROUP BY COALESCE(content_class, 'knowledge') ORDER BY content_class"
        ).fetchall()
        by_hour = connection.execute(
            "SELECT substr(created_at, 1, 13) || ':00:00Z' AS hour, COUNT(*) AS count FROM chunks "
            "WHERE created_at >= ? AND created_at <= ? GROUP BY hour ORDER BY hour",
            (_iso(window_start), _iso(generated)),
        ).fetchall()
        by_source_class = connection.execute(
            "SELECT source_class, COUNT(*) AS count, SUM(CASE WHEN created_at >= ? AND created_at <= ? THEN 1 ELSE 0 END) AS in_window "
            "FROM chunks GROUP BY source_class ORDER BY source_class IS NOT NULL DESC, source_class",
            (_iso(window_start), _iso(generated)),
        ).fetchall()
        by_emitter: dict[tuple[str, str], int] = {}
        for row in window:
            key = _emitter(row)
            by_emitter[key] = by_emitter.get(key, 0) + 1
        latest = []
        for row in sorted(window, key=lambda item: item["created_at"], reverse=True)[:5]:
            emitter, _ = _emitter(row)
            latest.append({"chunk_id": row["id"], "emitter": emitter, "preview": _preview(row["content"]), "source_class": row["source_class"], "stored_at": row["created_at"]})
        live_count = connection.execute("SELECT COUNT(*) FROM chunks WHERE archived_at IS NULL").fetchone()[0]
        unknown_count = connection.execute("SELECT COUNT(*) FROM chunks WHERE provenance_class = 'unknown' AND archived_at IS NULL").fetchone()[0]
        never_count = connection.execute("SELECT COUNT(*) FROM chunks WHERE (provenance_class IS NULL OR source_class IS NULL) AND archived_at IS NULL").fetchone()[0]
        top_files = connection.execute(
            "SELECT source_file, COUNT(*) AS count FROM chunks WHERE archived_at IS NULL AND "
            "(provenance_class IS NULL OR source_class IS NULL OR provenance_class = 'unknown') "
            "GROUP BY source_file ORDER BY count DESC, source_file LIMIT 2"
        ).fetchall()
        trend = []
        for offset in range(6, -1, -1):
            day = (generated - timedelta(days=offset)).date().isoformat()
            trend.append({
                "classified_unknown": connection.execute("SELECT COUNT(*) FROM chunks WHERE provenance_class = 'unknown' AND substr(created_at, 1, 10) = ?", (day,)).fetchone()[0],
                "day": day,
                "never_classified": connection.execute("SELECT COUNT(*) FROM chunks WHERE (provenance_class IS NULL OR source_class IS NULL) AND substr(created_at, 1, 10) = ?", (day,)).fetchone()[0],
            })
        stores = {
            "by_content_class": [{"content_class": row["content_class"], "count": row["count"]} for row in by_class],
            "in_window": {"by_hour": [{"count": row["count"], "hour": row["hour"]} for row in by_hour], "count": len(window)},
            "inputs": inputs,
            "latest": latest,
            "reason": "",
            "state": "measured",
            "total_chunks": len(rows),
        }
        emitters = {
            "by_emitter": [{"count_in_window": count, "derived_from": source, "emitter": emitter} for (emitter, source), count in sorted(by_emitter.items())],
            "by_source_class": [{"count": row["count"], "in_window": row["in_window"], "source_class": row["source_class"]} for row in by_source_class],
            "derivation_note": "metadata.attributionAgent is absent in the 2026-09-13 census; emitter derives from source, then sender, then source_file, or the section is unmeasurable",
            "hidden_from_default_search": connection.execute("SELECT COUNT(*) FROM chunks WHERE source_class IN ('desktop', 'brain-worker')").fetchone()[0],
            "inputs": inputs,
            "reason": "",
            "state": "measured",
        }
        author_unknown = {
            "census_2026_09_13": template["author_unknown"]["census_2026_09_13"],
            "classified_unknown": {"count": unknown_count, "definition": "provenance_class = 'unknown', archived_at IS NULL", "share": round(unknown_count / live_count, 6) if live_count else 0.0},
            "inputs": inputs,
            "never_classified": {"count": never_count, "definition": "provenance_class IS NULL OR source_class IS NULL, archived_at IS NULL", "share": round(never_count / live_count, 6) if live_count else 0.0},
            "reason": "",
            "state": "measured",
            "top_source_files": [{"count": row["count"], "source_file": row["source_file"]} for row in top_files],
            "trend_7d": trend,
        }
        return {"stores": stores, "emitters": emitters, "author_unknown": author_unknown}
    finally:
        connection.close()


def derive(root: Path, golden_root: Path) -> None:
    manifest = json.loads((root / "cases.json").read_text())
    for case in manifest["cases"]:
        path = golden_root / case["golden"]
        if not path.exists():
            continue
        golden = json.loads(path.read_text())
        golden.update(_db_sections(case, root, golden))
        path.write_text(json.dumps(golden, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--golden-root", type=Path, required=True)
    args = parser.parse_args()
    derive(args.fixture_root.resolve(), args.golden_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
