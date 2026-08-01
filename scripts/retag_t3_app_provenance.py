#!/usr/bin/env python3
"""Re-tag existing Codex chunks whose session is explicitly linked by T3."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainlayer.t3_provenance import T3_APP_SESSION, codex_session_id_from_source, t3_app_codex_session_ids


def _candidates(connection: sqlite3.Connection, linked_session_ids: set[str]) -> list[tuple[str, str]]:
    rows = connection.execute(
        """
        SELECT id, source_file, provenance_class
        FROM chunks
        WHERE source_file LIKE '%/.codex/sessions/%'
        ORDER BY id
        """
    )
    return [
        (chunk_id, provenance_class)
        for chunk_id, source_file, provenance_class in rows
        if codex_session_id_from_source(source_file) in linked_session_ids and provenance_class != T3_APP_SESSION
    ]


def _write_rollback_artifact(path: Path, candidates: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as artifact:
        for chunk_id, provenance_class in candidates:
            artifact.write(json.dumps({"id": chunk_id, "provenance_class": provenance_class}) + "\n")


def retag_t3_app_chunks(
    *,
    db_path: str | Path,
    state_db: str | Path,
    apply: bool = False,
    rollback_artifact: str | Path | None = None,
    batch_size: int = 5_000,
) -> dict[str, int]:
    """Report or apply the deterministic T3 provenance re-tagging operation."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if apply and rollback_artifact is None:
        raise ValueError("--rollback-artifact is required with --apply")

    linked_session_ids = t3_app_codex_session_ids(state_db)
    path = Path(db_path).expanduser()
    connection = (
        sqlite3.connect(path, timeout=1.0)
        if apply
        else sqlite3.connect(f"{path.absolute().as_uri()}?mode=ro&immutable=0", uri=True, timeout=1.0)
    )
    try:
        if apply:
            connection.execute("PRAGMA busy_timeout = 30000")
        candidates = _candidates(connection, linked_session_ids)
        report = {
            "linked_sessions": len(linked_session_ids),
            "candidate_chunks": len(candidates),
            "retagged_chunks": 0,
        }
        if not apply:
            return report

        artifact_path = Path(rollback_artifact).expanduser()
        _write_rollback_artifact(artifact_path, candidates)
        connection.execute("PRAGMA wal_checkpoint(FULL)")
        for batch_start in range(0, len(candidates), batch_size):
            batch = candidates[batch_start : batch_start + batch_size]
            connection.executemany(
                "UPDATE chunks SET provenance_class = ? WHERE id = ?",
                [(T3_APP_SESSION, chunk_id) for chunk_id, _ in batch],
            )
            connection.commit()
            if ((batch_start // batch_size) + 1) % 3 == 0:
                connection.execute("PRAGMA wal_checkpoint(FULL)")
        connection.execute("PRAGMA wal_checkpoint(FULL)")
        report["retagged_chunks"] = len(candidates)
        return report
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=Path.home() / ".local/share/brainlayer/brainlayer.db")
    parser.add_argument("--state-db", type=Path, default=Path.home() / ".t3/userdata/state.sqlite")
    parser.add_argument("--apply", action="store_true", help="Perform writes; default is read-only dry run")
    parser.add_argument("--rollback-artifact", type=Path, help="JSONL (id, provenance_class) captured before writes")
    parser.add_argument("--batch-size", type=int, default=5_000)
    args = parser.parse_args()
    report: dict[str, Any] = retag_t3_app_chunks(
        db_path=args.db_path,
        state_db=args.state_db,
        apply=args.apply,
        rollback_artifact=args.rollback_artifact,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
