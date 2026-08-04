#!/usr/bin/env python3
"""Re-tag existing Codex chunks whose session is explicitly linked by T3."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainlayer.provenance import PROVENANCE_RANK
from brainlayer.t3_provenance import T3_APP_SESSION, codex_session_id_from_source, t3_app_codex_session_ids


def _candidates(connection: sqlite3.Connection, linked_session_ids: set[str]) -> list[tuple[str, str | None]]:
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
        if codex_session_id_from_source(source_file) in linked_session_ids
        and provenance_class == "codex-session"
        and provenance_class not in PROVENANCE_RANK
    ]


def _atomic_write_jsonl(path: Path, rows: list[tuple[str, str | None]]) -> None:
    """Durably replace a JSONL artifact without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as artifact:
            for chunk_id, provenance_class in rows:
                artifact.write(json.dumps({"id": chunk_id, "provenance_class": provenance_class}) + "\n")
            artifact.flush()
            os.fsync(artifact.fileno())
        os.replace(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _write_rollback_artifact(path: Path, candidates: list[tuple[str, str | None]]) -> None:
    existing = (
        {
            row["id"]: row["provenance_class"]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
            for row in [json.loads(line)]
        }
        if path.exists()
        else {}
    )
    existing.update(dict(candidates))
    _atomic_write_jsonl(path, sorted(existing.items()))


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
            "matched_chunks": 0,
            "retagged_chunks": 0,
        }
        if not apply:
            return report

        artifact_path = Path(rollback_artifact).expanduser()
        _write_rollback_artifact(artifact_path, candidates)
        connection.execute("PRAGMA wal_checkpoint(FULL)")
        for batch_start in range(0, len(candidates), batch_size):
            batch = candidates[batch_start : batch_start + batch_size]
            cursor = connection.executemany(
                "UPDATE chunks SET provenance_class = ? WHERE id = ? AND provenance_class = ?",
                [(T3_APP_SESSION, chunk_id, "codex-session") for chunk_id, _ in batch],
            )
            report["matched_chunks"] += cursor.rowcount
            connection.commit()
            if ((batch_start // batch_size) + 1) % 3 == 0:
                connection.execute("PRAGMA wal_checkpoint(FULL)")
        connection.execute("PRAGMA wal_checkpoint(FULL)")
        report["retagged_chunks"] = report["matched_chunks"]
        return report
    finally:
        connection.close()


def rollback_t3_app_chunks(
    *,
    db_path: str | Path,
    rollback_artifact: str | Path,
    only_null_prior_values: bool = False,
    pre_restore_artifact: str | Path | None = None,
    batch_size: int = 5_000,
) -> dict[str, int]:
    """Restore provenance values recorded by a prior T3 re-tag artifact."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    artifact_path = Path(rollback_artifact).expanduser()
    rows = [json.loads(line) for line in artifact_path.read_text(encoding="utf-8").splitlines() if line]
    candidates = [(row["id"], row["provenance_class"]) for row in rows]
    if only_null_prior_values:
        candidates = [
            (chunk_id, provenance_class) for chunk_id, provenance_class in candidates if provenance_class is None
        ]
    if pre_restore_artifact is not None:
        pre_restore_path = Path(pre_restore_artifact).expanduser()
        if pre_restore_path.exists():
            raise ValueError(f"pre-restore artifact already exists: {pre_restore_path}")
    else:
        pre_restore_path = None
    connection = sqlite3.connect(Path(db_path).expanduser(), timeout=1.0)
    try:
        connection.execute("PRAGMA busy_timeout = 30000")
        current_values = (
            {
                chunk_id: provenance_class
                for chunk_id, provenance_class in connection.execute(
                    "SELECT id, provenance_class FROM chunks WHERE id IN ({})".format(
                        ", ".join("?" for _ in candidates)
                    ),
                    [chunk_id for chunk_id, _ in candidates],
                )
            }
            if candidates
            else {}
        )
        missing_chunks = sum(chunk_id not in current_values for chunk_id, _ in candidates)
        matched_candidates = [
            (chunk_id, provenance_class)
            for chunk_id, provenance_class in candidates
            if current_values.get(chunk_id) == T3_APP_SESSION
        ]
        skipped_non_t3_chunks = len(candidates) - missing_chunks - len(matched_candidates)
        if pre_restore_path is not None and matched_candidates:
            _atomic_write_jsonl(
                pre_restore_path,
                [(chunk_id, current_values[chunk_id]) for chunk_id, _ in matched_candidates],
            )
        restored_chunks = 0
        for batch_start in range(0, len(matched_candidates), batch_size):
            batch = matched_candidates[batch_start : batch_start + batch_size]
            cursor = connection.executemany(
                "UPDATE chunks SET provenance_class = ? WHERE id = ? AND provenance_class = ?",
                [(provenance_class, chunk_id, T3_APP_SESSION) for chunk_id, provenance_class in batch],
            )
            restored_chunks += cursor.rowcount
            connection.commit()
            if ((batch_start // batch_size) + 1) % 3 == 0:
                connection.execute("PRAGMA wal_checkpoint(FULL)")
        connection.execute("PRAGMA wal_checkpoint(FULL)")
        return {
            "requested_chunks": len(candidates),
            "matched_chunks": len(matched_candidates),
            "restored_chunks": restored_chunks,
            "missing_chunks": missing_chunks,
            "skipped_non_t3_chunks": skipped_non_t3_chunks,
        }
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=Path.home() / ".local/share/brainlayer/brainlayer.db")
    parser.add_argument("--state-db", type=Path, default=Path.home() / ".t3/userdata/state.sqlite")
    parser.add_argument("--apply", action="store_true", help="Perform writes; default is read-only dry run")
    parser.add_argument("--rollback", action="store_true", help="Restore values from --rollback-artifact")
    parser.add_argument("--only-null-prior-values", action="store_true")
    parser.add_argument("--pre-restore-artifact", type=Path)
    parser.add_argument("--rollback-artifact", type=Path, help="JSONL (id, provenance_class) captured before writes")
    parser.add_argument("--batch-size", type=int, default=5_000)
    args = parser.parse_args()
    if args.rollback:
        if args.rollback_artifact is None:
            parser.error("--rollback-artifact is required with --rollback")
        report: dict[str, Any] = rollback_t3_app_chunks(
            db_path=args.db_path,
            rollback_artifact=args.rollback_artifact,
            only_null_prior_values=args.only_null_prior_values,
            pre_restore_artifact=args.pre_restore_artifact,
            batch_size=args.batch_size,
        )
    else:
        report = retag_t3_app_chunks(
            db_path=args.db_path,
            state_db=args.state_db,
            apply=args.apply,
            rollback_artifact=args.rollback_artifact,
            batch_size=args.batch_size,
        )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
