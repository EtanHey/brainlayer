"""Isolated ABCDE experiment storage.

The ABCDE experiment is allowed to read a snapshot and write only to its own
experiment namespace DB. This module deliberately avoids VectorStore so it
cannot share a live BrainLayer connection or write into live FTS/vector tables.

Snapshot materialization is an explicit operator action:

    python -m brainlayer.eval.experiment_store materialize-snapshot \
      --backup-gzip-path "/path/to/Brain Drive/06_ARCHIVE/backups/brainlayer-db/<backup>.db.gz"

Do not run that command in CI. Unit tests should use tiny synthetic SQLite DBs.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from brainlayer import paths

EXPERIMENT_DIR = Path("~/.local/share/brainlayer/experiments").expanduser()
DEFAULT_EXPERIMENT_DB_PATH = EXPERIMENT_DIR / "abcde-experiment.db"
DEFAULT_SNAPSHOT_DB_PATH = EXPERIMENT_DIR / "abcde-snapshot.db"
CANONICAL_LIVE_DB_PATH = Path("~/.local/share/brainlayer/brainlayer.db").expanduser()
VALID_VARIANT_IDS = frozenset({"A", "B", "C", "D", "E"})
VALID_JUDGMENT_SOURCES = frozenset({"llm", "human"})


def _normalized_path(path: str | Path) -> Path:
    return Path(path).expanduser().absolute()


def _path_key(path: str | Path) -> str:
    expanded = Path(path).expanduser()
    try:
        return str(expanded.resolve())
    except OSError:
        return str(expanded.absolute())


def _live_db_keys() -> set[str]:
    return {_path_key(CANONICAL_LIVE_DB_PATH), _path_key(paths.DEFAULT_DB_PATH), _path_key(paths.get_db_path())}


def _guard_not_live_db(path: str | Path, *, role: str) -> Path:
    guarded = _normalized_path(path)
    if _path_key(guarded) in _live_db_keys():
        raise ValueError(f"Refusing to open the live BrainLayer DB as {role}: {guarded}")
    return guarded


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


class ExperimentStore:
    """SQLite store for the isolated ABCDE experiment namespace."""

    def __init__(
        self,
        *,
        experiment_db_path: str | Path = DEFAULT_EXPERIMENT_DB_PATH,
        snapshot_db_path: str | Path = DEFAULT_SNAPSHOT_DB_PATH,
    ) -> None:
        self.experiment_db_path = _guard_not_live_db(experiment_db_path, role="experiment DB")
        self.snapshot_db_path = _guard_not_live_db(snapshot_db_path, role="experiment snapshot DB")
        self.snapshot_conn = self._open_snapshot_readonly(self.snapshot_db_path)
        self.experiment_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.experiment_db_path)
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_schema()

    @staticmethod
    def _open_snapshot_readonly(snapshot_db_path: Path) -> sqlite3.Connection:
        if not snapshot_db_path.exists():
            raise FileNotFoundError(
                f"Experiment snapshot DB does not exist: {snapshot_db_path}. "
                "Run materialize-snapshot outside CI before a real experiment run."
            )
        conn = sqlite3.connect(f"file:{snapshot_db_path}?mode=ro", uri=True)
        conn.execute("PRAGMA query_only=ON")
        return conn

    def _ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS exp_chunks (
                chunk_id TEXT PRIMARY KEY,
                source_chunk_id TEXT NOT NULL UNIQUE,
                raw_text TEXT NOT NULL,
                content_type TEXT NOT NULL DEFAULT '',
                content_class TEXT NOT NULL DEFAULT '',
                strata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS exp_variants (
                chunk_id TEXT NOT NULL,
                variant_id TEXT NOT NULL CHECK (variant_id IN ('A', 'B', 'C', 'D', 'E')),
                enrichment_json TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                grader_scores_json TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (chunk_id, variant_id),
                FOREIGN KEY (chunk_id) REFERENCES exp_chunks(chunk_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS exp_judgments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT NOT NULL,
                variant_id TEXT NOT NULL CHECK (variant_id IN ('A', 'B', 'C', 'D', 'E')),
                source TEXT NOT NULL CHECK (source IN ('llm', 'human')),
                scores_json TEXT NOT NULL,
                better_option_flag INTEGER NOT NULL CHECK (better_option_flag IN (0, 1)),
                rationale TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chunk_id, variant_id) REFERENCES exp_variants(chunk_id, variant_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS exp_index_documents (
                chunk_id TEXT NOT NULL,
                variant_id TEXT NOT NULL CHECK (variant_id IN ('A', 'B', 'C', 'D', 'E')),
                index_payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (chunk_id, variant_id),
                FOREIGN KEY (chunk_id, variant_id) REFERENCES exp_variants(chunk_id, variant_id)
                    ON DELETE CASCADE
            );
            """
        )
        self.conn.commit()

    def upsert_chunk(
        self,
        *,
        source_chunk_id: str,
        raw_text: str,
        content_type: str,
        content_class: str,
        strata: dict[str, Any],
    ) -> str:
        chunk_id = source_chunk_id
        self.conn.execute(
            """
            INSERT INTO exp_chunks (
                chunk_id, source_chunk_id, raw_text, content_type, content_class, strata
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_chunk_id) DO UPDATE SET
                raw_text = excluded.raw_text,
                content_type = excluded.content_type,
                content_class = excluded.content_class,
                strata = excluded.strata,
                updated_at = CURRENT_TIMESTAMP
            """,
            (chunk_id, source_chunk_id, raw_text, content_type, content_class, _json_dumps(strata)),
        )
        self.conn.commit()
        return chunk_id

    def upsert_variant(
        self,
        *,
        chunk_id: str,
        variant_id: str,
        enrichment: dict[str, Any],
        model: str,
        prompt_hash: str,
        grader_scores: dict[str, Any] | None = None,
    ) -> None:
        if variant_id not in VALID_VARIANT_IDS:
            raise ValueError(f"variant_id must be one of A..E, got {variant_id!r}")
        self.conn.execute(
            """
            INSERT INTO exp_variants (
                chunk_id, variant_id, enrichment_json, model, prompt_hash, grader_scores_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id, variant_id) DO UPDATE SET
                enrichment_json = excluded.enrichment_json,
                model = excluded.model,
                prompt_hash = excluded.prompt_hash,
                grader_scores_json = excluded.grader_scores_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                chunk_id,
                variant_id,
                _json_dumps(enrichment),
                model,
                prompt_hash,
                None if grader_scores is None else _json_dumps(grader_scores),
            ),
        )
        self.conn.commit()

    def add_judgment(
        self,
        *,
        chunk_id: str,
        variant_id: str,
        source: str,
        scores: dict[str, Any],
        better_option_flag: bool,
        rationale: str,
    ) -> None:
        if variant_id not in VALID_VARIANT_IDS:
            raise ValueError(f"variant_id must be one of A..E, got {variant_id!r}")
        if source not in VALID_JUDGMENT_SOURCES:
            raise ValueError(f"source must be one of llm|human, got {source!r}")
        self.conn.execute(
            """
            INSERT INTO exp_judgments (
                chunk_id, variant_id, source, scores_json, better_option_flag, rationale
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (chunk_id, variant_id, source, _json_dumps(scores), int(better_option_flag), rationale),
        )
        self.conn.commit()

    def reserve_exp_index_document(self, *, chunk_id: str, variant_id: str, index_payload: dict[str, Any]) -> None:
        """Reserve the future isolated re-index write path inside the experiment DB only."""
        if variant_id not in VALID_VARIANT_IDS:
            raise ValueError(f"variant_id must be one of A..E, got {variant_id!r}")
        self.conn.execute(
            """
            INSERT INTO exp_index_documents (chunk_id, variant_id, index_payload_json)
            VALUES (?, ?, ?)
            ON CONFLICT(chunk_id, variant_id) DO UPDATE SET
                index_payload_json = excluded.index_payload_json,
                created_at = CURRENT_TIMESTAMP
            """,
            (chunk_id, variant_id, _json_dumps(index_payload)),
        )
        self.conn.commit()

    def close(self) -> None:
        self.snapshot_conn.close()
        self.conn.close()

    def __enter__(self) -> "ExperimentStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def materialize_snapshot(
    *,
    backup_gzip_path: str | Path | None,
    snapshot_db_path: str | Path = DEFAULT_SNAPSHOT_DB_PATH,
    overwrite: bool = False,
) -> Path:
    """Decompress the verified Brain Drive DB backup into the experiment snapshot path.

    This is intentionally opt-in and requires an explicit gzip path so CI cannot
    accidentally download or decompress the 4GB backup. The target path is
    guarded against the live BrainLayer DB and should normally be:
    ~/.local/share/brainlayer/experiments/abcde-snapshot.db.
    """
    if backup_gzip_path is None:
        raise ValueError(
            "backup_gzip_path is required; use the verified Brain Drive backup under "
            "Brain Drive/06_ARCHIVE/backups/brainlayer-db"
        )
    source = Path(backup_gzip_path).expanduser().absolute()
    target = _guard_not_live_db(snapshot_db_path, role="experiment snapshot DB")
    if not source.exists():
        raise FileNotFoundError(f"Verified backup gzip not found: {source}")
    if target.exists() and not overwrite:
        raise FileExistsError(f"Snapshot already exists: {target}. Pass overwrite=True to replace it.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_target = target.with_suffix(target.suffix + ".tmp")
    with gzip.open(source, "rb") as compressed, temporary_target.open("wb") as output:
        shutil.copyfileobj(compressed, output)
    temporary_target.replace(target)
    target.chmod(0o444)
    return target


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ABCDE experiment store utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser(
        "materialize-snapshot",
        help="Decompress the verified Brain Drive backup into the experiment snapshot DB",
    )
    materialize.add_argument("--backup-gzip-path", required=True)
    materialize.add_argument("--snapshot-db-path", default=str(DEFAULT_SNAPSHOT_DB_PATH))
    materialize.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "materialize-snapshot":
        target = materialize_snapshot(
            backup_gzip_path=args.backup_gzip_path,
            snapshot_db_path=args.snapshot_db_path,
            overwrite=args.overwrite,
        )
        print(target)
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
