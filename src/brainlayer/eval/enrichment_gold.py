"""Snapshot-only gold-set sampler for BrainLayer enrichment label evaluation."""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from brainlayer.eval.benchmark import ReadOnlyBenchmarkStore, canonical_eval_doc_id

DEFAULT_TARGET_SIZE = 60
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "fixtures" / "enrichment-gold.jsonl"
LIVE_BRAINLAYER_DB = Path("~/.local/share/brainlayer/brainlayer.db").expanduser()


@dataclass(frozen=True)
class Stratum:
    name: str
    patterns: tuple[str, ...] = ()
    columns: tuple[str, ...] = ("content", "tags", "summary")
    extra_conditions: tuple[str, ...] = ()


STRATA: tuple[Stratum, ...] = (
    Stratum(
        "decision",
        patterns=("decision", "decided", "deciding", "choose", "chosen", "tradeoff", "because"),
        columns=("content", "tags", "summary", "intent"),
    ),
    Stratum(
        "code",
        patterns=("code", "def ", "class ", "function ", "import ", "pytest", "```", ".py", ".ts", ".tsx"),
        columns=("content", "source_file", "content_type", "tags", "primary_symbols"),
    ),
    Stratum(
        "conversation",
        patterns=("user:", "assistant:", "claude_counter", "<user", "<assistant"),
        columns=("content", "metadata", "source_file"),
    ),
    Stratum(
        "correction",
        patterns=("correction", "wrong", "mistake", "missed", "actually", "fix", "regression"),
        columns=("content", "tags", "summary", "sentiment_label"),
    ),
    Stratum(
        "entity_bio",
        patterns=("bio", "person", "owner", "founder", "works at", "is the", "profile"),
        columns=("content", "tags", "summary", "entities"),
    ),
    Stratum(
        "short_conversational",
        patterns=("ok", "yes", "no", "thanks", "done", "go"),
        columns=("content", "sender", "metadata"),
        extra_conditions=("length(content) <= 120",),
    ),
    Stratum(
        "long_truncation",
        extra_conditions=("length(content) > 8000",),
    ),
    Stratum(
        "meta_research",
        patterns=("brain_search(", "brain_entity(", "research", "meta-research", "scope doc", "benchmark"),
        columns=("content", "tags", "summary"),
    ),
    Stratum(
        "sentiment",
        patterns=("frustrat", "confus", "angry", "upset", "satisfaction", "positive", "excited"),
        columns=("content", "tags", "summary", "sentiment_label", "sentiment_signals"),
        extra_conditions=("coalesce(sentiment_label, '') not in ('', 'neutral')",),
    ),
)

OUTPUT_COLUMNS = (
    "id",
    "content",
    "source_file",
    "project",
    "content_type",
    "tags",
    "summary",
    "sentiment_label",
    "sender",
    "created_at",
)


def _guard_snapshot_path(snapshot_path: str | Path) -> Path:
    path = Path(snapshot_path).expanduser()
    if path == LIVE_BRAINLAYER_DB:
        raise ValueError(f"Refusing to sample the live BrainLayer DB: {LIVE_BRAINLAYER_DB}")
    if path.exists() and LIVE_BRAINLAYER_DB.exists() and path.resolve() == LIVE_BRAINLAYER_DB.resolve():
        raise ValueError(f"Refusing to sample the live BrainLayer DB: {LIVE_BRAINLAYER_DB}")
    return path


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return {str(row[1]) for row in cursor.fetchall()}


def _condition_for_stratum(stratum: Stratum, columns: set[str]) -> str:
    extra_conditions = list(stratum.extra_conditions)
    pattern_conditions: list[str] = []
    for column in stratum.columns:
        if column not in columns:
            continue
        for pattern in stratum.patterns:
            pattern_conditions.append(f"lower(coalesce({column}, '')) like ?")
    if extra_conditions and pattern_conditions:
        return "(" + " AND ".join(extra_conditions) + " AND (" + " OR ".join(pattern_conditions) + "))"
    conditions = extra_conditions + pattern_conditions
    if not conditions:
        return "1 = 1"
    return "(" + " OR ".join(conditions) + ")"


def _params_for_stratum(stratum: Stratum, columns: set[str]) -> list[str]:
    params: list[str] = []
    for column in stratum.columns:
        if column not in columns:
            continue
        params.extend(f"%{pattern.lower()}%" for pattern in stratum.patterns)
    return params


def _quotas(target_size: int) -> dict[str, int]:
    if target_size < 1:
        raise ValueError("--target-size must be at least 1")
    base = target_size // len(STRATA)
    remainder = target_size % len(STRATA)
    return {stratum.name: base + (1 if index < remainder else 0) for index, stratum in enumerate(STRATA)}


def _fetch_candidates(
    store: ReadOnlyBenchmarkStore,
    stratum: Stratum,
    columns: set[str],
    *,
    candidate_limit: int,
) -> list[dict[str, Any]]:
    selected_columns = [column for column in OUTPUT_COLUMNS if column in columns]
    if "id" not in selected_columns or "content" not in selected_columns:
        raise ValueError("Snapshot chunks table must include id and content columns")

    condition = _condition_for_stratum(stratum, columns)
    params = _params_for_stratum(stratum, columns)
    order_column = "created_at" if "created_at" in columns else "id"
    sql = f"""
        SELECT {", ".join(selected_columns)}
        FROM chunks
        WHERE content IS NOT NULL
          AND length(trim(content)) > 0
          AND {condition}
        ORDER BY {order_column} DESC, id ASC
        LIMIT ?
    """
    cursor = store._read_cursor()
    cursor.execute(sql, [*params, candidate_limit])
    rows = cursor.fetchall()
    return [dict(zip(selected_columns, row, strict=True)) for row in rows]


def _record_from_row(row: dict[str, Any], stratum: str) -> dict[str, Any]:
    chunk_id = str(row["id"])
    source_file = str(row.get("source_file") or "")
    base_doc_id = source_file if source_file else chunk_id
    return {
        "chunk_id": chunk_id,
        "eval_doc_id": f"{canonical_eval_doc_id(base_doc_id)}#{chunk_id}",
        "stratum": stratum,
        "content": str(row["content"]),
        "content_length": len(str(row["content"])),
        "source_file": source_file or None,
        "project": row.get("project"),
        "content_type": row.get("content_type"),
        "tags": row.get("tags"),
        "summary": row.get("summary"),
        "sentiment_label": row.get("sentiment_label"),
        "sender": row.get("sender"),
        "created_at": row.get("created_at"),
    }


def sample_enrichment_gold(
    *,
    snapshot_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    target_size: int = DEFAULT_TARGET_SIZE,
    seed: int = 20260601,
    candidate_limit_per_stratum: int | None = None,
) -> list[dict[str, Any]]:
    """Sample stratified raw chunks from a read-only snapshot and write JSONL.

    Chunk content is untrusted data. This sampler copies it as inert labeler input only;
    it must not be interpreted as agent instructions.

    # TODO(PR1-followup): run the real 60-chunk pull from a vacuumed snapshot when
    # machine load permits, then commit/update fixtures/enrichment-gold.jsonl.
    """

    snapshot = _guard_snapshot_path(snapshot_path)
    output = Path(output_path).expanduser()
    quotas = _quotas(target_size)
    candidate_limit = candidate_limit_per_stratum or max(100, target_size * 20)
    rng = random.Random(seed)

    records: list[dict[str, Any]] = []
    used_chunk_ids: set[str] = set()
    leftovers: list[tuple[str, dict[str, Any]]] = []

    with ReadOnlyBenchmarkStore(snapshot) as store:
        columns = _table_columns(store.conn, "chunks")
        for stratum in STRATA:
            candidates = _fetch_candidates(
                store,
                stratum,
                columns,
                candidate_limit=candidate_limit,
            )
            rng.shuffle(candidates)
            quota = quotas[stratum.name]
            picked = 0
            for row in candidates:
                chunk_id = str(row["id"])
                if chunk_id in used_chunk_ids:
                    continue
                if picked < quota:
                    records.append(_record_from_row(row, stratum.name))
                    used_chunk_ids.add(chunk_id)
                    picked += 1
                else:
                    leftovers.append((stratum.name, row))

        if len(records) < target_size:
            rng.shuffle(leftovers)
            for stratum_name, row in leftovers:
                chunk_id = str(row["id"])
                if chunk_id in used_chunk_ids:
                    continue
                records.append(_record_from_row(row, stratum_name))
                used_chunk_ids.add(chunk_id)
                if len(records) >= target_size:
                    break

    records = records[:target_size]
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records)
    output.write_text(f"{payload}\n" if payload else "", encoding="utf-8")
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample enrichment gold chunks from a read-only BrainLayer snapshot.")
    parser.add_argument(
        "--snapshot-path", required=True, help="Path to a read-only SQLite snapshot. Live DB is rejected."
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT_PATH), help=f"JSONL output path (default: {DEFAULT_OUTPUT_PATH})"
    )
    parser.add_argument(
        "--target-size", type=int, default=DEFAULT_TARGET_SIZE, help="Target number of chunks to sample."
    )
    parser.add_argument("--seed", type=int, default=20260601, help="Deterministic shuffle seed.")
    parser.add_argument(
        "--candidate-limit-per-stratum",
        type=int,
        default=None,
        help="Max candidate rows read per stratum before Python-side sampling.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = sample_enrichment_gold(
        snapshot_path=args.snapshot_path,
        output_path=args.output,
        target_size=args.target_size,
        seed=args.seed,
        candidate_limit_per_stratum=args.candidate_limit_per_stratum,
    )
    print(f"Wrote {len(records)} enrichment gold records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
