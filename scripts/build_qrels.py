"""Build heuristic qrels for the BrainLayer search benchmark."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from brainlayer._helpers import _escape_fts5_query
from brainlayer.eval.benchmark import (
    DEFAULT_QUERY_SUITE,
    ReadOnlyBenchmarkStore,
    pipeline_fts5_only,
    pipeline_hybrid_rrf,
)
from brainlayer.paths import get_db_path
from brainlayer.vector_store import VectorStore


def heuristic_grade(query: str, content: str, summary: str | None, tags: list[str]) -> int:
    query_terms = [term.lower() for term in query.split() if term.strip()]
    content_lower = content.lower()
    summary_lower = (summary or "").lower()
    tags_lower = [tag.lower() for tag in tags]

    if (
        query_terms
        and all(term in content_lower for term in query_terms)
        and all(term in summary_lower for term in query_terms)
    ):
        return 3
    if query_terms and any(term in content_lower for term in query_terms):
        return 2
    if query_terms and any(term in tags_lower for term in query_terms):
        return 1
    return 0


def fallback_fts_candidates(store, query: str, n_results: int) -> list[tuple[str, float]]:
    cursor = store._read_cursor()
    relaxed_queries = []

    or_query = _escape_fts5_query(query, match_mode="or")
    if or_query:
        relaxed_queries.append(or_query)

    keywords = re.findall(r"[\w\u0590-\u05FF-]+", query)
    for keyword in keywords:
        keyword_query = _escape_fts5_query(keyword, match_mode="or")
        if keyword_query:
            relaxed_queries.append(keyword_query)

    seen: set[str] = set()
    for relaxed_query in relaxed_queries:
        rows = list(
            cursor.execute(
                """
                SELECT f.chunk_id, bm25(chunks_fts) AS score
                FROM chunks_fts f
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (relaxed_query, n_results),
            )
        )
        if rows:
            return [
                (chunk_id, float(-score)) for chunk_id, score in rows if not (chunk_id in seen or seen.add(chunk_id))
            ]
    return []


def collect_candidate_ids(store, query_text: str, n_results: int, mode: str) -> list[str]:
    candidates = pipeline_fts5_only(store, query_text, n_results=n_results)
    if not candidates:
        candidates = fallback_fts_candidates(store, query_text, n_results=n_results)

    if mode == "fts-only":
        return [chunk_id for chunk_id, _score in candidates]

    pooled: list[tuple[str, float]] = list(candidates)
    pooled.extend(pipeline_hybrid_rrf(store, query_text, n_results=n_results))

    seen: set[str] = set()
    return [chunk_id for chunk_id, _score in pooled if not (chunk_id in seen or seen.add(chunk_id))]


def build_qrels(store, n_results: int = 20, mode: str = "pool") -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    cursor = store._read_cursor()
    for query_id, query_text in DEFAULT_QUERY_SUITE:
        if query_id in {"q1", "q2"}:
            continue
        print(f"\n# {query_id}: {query_text}")
        judgments: dict[str, int] = {}
        for chunk_id in collect_candidate_ids(store, query_text, n_results, mode):
            row = next(
                cursor.execute(
                    "SELECT content, summary, tags FROM chunks WHERE id = ?",
                    (chunk_id,),
                ),
                None,
            )
            if row is None:
                continue
            content, summary, raw_tags = row
            judgments[chunk_id] = heuristic_grade(
                query_text,
                content,
                summary,
                json.loads(raw_tags) if raw_tags else [],
            )
            print(f"- {chunk_id}: grade={judgments[chunk_id]} summary={(summary or '')[:120]}")
        qrels[query_id] = judgments
    return qrels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default=str(get_db_path()), help="Path to the BrainLayer SQLite DB")
    parser.add_argument(
        "--output",
        default="tests/eval_qrels.json",
        help="Path to write the generated qrels JSON",
    )
    parser.add_argument("--n-results", type=int, default=20, help="Results to grade per query from each pipeline")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--pool", dest="mode", action="store_const", const="pool", help="Pool FTS5 and hybrid RRF candidates"
    )
    mode_group.add_argument(
        "--fts-only", dest="mode", action="store_const", const="fts-only", help="Use legacy FTS5-only candidates"
    )
    parser.set_defaults(mode="pool")
    args = parser.parse_args()

    store_factory = VectorStore if args.mode == "pool" else ReadOnlyBenchmarkStore
    with store_factory(Path(args.db_path)) as store:
        qrels = build_qrels(store, n_results=args.n_results, mode=args.mode)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(qrels, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(qrels)} queries to {output_path}")


if __name__ == "__main__":
    main()
