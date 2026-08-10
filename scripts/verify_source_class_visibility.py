#!/usr/bin/env python3
"""Verify source-class search visibility and exact expansion on a migrated DB."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brainlayer.vector_store import VectorStore

VISIBLE_CLASSES = ("cli-agent", "subagent", "fleet-coordination")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]{7,}")


def _ids(result: dict[str, list]) -> set[str]:
    return set(result.get("ids", [[]])[0])


def _candidate_rows(store: VectorStore, source_class: str | None) -> list[tuple[str, str]]:
    predicate = "c.source_class IS NULL" if source_class is None else "c.source_class = ?"
    params = () if source_class is None else (source_class,)
    return list(
        store.conn.cursor().execute(
            f"""
            SELECT c.id, c.content
            FROM chunks c
            JOIN chunk_fts_rowids r ON r.chunk_id = c.id
            WHERE {predicate}
              AND COALESCE(c.archived, 0) = 0
              AND c.superseded_by IS NULL
              AND c.aggregated_into IS NULL
              AND c.archived_at IS NULL
              AND LENGTH(COALESCE(c.content, '')) >= 16
            ORDER BY c.rowid
            LIMIT 250
            """,
            params,
        )
    )


def _find_search_probe(store: VectorStore, source_class: str | None) -> dict[str, object]:
    expected_default = source_class in VISIBLE_CLASSES or source_class is None
    expected_opt_in = source_class != "brain-worker"
    for chunk_id, content in _candidate_rows(store, source_class):
        for token in dict.fromkeys(_TOKEN_RE.findall(content)):
            count_row = (
                store.conn.cursor()
                .execute(
                    "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?",
                    (token,),
                )
                .fetchone()
            )
            if count_row is None or not 1 <= int(count_row[0]) <= 50:
                continue
            default_ids = _ids(
                store.search(
                    query_text=token,
                    n_results=100,
                    include_audit=True,
                    include_operational=True,
                    include_checkpoints=True,
                )
            )
            opt_in_ids = _ids(
                store.search(
                    query_text=token,
                    n_results=100,
                    include_audit=True,
                    include_operational=True,
                    include_checkpoints=True,
                    include_hidden_source_classes=True,
                )
            )
            if (chunk_id in default_ids) != expected_default:
                continue
            if (chunk_id in opt_in_ids) != expected_opt_in:
                continue
            return {
                "chunk_id": chunk_id,
                "token": token,
                "default_visible": chunk_id in default_ids,
                "desktop_opt_in_visible": chunk_id in opt_in_ids,
            }
    label = "NULL" if source_class is None else source_class
    raise RuntimeError(f"could not find a deterministic FTS visibility probe for {label}")


def _audit_hidden_class_default_visibility(store: VectorStore, source_class: str) -> dict[str, object]:
    """Fail on any hidden-class id returned by the sampled default-search corpus."""
    checked_tokens: set[str] = set()
    leaked_ids: set[str] = set()
    cursor = store.conn.cursor()
    for _chunk_id, content in _candidate_rows(store, source_class):
        for token in dict.fromkeys(_TOKEN_RE.findall(content)):
            if token in checked_tokens:
                continue
            count_row = cursor.execute(
                "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?",
                (token,),
            ).fetchone()
            if count_row is None or not 1 <= int(count_row[0]) <= 50:
                continue
            checked_tokens.add(token)
            result_ids = _ids(
                store.search(
                    query_text=token,
                    n_results=100,
                    include_audit=True,
                    include_operational=True,
                    include_checkpoints=True,
                )
            )
            if result_ids:
                placeholders = ",".join("?" for _ in result_ids)
                leaked_ids.update(
                    str(row[0])
                    for row in cursor.execute(
                        f"SELECT id FROM chunks WHERE source_class = ? AND id IN ({placeholders})",
                        (source_class, *sorted(result_ids)),
                    )
                )
            if len(checked_tokens) >= 250:
                break
        if len(checked_tokens) >= 250:
            break
    if not checked_tokens:
        raise RuntimeError(f"could not build aggregate visibility audit for {source_class}")
    if leaked_ids:
        raise RuntimeError(f"{source_class} default-search leak: {sorted(leaked_ids)[:20]}")
    return {"sampled_tokens": len(checked_tokens), "leaked_ids": []}


def _brain_worker_index_counts(store: VectorStore) -> dict[str, int]:
    """Count indexed rows across the complete brain-worker class, not one sample."""
    tables = {
        str(row[0])
        for row in store.conn.cursor().execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
    }
    counts: dict[str, int] = {}
    for table in (
        "chunks_fts",
        "chunks_fts_operational",
        "chunks_fts_trigram",
        "chunk_fts_rowids",
        "chunk_vectors",
        "chunk_vectors_binary",
    ):
        if table not in tables:
            continue
        counts[table] = int(
            store.conn.cursor()
            .execute(
                f'SELECT COUNT(*) FROM "{table}" i JOIN chunks c ON c.id = i.chunk_id '
                "WHERE c.source_class = 'brain-worker'"
            )
            .fetchone()[0]
        )
    return counts


def verify(db_path: Path) -> dict[str, object]:
    store = VectorStore(db_path.expanduser().resolve(), readonly=True)
    try:
        receipt: dict[str, object] = {}
        for source_class in (*VISIBLE_CLASSES, "desktop", None):
            label = "NULL" if source_class is None else source_class
            probe = _find_search_probe(store, source_class)
            context = store.get_context(str(probe["chunk_id"]), include_audit=True, include_checkpoints=True)
            probe["exact_expansion"] = (context.get("target") or {}).get("id") == probe["chunk_id"]
            if not probe["exact_expansion"]:
                raise RuntimeError(f"exact expansion failed for {label}")
            receipt[label] = probe

        receipt["desktop"]["aggregate_default_visibility"] = _audit_hidden_class_default_visibility(store, "desktop")

        brain_worker_row = (
            store.conn.cursor()
            .execute("SELECT id FROM chunks WHERE source_class = 'brain-worker' ORDER BY rowid LIMIT 1")
            .fetchone()
        )
        if brain_worker_row is None:
            raise RuntimeError("no brain-worker row exists for exact-expansion verification")
        brain_worker_id = str(brain_worker_row[0])
        context = store.get_context(brain_worker_id, include_audit=True, include_checkpoints=True)
        index_counts = _brain_worker_index_counts(store)
        if (context.get("target") or {}).get("id") != brain_worker_id or any(index_counts.values()):
            raise RuntimeError("brain-worker must expand exactly while remaining outside search indexes")
        receipt["brain-worker"] = {
            "chunk_id": brain_worker_id,
            "default_visible": False,
            "desktop_opt_in_visible": False,
            "exact_expansion": True,
            "index_rows": index_counts,
        }
        return receipt
    finally:
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.db), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
