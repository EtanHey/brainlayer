"""Issue #711 fail-first fixture: chunk_fts_rowids vs lifecycle-filtered truth.

Report/fixture only. The `chunks_fts_update` trigger in vector_store.py fires on
`content, summary, tags, resolved_query, key_facts, resolved_queries, content_class`
only, and the insert triggers filter on content_class only. Nothing reacts to
`archived_at`, `superseded_by`, `aggregated_into` or `status`, and nothing filters
empty content. Each case below drives a synthetic private DB through the real
schema and real write functions, then compares the trigger-maintained map with the
lifecycle-filtered truth. Cases expected to drift are `xfail(strict=True)`, so a
repair that fixes them flips the suite red until the marker is removed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainlayer.chunk_write import insert_canonical_chunk  # noqa: E402
from brainlayer.vector_store import VectorStore  # noqa: E402

TRUTH_SQL = """
SELECT id FROM chunks
WHERE archived_at IS NULL AND superseded_by IS NULL AND aggregated_into IS NULL
  AND COALESCE(status, 'active') = 'active'
  AND COALESCE(TRIM(content), '') <> ''
  AND COALESCE(content_class, 'knowledge') NOT IN ('operational', 'test', 'benchmark', 'cold')
"""
MAP_SQL = "SELECT chunk_id FROM chunk_fts_rowids WHERE fts_rowid IS NOT NULL"

DRIFT = pytest.mark.xfail(strict=True, reason="#711: lifecycle columns do not maintain chunk_fts_rowids")


@pytest.fixture
def store(tmp_path: Path):
    instance = VectorStore(tmp_path / "issue711.db")
    cursor = instance.conn.cursor()
    for chunk_id in ("a", "b"):
        insert_canonical_chunk(
            cursor,
            {"id": chunk_id, "content": "alpha beta gamma delta", "source": "test", "content_class": "knowledge"},
        )
    yield instance
    instance.close()


def _ids(store: VectorStore, sql: str) -> set[str]:
    return {row[0] for row in store.conn.cursor().execute(sql)}


def _sql(statement: str):
    return lambda store: store.conn.cursor().execute(statement)


def _assert_map_matches_truth(store: VectorStore) -> None:
    assert _ids(store, MAP_SQL) == _ids(store, TRUTH_SQL)


def test_baseline_map_matches_truth(store: VectorStore) -> None:
    _assert_map_matches_truth(store)


CASES = [
    pytest.param([lambda s: s.archive_chunk("a")], id="archive", marks=DRIFT),
    pytest.param(
        [lambda s: s.archive_chunk("a"), _sql("UPDATE chunks SET archived_at = NULL WHERE id = 'a'")],
        id="archive-then-unarchive",
    ),
    pytest.param([lambda s: s.supersede_chunk("a", "b")], id="supersede", marks=DRIFT),
    pytest.param(
        [
            lambda s: s.supersede_chunk("a", "b"),
            _sql("UPDATE chunks SET superseded_by = NULL, status = 'active' WHERE id = 'a'"),
        ],
        id="supersede-then-unsupersede",
    ),
    pytest.param([_sql("UPDATE chunks SET aggregated_into = 'b' WHERE id = 'a'")], id="aggregate", marks=DRIFT),
    pytest.param(
        [
            _sql("UPDATE chunks SET aggregated_into = 'b' WHERE id = 'a'"),
            _sql("UPDATE chunks SET aggregated_into = NULL WHERE id = 'a'"),
        ],
        id="aggregate-then-deaggregate",
    ),
    pytest.param([_sql("UPDATE chunks SET status = 'archived' WHERE id = 'a'")], id="status-retire", marks=DRIFT),
    pytest.param(
        [
            _sql("UPDATE chunks SET status = 'archived' WHERE id = 'a'"),
            _sql("UPDATE chunks SET status = 'active' WHERE id = 'a'"),
        ],
        id="status-retire-then-restore",
    ),
    pytest.param([_sql("UPDATE chunks SET content = '' WHERE id = 'a'")], id="content-emptied", marks=DRIFT),
    pytest.param(
        [
            _sql("UPDATE chunks SET content = '' WHERE id = 'a'"),
            _sql("UPDATE chunks SET content = 'alpha beta' WHERE id = 'a'"),
        ],
        id="content-emptied-then-restored",
    ),
]


@pytest.mark.parametrize("steps", CASES)
def test_rowid_map_tracks_lifecycle_transition(store: VectorStore, steps) -> None:
    for step in steps:
        step(store)
    _assert_map_matches_truth(store)


def test_canonical_insert_rejects_empty_content(store: VectorStore) -> None:
    """Empty content cannot enter through the canonical insert path, so the only
    empty-content drift route is a later UPDATE (covered by `content-emptied`)."""
    with pytest.raises(ValueError, match="content must be non-empty"):
        insert_canonical_chunk(
            store.conn.cursor(), {"id": "empty", "content": "", "source": "test", "content_class": "knowledge"}
        )
    _assert_map_matches_truth(store)
