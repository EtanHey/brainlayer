"""#623: a content-changing dedupe merge must keep content_hash == sha256(content.strip()).

Both merge writers run against the production ``VectorStore`` schema (FTS
triggers, vector tables, content_hash index) on a private tmp DB.
"""

from __future__ import annotations

import hashlib

import apsw
import pytest

from brainlayer.dedupe import merge_duplicate_chunk, merge_existing_chunk_content
from brainlayer.vector_store import VectorStore

ORIGINAL = "Quartz retrieval original about the lantern ledger"
INCOMING = "Quartz retrieval original about the lantern ledger, now with the amber footnote"


def _sha(content: str) -> str:
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()


@pytest.fixture
def store(tmp_path):
    vs = VectorStore(tmp_path / "content-hash.db")
    vs.conn.execute(
        """INSERT INTO chunks(id, content, metadata, source_file, content_hash, created_at, seen_count)
           VALUES ('canonical', ?, '{}', 'test.jsonl', ?, '2026-10-22T01:00:00Z', 1)""",
        (ORIGINAL, _sha(ORIGINAL)),
    )
    try:
        yield vs
    finally:
        vs.close()


def _merge(conn, merge_kind: str, content: str) -> bool:
    incoming = {"content": content, "created_at": "2026-10-22T02:00:00Z"}
    conn.execute("BEGIN IMMEDIATE")
    try:
        if merge_kind == "duplicate":
            changed = merge_duplicate_chunk(
                conn,
                canonical_id="canonical",
                duplicate_id="duplicate",
                incoming=incoming,
                mechanism="simhash",
                hamming_distance_value=3,
                ensure_schema=False,
            )
        else:
            changed = merge_existing_chunk_content(conn, chunk_id="canonical", incoming=incoming, ensure_schema=False)
        conn.execute("COMMIT")
        return changed
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _row(conn):
    return conn.execute(
        "SELECT content, content_hash, dedupe_hash, simhash FROM chunks WHERE id = 'canonical'"
    ).fetchone()


@pytest.mark.parametrize("merge_kind", ["duplicate", "content"])
def test_content_changing_merge_rewrites_content_hash(store, merge_kind):
    assert _merge(store.conn, merge_kind, INCOMING) is True

    content, content_hash, dedupe_hash, simhash = _row(store.conn)
    assert content.startswith("[Merged duplicate originals]")
    assert ORIGINAL in content and INCOMING in content  # retained originals
    assert dedupe_hash and simhash
    assert content_hash == _sha(content)


@pytest.mark.parametrize("merge_kind", ["duplicate", "content"])
def test_replayed_merge_is_idempotent_and_leaves_hash_alone(store, merge_kind):
    _merge(store.conn, merge_kind, INCOMING)
    first = _row(store.conn)

    assert _merge(store.conn, merge_kind, INCOMING) is False
    assert _merge(store.conn, merge_kind, ORIGINAL) is False
    assert _row(store.conn) == first


@pytest.mark.parametrize("merge_kind", ["duplicate", "content"])
def test_no_op_merge_does_not_touch_content_hash(store, merge_kind):
    # A no-op merge is not a repair path: whatever content_hash holds stays.
    store.conn.execute("UPDATE chunks SET content_hash = 'legacy-16charhash' WHERE id = 'canonical'")
    assert _merge(store.conn, merge_kind, ORIGINAL) is False
    assert _row(store.conn)[:2] == (ORIGINAL, "legacy-16charhash")


@pytest.mark.parametrize("merge_kind", ["duplicate", "content"])
def test_failed_merge_rolls_back_content_and_hash_together(store, merge_kind):
    before = _row(store.conn)
    store.conn.execute(
        """CREATE TRIGGER fail_after_chunk_update BEFORE INSERT ON dedupe_audit BEGIN
               SELECT RAISE(ABORT, 'injected audit failure');
           END"""
    )
    with pytest.raises(apsw.ConstraintError, match="injected audit failure"):
        _merge(store.conn, merge_kind, INCOMING)
    assert _row(store.conn) == before


def test_drain_same_id_merge_lets_fresh_enrichment_apply(store):
    """The issue's symptom on the drain writer: a fresh event for the merged body was rejected."""
    from brainlayer.drain import _apply_enrichment, _insert_or_merge_chunk

    conn = store.conn
    conn.execute("BEGIN IMMEDIATE")
    _insert_or_merge_chunk(
        conn,
        {
            "id": "canonical",
            "content": INCOMING,
            "metadata": "{}",
            "source_file": "test.jsonl",
            "created_at": "2026-10-22T02:00:00Z",
        },
    )
    conn.execute("COMMIT")
    content = _row(conn)[0]
    assert content.startswith("[Merged duplicate originals]")

    event = {"chunk_id": "canonical", "content_hash": _sha(content), "enrichment": {"summary": "fresh"}}
    _apply_enrichment(conn, event)
    summary, stored_hash = conn.execute("SELECT summary, content_hash FROM chunks WHERE id = 'canonical'").fetchone()
    assert summary == "fresh"
    assert stored_hash == _sha(content)
