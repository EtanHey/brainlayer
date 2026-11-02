import json
import logging
import os
import shutil
import sqlite3
from pathlib import Path

import pytest

from brainlayer._helpers import serialize_f32
from brainlayer.vector_store import VectorStore

# Repair (c): duplicate merge archives via archived_at only.


@pytest.mark.parametrize("canonical_kind", ["long-merged", "short", "long-first-merge"])
@pytest.mark.parametrize("merge_api", ["duplicate", "same-id", "upsert"])
def test_duplicate_merge_invalidates_only_when_embedding_input_changes(
    tmp_path, caplog, monkeypatch, canonical_kind, merge_api
):
    from brainlayer.dedupe import (
        DuplicateHit,
        _merged_content,
        compute_dedupe_fields,
        merge_duplicate_chunk,
        merge_existing_chunk_content,
    )
    from brainlayer.embeddings import MAX_PASSAGE_CHARS, embedding_input

    original = ("original " * 2000).strip() if canonical_kind != "short" else "short original"
    content = _merged_content(original, "earlier distinct original") if canonical_kind == "long-merged" else original
    incoming = "new distinct near-duplicate original"
    if canonical_kind != "short":
        assert len(content) > MAX_PASSAGE_CHARS
    store = VectorStore(tmp_path / "merge-prefix.db")
    try:
        store.conn.execute(
            "INSERT INTO chunks (id,content,metadata,source_file,created_at) VALUES ('canonical',?,'{}','test',?)",
            (content, "2026-10-31T00:00:00Z"),
        )
        store._upsert_chunk_vector(store.conn.cursor(), "canonical", [0.25] * 1024)
        vectors_before = {
            table: store.conn.execute(f"SELECT embedding FROM {table} WHERE chunk_id='canonical'").fetchone()
            for table in ("chunk_vectors", "chunk_vectors_binary")
        }
        with caplog.at_level(logging.INFO, logger="brainlayer.dedupe"):
            values = {"content": incoming, "created_at": "2026-11-02T00:00:00Z"}
            if merge_api == "duplicate":
                changed = merge_duplicate_chunk(
                    store.conn,
                    canonical_id="canonical",
                    duplicate_id="new-original",
                    incoming=values,
                    mechanism="simhash",
                    hamming_distance_value=1,
                )
            elif merge_api == "same-id":
                changed = merge_existing_chunk_content(store.conn, chunk_id="canonical", incoming=values)
            else:
                monkeypatch.setattr(
                    "brainlayer.vector_store.find_duplicate",
                    lambda *args, **kwargs: (
                        DuplicateHit("canonical", "simhash", 1),
                        compute_dedupe_fields(incoming, values["created_at"]),
                    ),
                )
                changed = (
                    store.upsert_chunks(
                        [{**values, "id": "new-original", "metadata": {}, "source_file": "test"}],
                        [[0.5] * 1024],
                    )
                    == 1
                )

        merged, created_at, seen_count = store.conn.execute(
            "SELECT content,created_at,seen_count FROM chunks WHERE id='canonical'"
        ).fetchone()
        assert changed is True
        assert original in merged and incoming in merged
        if canonical_kind == "long-merged":
            assert "earlier distinct original" in merged
        assert len(merged) > len(content)
        assert (created_at, seen_count) == ("2026-10-31T00:00:00Z", 2)
        retained = canonical_kind == "long-merged"
        assert (embedding_input(content) == embedding_input(merged)) is retained
        for table, before in vectors_before.items():
            after = store.conn.execute(f"SELECT embedding FROM {table} WHERE chunk_id='canonical'").fetchone()
            assert after == (before if retained else None)
        assert ("skipped_invalidation=1" in caplog.text) is retained
        assert store.conn.execute("SELECT COUNT(*) FROM dedupe_audit WHERE chunk_id_kept='canonical'").fetchone() == (
            1,
        )
    finally:
        store.close()


@pytest.mark.parametrize("merge_kind", ["duplicate", "content"])
@pytest.mark.parametrize("legacy", [False, True], ids=["framed", "legacy"])
def test_merge_idempotent_live_numbered_original(indexed_merge_store, merge_kind, legacy):
    """Exact 1,741-char public fleet rules copied read-only from the live last original.

    Source: rt-rollout--282da4862dd65530, 2026-11-02. No session or secret text.
    """
    import hashlib

    from brainlayer.dedupe import _merged_originals

    original = (Path(__file__).parent / "fixtures/dedupe/rollout_numbered_original.txt").read_text()
    assert len(original) == 1741
    assert hashlib.sha256(original.encode()).hexdigest() == (
        "43b08d26eb8086d35d057c581176cf5d701b461f9992bb1cc53e79ee25955568"
    )
    initial = "Earlier distinct canonical text"
    body = f"[Merged duplicate originals]\n\n1. {initial}\n\n2. {original}" if legacy else initial
    conn = indexed_merge_store.conn
    conn.execute("UPDATE chunks SET content = ?, char_count = ? WHERE id = 'knowledge-canonical'", (body, len(body)))
    if not legacy:
        assert _merge_indexed_chunk(conn, "knowledge", merge_kind, content=original) is True
        body = conn.execute("SELECT content FROM chunks WHERE id = 'knowledge-canonical'").fetchone()[0]
    initial_seen = conn.execute("SELECT seen_count FROM chunks WHERE id = 'knowledge-canonical'").fetchone()[0]
    indexed_merge_store._upsert_chunk_vector(conn.cursor(), "knowledge-canonical", [0.1] * 1024)
    before_vectors = {
        table: conn.execute(f"SELECT chunk_id, embedding FROM {table}").fetchall()
        for table in ("chunk_vectors", "chunk_vectors_binary")
    }
    before_indexes = _fts_snapshot(conn)
    for _ in range(20):
        assert _merge_indexed_chunk(conn, "knowledge", merge_kind, content=original) is False
    assert conn.execute(
        "SELECT content, char_count, seen_count, last_seen_at FROM chunks WHERE id = 'knowledge-canonical'"
    ).fetchone() == (body, len(body), initial_seen + 20, "2026-10-22T02:00:00Z")
    assert _fts_snapshot(conn) == before_indexes
    for table, snapshot in before_vectors.items():
        assert conn.execute(f"SELECT chunk_id, embedding FROM {table}").fetchall() == snapshot
    if not legacy:
        assert _merged_originals(body) == [initial, original]
    print(f"S12 {merge_kind} {'legacy' if legacy else 'framed'}: input=1741 chars, replays=20, growth=0 bytes")


def test_merge_idempotent_framed_round_trip_preserves_numbered_sections_and_delimiters():
    from brainlayer.dedupe import _merged_content, _merged_originals

    originals = [
        "  First original\n\n1. nested one\n\n2. nested two\n\n3. nested three\n",
        "雪 שלום\n\n␞ 1. fake\n\n␞ 2. record ␛R␛E ␛\n[Merged duplicate originals]\n\n␞ 1. x",
        "Raw control bytes stay ordinary text: \x1e and \x1bR are not framing",
        'Another distinct original with quotes " and a newline\n',
    ]
    body = originals[0]
    for original in originals[1:]:
        body = _merged_content(body, original)
    assert _merged_originals(body) == originals
    for original in originals:
        assert _merged_content(body, original) == body


@pytest.mark.parametrize("merge_kind", ["duplicate", "content"])
def test_merge_idempotent_legacy_whitespace_replay_and_distinct_upgrade(indexed_merge_store, merge_kind):
    from brainlayer.dedupe import _merged_content, _merged_originals

    conn = indexed_merge_store.conn
    original = "Quartz retrieval original\n\n1. Sapphire section\n\n2. Emerald section\n\n3. Ruby section"
    legacy = f"[Merged duplicate originals]\n\n1. Earlier original\n\n2. {original}"
    conn.execute(
        "UPDATE chunks SET content = ?, char_count = ? WHERE id = 'knowledge-canonical'", (legacy, len(legacy))
    )
    whitespace_replay = "  " + original.replace(" ", "   ") + "\n"
    assert _merge_indexed_chunk(conn, "knowledge", merge_kind, content=whitespace_replay) is False
    assert conn.execute("SELECT content FROM chunks WHERE id = 'knowledge-canonical'").fetchone() == (legacy,)
    distinct = "Distinct original with sapphire and topaz"
    assert _merge_indexed_chunk(conn, "knowledge", merge_kind, content=distinct) is True
    upgraded = conn.execute("SELECT content FROM chunks WHERE id = 'knowledge-canonical'").fetchone()[0]
    assert upgraded.startswith("[Merged duplicate originals]\n\n␞ 1. ")
    assert _merged_originals(upgraded)[-1] == distinct
    assert _merged_originals(upgraded) == [legacy, distinct]  # Ambiguous legacy text is retained verbatim.
    assert _merged_content(upgraded, whitespace_replay) == upgraded
    assert _merged_content(upgraded, distinct + "\n") == upgraded
    # Identity is exact text: a case variant is a different original and is kept.
    assert _merged_originals(_merged_content(upgraded, distinct.upper()))[-1] == distinct.upper()
    assert _fts_matches(conn, "chunks_fts", "topaz") == [("knowledge-canonical",)]
    assert _fts_matches(conn, "chunks_fts", "sapphire") == [("knowledge-canonical",)]
    assert _merge_indexed_chunk(conn, "knowledge", merge_kind, content=distinct) is False
    assert _merge_indexed_chunk(conn, "knowledge", merge_kind, content="topaz") is True  # A substring is distinct.


def test_merge_idempotent_legacy_replay_requires_original_boundaries():
    from brainlayer.dedupe import _merged_content, _merged_originals

    legacy = "[Merged duplicate originals]\n\n1. Larger original containing sapphire text\n\n2. Other original"
    assert _merged_originals(_merged_content(legacy, "sapphire"))[-1] == "sapphire"


def test_merge_idempotent_simple_legacy_upgrades_and_matches_every_original():
    from brainlayer.dedupe import _merged_content, _merged_originals

    legacy = "[Merged duplicate originals]\n\n1. Quartz original\n\n2. Sapphire original"
    assert _merged_content(legacy, " Sapphire   original\n") == legacy
    upgraded = _merged_content(legacy, "Ruby original")
    assert _merged_originals(upgraded) == ["Quartz original", "Sapphire original", "Ruby original"]
    for original in _merged_originals(upgraded):
        assert _merged_content(upgraded, f"\n{original}  ") == upgraded


def test_merge_idempotent_malformed_frame_kept_verbatim():
    from brainlayer.dedupe import _merged_content, _merged_originals

    malformed = "[Merged duplicate originals]\n\n␞ 1. partial␛X"
    assert _merged_originals(_merged_content(malformed, "new original")) == [malformed, "new original"]


_DISTINCT_PAIRS = {
    # Opposite meanings; both normalize to "" under normalize_for_exact_hash.
    "hebrew": ("אל תמחק את הגיבוי", "מחק את הגיבוי עכשיו"),
    "emoji": ("🔥🔥", "🎉"),
    "cjk": ("不要删除备份", "现在删除备份"),
    # Same normalized tokens ("migrate b"), different order of stopwords.
    "stopword-reorder": ("migrate from A to B", "migrate to A from B"),
}


@pytest.mark.parametrize("pair", list(_DISTINCT_PAIRS), ids=list(_DISTINCT_PAIRS))
@pytest.mark.parametrize("merge_kind", ["duplicate", "content"])
@pytest.mark.parametrize("legacy", [False, True], ids=["framed", "legacy"])
def test_merge_keeps_distinct_originals_that_normalize_alike(indexed_merge_store, merge_kind, legacy, pair):
    from brainlayer.dedupe import _merged_originals

    first, second = _DISTINCT_PAIRS[pair]
    conn = indexed_merge_store.conn
    initial = "Earlier distinct canonical text"
    body = f"[Merged duplicate originals]\n\n1. {initial}\n\n2. {first}" if legacy else initial
    conn.execute("UPDATE chunks SET content = ?, char_count = ? WHERE id = 'knowledge-canonical'", (body, len(body)))

    def stored():
        return conn.execute("SELECT content FROM chunks WHERE id = 'knowledge-canonical'").fetchone()[0]

    if not legacy:
        assert _merge_indexed_chunk(conn, "knowledge", merge_kind, content=first) is True
    # An identical replay of the first original is a no-op in either format.
    before = stored()
    assert _merge_indexed_chunk(conn, "knowledge", merge_kind, content=first) is False
    assert stored() == before
    # The second original is different text: it is appended, never dropped.
    assert _merge_indexed_chunk(conn, "knowledge", merge_kind, content=second) is True
    merged = stored()
    assert first in merged and second in merged
    assert _merged_originals(merged)[-1] == second
    for replay in (first, second, first):
        assert _merge_indexed_chunk(conn, "knowledge", merge_kind, content=replay) is False
    assert stored() == merged


def test_merge_legacy_empty_span_never_absorbs_incoming():
    """Reviewer fuzz shape: an empty legacy span matched any empty-normalized text."""
    from brainlayer.dedupe import _merged_content, _merged_originals

    legacy = "[Merged duplicate originals]\n\n1. \n\n2. something"
    for incoming in ("é", "a", "\x1bE", "יאללה"):
        merged = _merged_content(legacy, incoming)
        assert merged != legacy and _merged_originals(merged)[-1] == incoming
    # The same span nested as a framed record (an ambiguous legacy original).
    framed = _merged_content("[Merged duplicate originals]\n\n1. \n\n2. \n\n1. x", "plain")
    merged = _merged_content(framed, "é")
    assert merged != framed and _merged_originals(merged)[-1] == "é"
    assert _merged_content(legacy, "   ") == legacy  # Whitespace-only input adds nothing.


def test_merged_body_is_readable_and_terminal_safe_in_every_reader():
    """No framing control byte (RS/ESC, any C0 control) reaches previews, snippets or expand."""
    import re as _re

    from brainlayer.dedupe import _merged_content
    from brainlayer.mcp._format import format_recalled_context, format_search_results
    from brainlayer.mcp._shared import _build_compact_result

    body = "first original text"
    for original in (
        "second original text\n\n1. nested\n\n2. sections",
        "אל תמחק את הגיבוי",
        "symbols ␞ and ␛ survive escaped",
    ):
        body = _merged_content(body, original)
    control = _re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
    item = {"chunk_id": "c1", "content": body, "project": "p", "source_file": "s.jsonl", "date": "2026-11-02"}
    snippet = _build_compact_result(item)["snippet"]
    surfaces = {
        # brain_expand/recall full-content surfaces (BrainBar included) return the stored body as-is.
        "stored body / expand": body,
        "brain_search snippet": snippet,
        "compact preview": format_search_results("q", [{**item, "snippet": snippet}], 1),
        "full detail": format_search_results("q", [item], 1, detail="full"),
        "recall": format_recalled_context("q", [{**item, "is_target": True}, item]),
    }
    for name, text in surfaces.items():
        assert not control.search(text), name
        assert "escaped-records" not in text, name
    assert snippet.startswith("[Merged duplicate originals]\n\n␞ 1. first original text\n\n␞ 2. second")


def _chunk(
    chunk_id: str,
    content: str,
    *,
    created_at: str,
    tags=None,
    importance=None,
    half_life_days=None,
    project="brainlayer",
    content_type="note",
):
    chunk = {
        "id": chunk_id,
        "content": content,
        "metadata": {"session_id": "session-a"},
        "source_file": "test.jsonl",
        "project": project,
        "content_type": content_type,
        "value_type": "HIGH",
        "char_count": len(content),
        "source": "test",
        "created_at": created_at,
    }
    if tags is not None:
        chunk["tags"] = tags
    if importance is not None:
        chunk["importance"] = importance
    if half_life_days is not None:
        chunk["half_life_days"] = half_life_days
    return chunk


@pytest.fixture
def indexed_merge_store(tmp_path, request):
    """Use production FTS triggers, with a later row so a rebuild changes rowids."""
    store = VectorStore(tmp_path / "indexed-merges.db")
    stored_tags = getattr(request, "param", '["amber", "zebra"]')
    for content_class in ("knowledge", "operational"):
        for chunk_id, content in (("canonical", "Quartz retrieval original"), ("later", "Unrelated orchard")):
            store.conn.execute(
                """INSERT INTO chunks(id, content, metadata, source_file, tags, content_class, created_at, seen_count)
                   VALUES (?, ?, '{}', 'test.jsonl', ?, ?, ?, 1)""",
                (f"{content_class}-{chunk_id}", content, stored_tags, content_class, "2026-10-22T01:00:00Z"),
            )
    try:
        yield store
    finally:
        store.close()


def _merge_indexed_chunk(conn, content_class, merge_kind, *, content="Quartz retrieval original", tags=None):
    from brainlayer.dedupe import merge_duplicate_chunk, merge_existing_chunk_content, merge_existing_chunk_seen

    chunk_id = f"{content_class}-canonical"
    incoming = {
        "content": content,
        "tags": ["zebra", "amber"] if tags is None else tags,
        "created_at": "2026-10-22T02:00:00Z",
        "importance": 9,
        "half_life_days": 90,
    }
    # Match drain's transaction ownership; merge helpers retain their savepoints.
    conn.execute("BEGIN IMMEDIATE")
    try:
        if merge_kind == "duplicate":
            changed = merge_duplicate_chunk(
                conn,
                canonical_id=chunk_id,
                duplicate_id=f"{content_class}-duplicate",
                incoming=incoming,
                mechanism="sha256" if content == "Quartz retrieval original" else "simhash",
                hamming_distance_value=0,
                ensure_schema=False,
            )
        else:
            merge = merge_existing_chunk_content if merge_kind == "content" else merge_existing_chunk_seen
            changed = merge(conn, chunk_id=chunk_id, incoming=incoming, ensure_schema=False)
        conn.execute("COMMIT")
        return changed
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _fts_snapshot(conn):
    return {
        table: conn.execute(f"SELECT rowid, * FROM {table} ORDER BY rowid").fetchall()
        for table in ("chunks_fts", "chunks_fts_operational", "chunks_fts_trigram", "chunk_fts_rowids", "chunk_tags")
    }


def _fts_matches(conn, table, query):
    return conn.execute(f"SELECT chunk_id FROM {table} WHERE {table} MATCH ? ORDER BY chunk_id", (query,)).fetchall()


@pytest.mark.parametrize("content_class", ["knowledge", "operational"])
@pytest.mark.parametrize("merge_kind", ["duplicate", "content", "seen"])
def test_exact_duplicate_merge_preserves_fts_rowids_and_retrieval(indexed_merge_store, content_class, merge_kind):
    conn = indexed_merge_store.conn
    conn.execute("CREATE TEMP TABLE merge_tag_writes (chunk_id TEXT)")
    conn.execute("""CREATE TEMP TRIGGER record_merge_tag_write AFTER UPDATE OF tags ON chunks BEGIN
        INSERT INTO merge_tag_writes VALUES (new.id);
    END""")
    before = _fts_snapshot(conn)
    table = "chunks_fts" if content_class == "knowledge" else "chunks_fts_operational"
    matches = _fts_matches(conn, table, "quartz")
    assert matches == [(f"{content_class}-canonical",)]

    _merge_indexed_chunk(conn, content_class, merge_kind)

    assert _fts_snapshot(conn) == before
    assert conn.execute("SELECT * FROM merge_tag_writes").fetchall() == []
    assert _fts_matches(conn, table, "quartz") == matches
    assert conn.execute(
        "SELECT seen_count, last_seen_at, importance, half_life_days FROM chunks WHERE id = ?",
        (f"{content_class}-canonical",),
    ).fetchone() == (2, "2026-10-22T02:00:00Z", 9.0, 90.0)
    expected_dropped = f"{content_class}-duplicate" if merge_kind == "duplicate" else f"{content_class}-canonical"
    assert conn.execute("SELECT chunk_id_dropped, chunk_id_kept FROM dedupe_audit").fetchall() == [
        (expected_dropped, f"{content_class}-canonical")
    ]
    assert conn.execute("SELECT old_chunk_id, canonical_chunk_id FROM chunk_id_alias").fetchall() == (
        [(expected_dropped, f"{content_class}-canonical")] if merge_kind == "duplicate" else []
    )


@pytest.mark.parametrize("content_class", ["knowledge", "operational"])
@pytest.mark.parametrize("merge_kind", ["duplicate", "content", "seen"])
def test_duplicate_merge_preserves_historical_audit_rows(indexed_merge_store, content_class, merge_kind):
    conn = indexed_merge_store.conn
    conn.execute(
        "INSERT INTO dedupe_audit VALUES (?, ?, ?, ?, ?)",
        ("old-canary-雪", "historical-kept", "historical mechanism  ", None, "2000-01-01T00:00:00+00:00"),
    )
    old_id = conn.execute("SELECT rowid FROM dedupe_audit").fetchone()[0]
    old_bytes_sql = """SELECT rowid, CAST(chunk_id_dropped AS BLOB), CAST(chunk_id_kept AS BLOB),
        CAST(mechanism AS BLOB), hamming_distance, CAST(ts AS BLOB)
        FROM dedupe_audit WHERE rowid = ?"""
    before = conn.execute(old_bytes_sql, (old_id,)).fetchall()
    before_indexes = _fts_snapshot(conn)
    before_count = conn.execute("SELECT count(*) FROM dedupe_audit").fetchone()[0]

    _merge_indexed_chunk(conn, content_class, merge_kind)

    assert conn.execute("SELECT count(*) FROM dedupe_audit").fetchone()[0] == before_count + 1
    assert conn.execute(old_bytes_sql, (old_id,)).fetchall() == before
    dropped = f"{content_class}-duplicate" if merge_kind == "duplicate" else f"{content_class}-canonical"
    assert conn.execute(
        "SELECT chunk_id_dropped, chunk_id_kept FROM dedupe_audit WHERE rowid != ?", (old_id,)
    ).fetchall() == [(dropped, f"{content_class}-canonical")]
    assert _fts_snapshot(conn) == before_indexes


@pytest.mark.parametrize("indexed_merge_store", ["zebra, amber", '["zebra","amber"]'], indirect=True)
@pytest.mark.parametrize("content_class", ["knowledge", "operational"])
@pytest.mark.parametrize("merge_kind", ["duplicate", "content"])
def test_duplicate_merge_normalizes_legacy_tags_and_refreshes_tag_search(
    indexed_merge_store, content_class, merge_kind
):
    conn = indexed_merge_store.conn
    chunk_id = f"{content_class}-canonical"
    before_tags = conn.execute("SELECT tags FROM chunks WHERE id = ?", (chunk_id,)).fetchone()[0]
    if before_tags == "zebra, amber":
        assert conn.execute("SELECT tag FROM chunk_tags WHERE chunk_id = ?", (chunk_id,)).fetchall() == []

    assert _merge_indexed_chunk(conn, content_class, merge_kind) is False

    assert conn.execute("SELECT tags FROM chunks WHERE id = ?", (chunk_id,)).fetchone() == ('["amber", "zebra"]',)
    assert conn.execute("SELECT tag FROM chunk_tags WHERE chunk_id = ? ORDER BY tag", (chunk_id,)).fetchall() == [
        ("amber",),
        ("zebra",),
    ]
    assert conn.execute(
        """SELECT chunks.id FROM chunks JOIN chunk_tags ON chunk_tags.chunk_id = chunks.id
           WHERE chunk_tags.tag = 'amber' AND chunks.id = ?""",
        (chunk_id,),
    ).fetchall() == [(chunk_id,)]


@pytest.mark.parametrize("content_class", ["knowledge", "operational"])
@pytest.mark.parametrize("merge_kind", ["duplicate", "content"])
@pytest.mark.parametrize("changed_field", ["content", "tags"])
def test_changed_duplicate_merge_refreshes_retrieval(indexed_merge_store, content_class, merge_kind, changed_field):
    conn = indexed_merge_store.conn
    tables = ["chunks_fts", "chunks_fts_trigram"] if content_class == "knowledge" else ["chunks_fts_operational"]
    before = _fts_snapshot(conn)["chunk_fts_rowids"]
    for table in tables:
        assert _fts_matches(conn, table, "sapphire") == []
    change = {"content": "Quartz retrieval original sapphire"} if changed_field == "content" else {"tags": ["sapphire"]}

    assert _merge_indexed_chunk(conn, content_class, merge_kind, **change) == (changed_field == "content")

    assert _fts_snapshot(conn)["chunk_fts_rowids"] != before
    for table in tables:
        assert _fts_matches(conn, table, "sapphire") == [(f"{content_class}-canonical",)]
        assert _fts_matches(conn, table, "quartz") == [(f"{content_class}-canonical",)]
    if changed_field == "tags":
        assert conn.execute(
            "SELECT tag FROM chunk_tags WHERE chunk_id = ? ORDER BY tag", (f"{content_class}-canonical",)
        ).fetchall() == [("amber",), ("sapphire",), ("zebra",)]


def test_unchanged_duplicate_merge_avoids_fts_writes_at_scale(indexed_merge_store):
    """Count actual FTS shadow-table writes, without a machine-dependent time limit."""
    conn = indexed_merge_store.conn
    for index in range(96):
        conn.execute(
            "INSERT INTO chunks(id, content, metadata, source_file) VALUES (?, ?, '{}', 'test.jsonl')",
            (f"filler-{index}", f"Orchard filler {index}"),
        )
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone() == (100,)
    writes = []
    conn.set_update_hook(
        lambda operation, database, table, rowid: writes.append(table) if table.startswith("chunks_fts") else None
    )
    try:
        # Reproduce the old assignment behavior on the same full-schema temp DB.
        conn.execute("UPDATE chunks SET content = content, tags = tags WHERE id = 'knowledge-canonical'")
        legacy_writes = len(writes)
        writes.clear()
        for _ in range(20):
            _merge_indexed_chunk(conn, "knowledge", "duplicate")
        print(f"N=100: legacy FTS shadow writes/merge={legacy_writes}; unchanged merges=20, FTS writes={len(writes)}")
        assert legacy_writes > 0
        assert len(writes) == 0
    finally:
        conn.set_update_hook(None)


@pytest.mark.parametrize("merge_kind", ["duplicate", "content"], ids=["cross-id", "same-id"])
@pytest.mark.parametrize("fingerprint_state", ["missing", "stale", "valid"])
def test_unchanged_merge_repairs_fingerprints_for_later_duplicate_lookup(
    indexed_merge_store, merge_kind, fingerprint_state
):
    """Repair lookup metadata without rebuilding retrieval or changing peer rows."""
    from brainlayer.dedupe import (
        compute_dedupe_fields,
        find_duplicate,
        hamming_distance,
        merge_duplicate_chunk,
        merge_existing_chunk_content,
    )

    conn = indexed_merge_store.conn
    chunk_id = "knowledge-canonical"
    created_at = "2026-10-22T01:00:00Z"
    words = [f"token{index}" for index in range(100)]
    content = " ".join(words)
    near_content = " ".join(["changed0", *words[1:]])
    expected = compute_dedupe_fields(content, created_at)
    near_fields = compute_dedupe_fields(near_content, created_at)
    # The second lookup must exercise band selection and SimHash, not exact hash.
    assert near_fields.dedupe_hash != expected.dedupe_hash
    assert hamming_distance(expected.simhash, near_fields.simhash) <= 3
    assert any(left == right for left, right in zip(expected.bands, near_fields.bands))
    expected_values = (expected.dedupe_hash, expected.simhash, *expected.bands)
    stored_values = expected_values
    if fingerprint_state == "missing":
        stored_values = (None,) * 6
    elif fingerprint_state == "stale":
        stale_simhash = f"{int(expected.simhash, 16) ^ ((1 << 64) - 1):016x}"
        stored_values = ("stale-exact-hash", stale_simhash, *(stale_simhash[i : i + 4] for i in range(0, 16, 4)))
    conn.execute(
        """UPDATE chunks SET content = ?, char_count = ?, dedupe_hash = ?, simhash = ?,
           simhash_band_0 = ?, simhash_band_1 = ?, simhash_band_2 = ?, simhash_band_3 = ? WHERE id = ?""",
        (content, len(content), *stored_values, chunk_id),
    )
    before_fts = _fts_snapshot(conn)
    peers = conn.execute("SELECT * FROM chunks WHERE id != ? ORDER BY id", (chunk_id,)).fetchall()
    incoming = {"content": content, "tags": ["zebra", "amber"], "created_at": "2026-10-22T02:00:00Z"}

    conn.execute("BEGIN IMMEDIATE")
    try:
        if merge_kind == "duplicate":
            changed = merge_duplicate_chunk(
                conn,
                canonical_id=chunk_id,
                duplicate_id="fingerprint-duplicate",
                incoming=incoming,
                mechanism="sha256",
                hamming_distance_value=0,
                ensure_schema=False,
            )
        else:
            changed = merge_existing_chunk_content(conn, chunk_id=chunk_id, incoming=incoming, ensure_schema=False)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    assert changed is False
    assert _fts_snapshot(conn) == before_fts
    assert conn.execute("SELECT * FROM chunks WHERE id != ? ORDER BY id", (chunk_id,)).fetchall() == peers
    for probe_content, mechanism in ((content, "sha256"), (near_content, "simhash")):
        hit, _ = find_duplicate(conn, chunk_id="later-probe", content=probe_content, created_at=created_at)
        assert hit is not None
        assert hit.canonical_chunk_id == chunk_id
        assert hit.mechanism == mechanism
    assert (
        conn.execute(
            """SELECT dedupe_hash, simhash, simhash_band_0, simhash_band_1, simhash_band_2, simhash_band_3
           FROM chunks WHERE id = ?""",
            (chunk_id,),
        ).fetchone()
        == expected_values
    )
    assert conn.execute("SELECT content, tags, seen_count FROM chunks WHERE id = ?", (chunk_id,)).fetchone() == (
        content,
        '["amber", "zebra"]',
        2,
    )


def test_normalized_exact_hash_preserves_timestamps_but_ignores_stopwords_and_whitespace():
    from brainlayer.dedupe import normalized_exact_hash

    left = "The API was ready\n\nfor the launch"
    right = "api ready launch"
    first_deadline = "Deploy at 2026-05-16T10:00:00Z"
    second_deadline = "Deploy at 2026-05-17T10:00:00Z"

    assert normalized_exact_hash(left) == normalized_exact_hash(right)
    assert normalized_exact_hash(first_deadline) != normalized_exact_hash(second_deadline)


def test_simhash_is_content_only_across_week_buckets():
    from brainlayer.dedupe import hamming_distance, is_near_duplicate, simhash64

    content = "Weekly standup shipped ingest dedupe tests and opened the review loop"

    left = simhash64(content, created_at="2026-05-04T09:00:00Z")
    right = simhash64(content, created_at="2026-05-11T09:00:00Z")

    assert hamming_distance(left, right) == 0
    assert is_near_duplicate(
        content,
        content,
        created_at_a="2026-05-04T09:00:00Z",
        created_at_b="2026-05-11T09:00:00Z",
    )


def test_simhash_numeric_guard_preserves_timestamp_only_changes_in_long_content():
    from brainlayer.dedupe import is_near_duplicate

    left = (
        "Incident deploy checkpoint completed at 2026-05-16T10:00:00Z with rollout validation, "
        "service smoke checks, alert review, owner handoff, recovery notes, and follow up tracking."
    )
    right = (
        "Incident deploy checkpoint completed at 2026-05-17T14:00:00Z with rollout validation, "
        "service smoke checks, alert review, owner handoff, recovery notes, and follow up tracking."
    )

    assert not is_near_duplicate(
        left,
        right,
        created_at_a="2026-05-16T10:00:00Z",
        created_at_b="2026-05-16T10:05:00Z",
    )


def test_identical_reposts_collapse_to_single_chunk_with_seen_count_and_alias(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)

    content = "Identical repost about the FM1 ingest dedupe implementation milestone"
    store.upsert_chunks(
        [
            _chunk("chunk-a", content, created_at="2026-05-16T09:00:00Z", tags=["fm1"], importance=4),
            _chunk("chunk-b", content, created_at="2026-05-16T10:00:00Z", tags=["dedupe"], importance=9),
        ],
        [[0.1] * 1024, [0.2] * 1024],
    )

    row = (
        store.conn.cursor()
        .execute("SELECT id, seen_count, importance, tags, last_seen_at FROM chunks WHERE archived_at IS NULL")
        .fetchone()
    )
    alias = (
        store.conn.cursor()
        .execute("SELECT canonical_chunk_id FROM chunk_id_alias WHERE old_chunk_id = 'chunk-b'")
        .fetchone()
    )
    audit = (
        store.conn.cursor()
        .execute("SELECT chunk_id_dropped, chunk_id_kept, mechanism, hamming_distance FROM dedupe_audit")
        .fetchone()
    )

    assert row[0] == "chunk-a"
    assert row[1] == 2
    assert row[2] == 9
    assert set(json.loads(row[3])) == {"fm1", "dedupe"}
    assert row[4] == "2026-05-16T10:00:00Z"
    assert alias == ("chunk-a",)
    assert audit == ("chunk-b", "chunk-a", "sha256", 0)
    assert store.get_chunk("chunk-b")["id"] == "chunk-a"

    store.close()


def test_same_chunk_id_reposts_increment_seen_count(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)
    content = "Same deterministic chunk id should still count repeated sightings"

    store.upsert_chunks(
        [_chunk("same-id", content, created_at="2026-05-16T09:00:00Z", tags=["first"], importance=4)],
        [[0.1] * 1024],
    )
    store.upsert_chunks(
        [_chunk("same-id", content, created_at="2026-05-16T10:00:00Z", tags=["second"], importance=8)],
        [[0.2] * 1024],
    )

    row = (
        store.conn.cursor()
        .execute("SELECT seen_count, importance, tags, last_seen_at FROM chunks WHERE id = 'same-id'")
        .fetchone()
    )
    audit = (
        store.conn.cursor().execute("SELECT chunk_id_dropped, chunk_id_kept, mechanism FROM dedupe_audit").fetchone()
    )

    assert row == (2, 8.0, '["first", "second"]', "2026-05-16T10:00:00Z")
    assert audit == ("same-id", "same-id", "sha256_same_id")
    store.close()


def test_identical_content_in_different_projects_stays_separate(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)
    content = "Identical scoped content should not disappear from project-filtered recall"

    store.upsert_chunks(
        [
            _chunk("project-a", content, created_at="2026-05-16T09:00:00Z", project="alpha"),
            _chunk("project-b", content, created_at="2026-05-16T10:00:00Z", project="beta"),
        ],
        [[0.1] * 1024, [0.2] * 1024],
    )

    rows = store.conn.cursor().execute("SELECT id, project, seen_count FROM chunks ORDER BY id").fetchall()
    aliases = store.conn.cursor().execute("SELECT COUNT(*) FROM chunk_id_alias").fetchone()[0]

    assert rows == [("project-a", "alpha", 1), ("project-b", "beta", 1)]
    assert aliases == 0
    store.close()


def test_identical_content_with_different_content_types_stays_separate(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)
    content = "Identical typed content should keep separate type-filtered recall"

    store.upsert_chunks(
        [
            _chunk("todo-row", content, created_at="2026-05-16T09:00:00Z", content_type="todo"),
            _chunk("decision-row", content, created_at="2026-05-16T10:00:00Z", content_type="decision"),
        ],
        [[0.1] * 1024, [0.2] * 1024],
    )

    rows = store.conn.cursor().execute("SELECT id, content_type, seen_count FROM chunks ORDER BY id").fetchall()
    aliases = store.conn.cursor().execute("SELECT COUNT(*) FROM chunk_id_alias").fetchone()[0]

    assert rows == [("decision-row", "decision", 1), ("todo-row", "todo", 1)]
    assert aliases == 0
    store.close()


def test_simhash_merge_defers_canonical_vector_until_reembedded(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)
    first_words = [f"token{i}" for i in range(100)]
    second_words = first_words.copy()
    second_words[0] = "changed0"
    first = " ".join(first_words)
    second = " ".join(second_words)

    store.upsert_chunks(
        [
            _chunk("near-a", first, created_at="2026-05-16T09:00:00Z"),
            _chunk("near-b", second, created_at="2026-05-16T09:05:00Z"),
        ],
        [[0.1] * 1024, [0.7] * 1024],
    )

    rows = store.conn.cursor().execute("SELECT chunk_id, embedding FROM chunk_vectors").fetchall()

    assert rows == [], "incoming/previous vectors do not describe canonical merged content"
    assert store.conn.execute("SELECT count(*) FROM chunk_vectors_binary").fetchone() == (0,)
    from brainlayer.store import embed_hot_chunk

    encoded = []

    def encode_canonical(content):
        encoded.append(content)
        return [0.9] * 1024

    assert embed_hot_chunk(store, encode_canonical, "near-a")
    assert encoded == [store.get_chunk("near-a")["content"]]
    assert list(store.conn.execute("SELECT chunk_id, embedding FROM chunk_vectors")) == [
        ("near-a", serialize_f32([0.9] * 1024))
    ]
    assert store.conn.execute("SELECT count(*) FROM chunk_vectors_binary").fetchone() == (1,)
    store.close()


def test_merged_content_list_does_not_double_number_existing_items():
    from brainlayer.dedupe import _merged_content

    first = _merged_content("original milestone content", "second milestone content")
    second = _merged_content(first, "third milestone content")

    assert "1. 1." not in second
    assert "\n\n\u241e 1. original milestone content" in second
    assert "\n\n\u241e 2. second milestone content" in second
    assert "\n\n\u241e 3. third milestone content" in second


def test_weekly_standups_remain_distinct_chunks(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)
    first_words = [f"token{i}" for i in range(100)]
    second_words = first_words.copy()
    second_words[0] = "changed0"

    store.upsert_chunks(
        [
            _chunk(
                "week-1",
                " ".join(first_words),
                created_at="2026-05-04T09:00:00Z",
            ),
            _chunk(
                "week-2",
                " ".join(second_words),
                created_at="2026-05-11T09:00:00Z",
            ),
        ],
        [[0.1] * 1024, [0.2] * 1024],
    )

    count = store.conn.cursor().execute("SELECT COUNT(*) FROM chunks WHERE archived_at IS NULL").fetchone()[0]
    aliases = store.conn.cursor().execute("SELECT COUNT(*) FROM chunk_id_alias").fetchone()[0]

    assert count == 2
    assert aliases == 0

    store.close()


def test_simhash_scope_allows_unknown_week_to_unknown_week(tmp_path):
    from brainlayer.dedupe import compute_dedupe_fields, find_duplicate

    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)
    first_words = [f"token{i}" for i in range(100)]
    second_words = first_words.copy()
    second_words[0] = "changed0"
    candidate_content = " ".join(first_words)
    incoming_content = " ".join(second_words)
    fields = compute_dedupe_fields(candidate_content, None)

    store.conn.cursor().execute(
        """
        INSERT INTO chunks(id, content, metadata, source_file, project, content_type, created_at,
                           dedupe_hash, simhash, simhash_band_0, simhash_band_1, simhash_band_2, simhash_band_3)
        VALUES (?, ?, '{}', 'seed', 'brainlayer', 'note', NULL, ?, ?, ?, ?, ?, ?)
        """,
        (
            "unknown-week",
            candidate_content,
            fields.dedupe_hash,
            fields.simhash,
            *fields.bands,
        ),
    )

    hit, _ = find_duplicate(
        store.conn,
        chunk_id="incoming",
        content=incoming_content,
        created_at=None,
        project="brainlayer",
        content_type="note",
    )

    assert hit is not None
    assert hit.canonical_chunk_id == "unknown-week"
    assert hit.mechanism == "simhash"
    store.close()


def test_simhash_scope_blocks_unknown_week_to_dated_week(tmp_path):
    from brainlayer.dedupe import compute_dedupe_fields, find_duplicate

    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)
    first_words = [f"token{i}" for i in range(100)]
    second_words = first_words.copy()
    second_words[0] = "changed0"
    candidate_content = " ".join(first_words)
    incoming_content = " ".join(second_words)
    fields = compute_dedupe_fields(candidate_content, "2026-05-11T09:00:00Z")

    store.conn.cursor().execute(
        """
        INSERT INTO chunks(id, content, metadata, source_file, project, content_type, created_at,
                           dedupe_hash, simhash, simhash_band_0, simhash_band_1, simhash_band_2, simhash_band_3)
        VALUES (?, ?, '{}', 'seed', 'brainlayer', 'note', ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "dated-week",
            candidate_content,
            "2026-05-11T09:00:00Z",
            fields.dedupe_hash,
            fields.simhash,
            *fields.bands,
        ),
    )

    hit, _ = find_duplicate(
        store.conn,
        chunk_id="incoming",
        content=incoming_content,
        created_at=None,
        project="brainlayer",
        content_type="note",
    )

    assert hit is None
    store.close()


def test_timestamp_only_changes_remain_distinct_chunks(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)

    store.upsert_chunks(
        [
            _chunk("deadline-a", "Deploy at 2026-05-16T10:00:00Z", created_at="2026-05-16T09:00:00Z"),
            _chunk("deadline-b", "Deploy at 2026-05-17T10:00:00Z", created_at="2026-05-16T09:05:00Z"),
        ],
        [[0.1] * 1024, [0.2] * 1024],
    )

    count = store.conn.cursor().execute("SELECT COUNT(*) FROM chunks WHERE archived_at IS NULL").fetchone()[0]
    aliases = store.conn.cursor().execute("SELECT COUNT(*) FROM chunk_id_alias").fetchone()[0]

    assert count == 2
    assert aliases == 0
    store.close()


def test_enrichment_content_hash_does_not_disable_dedupe(tmp_path):
    from brainlayer.enrichment_replay import _content_hash

    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)
    content = "Enrichment can overwrite raw content hash without breaking dedupe"
    store.upsert_chunks([_chunk("hash-a", content, created_at="2026-05-16T09:00:00Z")], [[0.1] * 1024])
    store.conn.cursor().execute(
        "UPDATE chunks SET content_hash = ? WHERE id = ?",
        (_content_hash(content), "hash-a"),
    )

    store.upsert_chunks([_chunk("hash-b", content, created_at="2026-05-16T10:00:00Z")], [[0.2] * 1024])

    row = store.conn.cursor().execute("SELECT id, seen_count FROM chunks WHERE archived_at IS NULL").fetchone()
    alias = (
        store.conn.cursor()
        .execute("SELECT canonical_chunk_id FROM chunk_id_alias WHERE old_chunk_id = 'hash-b'")
        .fetchone()
    )

    assert row == ("hash-a", 2)
    assert alias == ("hash-a",)
    store.close()


def test_upsert_archives_existing_row_before_aliasing_same_id_to_canonical(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)
    canonical = "Canonical content that a deterministic id later duplicates"
    stale = "Previously active stale content for the deterministic id"

    store.upsert_chunks(
        [
            _chunk("canonical-row", canonical, created_at="2026-05-16T09:00:00Z"),
            _chunk("deterministic-row", stale, created_at="2026-05-16T09:01:00Z"),
        ],
        [[0.1] * 1024, [0.8] * 1024],
    )
    store.upsert_chunks(
        [_chunk("deterministic-row", canonical, created_at="2026-05-16T10:00:00Z")],
        [[0.2] * 1024],
    )

    active_rows = store.conn.cursor().execute("SELECT id FROM chunks WHERE archived_at IS NULL ORDER BY id").fetchall()
    archived_row = (
        store.conn.cursor()
        .execute("SELECT archived_at IS NOT NULL, superseded_by FROM chunks WHERE id = 'deterministic-row'")
        .fetchone()
    )
    alias = (
        store.conn.cursor()
        .execute("SELECT canonical_chunk_id FROM chunk_id_alias WHERE old_chunk_id = 'deterministic-row'")
        .fetchone()
    )

    assert active_rows == [("canonical-row",)]
    assert archived_row == (1, "canonical-row")
    assert alias == ("canonical-row",)
    assert store.get_chunk("deterministic-row")["id"] == "canonical-row"
    store.close()


def test_archiving_canonical_duplicate_retargets_existing_aliases(tmp_path):
    from brainlayer.dedupe import compute_dedupe_fields, merge_duplicate_chunk, write_alias

    store = VectorStore(tmp_path / "alias-chain.db")
    cursor = store.conn.cursor()
    first_fields = compute_dedupe_fields("First canonical duplicate chain memory", "2026-05-16T09:00:00Z")
    second_fields = compute_dedupe_fields("Second canonical duplicate chain memory", "2026-05-16T10:00:00Z")
    cursor.execute(
        """
        INSERT INTO chunks(id, content, metadata, source_file, project, content_type, created_at,
                           dedupe_hash, simhash, simhash_band_0, simhash_band_1, simhash_band_2, simhash_band_3)
        VALUES (?, ?, '{}', 'seed', 'brainlayer', 'note', ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "chain-a",
            "First canonical duplicate chain memory",
            "2026-05-16T09:00:00Z",
            first_fields.dedupe_hash,
            first_fields.simhash,
            *first_fields.bands,
        ),
    )
    cursor.execute(
        """
        INSERT INTO chunks(id, content, metadata, source_file, project, content_type, created_at,
                           dedupe_hash, simhash, simhash_band_0, simhash_band_1, simhash_band_2, simhash_band_3)
        VALUES (?, ?, '{}', 'seed', 'brainlayer', 'note', ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "chain-c",
            "Second canonical duplicate chain memory",
            "2026-05-16T10:00:00Z",
            second_fields.dedupe_hash,
            second_fields.simhash,
            *second_fields.bands,
        ),
    )
    write_alias(store.conn, old_chunk_id="chain-b", canonical_chunk_id="chain-a")

    merge_duplicate_chunk(
        store.conn,
        canonical_id="chain-c",
        duplicate_id="chain-a",
        incoming={
            "id": "chain-a",
            "content": "First canonical duplicate chain memory",
            "tags": None,
            "created_at": "2026-05-16T09:00:00Z",
        },
        mechanism="simhash",
        hamming_distance_value=2,
        archive_existing_duplicate=True,
    )

    aliases = dict(cursor.execute("SELECT old_chunk_id, canonical_chunk_id FROM chunk_id_alias ORDER BY old_chunk_id"))
    assert aliases == {"chain-a": "chain-c", "chain-b": "chain-c"}
    assert store.resolve_chunk_id("chain-b") == "chain-c"
    store.close()


def test_backfill_merges_snapshot_duplicates_and_preserves_alias_refs(tmp_path):
    from brainlayer.dedupe import backfill_dedupe_database

    live_path = tmp_path / "live.db"
    snapshot_path = tmp_path / "snapshot.db"
    store = VectorStore(live_path)
    content = "Backfill duplicate milestone should collapse through the snapshot only"
    cursor = store.conn.cursor()
    cursor.execute(
        """
        INSERT INTO chunks(id, content, metadata, source_file, tags, importance, created_at, archived)
        VALUES (?, ?, '{}', 'seed', ?, ?, ?, 0)
        """,
        ("old-a", content, '["old"]', 3, "2026-05-16T09:00:00Z"),
    )
    cursor.execute(
        """
        INSERT INTO chunks(id, content, metadata, source_file, tags, importance, created_at, archived)
        VALUES (?, ?, '{}', 'seed', ?, ?, ?, 0)
        """,
        ("old-b", content, '["new"]', 8, "2026-05-16T11:00:00Z"),
    )
    cursor.execute(
        "INSERT INTO kg_entities(id, name, entity_type) VALUES (?, ?, ?)", ("entity-1", "Entity One", "project")
    )
    cursor.execute(
        """
        INSERT INTO kg_entity_chunks(entity_id, chunk_id, relevance, context)
        VALUES (?, ?, ?, ?)
        """,
        ("entity-1", "old-b", 0.9, "duplicate link"),
    )
    store.close()
    shutil.copy2(live_path, snapshot_path)

    result = backfill_dedupe_database(snapshot_path, batch_size=1)

    assert result.merged == 1

    checked = VectorStore(snapshot_path)
    active_rows = (
        checked.conn.cursor()
        .execute("SELECT id, seen_count, importance, tags FROM chunks WHERE archived_at IS NULL")
        .fetchall()
    )
    alias = (
        checked.conn.cursor()
        .execute("SELECT canonical_chunk_id FROM chunk_id_alias WHERE old_chunk_id = 'old-b'")
        .fetchone()
    )
    entity_link = (
        checked.conn.cursor().execute("SELECT chunk_id FROM kg_entity_chunks WHERE entity_id = 'entity-1'").fetchone()
    )
    entity_link_count = checked.conn.cursor().execute("SELECT COUNT(*) FROM kg_entity_chunks").fetchone()[0]

    assert active_rows == [("old-a", 2, 8.0, '["new", "old"]')]
    assert alias == ("old-a",)
    assert entity_link == ("old-a",)
    assert entity_link_count == 1
    assert checked.get_chunk("old-b")["id"] == "old-a"
    checked.close()


def test_backfill_refuses_default_live_db_without_explicit_allow(monkeypatch, tmp_path):
    from brainlayer import dedupe
    from brainlayer.dedupe import backfill_dedupe_database

    live_path = tmp_path / "brainlayer.db"
    VectorStore(live_path).close()
    monkeypatch.setattr(dedupe, "get_db_path", lambda: live_path)

    try:
        backfill_dedupe_database(live_path)
    except ValueError as exc:
        assert "snapshot" in str(exc)
    else:
        raise AssertionError("live default DB backfill should require allow_live=True")


def test_resimhash_recomputes_stored_simhashes_without_merging(tmp_path):
    from brainlayer.dedupe import compute_dedupe_fields, resimhash_dedupe_database

    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)
    content = "Resimhash migration should recompute stale simhash fields only"
    expected_fields = compute_dedupe_fields(content, "2026-05-11T09:00:00Z")
    cursor = store.conn.cursor()
    cursor.execute(
        """
        INSERT INTO chunks(id, content, metadata, source_file, created_at,
                           dedupe_hash, simhash, simhash_band_0, simhash_band_1, simhash_band_2, simhash_band_3)
        VALUES (?, ?, '{}', 'seed', ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "stale-row",
            content,
            "2026-05-11T09:00:00Z",
            expected_fields.dedupe_hash,
            "0000000000000000",
            "0000",
            "0000",
            "0000",
            "0000",
        ),
    )
    store.close()

    result = resimhash_dedupe_database(db_path, batch_size=1, checkpoint_every=1)

    checked = VectorStore(db_path)
    row = (
        checked.conn.cursor()
        .execute(
            """
            SELECT simhash, simhash_band_0, simhash_band_1, simhash_band_2, simhash_band_3
            FROM chunks
            WHERE id = 'stale-row'
            """
        )
        .fetchone()
    )
    aliases = checked.conn.cursor().execute("SELECT COUNT(*) FROM chunk_id_alias").fetchone()[0]

    assert result.scanned == 1
    assert result.updated == 1
    assert row == (expected_fields.simhash, *expected_fields.bands)
    assert aliases == 0
    checked.close()


def test_resimhash_resumes_after_recorded_last_id(tmp_path):
    from brainlayer.dedupe import compute_dedupe_fields, resimhash_dedupe_database

    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)
    cursor = store.conn.cursor()
    new_fields = compute_dedupe_fields("second migration resume row", "2026-05-11T09:00:00Z")
    stale_simhash = "0000000000000000"
    stale_bands = ("0000", "0000", "0000", "0000")
    for chunk_id, content in [
        ("a-first", "first migration resume row"),
        ("b-second", "second migration resume row"),
    ]:
        fields = compute_dedupe_fields(content, "2026-05-11T09:00:00Z")
        cursor.execute(
            """
            INSERT INTO chunks(id, content, metadata, source_file, created_at,
                               dedupe_hash, simhash, simhash_band_0, simhash_band_1, simhash_band_2, simhash_band_3)
            VALUES (?, ?, '{}', 'seed', '2026-05-11T09:00:00Z', ?, ?, ?, ?, ?, ?)
            """,
            (chunk_id, content, fields.dedupe_hash, stale_simhash, *stale_bands),
        )
    cursor.execute(
        """
        CREATE TABLE dedupe_resimhash_progress (
            migration_name TEXT PRIMARY KEY,
            last_id TEXT,
            scanned INTEGER NOT NULL DEFAULT 0,
            updated INTEGER NOT NULL DEFAULT 0,
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT
        )
        """
    )
    cursor.execute(
        """
        INSERT INTO dedupe_resimhash_progress(migration_name, last_id, scanned, updated, started_at, updated_at)
        VALUES ('test-resume', 'a-first', 1, 1, '2026-06-11T00:00:00Z', '2026-06-11T00:00:00Z')
        """
    )
    store.close()

    result = resimhash_dedupe_database(db_path, batch_size=1, checkpoint_every=1, migration_name="test-resume")

    checked = VectorStore(db_path)
    rows = dict(
        checked.conn.cursor().execute("SELECT id, simhash FROM chunks WHERE id IN ('a-first', 'b-second') ORDER BY id")
    )
    assert result.scanned == 2
    assert rows == {"a-first": stale_simhash, "b-second": new_fields.simhash}
    checked.close()


def test_alias_resolution_expires_after_grace_period(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)
    store.upsert_chunks(
        [_chunk("canonical", "Canonical alias target", created_at="2026-05-16T09:00:00Z")],
        [[0.1] * 1024],
    )
    store.conn.cursor().execute(
        """
        INSERT INTO chunk_id_alias(old_chunk_id, canonical_chunk_id, deprecated_at)
        VALUES (?, ?, datetime('now', '-91 days'))
        """,
        ("old-expired", "canonical"),
    )

    assert store.resolve_chunk_id("old-expired") == "old-expired"
    assert store.get_chunk("old-expired") is None

    store.close()


def test_alias_resolution_falls_back_when_alias_table_missing_on_readonly_legacy_db(tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                content TEXT,
                metadata TEXT,
                source_file TEXT,
                project TEXT,
                content_type TEXT,
                value_type TEXT,
                tags TEXT,
                importance REAL,
                created_at TEXT,
                summary TEXT,
                superseded_by TEXT,
                aggregated_into TEXT,
                archived_at TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO chunks(id, content, metadata, source_file, project, content_type, value_type)
            VALUES (?, ?, '{}', 'legacy', 'brainlayer', 'note', 'HIGH')
            """,
            (
                "legacy-id",
                "legacy chunk",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    os.chmod(db_path, 0o444)
    store = None
    try:
        store = VectorStore(db_path)
        chunk = store.get_chunk("legacy-id")
    finally:
        if store is not None:
            store.close()
        os.chmod(db_path, 0o644)

    assert chunk["id"] == "legacy-id"


def test_dedupe_schema_exists_on_minimal_sqlite_connection(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    VectorStore(db_path).close()

    conn = sqlite3.connect(db_path)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()

    assert {"seen_count", "last_seen_at", "content_hash", "dedupe_hash", "simhash"}.issubset(cols)
    assert {"dedupe_audit", "chunk_id_alias"}.issubset(tables)
