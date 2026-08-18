import re
import sqlite3
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _init_chunks(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT,
            summary TEXT,
            tags TEXT,
            resolved_query TEXT,
            key_facts TEXT,
            resolved_queries TEXT,
            created_at TEXT,
            project TEXT,
            content_type TEXT
        );
        CREATE TABLE chunk_fts_rowids (
            chunk_id TEXT PRIMARY KEY,
            fts_rowid INTEGER,
            trigram_rowid INTEGER
        );
        """
    )


def test_guard_refuses_live_path_without_explicit_override(tmp_path, monkeypatch):
    from brainlayer import db_shrink

    live_path = tmp_path / "brainlayer.db"
    live_path.touch()
    monkeypatch.setattr(db_shrink, "get_db_path", lambda: live_path)

    with pytest.raises(ValueError, match="Refusing to write to the canonical live DB"):
        db_shrink.assert_not_live_db(live_path)

    db_shrink.assert_not_live_db(live_path, allow_live=True)


def test_migrate_fts_single_trigram_drops_redundant_table(tmp_path):
    from brainlayer.db_shrink import migrate_fts_single_trigram

    db_path = tmp_path / "fts.db"
    conn = sqlite3.connect(db_path)
    _init_chunks(conn)
    conn.executescript(
        """
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id UNINDEXED,
            prefix='2 3 4', tokenize='unicode61 remove_diacritics 2'
        );
        CREATE VIRTUAL TABLE chunks_fts_trigram USING fts5(
            content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id UNINDEXED,
            tokenize='trigram'
        );
        """
    )
    conn.execute(
        """
        INSERT INTO chunks(id, content, summary, tags, resolved_query, key_facts, resolved_queries, created_at)
        VALUES ('c1', 'searchable abcdef memory', '', '', '', '', '', '2026-01-01')
        """
    )
    conn.execute(
        """
        INSERT INTO chunks_fts(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
        SELECT content, summary, tags, resolved_query, key_facts, resolved_queries, id FROM chunks
        """
    )
    conn.execute(
        """
        INSERT INTO chunks_fts_trigram(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
        SELECT content, summary, tags, resolved_query, key_facts, resolved_queries, id FROM chunks
        """
    )
    conn.commit()
    conn.close()

    result = migrate_fts_single_trigram(db_path)

    checked = sqlite3.connect(db_path)
    schema = checked.execute("SELECT sql FROM sqlite_master WHERE name = 'chunks_fts'").fetchone()[0]
    trigram_table = checked.execute("SELECT 1 FROM sqlite_master WHERE name = 'chunks_fts_trigram'").fetchone()
    fts_count = checked.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    meta_mode = checked.execute("SELECT value FROM brainlayer_meta WHERE key = 'fts_mode'").fetchone()[0]
    checked.close()

    assert result.chunk_count == 1
    assert result.fts_count == 1
    assert "tokenize='trigram'" in schema
    assert trigram_table is None
    assert fts_count == 1
    assert meta_mode == "single_trigram"


def test_migrate_fts_compact_dual_preserves_trigram_table(tmp_path):
    from brainlayer.db_shrink import migrate_fts_compact_dual

    db_path = tmp_path / "fts-dual.db"
    conn = sqlite3.connect(db_path)
    _init_chunks(conn)
    conn.executescript(
        """
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id UNINDEXED,
            prefix='2 3 4', tokenize='unicode61 remove_diacritics 2'
        );
        CREATE VIRTUAL TABLE chunks_fts_trigram USING fts5(
            content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id UNINDEXED,
            tokenize='trigram'
        );
        """
    )
    conn.execute(
        """
        INSERT INTO chunks(id, content, summary, tags, resolved_query, key_facts, resolved_queries, created_at)
        VALUES ('c1', 'searchable abcdef memory', '', '', '', '', '', '2026-01-01')
        """
    )
    conn.commit()
    conn.close()

    result = migrate_fts_compact_dual(db_path)

    checked = sqlite3.connect(db_path)
    schema = checked.execute("SELECT sql FROM sqlite_master WHERE name = 'chunks_fts'").fetchone()[0]
    trigram_schema = checked.execute("SELECT sql FROM sqlite_master WHERE name = 'chunks_fts_trigram'").fetchone()[0]
    counts = checked.execute(
        "SELECT (SELECT COUNT(*) FROM chunks_fts), (SELECT COUNT(*) FROM chunks_fts_trigram)"
    ).fetchone()
    checked.close()

    assert result.mode == "compact_dual"
    assert "prefix=" not in schema
    assert "tokenize='trigram'" in trigram_schema
    assert counts == (1, 1)


# --- repair (e): the physical-delete dedupe path must stay gone ---------------
# db_shrink once carried apply_content_dedup -> _merge_duplicate_references ->
# `DELETE FROM chunks`, stamped mechanism='normalized_content_physical_delete'
# and keyed on the lossy normalized_exact_hash. It contradicted the lifecycle law
# (duplicates archive with aggregated_into lineage, never delete) and had never
# run on the canonical DB. Merging now lives in brainlayer.dedupe_merge, which
# only writes lifecycle columns.

REMOVED_DEDUP_SYMBOLS = (
    "apply_content_dedup",
    "analyze_content_duplicates",
    "_merge_duplicate_references",
    "_record_alias",
    "_delete_by_chunk_id",
    "_delete_fts_rows_for_chunk",
    "_insert_or_ignore_repoint",
    "_bulk_repoint_direct_refs",
    "_bulk_repoint_chunk_self_refs",
    "load_protected_qrel_ids",
    "DedupResult",
)


def test_physical_delete_dedup_path_is_gone():
    from brainlayer import db_shrink

    present = [name for name in REMOVED_DEDUP_SYMBOLS if hasattr(db_shrink, name)]
    assert present == [], f"physical-delete dedupe path reintroduced: {present}"


def test_db_shrink_never_deletes_chunk_rows():
    source = (REPO_ROOT / "src/brainlayer/db_shrink.py").read_text(encoding="utf-8")
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
    body = code.split('"""', 2)[-1] if code.count('"""') >= 2 else code
    offenders = re.findall(r"DELETE\s+FROM\s+chunks\b(?!_fts)", body, re.I)
    assert offenders == [], f"db_shrink must not delete chunk rows: {offenders}"


def test_no_physical_delete_mechanism_anywhere_in_production_code():
    """No writer may stamp a physical-delete dedupe mechanism.

    Scans real string VALUES via the AST, not raw text: a docstring explaining
    why the path was removed is documentation, while a literal carrying the
    mechanism name is a writer about to stamp it.
    """
    import ast

    offenders = []
    for root in (REPO_ROOT / "src/brainlayer", REPO_ROOT / "scripts", REPO_ROOT / "hooks"):
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            docstrings = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    doc = node.body[0] if node.body else None
                    if isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant):
                        docstrings.add(id(doc.value))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in docstrings
                    and "physical_delete" in node.value
                ):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert offenders == [], f"physical-delete dedupe mechanism present: {offenders}"
