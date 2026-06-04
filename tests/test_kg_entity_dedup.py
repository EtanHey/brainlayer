import importlib.util
import sqlite3
from pathlib import Path

import apsw
import pytest


def _load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "kg_entity_dedup.py"
    spec = importlib.util.spec_from_file_location("kg_entity_dedup_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _connect_fixture(tmp_path):
    conn = sqlite3.connect(tmp_path / "kg.db")
    conn.executescript(
        """
        CREATE TABLE kg_entities (
            id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            metadata TEXT DEFAULT '{}',
            canonical_name TEXT,
            status TEXT DEFAULT 'active',
            created_at TEXT,
            updated_at TEXT,
            expired_at TEXT,
            UNIQUE(entity_type, name)
        );
        CREATE TABLE kg_relations (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            properties TEXT DEFAULT '{}',
            confidence REAL DEFAULT 1.0,
            user_verified INTEGER DEFAULT 0,
            fact TEXT,
            importance REAL DEFAULT 0.5,
            valid_from TEXT,
            valid_until TEXT,
            expired_at TEXT,
            source_chunk_id TEXT,
            UNIQUE(source_id, target_id, relation_type)
        );
        CREATE TABLE kg_entity_chunks (
            entity_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            relevance REAL DEFAULT 1.0,
            context TEXT,
            mention_type TEXT,
            relation_tier INTEGER DEFAULT 4,
            weight REAL DEFAULT 0.25,
            PRIMARY KEY (entity_id, chunk_id)
        );
        CREATE TABLE kg_entity_aliases (
            alias TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            alias_type TEXT DEFAULT 'name',
            created_at TEXT,
            valid_from TEXT,
            valid_to TEXT,
            PRIMARY KEY (alias, entity_id)
        );
        CREATE TABLE kg_vec_entities (
            entity_id TEXT PRIMARY KEY,
            embedding BLOB
        );
        """
    )
    return conn


def _connect_apsw_fixture(tmp_path):
    conn = apsw.Connection(str(tmp_path / "kg-apsw.db"))
    conn.execute(
        """
        CREATE TABLE kg_entities (
            id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            metadata TEXT DEFAULT '{}',
            canonical_name TEXT,
            status TEXT DEFAULT 'active',
            created_at TEXT,
            updated_at TEXT,
            expired_at TEXT,
            UNIQUE(entity_type, name)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE kg_entity_chunks (
            entity_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            relevance REAL DEFAULT 1.0,
            context TEXT,
            mention_type TEXT,
            relation_tier INTEGER DEFAULT 4,
            weight REAL DEFAULT 0.25,
            PRIMARY KEY (entity_id, chunk_id)
        )
        """
    )
    return conn


def _entity(conn, entity_id, entity_type, name):
    conn.execute(
        "INSERT INTO kg_entities (id, entity_type, name, canonical_name) VALUES (?, ?, ?, ?)",
        (entity_id, entity_type, name, name.casefold()),
    )


def _scalar(conn, sql, params=()):
    return conn.execute(sql, params).fetchone()[0]


def test_approved_merge_repoints_relations_and_chunks(tmp_path):
    dedup = _load_script()
    conn = _connect_fixture(tmp_path)
    _entity(conn, "person-etan", "person", "Etan Heyman")
    _entity(conn, "person-etan-fragment", "person", "Etan H.")
    _entity(conn, "project-brainlayer", "project", "BrainLayer")
    conn.execute(
        """
        INSERT INTO kg_relations (id, source_id, target_id, relation_type, fact)
        VALUES ('rel-source', 'person-etan-fragment', 'project-brainlayer', 'maintains', 'fragment maintains project')
        """
    )
    conn.execute(
        """
        INSERT INTO kg_relations (id, source_id, target_id, relation_type, fact)
        VALUES ('rel-target', 'project-brainlayer', 'person-etan-fragment', 'mentions', 'project mentions fragment')
        """
    )
    conn.execute(
        """
        INSERT INTO kg_entity_chunks (entity_id, chunk_id, relevance, context, mention_type)
        VALUES ('person-etan-fragment', 'chunk-a', 0.8, 'fragment context', 'explicit')
        """
    )

    stats = dedup.apply_approved_mapping(
        conn,
        [
            {
                "label": "Etan canonical",
                "canonical": {"id": "person-etan", "entity_type": "person", "name": "Etan Heyman"},
                "sources": ["person-etan-fragment"],
                "aliases": ["Etan"],
            }
        ],
    )

    assert stats["merge_groups"] == 1
    assert stats["merge_entities_deleted"] == 1
    assert conn.execute("SELECT name FROM kg_entities WHERE id = 'person-etan'").fetchone()[0] == "Etan Heyman"
    assert conn.execute("SELECT id FROM kg_entities WHERE id = 'person-etan-fragment'").fetchone() is None
    assert _scalar(conn, "SELECT count(*) FROM kg_entity_chunks WHERE entity_id = 'person-etan-fragment'") == 0
    assert _scalar(conn, "SELECT count(*) FROM kg_entity_chunks WHERE entity_id = 'person-etan'") == 1
    assert conn.execute("SELECT source_id FROM kg_relations WHERE id = 'rel-source'").fetchone()[0] == "person-etan"
    assert conn.execute("SELECT target_id FROM kg_relations WHERE id = 'rel-target'").fetchone()[0] == "person-etan"
    aliases = {row[0] for row in conn.execute("SELECT alias FROM kg_entity_aliases WHERE entity_id = 'person-etan'")}
    assert {"Etan", "Etan H."}.issubset(aliases)


def test_mapping_driven_merge_does_not_fuzzy_merge_david_heyman(tmp_path):
    dedup = _load_script()
    conn = _connect_fixture(tmp_path)
    _entity(conn, "person-etan", "person", "Etan Heyman")
    _entity(conn, "person-etan-fragment", "person", "Etan")
    _entity(conn, "person-david", "person", "David Heyman")
    conn.execute(
        """
        INSERT INTO kg_entity_chunks (entity_id, chunk_id, relevance, context)
        VALUES ('person-david', 'family-chunk', 0.9, 'David context')
        """
    )
    conn.execute(
        """
        INSERT INTO kg_entity_chunks (entity_id, chunk_id, relevance, context)
        VALUES ('person-etan-fragment', 'family-chunk', 0.7, 'Etan context')
        """
    )

    dedup.apply_approved_mapping(
        conn,
        [
            {
                "label": "Etan only",
                "canonical": {"id": "person-etan", "entity_type": "person", "name": "Etan Heyman"},
                "sources": ["person-etan-fragment"],
            }
        ],
    )

    assert conn.execute("SELECT name FROM kg_entities WHERE id = 'person-david'").fetchone()[0] == "David Heyman"
    assert _scalar(conn, "SELECT count(*) FROM kg_entity_chunks WHERE entity_id = 'person-david'") == 1


def test_reviewed_name_mapping_merges_exact_etan_fragments_without_company_or_david(tmp_path):
    dedup = _load_script()
    conn = _connect_fixture(tmp_path)
    _entity(conn, "person-etan", "person", "Etan Heyman")
    _entity(conn, "person-etan-short", "person", "Etan")
    _entity(conn, "person-eitan", "person", "Eitan")
    _entity(conn, "person-typo", "person", "Eitan Heysman")
    _entity(conn, "person-david", "person", "David Heyman")
    _entity(conn, "company-eitan", "company", "Eitan Heyman Development")
    conn.execute(
        """
        INSERT INTO kg_entity_chunks (entity_id, chunk_id, relevance, context)
        VALUES ('person-eitan', 'chunk-person', 0.9, 'Eitan context')
        """
    )
    conn.execute(
        """
        INSERT INTO kg_relations (id, source_id, target_id, relation_type)
        VALUES ('rel-person', 'person-typo', 'company-eitan', 'mentions')
        """
    )

    stats = dedup.apply_reviewed_name_mapping(
        conn,
        [
            {
                "label": "Etan exact reviewed fragments",
                "canonical": {"entity_type": "person", "name": "Etan Heyman"},
                "source_names": [
                    {"entity_type": "person", "name": "Etan"},
                    {"entity_type": "person", "name": "Eitan"},
                    {"entity_type": "person", "name": "Eitan Heysman"},
                ],
                "aliases": ["EHeyman", "ETAN HEYMAN"],
            }
        ],
    )

    assert stats["name_merge_groups"] == 1
    assert stats["merge_sources"] == 3
    assert (
        _scalar(
            conn, "SELECT count(*) FROM kg_entities WHERE id IN ('person-etan-short', 'person-eitan', 'person-typo')"
        )
        == 0
    )
    assert conn.execute("SELECT name FROM kg_entities WHERE id = 'person-david'").fetchone()[0] == "David Heyman"
    assert (
        conn.execute("SELECT name FROM kg_entities WHERE id = 'company-eitan'").fetchone()[0]
        == "Eitan Heyman Development"
    )
    assert conn.execute("SELECT source_id FROM kg_relations WHERE id = 'rel-person'").fetchone()[0] == "person-etan"
    aliases = {row[0] for row in conn.execute("SELECT alias FROM kg_entity_aliases WHERE entity_id = 'person-etan'")}
    assert {"Etan", "Eitan", "Eitan Heysman", "EHeyman", "ETAN HEYMAN"}.issubset(aliases)


def test_reviewed_name_mapping_rejects_david_heyman_even_if_requested(tmp_path):
    dedup = _load_script()
    conn = _connect_fixture(tmp_path)
    _entity(conn, "person-etan", "person", "Etan Heyman")
    _entity(conn, "person-david", "person", "David Heyman")

    with pytest.raises(RuntimeError, match="blocked never-merge name"):
        dedup.apply_reviewed_name_mapping(
            conn,
            [
                {
                    "label": "bad person merge",
                    "canonical": {"entity_type": "person", "name": "Etan Heyman"},
                    "source_names": [{"entity_type": "person", "name": "David Heyman"}],
                }
            ],
        )


def test_suggest_lists_candidates_without_applying_them(tmp_path):
    dedup = _load_script()
    conn = _connect_fixture(tmp_path)
    _entity(conn, "project-brainlayer", "project", "BrainLayer")
    _entity(conn, "project-brain-layer", "project", "brain layer")

    suggestions = dedup.suggest_candidate_clusters(conn)

    normalized_clusters = [cluster for cluster in suggestions if cluster["reason"] == "normalized-name"]
    assert normalized_clusters
    assert {row["id"] for row in normalized_clusters[0]["entities"]} == {"project-brainlayer", "project-brain-layer"}
    assert _scalar(conn, "SELECT count(*) FROM kg_entities") == 2


def test_suggest_lists_cross_type_domain_fragments_without_applying_them(tmp_path):
    dedup = _load_script()
    conn = _connect_fixture(tmp_path)
    _entity(conn, "domain-project", "project", "brainlayer.etanheyman.com")
    _entity(conn, "domain-tool", "tool", "brainlayer.etanheyman.com")
    _entity(conn, "domain-concept", "concept", "brainlayer.etanheyman.com")

    suggestions = dedup.suggest_candidate_clusters(conn)

    cross_type = [cluster for cluster in suggestions if cluster["reason"] == "normalized-name-cross-type"]
    assert cross_type
    assert {row["id"] for row in cross_type[0]["entities"]} == {"domain-project", "domain-tool", "domain-concept"}
    assert _scalar(conn, "SELECT count(*) FROM kg_entities") == 3


def test_apsw_fetch_dicts_handles_completed_zero_row_selects(tmp_path):
    dedup = _load_script()
    conn = _connect_apsw_fixture(tmp_path)

    assert dedup.fetch_dicts(conn, "SELECT id, name FROM kg_entities WHERE name = ?", ("missing",)) == []
    assert dedup.suggest_candidate_clusters(conn) == []


def test_cursor_description_only_swallows_apsw_completed_execution():
    dedup = _load_script()

    class ExecutionCompleteError(Exception):
        pass

    class FakeCursor:
        @property
        def description(self):
            raise ExecutionCompleteError("not APSW")

    with pytest.raises(ExecutionCompleteError, match="not APSW"):
        dedup._cursor_description(FakeCursor())


def test_path_purge_deletes_only_path_shaped_entities(tmp_path):
    dedup = _load_script()
    conn = _connect_fixture(tmp_path)
    _entity(conn, "path-abs", "topic", "/Users/etanheyman/Gits/brainlayer/src/brainlayer/store.py")
    _entity(conn, "path-rel", "topic", "src/brainlayer/vector_store.py")
    _entity(conn, "nodejs", "technology", "Node.js")
    _entity(conn, "acdc", "concept", "AC/DC")
    conn.execute(
        "INSERT INTO kg_entity_chunks (entity_id, chunk_id, context) VALUES ('path-abs', 'chunk-path', 'path context')"
    )
    conn.execute(
        """
        INSERT INTO kg_relations (id, source_id, target_id, relation_type)
        VALUES ('rel-path', 'path-abs', 'nodejs', 'mentions')
        """
    )
    conn.execute(
        """
        INSERT INTO kg_relations (id, source_id, target_id, relation_type)
        VALUES ('rel-normal', 'nodejs', 'acdc', 'related_to')
        """
    )
    conn.execute("INSERT INTO kg_entity_aliases (alias, entity_id, alias_type) VALUES ('store.py', 'path-abs', 'path')")

    candidates = dedup.find_path_pseudo_entities(conn)
    assert {row["id"] for row in candidates} == {"path-abs", "path-rel"}

    stats = dedup.purge_entities(conn, candidates)

    assert stats["purged_entities"] == 2
    assert conn.execute("SELECT id FROM kg_entities WHERE id = 'path-abs'").fetchone() is None
    assert conn.execute("SELECT id FROM kg_entities WHERE id = 'path-rel'").fetchone() is None
    assert conn.execute("SELECT id FROM kg_entities WHERE id = 'nodejs'").fetchone()[0] == "nodejs"
    assert conn.execute("SELECT id FROM kg_entities WHERE id = 'acdc'").fetchone()[0] == "acdc"
    assert _scalar(conn, "SELECT count(*) FROM kg_relations WHERE id = 'rel-path'") == 0
    assert _scalar(conn, "SELECT count(*) FROM kg_relations WHERE id = 'rel-normal'") == 1
    assert _scalar(conn, "SELECT count(*) FROM kg_entity_chunks WHERE entity_id LIKE 'path-%'") == 0
    assert _scalar(conn, "SELECT count(*) FROM kg_entity_aliases WHERE entity_id LIKE 'path-%'") == 0
