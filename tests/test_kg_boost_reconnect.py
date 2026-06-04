import sqlite3

import apsw


def test_kg_boost_feature_flag_defaults_off(monkeypatch):
    from brainlayer.mcp.search_handler import _kg_boost_enabled

    monkeypatch.delenv("BRAINLAYER_KG_BOOST", raising=False)
    assert _kg_boost_enabled() is False

    monkeypatch.setenv("BRAINLAYER_KG_BOOST", "1")
    assert _kg_boost_enabled() is True


def _execute(conn, sql, params=()):
    return conn.cursor().execute(sql, params)


def _create_kg_fixture(conn):
    statements = [
        """
        CREATE TABLE kg_entities (
            id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            metadata TEXT DEFAULT '{}',
            created_at TEXT,
            updated_at TEXT,
            UNIQUE(entity_type, name)
        )
        """,
        """
        CREATE VIRTUAL TABLE kg_entities_fts USING fts5(
            name, metadata, entity_id UNINDEXED
        )
        """,
        """
        CREATE TABLE kg_entity_aliases (
            alias TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            alias_type TEXT DEFAULT 'name',
            created_at TEXT,
            PRIMARY KEY (alias, entity_id)
        )
        """,
        """
        CREATE TABLE kg_entity_chunks (
            entity_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            relevance REAL DEFAULT 1.0,
            PRIMARY KEY (entity_id, chunk_id)
        )
        """,
    ]
    for statement in statements:
        _execute(conn, statement)
    return conn


def _sqlite_fixture(tmp_path):
    return _create_kg_fixture(sqlite3.connect(tmp_path / "kg-boost.db"))


def _apsw_fixture(tmp_path):
    return _create_kg_fixture(apsw.Connection(str(tmp_path / "kg-boost-apsw.db")))


def _entity(conn, entity_id, entity_type, name):
    _execute(
        conn,
        "INSERT INTO kg_entities (id, entity_type, name, metadata) VALUES (?, ?, ?, '{}')",
        (entity_id, entity_type, name),
    )
    _execute(
        conn,
        "INSERT INTO kg_entities_fts (name, metadata, entity_id) VALUES (?, '{}', ?)",
        (name, entity_id),
    )


def _alias(conn, alias, entity_id, alias_type="name"):
    _execute(
        conn,
        "INSERT INTO kg_entity_aliases (alias, entity_id, alias_type) VALUES (?, ?, ?)",
        (alias, entity_id, alias_type),
    )


class _Store:
    def __init__(self, conn):
        self.conn = conn

    def _read_cursor(self):
        return self.conn.cursor()


def _assert_fts_entity_lookup_resolves_deduped_canonical(conn):
    from brainlayer import search_repo

    _entity(conn, "person-etan", "person", "Etan Heyman")
    _alias(conn, "ETAN HEYMAN", "person-etan")
    _alias(conn, "etanheyman", "person-etan", alias_type="handle")

    store = _Store(conn)

    spaced_matches = search_repo._kg_boost_entity_matches(store, "What did ETAN HEYMAN decide?", limit=10)
    assert spaced_matches == [{"id": "person-etan", "name": "Etan Heyman", "entity_type": "person"}]

    handle_matches = search_repo._kg_boost_entity_matches(store, "etanheyman search relevance", limit=10)
    assert handle_matches == [{"id": "person-etan", "name": "Etan Heyman", "entity_type": "person"}]


def test_fts_entity_lookup_resolves_deduped_canonical_sqlite(tmp_path):
    _assert_fts_entity_lookup_resolves_deduped_canonical(_sqlite_fixture(tmp_path))


def test_fts_entity_lookup_resolves_deduped_canonical_apsw(tmp_path):
    _assert_fts_entity_lookup_resolves_deduped_canonical(_apsw_fixture(tmp_path))


def _assert_kg_linked_chunks_only_when_entity_detected(conn):
    from brainlayer import search_repo

    _entity(conn, "person-etan", "person", "Etan Heyman")
    _entity(conn, "concept-architecture", "concept", "Architecture")
    _execute(
        conn,
        "INSERT INTO kg_entity_chunks (entity_id, chunk_id) VALUES (?, ?)",
        ("person-etan", "chunk-etan"),
    )
    _execute(
        conn,
        "INSERT INTO kg_entity_chunks (entity_id, chunk_id) VALUES (?, ?)",
        ("concept-architecture", "chunk-architecture"),
    )

    store = _Store(conn)

    assert search_repo._kg_linked_chunk_ids_for_query(store, "Etan Heyman BrainLayer work") == {"chunk-etan"}
    assert search_repo._kg_linked_chunk_ids_for_query(store, "architecture implementation details") == set()


def test_kg_linked_chunks_only_when_entity_detected_sqlite(tmp_path):
    _assert_kg_linked_chunks_only_when_entity_detected(_sqlite_fixture(tmp_path))


def test_kg_linked_chunks_only_when_entity_detected_apsw(tmp_path):
    _assert_kg_linked_chunks_only_when_entity_detected(_apsw_fixture(tmp_path))
