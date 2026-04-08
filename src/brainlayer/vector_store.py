"""SQLite-vec based vector store for fast search.

Thin facade: VectorStore inherits from focused mixin modules.
See search_repo.py, kg_repo.py, session_repo.py for the extracted methods.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import apsw
import apsw.bestpractice
import sqlite_vec

from ._helpers import (  # noqa: I001
    _DEFAULT_MIN_CHARS as _DEFAULT_MIN_CHARS,
)
from ._helpers import (
    _SOURCE_MIN_CHARS as _SOURCE_MIN_CHARS,
)
from ._helpers import (
    _escape_fts5_query as _escape_fts5_query,
)
from ._helpers import (
    _safe_json_loads as _safe_json_loads,
)
from ._helpers import (
    serialize_f32 as serialize_f32,
)
from ._helpers import (
    source_aware_min_chars as source_aware_min_chars,
)
from .kg_repo import KGMixin
from .search_repo import SearchMixin
from .session_repo import SessionMixin


def _set_busy_timeout_hook(conn: apsw.Connection) -> None:
    """Set busy_timeout on every new connection before any other hooks.

    APSW bestpractice hooks (connection_optimize) run PRAGMA optimize inside
    the Connection() constructor. Without busy_timeout set first, this PRAGMA
    fails with BusyError when other processes hold the DB lock.
    """
    conn.setbusytimeout(30_000)  # 30 seconds — needed for parallel enrichment workers + MCP


# Register busy_timeout hook BEFORE bestpractice hooks so it fires first.
# bestpractice.apply() adds hooks that run PRAGMA optimize inside Connection(),
# which needs busy_timeout active or it crashes under contention.
apsw.connection_hooks.insert(0, _set_busy_timeout_hook)
apsw.bestpractice.apply(apsw.bestpractice.recommended)


class VectorStore(SearchMixin, KGMixin, SessionMixin):
    """SQLite-vec based vector store.

    Core chunk CRUD and schema management live here.
    Search, KG, and session methods are inherited from mixin classes.
    """

    # Retry settings for DB init under contention (multiple MCP instances + enrichment)
    _INIT_MAX_RETRIES = 5
    _INIT_BASE_DELAY = 0.5  # seconds

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._fts5_health_cache: dict[str, Any] = {}
        self._retrieval_strengthening_pending: dict[str, dict[str, float]] = {}
        self._retrieval_strengthening_query_count = 0
        self._retrieval_strengthening_flush_threshold = 100
        self._retrieval_strengthening_lock = threading.Lock()
        self._readonly = self.db_path.exists() and not os.access(self.db_path, os.W_OK)
        if self._readonly:
            self._init_readonly_db()
        else:
            self._init_db_with_retry()

    def _init_readonly_db(self) -> None:
        """Open an existing DB in readonly mode without running migrations."""
        self.conn = apsw.Connection(str(self.db_path), flags=apsw.SQLITE_OPEN_READONLY)
        self.conn.setbusytimeout(10_000)
        self.conn.enableloadextension(True)
        self.conn.loadextension(sqlite_vec.loadable_path())
        self.conn.enableloadextension(False)

        cursor = self.conn.cursor()
        existing_tables = {
            row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
        }
        self._binary_index_available = "chunk_vectors_binary" in existing_tables
        self._local = threading.local()

    def _init_db_with_retry(self) -> None:
        """Initialize DB with retry on BusyError.

        Multiple BrainLayer processes (MCP instances, daemon, enrichment) may
        contend for write locks during DDL. Retry with exponential backoff
        instead of crashing on the first BusyError.
        """
        import time

        last_err = None
        for attempt in range(self._INIT_MAX_RETRIES):
            try:
                self._init_db()
                return
            except apsw.BusyError as e:
                last_err = e
                delay = self._INIT_BASE_DELAY * (2**attempt)
                import sys

                print(
                    f"  DB init BusyError (attempt {attempt + 1}/{self._INIT_MAX_RETRIES}), "
                    f"retrying in {delay:.1f}s...",
                    file=sys.stderr,
                )
                time.sleep(delay)
        raise last_err  # type: ignore[misc]

    def _init_db(self) -> None:
        """Initialize database with vector extension."""
        self.conn = apsw.Connection(str(self.db_path))

        # Set busy timeout IMMEDIATELY via APSW native method — before any DDL.
        self.conn.setbusytimeout(10_000)  # 10 seconds

        self.conn.enableloadextension(True)
        self.conn.loadextension(sqlite_vec.loadable_path())
        self.conn.enableloadextension(False)

        cursor = self.conn.cursor()

        # WAL mode is persistent on the DB file — set it every time
        cursor.execute("PRAGMA journal_mode = WAL")

        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                source_file TEXT NOT NULL,
                project TEXT,
                content_type TEXT,
                value_type TEXT,
                char_count INTEGER,
                source TEXT,
                sender TEXT,
                language TEXT,
                conversation_id TEXT,
                position INTEGER,
                context_summary TEXT
            )
        """)

        # Add columns if upgrading existing DB
        existing_cols = {row[1] for row in cursor.execute("PRAGMA table_info(chunks)")}
        for col, typ in [
            ("source", "TEXT"),
            ("sender", "TEXT"),
            ("language", "TEXT"),
            ("conversation_id", "TEXT"),
            ("position", "INTEGER"),
            ("context_summary", "TEXT"),
            ("tags", "TEXT"),
            ("tag_confidence", "REAL"),
            ("summary", "TEXT"),
            ("importance", "REAL"),
            ("intent", "TEXT"),
            ("enriched_at", "TEXT"),
            ("primary_symbols", "TEXT"),
            ("resolved_query", "TEXT"),
            ("key_facts", "TEXT"),
            ("resolved_queries", "TEXT"),
            ("epistemic_level", "TEXT"),
            ("version_scope", "TEXT"),
            ("debt_impact", "TEXT"),
            ("external_deps", "TEXT"),
            ("created_at", "TEXT"),
            ("sentiment_label", "TEXT"),
            ("sentiment_score", "REAL"),
            ("sentiment_signals", "TEXT"),
            ("half_life_days", "REAL DEFAULT 30.0"),
            ("last_retrieved", "REAL DEFAULT NULL"),
            ("retrieval_count", "INTEGER DEFAULT 0"),
            ("decay_score", "REAL DEFAULT 1.0"),
            ("pinned", "INTEGER DEFAULT 0"),
            ("archived", "INTEGER DEFAULT 0"),
            # Lifecycle columns (chunk lifecycle management)
            ("superseded_by", "TEXT"),
            ("aggregated_into", "TEXT"),
            ("archived_at", "TEXT"),
        ]:
            if col not in existing_cols:
                cursor.execute(f"ALTER TABLE chunks ADD COLUMN {col} {typ}")

        cursor.execute("""
            UPDATE chunks
            SET archived = 1
            WHERE value_type = 'ARCHIVED' AND COALESCE(archived, 0) = 0
        """)

        # Indexes for filtering
        for idx, col in [
            ("idx_chunks_source", "source"),
            ("idx_chunks_sender", "sender"),
            ("idx_chunks_conversation", "conversation_id"),
            ("idx_chunks_intent", "intent"),
            ("idx_chunks_importance", "importance"),
            ("idx_chunks_enriched", "enriched_at"),
            ("idx_chunks_created", "created_at"),
            ("idx_chunks_sentiment", "sentiment_label"),
            ("idx_chunks_project", "project"),
            ("idx_chunks_content_type", "content_type"),
            ("idx_chunks_language", "language"),
        ]:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx} ON chunks({col})")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_decay_score ON chunks(decay_score) WHERE archived = 0")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_last_retrieved ON chunks(last_retrieved) WHERE archived = 0"
        )

        # Vector table (1024 dims for bge-large-en-v1.5)
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding FLOAT[1024]
            )
        """)
        existing_tables = {
            row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
        }
        self._binary_index_available = "chunk_vectors_binary" in existing_tables
        if not self._binary_index_available:
            try:
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors_binary USING vec0(
                        chunk_id TEXT PRIMARY KEY,
                        embedding BIT[1024]
                    )
                """)
                self._binary_index_available = True
            except apsw.ReadOnlyError:
                self._binary_index_available = False

        # FTS5 full-text search — indexes content + enrichment metadata
        # for better keyword matches on summaries, tags, and resolved queries.
        _FTS5_COLUMNS = "content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id UNINDEXED"

        # Detect old single-column FTS5 schema and rebuild if needed.
        # FTS5 virtual tables can't be ALTERed — must drop and recreate.
        _needs_fts_rebuild = False
        try:
            fts_cols = {row[1] for row in cursor.execute("PRAGMA table_info(chunks_fts)")}
            if fts_cols and ("summary" not in fts_cols or "key_facts" not in fts_cols):
                _needs_fts_rebuild = True
        except Exception:
            pass  # Table doesn't exist yet, will be created below

        if _needs_fts_rebuild:
            cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_insert")
            cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_delete")
            cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_update")
            cursor.execute("DROP TABLE IF EXISTS chunks_fts")

        cursor.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                {_FTS5_COLUMNS}
            )
        """)

        # FTS5 sync triggers — keep summary/tags/resolved_query in sync
        cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_insert")
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
                VALUES (
                    new.content,
                    new.summary,
                    new.tags,
                    new.resolved_query,
                    new.key_facts,
                    new.resolved_queries,
                    new.id
                );
            END
        """)
        cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_delete")
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
                DELETE FROM chunks_fts WHERE chunk_id = old.id;
            END
        """)
        cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_update")
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_update
            AFTER UPDATE OF content, summary, tags, resolved_query, key_facts, resolved_queries ON chunks BEGIN
                DELETE FROM chunks_fts WHERE chunk_id = old.id;
                INSERT INTO chunks_fts(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
                VALUES (
                    new.content,
                    new.summary,
                    new.tags,
                    new.resolved_query,
                    new.key_facts,
                    new.resolved_queries,
                    new.id
                );
            END
        """)

        # ── Tag junction table (replaces json_each scanning) ──────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunk_tags (
                chunk_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                PRIMARY KEY (chunk_id, tag)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_tags_tag ON chunk_tags(tag)")

        # Sync triggers: keep chunk_tags in sync with chunks.tags JSON
        cursor.execute("DROP TRIGGER IF EXISTS chunk_tags_insert")
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunk_tags_insert AFTER INSERT ON chunks
            WHEN new.tags IS NOT NULL AND json_valid(new.tags) = 1 BEGIN
                INSERT OR IGNORE INTO chunk_tags(chunk_id, tag)
                SELECT new.id, value FROM json_each(new.tags);
            END
        """)
        cursor.execute("DROP TRIGGER IF EXISTS chunk_tags_update")
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunk_tags_update
            AFTER UPDATE OF tags ON chunks
            WHEN new.tags IS NOT NULL AND json_valid(new.tags) = 1 BEGIN
                DELETE FROM chunk_tags WHERE chunk_id = new.id;
                INSERT OR IGNORE INTO chunk_tags(chunk_id, tag)
                SELECT new.id, value FROM json_each(new.tags);
            END
        """)
        cursor.execute("DROP TRIGGER IF EXISTS chunk_tags_update_clear")
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunk_tags_update_clear
            AFTER UPDATE OF tags ON chunks
            WHEN new.tags IS NULL OR json_valid(new.tags) = 0 BEGIN
                DELETE FROM chunk_tags WHERE chunk_id = new.id;
            END
        """)
        cursor.execute("DROP TRIGGER IF EXISTS chunk_tags_delete")
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunk_tags_delete AFTER DELETE ON chunks BEGIN
                DELETE FROM chunk_tags WHERE chunk_id = old.id;
            END
        """)

        # Backfill chunk_tags from existing data (detects partial fills from crashes)
        tagged_chunks = cursor.execute(
            "SELECT COUNT(*) FROM chunks WHERE tags IS NOT NULL AND json_valid(tags) = 1"
        ).fetchone()[0]
        backfilled_chunks = cursor.execute("SELECT COUNT(DISTINCT chunk_id) FROM chunk_tags").fetchone()[0]
        if tagged_chunks > 0 and backfilled_chunks < tagged_chunks:
            cursor.execute("""
                INSERT OR IGNORE INTO chunk_tags(chunk_id, tag)
                SELECT c.id, j.value FROM chunks c, json_each(c.tags) j
                WHERE c.tags IS NOT NULL AND json_valid(c.tags) = 1
            """)

        # Session context table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_context (
                session_id TEXT PRIMARY KEY,
                project TEXT,
                branch TEXT,
                pr_number INTEGER,
                commit_shas TEXT,
                files_changed TEXT,
                started_at TEXT,
                ended_at TEXT,
                created_at TEXT
            )
        """)
        existing_sc_cols = {row[1] for row in cursor.execute("PRAGMA table_info(session_context)")}
        for col in ("plan_name", "plan_phase", "story_id"):
            if col not in existing_sc_cols:
                cursor.execute(f"ALTER TABLE session_context ADD COLUMN {col} TEXT")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                timestamp TEXT,
                session_id TEXT,
                action TEXT,
                chunk_id TEXT,
                project TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_interactions_path ON file_interactions(file_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_interactions_session ON file_interactions(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_context_project ON session_context(project)")

        # Operations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS operations (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                operation_type TEXT,
                chunk_ids TEXT,
                summary TEXT,
                outcome TEXT,
                started_at TEXT,
                ended_at TEXT,
                step_count INTEGER DEFAULT 0,
                created_at TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_operations_session ON operations(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_operations_type ON operations(operation_type)")

        # Topic chains table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topic_chains (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                session_a TEXT NOT NULL,
                session_b TEXT NOT NULL,
                shared_actions INTEGER DEFAULT 0,
                time_delta_hours REAL,
                project TEXT,
                created_at TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_topic_chains_file ON topic_chains(file_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_topic_chains_session ON topic_chains(session_a)")

        # Session enrichment table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_enrichments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL UNIQUE,
                file_path TEXT,
                enrichment_version TEXT NOT NULL DEFAULT '1.0',
                enrichment_model TEXT,
                enrichment_timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                session_start_time TEXT,
                session_end_time TEXT,
                duration_seconds INTEGER,
                message_count INTEGER NOT NULL DEFAULT 0,
                user_message_count INTEGER NOT NULL DEFAULT 0,
                assistant_message_count INTEGER NOT NULL DEFAULT 0,
                tool_call_count INTEGER NOT NULL DEFAULT 0,
                session_summary TEXT,
                primary_intent TEXT,
                outcome TEXT CHECK(outcome IN ('success','partial_success','failure','abandoned','ongoing')),
                complexity_score INTEGER CHECK(complexity_score BETWEEN 1 AND 10),
                session_quality_score INTEGER CHECK(session_quality_score BETWEEN 1 AND 10),
                decisions_made TEXT DEFAULT '[]',
                corrections TEXT DEFAULT '[]',
                learnings TEXT DEFAULT '[]',
                mistakes TEXT DEFAULT '[]',
                patterns TEXT DEFAULT '[]',
                topic_tags TEXT DEFAULT '[]',
                tool_usage_stats TEXT DEFAULT '[]',
                what_worked TEXT,
                what_failed TEXT,
                summary_embedding BLOB
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_enrichments_session ON session_enrichments(session_id)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_enrichments_project ON session_enrichments(primary_intent)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_enrichments_outcome ON session_enrichments(outcome)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_enrichments_quality ON session_enrichments(session_quality_score)"
        )

        # Session enrichment FTS5
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS session_enrichments_fts USING fts5(
                session_summary, what_worked, what_failed, session_id UNINDEXED
            )
        """)

        # Phase commits table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS phase_commits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                commit_hash TEXT NOT NULL,
                commit_message TEXT,
                phase_name TEXT,
                session_id TEXT,
                project TEXT,
                files_changed TEXT,
                confidence_score REAL,
                outcome TEXT,
                reversibility TEXT,
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_phase_commits_project ON phase_commits(project)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_phase_commits_phase ON phase_commits(phase_name)")

        # source_project_id column
        if "source_project_id" not in existing_cols:
            cursor.execute("ALTER TABLE chunks ADD COLUMN source_project_id TEXT")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source_project ON chunks(source_project_id)")

        # ── Chunk events audit table ──────────────────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunk_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT NOT NULL,
                action TEXT NOT NULL,
                timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                by_whom TEXT,
                reason TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_events_chunk ON chunk_events(chunk_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_events_action ON chunk_events(action)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_events_timestamp ON chunk_events(timestamp)")

        # ── Health events audit table ─────────────────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                details TEXT,
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_health_events_type ON health_events(event_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_health_events_severity ON health_events(severity)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_health_events_created ON health_events(created_at)")

        # ── Correction mining table ───────────────────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS correction_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                entity_name TEXT,
                attribute TEXT,
                old_value TEXT,
                new_value TEXT,
                confidence REAL DEFAULT 0.0,
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_correction_pairs_chunk ON correction_pairs(chunk_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_correction_pairs_pattern ON correction_pairs(pattern_type)")

        # ── Knowledge Graph tables ──────────────────────────────────────

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kg_entities (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                name TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                UNIQUE(entity_type, name)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_entities_type ON kg_entities(entity_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_entities_name ON kg_entities(name)")

        kg_entity_cols = {row[1] for row in cursor.execute("PRAGMA table_info(kg_entities)")}
        if "user_verified" not in kg_entity_cols:
            cursor.execute("ALTER TABLE kg_entities ADD COLUMN user_verified INTEGER DEFAULT 0")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kg_relations (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                properties TEXT DEFAULT '{}',
                confidence REAL DEFAULT 1.0,
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                UNIQUE(source_id, target_id, relation_type)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_relations_source ON kg_relations(source_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_relations_target ON kg_relations(target_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_relations_type ON kg_relations(relation_type)")

        kg_rel_cols = {row[1] for row in cursor.execute("PRAGMA table_info(kg_relations)")}
        if "user_verified" not in kg_rel_cols:
            cursor.execute("ALTER TABLE kg_relations ADD COLUMN user_verified INTEGER DEFAULT 0")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kg_entity_chunks (
                entity_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                relevance REAL DEFAULT 1.0,
                context TEXT,
                PRIMARY KEY (entity_id, chunk_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_ec_entity ON kg_entity_chunks(entity_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_ec_chunk ON kg_entity_chunks(chunk_id)")

        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS kg_vec_entities USING vec0(
                entity_id TEXT PRIMARY KEY,
                embedding FLOAT[1024]
            )
        """)

        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS kg_entities_fts USING fts5(
                name, metadata, entity_id UNINDEXED
            )
        """)

        # KG FTS5 sync triggers
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS kg_entities_fts_insert AFTER INSERT ON kg_entities BEGIN
                INSERT INTO kg_entities_fts(name, metadata, entity_id)
                VALUES (new.name, new.metadata, new.id);
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS kg_entities_fts_delete AFTER DELETE ON kg_entities BEGIN
                DELETE FROM kg_entities_fts WHERE entity_id = old.id;
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS kg_entities_fts_update
            AFTER UPDATE OF name, metadata ON kg_entities BEGIN
                DELETE FROM kg_entities_fts WHERE entity_id = old.id;
                INSERT INTO kg_entities_fts(name, metadata, entity_id)
                VALUES (new.name, new.metadata, new.id);
            END
        """)

        # Entity aliases table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kg_entity_aliases (
                alias TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                alias_type TEXT DEFAULT 'name',
                created_at TEXT,
                PRIMARY KEY (alias, entity_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_alias_entity ON kg_entity_aliases(entity_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_alias_lookup ON kg_entity_aliases(alias COLLATE NOCASE)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_alias_type ON kg_entity_aliases(alias_type)")

        # KG standard spec migrations
        for col, default in [
            ("canonical_name", "TEXT"),
            ("description", "TEXT"),
            ("confidence", "REAL DEFAULT 1.0"),
            ("importance", "REAL DEFAULT 0.5"),
            ("valid_from", "TEXT"),
            ("valid_until", "TEXT"),
            ("group_id", "TEXT"),
        ]:
            if col not in kg_entity_cols:
                cursor.execute(f"ALTER TABLE kg_entities ADD COLUMN {col} {default}")

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_kg_entities_canonical ON kg_entities(canonical_name, entity_type)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_entities_valid ON kg_entities(valid_from, valid_until)")

        for col, default in [
            ("fact", "TEXT"),
            ("importance", "REAL DEFAULT 0.5"),
            ("valid_from", "TEXT"),
            ("valid_until", "TEXT"),
            ("expired_at", "TEXT"),
            ("source_chunk_id", "TEXT"),
        ]:
            if col not in kg_rel_cols:
                cursor.execute(f"ALTER TABLE kg_relations ADD COLUMN {col} {default}")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_relations_validity ON kg_relations(valid_from, valid_until)")

        ec_cols = {row[1] for row in cursor.execute("PRAGMA table_info(kg_entity_chunks)")}
        if "mention_type" not in ec_cols:
            cursor.execute("ALTER TABLE kg_entity_chunks ADD COLUMN mention_type TEXT")

        # ── R49: Entity Contracts Schema ───────────────────────────────

        # R49: entity_contracts table — defines required/expected fields per entity type
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entity_contracts (
                entity_type TEXT NOT NULL,
                field_name TEXT NOT NULL,
                field_type TEXT NOT NULL DEFAULT 'text',
                requirement TEXT NOT NULL DEFAULT 'optional',
                description TEXT,
                PRIMARY KEY (entity_type, field_name)
            )
        """)

        # R49: entity_health table — completeness scores per entity
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entity_health (
                entity_name TEXT PRIMARY KEY,
                completeness_score REAL NOT NULL DEFAULT 0.0,
                health_level INTEGER NOT NULL DEFAULT 1,
                missing_required TEXT DEFAULT '[]',
                missing_expected TEXT DEFAULT '[]',
                chunk_count INTEGER NOT NULL DEFAULT 0,
                relationship_count INTEGER NOT NULL DEFAULT 0,
                last_scored_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            )
        """)

        # R49: entity_type_hierarchy — type taxonomy stored as data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entity_type_hierarchy (
                child_type TEXT PRIMARY KEY,
                parent_type TEXT,
                description TEXT
            )
        """)

        # R49: Seed type hierarchy with core types + subtypes
        _type_hierarchy_seed = [
            ("agent", "entity", "Autonomous AI agent or golem"),
            ("person", "entity", "Human individual"),
            ("tool", "entity", "Software tool or service"),
            ("project", "entity", "Software project or initiative"),
            ("concept", "entity", "Abstract concept, pattern, or domain idea"),
            ("topic", "concept", "Recurring subject or thematic area"),
            ("protocol", "topic", "Named workflow or protocol"),
            ("community", "entity", "Community, audience, or social group"),
            ("health_metric", "topic", "Health or wellness metric"),
            ("workflow", "concept", "Repeatable workflow or process"),
            ("device", "entity", "Hardware device or machine"),
            ("event", "entity", "Temporal event or occurrence"),
            ("organization", "entity", "Company or group"),
            ("golem", "agent", "Specialized AI agent in the golems ecosystem"),
            ("platform", "tool", "Software platform or framework"),
            ("skill", "concept", "Reusable AI skill or capability"),
            ("decision", "concept", "Architectural or design decision"),
        ]
        for child, parent, desc in _type_hierarchy_seed:
            cursor.execute(
                "INSERT OR IGNORE INTO entity_type_hierarchy (child_type, parent_type, description) VALUES (?, ?, ?)",
                (child, parent, desc),
            )

        # R49: ALTER kg_entities — add entity_subtype, status
        if "entity_subtype" not in kg_entity_cols:
            cursor.execute("ALTER TABLE kg_entities ADD COLUMN entity_subtype TEXT")
        if "status" not in kg_entity_cols:
            cursor.execute("ALTER TABLE kg_entities ADD COLUMN status TEXT DEFAULT 'active'")
        if "parent_id" not in kg_entity_cols:
            cursor.execute("ALTER TABLE kg_entities ADD COLUMN parent_id TEXT")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_entities_parent ON kg_entities(parent_id)")

        # R49: ALTER kg_entity_chunks — add relation_tier, weight
        if "relation_tier" not in ec_cols:
            cursor.execute("ALTER TABLE kg_entity_chunks ADD COLUMN relation_tier INTEGER DEFAULT 4")
        if "weight" not in ec_cols:
            cursor.execute("ALTER TABLE kg_entity_chunks ADD COLUMN weight REAL DEFAULT 0.25")

        # R49: Upgrade kg_entity_aliases — add valid_from, valid_to if missing
        alias_cols = {row[1] for row in cursor.execute("PRAGMA table_info(kg_entity_aliases)")}
        if "valid_to" not in alias_cols:
            cursor.execute("ALTER TABLE kg_entity_aliases ADD COLUMN valid_to TEXT")
        # valid_from may already exist from original schema (as created_at) — ensure both names exist
        if "valid_from" not in alias_cols:
            cursor.execute("ALTER TABLE kg_entity_aliases ADD COLUMN valid_from TEXT")

        # kg_current_facts view
        cursor.execute("DROP VIEW IF EXISTS kg_current_facts")
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS kg_current_facts AS
            SELECT * FROM kg_relations
            WHERE (valid_from IS NULL OR valid_from <= strftime('%Y-%m-%dT%H:%M:%fZ','now'))
              AND (valid_until IS NULL OR valid_until >= strftime('%Y-%m-%dT%H:%M:%fZ','now'))
              AND expired_at IS NULL
        """)

        # FTS5 backfill check — populate from chunks if FTS is empty (fresh rebuild or first run)
        fts_count = list(cursor.execute("SELECT COUNT(*) FROM chunks_fts"))[0][0]
        chunk_count = list(cursor.execute("SELECT COUNT(*) FROM chunks"))[0][0]
        if chunk_count > 0 and fts_count == 0:
            cursor.execute("""
                INSERT INTO chunks_fts(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
                SELECT content, summary, tags, resolved_query, key_facts, resolved_queries, id FROM chunks
            """)

        # Thread-local storage for per-thread read connections.
        # APSW connections are NOT thread-safe — each thread needs its own.
        # This prevents "Connection is busy in another thread" when parallel
        # MCP tool calls (e.g., brain_search) hit the same VectorStore.
        self._local = threading.local()

    def _get_read_conn(self) -> apsw.Connection:
        """Get or create a per-thread readonly connection."""
        conn = getattr(self._local, "read_conn", None)
        if conn is None:
            conn = apsw.Connection(str(self.db_path), flags=apsw.SQLITE_OPEN_READONLY)
            conn.enableloadextension(True)
            conn.loadextension(sqlite_vec.loadable_path())
            conn.enableloadextension(False)
            conn.setbusytimeout(30_000)
            self._local.read_conn = conn
        return conn

    def _read_cursor(self):
        """Return a cursor for read operations using a per-thread readonly connection."""
        return self._get_read_conn().cursor()

    def _log_health_event(self, event_type: str, severity: str, details: Dict[str, Any]) -> None:
        """Append a health event. Readonly stores skip writes silently."""
        if getattr(self, "_readonly", False):
            return
        try:
            self.conn.cursor().execute(
                "INSERT INTO health_events (event_type, severity, details) VALUES (?, ?, ?)",
                (event_type, severity, json.dumps(details, sort_keys=True)),
            )
        except (apsw.ReadOnlyError, apsw.BusyError):
            return

    def _get_fts5_counts(self) -> tuple[int, int]:
        """Read chunks and FTS counts using a single query on the readonly path."""
        row = (
            self._read_cursor()
            .execute("SELECT (SELECT COUNT(*) FROM chunks), (SELECT COUNT(*) FROM chunks_fts)")
            .fetchone()
        )
        return int(row[0]), int(row[1])

    @staticmethod
    def _build_fts5_health_result(chunk_count: int, fts_count: int, severity: str) -> Dict[str, Any]:
        """Shape a health payload from count data."""
        desync_pct = 0.0
        if chunk_count > 0:
            desync_pct = round(abs(chunk_count - fts_count) * 100.0 / chunk_count, 2)
        return {
            "synced": desync_pct <= 1.0,
            "chunk_count": chunk_count,
            "fts_count": fts_count,
            "desync_pct": desync_pct,
            "severity": severity,
        }

    def check_fts5_health(self, cache_ttl_seconds: int = 60) -> Dict[str, Any]:
        """Check FTS5 sync health with a short-lived cache for hot-path callers."""
        now = time.time()
        cache = self._fts5_health_cache
        if cache_ttl_seconds > 0 and cache.get("expires_at", 0) > now:
            return dict(cache["result"])

        chunk_count, fts_count = self._get_fts5_counts()
        desync_pct = 0.0 if chunk_count == 0 else abs(chunk_count - fts_count) * 100.0 / chunk_count

        if desync_pct > 20.0:
            self._log_health_event(
                "fts5_desync_critical",
                "emergency",
                {"chunk_count": chunk_count, "fts_count": fts_count, "desync_pct": round(desync_pct, 2)},
            )
            rebuild_result = self.rebuild_fts5()
            result = {
                "synced": rebuild_result["success"],
                "chunk_count": rebuild_result["chunk_count"],
                "fts_count": rebuild_result["fts_count"],
                "desync_pct": rebuild_result["desync_pct"],
                "severity": "emergency",
                "rebuild_triggered": True,
            }
        elif desync_pct > 5.0:
            result = self._build_fts5_health_result(chunk_count, fts_count, "critical")
            self._log_health_event("fts5_desync_critical", "critical", result)
        elif desync_pct > 1.0:
            result = self._build_fts5_health_result(chunk_count, fts_count, "warning")
            self._log_health_event("fts5_desync_warning", "warning", result)
        else:
            result = self._build_fts5_health_result(chunk_count, fts_count, "info")

        if cache_ttl_seconds > 0:
            self._fts5_health_cache = {"result": dict(result), "expires_at": now + cache_ttl_seconds}
        else:
            self._fts5_health_cache = {}
        return result

    def check_wal_health(self) -> Dict[str, Any]:
        """Check WAL size and passive checkpoint status."""
        wal_path = Path(f"{self.db_path}-wal")
        wal_size_bytes = wal_path.stat().st_size if wal_path.exists() else 0
        wal_size_mb = wal_size_bytes / (1024 * 1024)

        checkpoint_row = self.conn.cursor().execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
        checkpoint_status = {
            "busy": int(checkpoint_row[0]),
            "log_frames": int(checkpoint_row[1]),
            "checkpointed_frames": int(checkpoint_row[2]),
        }

        severity = "info"
        if wal_size_mb > 50:
            severity = "warning"
            self._log_health_event(
                "wal_bloat",
                "warning",
                {
                    "wal_path": str(wal_path),
                    "wal_size_bytes": wal_size_bytes,
                    "wal_size_mb": round(wal_size_mb, 2),
                    "checkpoint_status": checkpoint_status,
                },
            )

        return {
            "wal_path": str(wal_path),
            "wal_exists": wal_path.exists(),
            "wal_size_bytes": wal_size_bytes,
            "wal_size_mb": round(wal_size_mb, 2),
            "checkpoint_status": checkpoint_status,
            "severity": severity,
        }

    def deep_integrity_check(self) -> Dict[str, Any]:
        """Run an FTS integrity check plus a bounded spot-check of chunk IDs."""
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('integrity-check')")

        total_chunks = cursor.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        spot_check_count = min(100, total_chunks)
        sample_ids = [
            row[0]
            for row in cursor.execute(
                "SELECT id FROM chunks ORDER BY random() LIMIT ?",
                (spot_check_count,),
            )
        ]

        missing_chunk_ids: list[str] = []
        if sample_ids:
            placeholders = ", ".join("?" for _ in sample_ids)
            found_ids = {
                row[0]
                for row in cursor.execute(
                    f"SELECT chunk_id FROM chunks_fts WHERE chunk_id IN ({placeholders})",
                    sample_ids,
                )
            }
            missing_chunk_ids = sorted(set(sample_ids) - found_ids)

        ok = len(missing_chunk_ids) == 0
        details = {
            "spot_check_count": spot_check_count,
            "missing_chunk_ids": missing_chunk_ids,
        }
        if ok:
            self._log_health_event("fts5_integrity_ok", "info", details)
        else:
            self._log_health_event("integrity_fail", "critical", details)

        return {
            "ok": ok,
            "fts_integrity": "ok" if ok else "failed",
            "spot_check_count": spot_check_count,
            "missing_chunk_ids": missing_chunk_ids,
        }

    def rebuild_fts5(self) -> Dict[str, Any]:
        """Rebuild the FTS5 table and verify post-rebuild counts."""
        self._log_health_event("fts5_rebuild", "emergency", {"db_path": str(self.db_path)})
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        chunk_count, fts_count = self._get_fts5_counts()
        if chunk_count != fts_count:
            cursor.execute("DELETE FROM chunks_fts")
            cursor.execute("""
                INSERT INTO chunks_fts(content, summary, tags, resolved_query, chunk_id)
                SELECT content, summary, tags, resolved_query, id FROM chunks
            """)
        try:
            cursor.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except apsw.Error:
            pass

        self._fts5_health_cache = {}
        chunk_count, fts_count = self._get_fts5_counts()
        desync_pct = 0.0 if chunk_count == 0 else round(abs(chunk_count - fts_count) * 100.0 / chunk_count, 2)
        return {
            "success": chunk_count == fts_count,
            "chunk_count": chunk_count,
            "fts_count": fts_count,
            "desync_pct": desync_pct,
        }

    def build_binary_index(self) -> int:
        """Backfill binary-quantized vectors from existing float vectors."""
        if not getattr(self, "_binary_index_available", False):
            raise RuntimeError("Binary vector index is unavailable on this database")
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO chunk_vectors_binary(chunk_id, embedding)
            SELECT chunk_id, vec_quantize_binary(embedding) FROM chunk_vectors
        """)
        from .search_repo import clear_hybrid_search_cache

        clear_hybrid_search_cache(getattr(self, "db_path", None))
        return self.conn.changes()

    def _upsert_chunk_vector(self, cursor, chunk_id: str, embedding: List[float]) -> None:
        """Keep float and binary vector tables in sync for a chunk."""
        embedding_bytes = serialize_f32(embedding)
        cursor.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,))
        cursor.execute(
            """
            INSERT INTO chunk_vectors (chunk_id, embedding)
            VALUES (?, ?)
        """,
            (chunk_id, embedding_bytes),
        )
        if getattr(self, "_binary_index_available", False):
            cursor.execute("DELETE FROM chunk_vectors_binary WHERE chunk_id = ?", (chunk_id,))
            cursor.execute(
                """
                INSERT INTO chunk_vectors_binary (chunk_id, embedding)
                VALUES (?, vec_quantize_binary(?))
            """,
                (chunk_id, embedding_bytes),
            )

    def _delete_chunk_vector(self, cursor, chunk_id: str) -> None:
        """Delete a chunk from both float and binary vector tables."""
        cursor.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,))
        if getattr(self, "_binary_index_available", False):
            cursor.execute("DELETE FROM chunk_vectors_binary WHERE chunk_id = ?", (chunk_id,))

    # ── Chunk CRUD ──────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        """Upsert chunks with embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have same length")

        cursor = self.conn.cursor()

        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk["id"]

            cursor.execute(
                """
                INSERT INTO chunks
                (id, content, metadata, source_file, project,
                 content_type, value_type, char_count, source, created_at,
                 conversation_id, position, sender)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content = excluded.content,
                    metadata = excluded.metadata,
                    source_file = excluded.source_file,
                    project = excluded.project,
                    content_type = excluded.content_type,
                    value_type = excluded.value_type,
                    char_count = excluded.char_count,
                    source = excluded.source,
                    created_at = COALESCE(chunks.created_at, excluded.created_at),
                    conversation_id = COALESCE(excluded.conversation_id, chunks.conversation_id),
                    position = COALESCE(excluded.position, chunks.position),
                    sender = COALESCE(excluded.sender, chunks.sender)
            """,
                (
                    chunk_id,
                    chunk["content"],
                    json.dumps(chunk["metadata"]),
                    chunk["source_file"],
                    chunk.get("project"),
                    chunk.get("content_type"),
                    chunk.get("value_type"),
                    chunk.get("char_count", 0),
                    chunk.get("source", "claude_code"),
                    chunk.get("created_at"),
                    chunk.get("conversation_id"),
                    chunk.get("position"),
                    chunk.get("sender"),
                ),
            )

            self._upsert_chunk_vector(cursor, chunk_id, embedding)

        from .search_repo import clear_hybrid_search_cache

        clear_hybrid_search_cache(getattr(self, "db_path", None))

        return len(chunks)

    def update_chunk(
        self,
        chunk_id: str,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        importance: Optional[float] = None,
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """Update fields on an existing chunk. Returns True if chunk was found."""
        cursor = self.conn.cursor()
        rows = list(cursor.execute("SELECT id FROM chunks WHERE id = ?", (chunk_id,)))
        if not rows:
            return False

        if content is not None:
            cursor.execute(
                "UPDATE chunks SET content = ?, char_count = ?, summary = ? WHERE id = ?",
                (content, len(content), content[:200], chunk_id),
            )
        if tags is not None:
            cursor.execute(
                "UPDATE chunks SET tags = ? WHERE id = ?",
                (json.dumps(tags), chunk_id),
            )
        if importance is not None:
            cursor.execute(
                "UPDATE chunks SET importance = ? WHERE id = ?",
                (float(max(1, min(10, importance))), chunk_id),
            )
        if embedding is not None:
            self._upsert_chunk_vector(cursor, chunk_id, embedding)
        from .search_repo import clear_hybrid_search_cache

        clear_hybrid_search_cache(getattr(self, "db_path", None))
        return True

    def archive_chunk(self, chunk_id: str) -> bool:
        """Soft-delete a chunk by setting value_type to ARCHIVED and archived_at."""
        cursor = self.conn.cursor()
        rows = list(cursor.execute("SELECT id FROM chunks WHERE id = ?", (chunk_id,)))
        if not rows:
            return False
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        cursor.execute(
            "UPDATE chunks SET value_type = 'ARCHIVED', archived = 1, archived_at = ? WHERE id = ?",
            (now, chunk_id),
        )
        self._delete_chunk_vector(cursor, chunk_id)
        from .search_repo import clear_hybrid_search_cache

        clear_hybrid_search_cache(getattr(self, "db_path", None))
        return True

    def supersede_chunk(self, old_chunk_id: str, new_chunk_id: str) -> bool:
        """Mark old chunk as superseded by new chunk. Removes old from vector index."""
        cursor = self.conn.cursor()
        old_rows = list(cursor.execute("SELECT id FROM chunks WHERE id = ?", (old_chunk_id,)))
        if not old_rows:
            return False
        new_rows = list(cursor.execute("SELECT id FROM chunks WHERE id = ?", (new_chunk_id,)))
        if not new_rows:
            return False
        cursor.execute(
            "UPDATE chunks SET superseded_by = ? WHERE id = ?",
            (new_chunk_id, old_chunk_id),
        )
        self._delete_chunk_vector(cursor, old_chunk_id)
        from .search_repo import clear_hybrid_search_cache

        clear_hybrid_search_cache(getattr(self, "db_path", None))
        return True

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a single chunk by ID."""
        cursor = self.conn.cursor()
        rows = list(
            cursor.execute(
                """SELECT id, content, metadata, source_file, project, content_type,
                      value_type, tags, importance, created_at, summary,
                      superseded_by, aggregated_into, archived_at
               FROM chunks WHERE id = ?""",
                (chunk_id,),
            )
        )
        if not rows:
            return None
        r = rows[0]
        return {
            "id": r[0],
            "content": r[1],
            "metadata": r[2],
            "source_file": r[3],
            "project": r[4],
            "content_type": r[5],
            "value_type": r[6],
            "tags": r[7],
            "importance": r[8],
            "created_at": r[9],
            "summary": r[10],
            "superseded_by": r[11],
            "aggregated_into": r[12],
            "archived_at": r[13],
        }

    # ── Chunk events audit ─────────────────────────────────────────────

    def record_event(
        self,
        chunk_id: str,
        action: str,
        by_whom: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> int:
        """Record an audit event for a chunk. Returns the event row ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO chunk_events (chunk_id, action, by_whom, reason) VALUES (?, ?, ?, ?)",
            (chunk_id, action, by_whom, reason),
        )
        return self.conn.last_insert_rowid()

    def get_chunk_events(
        self,
        chunk_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get audit events for a chunk, newest first."""
        cursor = self._read_cursor()
        rows = list(
            cursor.execute(
                "SELECT id, chunk_id, action, timestamp, by_whom, reason "
                "FROM chunk_events WHERE chunk_id = ? ORDER BY id DESC LIMIT ?",
                (chunk_id, limit),
            )
        )
        return [
            {
                "id": r[0],
                "chunk_id": r[1],
                "action": r[2],
                "timestamp": r[3],
                "by_whom": r[4],
                "reason": r[5],
            }
            for r in rows
        ]

    # ── Context manager ─────────────────────────────────────────────────

    def close(self) -> None:
        """Close database connections."""
        # Close thread-local read connection if it exists
        if hasattr(self, "_local"):
            read_conn = getattr(self._local, "read_conn", None)
            if read_conn is not None:
                read_conn.close()
                self._local.read_conn = None
        if hasattr(self, "conn"):
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
