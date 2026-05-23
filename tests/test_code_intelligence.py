"""Tests for code_intelligence.py — project entity extraction from repo metadata."""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture
def tmp_repos(tmp_path: Path) -> Path:
    """Create temporary repo structures with package.json and pyproject.toml."""
    # Python project with pyproject.toml
    py_repo = tmp_path / "brainlayer"
    py_repo.mkdir()
    (py_repo / "pyproject.toml").write_text(
        """
[project]
name = "brainlayer"
version = "1.0.0"
description = "Memory for AI agents"
dependencies = [
    "apsw>=3.45.0",
    "fastapi>=0.100.0",
    "typer>=0.9.0",
    "rich>=13.0",
]

[project.scripts]
brainlayer = "brainlayer.cli:app"
brainlayer-mcp = "brainlayer.mcp:serve"
"""
    )

    # JS project with package.json
    js_repo = tmp_path / "voicelayer"
    js_repo.mkdir()
    (js_repo / "package.json").write_text(
        json.dumps(
            {
                "name": "voicelayer-mcp",
                "version": "2.0.0",
                "description": "Voice I/O for AI assistants",
                "dependencies": {
                    "@modelcontextprotocol/sdk": "^1.0.0",
                    "onnxruntime-node": "^1.0.0",
                    "zod": "^3.0.0",
                },
                "scripts": {"build": "tsc", "dev": "ts-node src/index.ts"},
            }
        )
    )
    (js_repo / "tsconfig.json").write_text("{}")

    # JS project with bun
    bun_repo = tmp_path / "golems"
    bun_repo.mkdir()
    (bun_repo / "package.json").write_text(
        json.dumps(
            {
                "name": "golems",
                "dependencies": {"@axiomhq/js": "^1.0.0", "react": "^18.0.0"},
                "devDependencies": {"typescript": "^5.0.0"},
                "scripts": {"test": "vitest"},
            }
        )
    )
    (bun_repo / "bun.lockb").write_bytes(b"")

    # Empty dir (no config — should be skipped)
    empty = tmp_path / "empty-repo"
    empty.mkdir()

    # Hidden dir (should be skipped)
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "package.json").write_text('{"name": "hidden"}')

    return tmp_path


@pytest.fixture
def tmp_db(tmp_path: Path) -> str:
    """Create a temporary KG database with required tables."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")

    conn.execute(
        """CREATE TABLE IF NOT EXISTS kg_entities (
            id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            metadata TEXT DEFAULT '{}',
            importance REAL DEFAULT 0.0,
            created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS kg_relations (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            properties TEXT DEFAULT '{}',
            confidence REAL DEFAULT 0.5,
            created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        )"""
    )
    conn.commit()
    conn.close()
    return db_path


class TestScanProjects:
    def test_finds_pyproject_and_packagejson(self, tmp_repos: Path) -> None:
        from brainlayer.pipeline.code_intelligence import scan_projects

        projects = scan_projects(tmp_repos)
        names = {p["name"] for p in projects}
        assert "brainlayer" in names
        assert "voicelayer-mcp" in names
        assert "golems" in names

    def test_skips_empty_and_hidden(self, tmp_repos: Path) -> None:
        from brainlayer.pipeline.code_intelligence import scan_projects

        projects = scan_projects(tmp_repos)
        names = {p["name"] for p in projects}
        assert "empty-repo" not in names
        assert "hidden" not in names

    def test_extracts_pyproject_metadata(self, tmp_repos: Path) -> None:
        from brainlayer.pipeline.code_intelligence import scan_projects

        projects = scan_projects(tmp_repos)
        py = next(p for p in projects if p["name"] == "brainlayer")
        assert py["version"] == "1.0.0"
        assert py["language"] == "python"
        assert py["description"] == "Memory for AI agents"
        assert "apsw" in py["dependencies"]
        assert "fastapi" in py["dependencies"]
        assert "brainlayer" in py["scripts"]

    def test_extracts_packagejson_metadata(self, tmp_repos: Path) -> None:
        from brainlayer.pipeline.code_intelligence import scan_projects

        projects = scan_projects(tmp_repos)
        js = next(p for p in projects if p["name"] == "voicelayer-mcp")
        assert js["version"] == "2.0.0"
        assert js["language"] == "typescript"  # tsconfig.json exists
        assert "@modelcontextprotocol/sdk" in js["dependencies"]

    def test_detects_bun_package_manager(self, tmp_repos: Path) -> None:
        from brainlayer.pipeline.code_intelligence import scan_projects

        projects = scan_projects(tmp_repos)
        bun = next(p for p in projects if p["name"] == "golems")
        assert bun["package_manager"] == "bun"

    def test_detects_framework(self, tmp_repos: Path) -> None:
        from brainlayer.pipeline.code_intelligence import scan_projects

        projects = scan_projects(tmp_repos)
        golems = next(p for p in projects if p["name"] == "golems")
        assert golems.get("framework") == "React"

        voice = next(p for p in projects if p["name"] == "voicelayer-mcp")
        assert voice.get("framework") == "MCP"

    def test_framework_detection_prefers_specific(self, tmp_path: Path) -> None:
        """React Native projects should not be classified as React."""
        from brainlayer.pipeline.code_intelligence import _detect_framework

        # Has both react and react-native — should detect React Native
        assert _detect_framework(["react", "react-native"]) == "React Native"
        # Has both react and expo — should detect Expo
        assert _detect_framework(["react", "expo"]) == "Expo"
        # Has both react and next — should detect Next.js
        assert _detect_framework(["react", "react-dom", "next"]) == "Next.js"
        # Only react — should detect React
        assert _detect_framework(["react", "react-dom"]) == "React"

    def test_tilde_version_specifier(self, tmp_path: Path) -> None:
        """PEP 440 ~= compatible release should be stripped correctly."""
        repo = tmp_path / "testproj"
        repo.mkdir()
        (repo / "pyproject.toml").write_text(
            '[project]\nname = "testproj"\ndependencies = ["numpy~=1.0", "pandas>=2.0"]\n'
        )

        from brainlayer.pipeline.code_intelligence import _extract_metadata

        meta = _extract_metadata(repo)
        assert meta is not None
        assert "numpy" in meta["dependencies"]
        assert "pandas" in meta["dependencies"]

    def test_nonexistent_dir_returns_empty(self, tmp_path: Path) -> None:
        from brainlayer.pipeline.code_intelligence import scan_projects

        projects = scan_projects(tmp_path / "nonexistent")
        assert projects == []


class TestEnrichProjects:
    def test_empty_scan_returns_all_keys(self, tmp_path: Path, tmp_db: str) -> None:
        """Early return when no projects found must include all stat keys."""
        from brainlayer.pipeline.code_intelligence import enrich_projects

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        stats = enrich_projects(db_path=tmp_db, base_dir=empty_dir)
        assert stats["projects_scanned"] == 0
        assert stats["dep_entities_created"] == 0  # The key that was missing

    def test_creates_project_entities(self, tmp_repos: Path, tmp_db: str) -> None:
        from brainlayer.pipeline.code_intelligence import enrich_projects

        stats = enrich_projects(db_path=tmp_db, base_dir=tmp_repos)
        assert stats["projects_scanned"] == 3
        assert stats["entities_created"] == 3

        conn = sqlite3.connect(tmp_db)
        entities = conn.execute("SELECT name, entity_type FROM kg_entities WHERE entity_type = 'project'").fetchall()
        conn.close()
        names = {e[0] for e in entities}
        assert "brainlayer" in names
        assert "voicelayer-mcp" in names
        assert "golems" in names

    def test_updates_existing_entities(self, tmp_repos: Path, tmp_db: str) -> None:
        from brainlayer.pipeline.code_intelligence import enrich_projects

        # First run creates
        enrich_projects(db_path=tmp_db, base_dir=tmp_repos)
        # Second run updates
        stats = enrich_projects(db_path=tmp_db, base_dir=tmp_repos)
        assert stats["entities_updated"] == 3
        assert stats["entities_created"] == 0

    def test_creates_dependency_relations(self, tmp_repos: Path, tmp_db: str) -> None:
        from brainlayer.pipeline.code_intelligence import enrich_projects

        enrich_projects(db_path=tmp_db, base_dir=tmp_repos)

        conn = sqlite3.connect(tmp_db)
        rels = conn.execute(
            "SELECT r.relation_type, e1.name, e2.name FROM kg_relations r "
            "JOIN kg_entities e1 ON r.source_id = e1.id "
            "JOIN kg_entities e2 ON r.target_id = e2.id "
            "WHERE r.relation_type = 'depends_on'"
        ).fetchall()
        conn.close()

        rel_pairs = {(r[1], r[2]) for r in rels}
        assert ("brainlayer", "fastapi") in rel_pairs
        assert ("brainlayer", "typer") in rel_pairs
        assert ("golems", "react") in rel_pairs
        assert ("voicelayer-mcp", "@modelcontextprotocol/sdk") in rel_pairs

    def test_creates_library_entities_for_deps(self, tmp_repos: Path, tmp_db: str) -> None:
        from brainlayer.pipeline.code_intelligence import enrich_projects

        stats = enrich_projects(db_path=tmp_db, base_dir=tmp_repos)
        assert stats["dep_entities_created"] > 0

        conn = sqlite3.connect(tmp_db)
        libs = conn.execute("SELECT name FROM kg_entities WHERE entity_type = 'library'").fetchall()
        conn.close()
        lib_names = {l[0] for l in libs}
        assert "fastapi" in lib_names
        assert "react" in lib_names

    def test_dry_run_makes_no_changes(self, tmp_repos: Path, tmp_db: str) -> None:
        from brainlayer.pipeline.code_intelligence import enrich_projects

        stats = enrich_projects(db_path=tmp_db, base_dir=tmp_repos, dry_run=True)
        assert stats["entities_created"] == 3  # Counts but doesn't persist

        conn = sqlite3.connect(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM kg_entities").fetchone()[0]
        conn.close()
        assert count == 0

    def test_stores_metadata_as_json(self, tmp_repos: Path, tmp_db: str) -> None:
        from brainlayer.pipeline.code_intelligence import enrich_projects

        enrich_projects(db_path=tmp_db, base_dir=tmp_repos)

        conn = sqlite3.connect(tmp_db)
        row = conn.execute(
            "SELECT metadata FROM kg_entities WHERE name = 'brainlayer' AND entity_type = 'project'"
        ).fetchone()
        conn.close()

        meta = json.loads(row[0])
        assert meta["language"] == "python"
        assert meta["version"] == "1.0.0"
        assert "brainlayer" in meta["repo_dir"]
        assert meta["package_manager"] == "pip"

    def test_idempotent_relations(self, tmp_repos: Path, tmp_db: str) -> None:
        """Running twice should not duplicate relations."""
        from brainlayer.pipeline.code_intelligence import enrich_projects

        enrich_projects(db_path=tmp_db, base_dir=tmp_repos)
        stats2 = enrich_projects(db_path=tmp_db, base_dir=tmp_repos)
        assert stats2["relations_added"] == 0

        conn = sqlite3.connect(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM kg_relations WHERE relation_type = 'depends_on'").fetchone()[0]
        conn.close()
        # Should be exactly the number from first run, not doubled
        assert count > 0

    def test_reconcile_removed_dependency_expires_relation(self, tmp_repos: Path, tmp_db: str) -> None:
        """regression-guard: removed code_intelligence dependencies are expired, not left current."""
        from brainlayer.pipeline.code_intelligence import enrich_projects

        enrich_projects(db_path=tmp_db, base_dir=tmp_repos)

        (tmp_repos / "brainlayer" / "pyproject.toml").write_text(
            """
[project]
name = "brainlayer"
version = "1.0.0"
description = "Memory for AI agents"
dependencies = [
    "apsw>=3.45.0",
    "typer>=0.9.0",
    "rich>=13.0",
]
"""
        )

        enrich_projects(db_path=tmp_db, base_dir=tmp_repos)

        conn = sqlite3.connect(tmp_db)
        row = conn.execute(
            """
            SELECT r.expired_at
            FROM kg_relations r
            JOIN kg_entities src ON r.source_id = src.id
            JOIN kg_entities tgt ON r.target_id = tgt.id
            WHERE src.name = 'brainlayer'
              AND tgt.name = 'fastapi'
              AND r.relation_type = 'depends_on'
            """
        ).fetchone()
        active_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM kg_current_facts r
            JOIN kg_entities src ON r.source_id = src.id
            JOIN kg_entities tgt ON r.target_id = tgt.id
            WHERE src.name = 'brainlayer'
              AND tgt.name = 'fastapi'
              AND r.relation_type = 'depends_on'
            """
        ).fetchone()[0]
        conn.close()

        assert row is not None
        assert row[0] is not None
        assert active_count == 0

    def test_reobserved_dependency_refreshes_valid_until_without_expiring(self, tmp_repos: Path, tmp_db: str) -> None:
        """regression-guard: positive re-observation revives the fact and extends validity."""
        from brainlayer.pipeline.code_intelligence import enrich_projects

        enrich_projects(db_path=tmp_db, base_dir=tmp_repos)

        conn = sqlite3.connect(tmp_db)
        conn.execute(
            """
            UPDATE kg_relations
            SET valid_until = '2026-01-01T00:00:00.000000Z',
                expired_at = '2026-01-01T00:00:00.000000Z'
            WHERE id IN (
                SELECT r.id
                FROM kg_relations r
                JOIN kg_entities src ON r.source_id = src.id
                JOIN kg_entities tgt ON r.target_id = tgt.id
                WHERE src.name = 'brainlayer'
                  AND tgt.name = 'fastapi'
                  AND r.relation_type = 'depends_on'
            )
            """
        )
        conn.commit()
        conn.close()

        enrich_projects(db_path=tmp_db, base_dir=tmp_repos)

        conn = sqlite3.connect(tmp_db)
        row = conn.execute(
            """
            SELECT r.valid_until, r.expired_at
            FROM kg_relations r
            JOIN kg_entities src ON r.source_id = src.id
            JOIN kg_entities tgt ON r.target_id = tgt.id
            WHERE src.name = 'brainlayer'
              AND tgt.name = 'fastapi'
              AND r.relation_type = 'depends_on'
            """
        ).fetchone()
        conn.close()

        assert row is not None
        assert row[0] is not None
        assert row[0] != "2026-01-01T00:00:00.000000Z"
        assert row[1] is None

    def test_reconcile_preserves_non_code_intelligence_dependency_owner(self, tmp_repos: Path, tmp_db: str) -> None:
        """regression-guard: code scanner must not take ownership of non-scanner relations."""
        from brainlayer.pipeline.code_intelligence import enrich_projects

        conn = sqlite3.connect(tmp_db)
        conn.execute(
            "INSERT INTO kg_entities (id, entity_type, name, metadata) VALUES ('proj-manual', 'project', 'brainlayer', '{}')"
        )
        conn.execute(
            "INSERT INTO kg_entities (id, entity_type, name, metadata) VALUES ('lib-manual', 'library', 'fastapi', '{}')"
        )
        conn.execute(
            """
            INSERT INTO kg_relations (id, source_id, target_id, relation_type, properties, confidence)
            VALUES ('rel-manual', 'proj-manual', 'lib-manual', 'depends_on', '{"source":"manual"}', 0.8)
            """
        )
        conn.commit()
        conn.close()

        enrich_projects(db_path=tmp_db, base_dir=tmp_repos)

        (tmp_repos / "brainlayer" / "pyproject.toml").write_text(
            """
[project]
name = "brainlayer"
version = "1.0.0"
description = "Memory for AI agents"
dependencies = [
    "apsw>=3.45.0",
    "typer>=0.9.0",
    "rich>=13.0",
]
"""
        )
        enrich_projects(db_path=tmp_db, base_dir=tmp_repos)

        conn = sqlite3.connect(tmp_db)
        row = conn.execute("SELECT properties, expired_at FROM kg_relations WHERE id = 'rel-manual'").fetchone()
        conn.close()

        assert row is not None
        assert json.loads(row[0]) == {"source": "manual"}
        assert row[1] is None

    def test_library_entity_not_expired_when_other_depends_on_remains(self, tmp_repos: Path, tmp_db: str) -> None:
        """regression-guard: expiring one source fact must not hide another active depends_on fact."""
        from brainlayer.pipeline.code_intelligence import enrich_projects

        enrich_projects(db_path=tmp_db, base_dir=tmp_repos)

        conn = sqlite3.connect(tmp_db)
        fastapi_id = conn.execute(
            "SELECT id FROM kg_entities WHERE entity_type = 'library' AND name = 'fastapi'"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO kg_entities (id, entity_type, name, metadata) VALUES ('proj-other', 'project', 'other', '{}')"
        )
        conn.execute(
            """
            INSERT INTO kg_relations (id, source_id, target_id, relation_type, properties, confidence)
            VALUES ('rel-other', 'proj-other', ?, 'depends_on', '{"source":"manual"}', 0.8)
            """,
            (fastapi_id,),
        )
        conn.commit()
        conn.close()

        (tmp_repos / "brainlayer" / "pyproject.toml").write_text(
            """
[project]
name = "brainlayer"
version = "1.0.0"
description = "Memory for AI agents"
dependencies = [
    "apsw>=3.45.0",
    "typer>=0.9.0",
    "rich>=13.0",
]
"""
        )
        enrich_projects(db_path=tmp_db, base_dir=tmp_repos)

        conn = sqlite3.connect(tmp_db)
        relation_expired_at, entity_expired_at = conn.execute(
            """
            SELECT r.expired_at, lib.expired_at
            FROM kg_relations r
            JOIN kg_entities src ON r.source_id = src.id
            JOIN kg_entities lib ON r.target_id = lib.id
            WHERE src.name = 'brainlayer'
              AND lib.name = 'fastapi'
              AND r.relation_type = 'depends_on'
            """
        ).fetchone()
        manual_active_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM kg_current_facts r
            JOIN kg_entities src ON r.source_id = src.id
            JOIN kg_entities lib ON r.target_id = lib.id
            WHERE src.name = 'other'
              AND lib.name = 'fastapi'
              AND r.relation_type = 'depends_on'
            """
        ).fetchone()[0]
        conn.close()

        assert relation_expired_at is not None
        assert entity_expired_at is None
        assert manual_active_count == 1

    def test_timestamp_precision_matches_sqlite_current_fact_view(self) -> None:
        """regression-guard: Python validity timestamps must sort with SQLite strftime('%f') values."""
        from brainlayer.pipeline.code_intelligence import _timestamp

        observed = datetime(2026, 5, 23, 10, 0, 45, 123456, tzinfo=timezone.utc)

        assert _timestamp(observed) == "2026-05-23T10:00:45.123Z"

    def test_refresh_preserves_code_intelligence_relation_metadata(self, tmp_repos: Path, tmp_db: str) -> None:
        """regression-guard: refreshing scanner-owned facts must not clobber extra provenance metadata."""
        from brainlayer.pipeline.code_intelligence import enrich_projects

        enrich_projects(db_path=tmp_db, base_dir=tmp_repos)

        conn = sqlite3.connect(tmp_db)
        conn.execute(
            """
            UPDATE kg_relations
            SET properties = '{"source":"code_intelligence","evidence":"manual-note"}'
            WHERE id IN (
                SELECT r.id
                FROM kg_relations r
                JOIN kg_entities src ON r.source_id = src.id
                JOIN kg_entities tgt ON r.target_id = tgt.id
                WHERE src.name = 'brainlayer'
                  AND tgt.name = 'fastapi'
                  AND r.relation_type = 'depends_on'
            )
            """
        )
        conn.commit()
        conn.close()

        enrich_projects(db_path=tmp_db, base_dir=tmp_repos)

        conn = sqlite3.connect(tmp_db)
        row = conn.execute(
            """
            SELECT r.properties
            FROM kg_relations r
            JOIN kg_entities src ON r.source_id = src.id
            JOIN kg_entities tgt ON r.target_id = tgt.id
            WHERE src.name = 'brainlayer'
              AND tgt.name = 'fastapi'
              AND r.relation_type = 'depends_on'
            """
        ).fetchone()
        conn.close()

        assert row is not None
        assert json.loads(row[0])["evidence"] == "manual-note"

    def test_reconcile_checkpoints_wal_around_bulk_writes(
        self, tmp_repos: Path, tmp_db: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """regression-guard: KG reconciliation bulk writes checkpoint WAL before and after the pass."""
        from brainlayer.pipeline import code_intelligence

        real_connect = sqlite3.connect
        executed_sql: list[str] = []

        class RecordingConnection:
            def __init__(self, conn: sqlite3.Connection) -> None:
                self._conn = conn

            def execute(self, sql: str, *args, **kwargs):
                executed_sql.append(sql)
                return self._conn.execute(sql, *args, **kwargs)

            def __getattr__(self, name: str):
                return getattr(self._conn, name)

        def recording_connect(*args, **kwargs):
            return RecordingConnection(real_connect(*args, **kwargs))

        monkeypatch.setattr(code_intelligence.sqlite3, "connect", recording_connect)

        code_intelligence.enrich_projects(db_path=tmp_db, base_dir=tmp_repos)

        checkpoint_count = sum(1 for sql in executed_sql if "PRAGMA wal_checkpoint(FULL)" in sql)
        assert checkpoint_count >= 2

    def test_writer_lock_rejects_existing_same_pid_owner(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """regression-guard: code-intelligence must not bypass another writer owned by this PID."""
        from brainlayer.pipeline.code_intelligence import (
            _code_intelligence_writer_lock,
            _pidfile_payload,
            _writer_pidfile_path,
        )

        monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(tmp_path / "locks"))
        db_path = tmp_path / "same-pid.db"
        pidfile = _writer_pidfile_path(db_path)
        pidfile.parent.mkdir(parents=True)
        pidfile.write_bytes(_pidfile_payload(os.getpid()))

        with pytest.raises(RuntimeError, match="another writer is using"):
            with _code_intelligence_writer_lock(db_path, dry_run=False):
                pass

        assert pidfile.exists()
