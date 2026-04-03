"""Tests for code_intelligence.py — project entity extraction from repo metadata."""

import json
import sqlite3
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

    def test_nonexistent_dir_returns_empty(self, tmp_path: Path) -> None:
        from brainlayer.pipeline.code_intelligence import scan_projects

        projects = scan_projects(tmp_path / "nonexistent")
        assert projects == []


class TestEnrichProjects:
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
