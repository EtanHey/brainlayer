"""Code Intelligence — auto-populate project entities from repo metadata.

Scans ~/Gits/ for repos with pyproject.toml or package.json, extracts
metadata (name, version, description, dependencies, scripts, language),
and upserts structured project entities into the KG.

Usage:
    python -m brainlayer.pipeline.code_intelligence [--base-dir ~/Gits] [--dry-run]
"""

import json
import logging
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_BASE_DIR = Path.home() / "Gits"


def scan_projects(base_dir: Path) -> list[dict[str, Any]]:
    """Scan a directory for repos with pyproject.toml or package.json.

    Returns a list of project metadata dicts.
    """
    projects = []
    if not base_dir.is_dir():
        logger.warning("Base directory %s does not exist", base_dir)
        return projects

    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue

        meta = _extract_metadata(entry)
        if meta:
            projects.append(meta)

    return projects


def _extract_metadata(repo_path: Path) -> dict[str, Any] | None:
    """Extract project metadata from pyproject.toml or package.json."""
    pyproject = repo_path / "pyproject.toml"
    packagejson = repo_path / "package.json"

    if pyproject.exists():
        return _extract_pyproject(repo_path, pyproject)
    elif packagejson.exists():
        return _extract_package_json(repo_path, packagejson)
    return None


def _extract_pyproject(repo_path: Path, path: Path) -> dict[str, Any]:
    """Extract metadata from pyproject.toml."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    with open(path, "rb") as f:
        data = tomllib.load(f)

    project = data.get("project", {})
    name = project.get("name", repo_path.name)
    version = project.get("version")
    description = project.get("description", "")

    # Extract dependency names (strip version specifiers)
    raw_deps = project.get("dependencies", [])
    deps = [re.split(r"[>=<~!\[;]", d)[0].strip() for d in raw_deps]

    scripts = list(project.get("scripts", {}).keys())

    return {
        "name": name,
        "repo_dir": str(repo_path),
        "version": version,
        "description": description,
        "language": "python",
        "package_manager": "pip",
        "dependencies": deps,
        "scripts": scripts,
        "config_file": "pyproject.toml",
    }


def _extract_package_json(repo_path: Path, path: Path) -> dict[str, Any]:
    """Extract metadata from package.json."""
    with open(path) as f:
        data = json.load(f)

    name = data.get("name", repo_path.name)
    version = data.get("version")
    description = data.get("description", "")

    deps = list(data.get("dependencies", {}).keys())
    dev_deps = list(data.get("devDependencies", {}).keys())
    scripts = list(data.get("scripts", {}).keys())

    # Detect package manager
    pkg_manager = "npm"
    if (repo_path / "bun.lockb").exists() or (repo_path / "bun.lock").exists():
        pkg_manager = "bun"
    elif (repo_path / "pnpm-lock.yaml").exists():
        pkg_manager = "pnpm"
    elif (repo_path / "yarn.lock").exists():
        pkg_manager = "yarn"

    # Detect framework from dependencies
    framework = _detect_framework(deps)

    result: dict[str, Any] = {
        "name": name,
        "repo_dir": str(repo_path),
        "version": version,
        "description": description,
        "language": "typescript" if (repo_path / "tsconfig.json").exists() else "javascript",
        "package_manager": pkg_manager,
        "dependencies": deps,
        "dev_dependencies": dev_deps,
        "scripts": scripts,
        "config_file": "package.json",
    }
    if framework:
        result["framework"] = framework
    return result


def _detect_framework(deps: list[str]) -> str | None:
    """Detect major framework from dependency list.

    Order matters: check specific frameworks before generic ones
    (e.g., react-native/expo before react, next before react).
    """
    # Ordered: most specific first
    framework_signals = [
        ("expo", "Expo"),
        ("react-native", "React Native"),
        ("next", "Next.js"),
        ("convex", "Convex"),
        ("@modelcontextprotocol/sdk", "MCP"),
        ("express", "Express"),
        ("fastify", "Fastify"),
        ("react", "React"),
    ]
    for dep, name in framework_signals:
        if dep in deps:
            return name
    return None


def enrich_projects(
    db_path: str | None = None,
    base_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Scan repos and enrich project entities in the KG.

    Returns stats dict with counts of created/updated entities and relations.
    """
    if db_path is None:
        from ..paths import get_db_path

        db_path = get_db_path()

    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR

    projects = scan_projects(base_dir)
    if not projects:
        logger.warning("No projects found in %s", base_dir)
        return {
            "projects_scanned": 0,
            "entities_created": 0,
            "entities_updated": 0,
            "relations_added": 0,
            "dep_entities_created": 0,
        }

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")

    stats = {
        "projects_scanned": len(projects),
        "entities_created": 0,
        "entities_updated": 0,
        "relations_added": 0,
        "dep_entities_created": 0,
    }

    try:
        for project in projects:
            _upsert_project(conn, project, stats, dry_run)

        if not dry_run:
            conn.commit()
    finally:
        conn.close()

    return stats


def _upsert_project(
    conn: sqlite3.Connection,
    project: dict[str, Any],
    stats: dict[str, int],
    dry_run: bool,
) -> None:
    """Create or update a project entity with its metadata and relations."""
    name = project["name"]

    # Check if entity already exists
    row = conn.execute(
        "SELECT id, metadata FROM kg_entities WHERE LOWER(name) = LOWER(?) AND entity_type = 'project'",
        (name,),
    ).fetchone()

    metadata = {
        "repo_dir": project["repo_dir"],
        "version": project.get("version"),
        "language": project["language"],
        "package_manager": project["package_manager"],
        "config_file": project["config_file"],
        "scripts": project.get("scripts", []),
        "dependency_count": len(project.get("dependencies", [])),
    }
    if project.get("framework"):
        metadata["framework"] = project["framework"]

    metadata_json = json.dumps(metadata)
    description = project.get("description", "")
    importance = 6.0  # Projects are moderate-high importance

    if row:
        entity_id = row[0]
        if not dry_run:
            conn.execute(
                """UPDATE kg_entities
                   SET description = ?, metadata = ?, importance = ?,
                       updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                   WHERE id = ?""",
                (description, metadata_json, importance, entity_id),
            )
        stats["entities_updated"] += 1
        logger.info("Updated project: %s (id=%s)", name, entity_id)
    else:
        entity_id = f"proj-{uuid.uuid4().hex[:12]}"
        if not dry_run:
            conn.execute(
                """INSERT INTO kg_entities (id, entity_type, name, description, metadata, importance, created_at, updated_at)
                   VALUES (?, 'project', ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'), strftime('%Y-%m-%dT%H:%M:%fZ','now'))""",
                (entity_id, name, description, metadata_json, importance),
            )
        stats["entities_created"] += 1
        logger.info("Created project: %s (id=%s)", name, entity_id)

    # Add depends_on relations for key dependencies
    for dep_name in project.get("dependencies", []):
        _add_dependency_relation(conn, entity_id, name, dep_name, stats, dry_run)


def _add_dependency_relation(
    conn: sqlite3.Connection,
    source_id: str,
    source_name: str,
    dep_name: str,
    stats: dict[str, int],
    dry_run: bool,
) -> None:
    """Add a depends_on relation from project to dependency.

    Creates the dependency as a 'library' entity if it doesn't exist.
    Only creates relations for notable dependencies (frameworks, SDKs, key tools).
    """
    # Only track notable dependencies to avoid noise
    notable_patterns = {
        # Frameworks & runtimes
        "react",
        "react-dom",
        "react-native",
        "next",
        "express",
        "fastify",
        "convex",
        "expo",
        "electron",
        # AI/ML
        "@anthropic-ai/sdk",
        "@modelcontextprotocol/sdk",
        "openai",
        "langchain",
        "sentence-transformers",
        "onnxruntime-node",
        # Databases
        "apsw",
        "sqlite-vec",
        "prisma",
        "@prisma/client",
        "drizzle-orm",
        # Infra
        "fastapi",
        "uvicorn",
        "@axiomhq/js",
        # Python notable
        "pyyaml",
        "typer",
        "rich",
        "textual",
        "mcp",
    }

    if dep_name not in notable_patterns:
        return

    # Find or create the dependency entity
    target_row = conn.execute(
        "SELECT id FROM kg_entities WHERE LOWER(name) = LOWER(?)",
        (dep_name,),
    ).fetchone()

    if not target_row:
        target_id = f"lib-{uuid.uuid4().hex[:12]}"
        if not dry_run:
            conn.execute(
                """INSERT INTO kg_entities (id, entity_type, name, importance, created_at, updated_at)
                   VALUES (?, 'library', ?, 3.0, strftime('%Y-%m-%dT%H:%M:%fZ','now'), strftime('%Y-%m-%dT%H:%M:%fZ','now'))""",
                (target_id, dep_name),
            )
        stats["dep_entities_created"] += 1
    else:
        target_id = target_row[0]

    # Check if relation already exists
    existing = conn.execute(
        "SELECT id FROM kg_relations WHERE source_id = ? AND target_id = ? AND relation_type = 'depends_on'",
        (source_id, target_id),
    ).fetchone()

    if existing:
        return

    rel_id = f"rel-{uuid.uuid4().hex[:12]}"
    props = json.dumps({"source": "code_intelligence"})

    if not dry_run:
        conn.execute(
            """INSERT OR IGNORE INTO kg_relations
               (id, source_id, target_id, relation_type, properties, confidence, created_at)
               VALUES (?, ?, ?, 'depends_on', ?, 0.99, strftime('%Y-%m-%dT%H:%M:%fZ','now'))""",
            (rel_id, source_id, target_id, props),
        )
    stats["relations_added"] += 1
    logger.info("Added relation: %s --depends_on--> %s", source_name, dep_name)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Populate project entities from repo metadata")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR, help="Directory containing repos")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying")
    args = parser.parse_args()

    result = enrich_projects(base_dir=args.base_dir, dry_run=args.dry_run)
    prefix = "[DRY RUN] " if args.dry_run else ""
    print(f"\n{prefix}Code intelligence scan complete:")
    print(f"  Projects scanned:      {result['projects_scanned']}")
    print(f"  Entities created:      {result['entities_created']}")
    print(f"  Entities updated:      {result['entities_updated']}")
    print(f"  Relations added:       {result['relations_added']}")
    print(f"  Library entities added: {result['dep_entities_created']}")
