"""Code Intelligence — auto-populate project entities from repo metadata.

Scans ~/Gits/ for repos with pyproject.toml or package.json, extracts
metadata (name, version, description, dependencies, scripts, language),
and upserts structured project entities into the KG.

Usage:
    python -m brainlayer.pipeline.code_intelligence [--base-dir ~/Gits] [--dry-run]
"""

import fcntl
import hashlib
import json
import logging
import os
import re
import sqlite3
import subprocess
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_BASE_DIR = Path.home() / "Gits"
OBSERVATION_VALIDITY_DAYS = 30
CODE_INTELLIGENCE_SOURCE = "code_intelligence"
SQLITE_BUSY_TIMEOUT_MS = 30_000
NOTABLE_DEPENDENCIES = {
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


def _timestamp(dt: datetime) -> str:
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return f"{dt:%Y-%m-%dT%H:%M:%S}.{dt.microsecond // 1000:03d}Z"


def _load_relation_properties(properties: str | None) -> dict[str, Any]:
    if not properties:
        return {}
    try:
        loaded = json.loads(properties)
    except (TypeError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _is_code_intelligence_relation(properties: str | None) -> bool:
    loaded = _load_relation_properties(properties)
    source = loaded.get("source")
    sources = loaded.get("sources")
    return (
        source == CODE_INTELLIGENCE_SOURCE
        or (isinstance(source, list) and CODE_INTELLIGENCE_SOURCE in source)
        or (isinstance(sources, list) and CODE_INTELLIGENCE_SOURCE in sources)
    )


def _code_intelligence_properties(properties: str | None = None) -> str:
    loaded = _load_relation_properties(properties)
    source = loaded.get("source")
    sources = loaded.get("sources")
    if source is None and sources is None:
        loaded["source"] = CODE_INTELLIGENCE_SOURCE
    elif isinstance(source, list):
        if CODE_INTELLIGENCE_SOURCE not in source:
            loaded["source"] = [*source, CODE_INTELLIGENCE_SOURCE]
    elif source != CODE_INTELLIGENCE_SOURCE:
        existing_sources = sources if isinstance(sources, list) else []
        loaded["sources"] = [*dict.fromkeys([*existing_sources, source, CODE_INTELLIGENCE_SOURCE])]
    return json.dumps(loaded, sort_keys=True)


def _writer_pidfile_path(db_path: str | Path) -> Path:
    pidfile_dir = Path(os.environ.get("BRAINLAYER_WRITER_PIDFILE_DIR", "/tmp")).expanduser()
    if not pidfile_dir.is_absolute():
        pidfile_dir = Path("/tmp") / pidfile_dir
    resolved_path = Path(db_path).expanduser().resolve()
    path_hash = hashlib.sha256(str(resolved_path).encode("utf-8")).hexdigest()[:16]
    return pidfile_dir.resolve() / f"brainlayer-writer-{path_hash}-{resolved_path.name}.pid"


def _pid_start_time(pid: int) -> str | None:
    try:
        stat_path = Path("/proc") / str(pid) / "stat"
        if stat_path.exists():
            fields = stat_path.read_text(encoding="utf-8").split()
            if len(fields) > 21:
                return fields[21]
    except OSError:
        pass
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            check=False,
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return " ".join(result.stdout.split()) or None


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pidfile_payload(pid: int) -> bytes:
    start_time = _pid_start_time(pid)
    if start_time:
        return f"{pid}\nstart_time={start_time}\n".encode("utf-8")
    return f"{pid}\n".encode("utf-8")


def _read_pidfile_owner(pidfile: Path) -> tuple[int | None, str | None]:
    try:
        lines = pidfile.read_text(encoding="utf-8").splitlines()
        if not lines:
            return None, None
        pid = int(lines[0].strip())
        start_time = None
        for line in lines[1:]:
            if line.startswith("start_time="):
                start_time = line.removeprefix("start_time=").strip() or None
                break
        return pid, start_time
    except (OSError, ValueError):
        return None, None


def _pidfile_owner_matches(pid: int, start_time: str | None) -> bool:
    if not _pid_is_alive(pid):
        return False
    if not start_time:
        return True
    current_start_time = _pid_start_time(pid)
    return current_start_time is None or current_start_time == start_time


@contextmanager
def _code_intelligence_writer_lock(db_path: str | Path, dry_run: bool):
    if dry_run:
        yield
        return

    pidfile = _writer_pidfile_path(db_path)
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    acquired = False

    for attempt in range(4):
        try:
            fd = os.open(pidfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                os.write(fd, _pidfile_payload(pid))
            finally:
                os.close(fd)
            acquired = True
            break
        except FileExistsError:
            owner_pid, owner_start_time = _read_pidfile_owner(pidfile)
            if owner_pid is not None and _pidfile_owner_matches(owner_pid, owner_start_time):
                raise RuntimeError(f"another writer is using {db_path} (pid {owner_pid})")
            try:
                pidfile.unlink()
            except FileNotFoundError:
                pass
            time.sleep(0.01 * (attempt + 1))

    if not acquired:
        raise RuntimeError(f"could not acquire writer pidfile for {db_path}")

    try:
        yield
    finally:
        if _read_pidfile_owner(pidfile)[0] == pid:
            try:
                pidfile.unlink()
            except FileNotFoundError:
                pass


def _ensure_reconciliation_schema(conn: sqlite3.Connection) -> None:
    """Apply the KG validity columns needed by code-intelligence reconciliation."""
    entity_cols = {row[1] for row in conn.execute("PRAGMA table_info(kg_entities)")}
    if "valid_until" not in entity_cols:
        conn.execute("ALTER TABLE kg_entities ADD COLUMN valid_until TEXT")
    if "expired_at" not in entity_cols:
        conn.execute("ALTER TABLE kg_entities ADD COLUMN expired_at TEXT")

    relation_cols = {row[1] for row in conn.execute("PRAGMA table_info(kg_relations)")}
    if "valid_from" not in relation_cols:
        conn.execute("ALTER TABLE kg_relations ADD COLUMN valid_from TEXT")
    if "valid_until" not in relation_cols:
        conn.execute("ALTER TABLE kg_relations ADD COLUMN valid_until TEXT")
    if "expired_at" not in relation_cols:
        conn.execute("ALTER TABLE kg_relations ADD COLUMN expired_at TEXT")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_kg_entities_expired ON kg_entities(expired_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_kg_relations_expired ON kg_relations(expired_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_kg_relations_validity ON kg_relations(valid_from, valid_until)")
    conn.execute("DROP VIEW IF EXISTS kg_current_facts")
    conn.execute("""
        CREATE VIEW IF NOT EXISTS kg_current_facts AS
        SELECT * FROM kg_relations
        WHERE (valid_from IS NULL OR valid_from <= strftime('%Y-%m-%dT%H:%M:%fZ','now'))
          AND (valid_until IS NULL OR valid_until >= strftime('%Y-%m-%dT%H:%M:%fZ','now'))
          AND expired_at IS NULL
    """)


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
            "relations_expired": 0,
            "dep_entities_expired": 0,
        }

    observed_dt = datetime.now(timezone.utc)
    observed_at = _timestamp(observed_dt)
    valid_until = _timestamp(observed_dt + timedelta(days=OBSERVATION_VALIDITY_DAYS))

    stats = {
        "projects_scanned": len(projects),
        "entities_created": 0,
        "entities_updated": 0,
        "relations_added": 0,
        "dep_entities_created": 0,
        "relations_expired": 0,
        "dep_entities_expired": 0,
    }

    with _code_intelligence_writer_lock(db_path, dry_run):
        conn = sqlite3.connect(db_path, timeout=SQLITE_BUSY_TIMEOUT_MS / 1000)
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA journal_mode = WAL")
        if not dry_run:
            conn.execute("PRAGMA wal_checkpoint(FULL)")
            _ensure_reconciliation_schema(conn)

        try:
            for project in projects:
                _upsert_project(conn, project, stats, dry_run, observed_at, valid_until)

            if not dry_run:
                conn.commit()
                conn.execute("PRAGMA wal_checkpoint(FULL)")
        finally:
            conn.close()

    return stats


def _upsert_project(
    conn: sqlite3.Connection,
    project: dict[str, Any],
    stats: dict[str, int],
    dry_run: bool,
    observed_at: str,
    valid_until: str,
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
                       valid_until = ?, expired_at = NULL,
                       updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                   WHERE id = ?""",
                (description, metadata_json, importance, valid_until, entity_id),
            )
        stats["entities_updated"] += 1
        logger.info("Updated project: %s (id=%s)", name, entity_id)
    else:
        entity_id = f"proj-{uuid.uuid4().hex[:12]}"
        if not dry_run:
            conn.execute(
                """INSERT INTO kg_entities
                   (id, entity_type, name, description, metadata, importance, valid_until, expired_at, created_at, updated_at)
                   VALUES (?, 'project', ?, ?, ?, ?, ?, NULL, strftime('%Y-%m-%dT%H:%M:%fZ','now'), strftime('%Y-%m-%dT%H:%M:%fZ','now'))""",
                (entity_id, name, description, metadata_json, importance, valid_until),
            )
        stats["entities_created"] += 1
        logger.info("Created project: %s (id=%s)", name, entity_id)

    # Add depends_on relations for key dependencies
    observed_targets: set[str] = set()
    for dep_name in project.get("dependencies", []):
        target_id = _add_dependency_relation(conn, entity_id, name, dep_name, stats, dry_run, observed_at, valid_until)
        if target_id is not None:
            observed_targets.add(target_id)

    _expire_stale_dependency_relations(conn, entity_id, observed_targets, stats, dry_run, observed_at)


def _add_dependency_relation(
    conn: sqlite3.Connection,
    source_id: str,
    source_name: str,
    dep_name: str,
    stats: dict[str, int],
    dry_run: bool,
    observed_at: str,
    valid_until: str,
) -> str | None:
    """Add a depends_on relation from project to dependency.

    Creates the dependency as a 'library' entity if it doesn't exist.
    Only creates relations for notable dependencies (frameworks, SDKs, key tools).
    """
    if dep_name not in NOTABLE_DEPENDENCIES:
        return None

    # Find or create the dependency entity
    target_row = conn.execute(
        "SELECT id FROM kg_entities WHERE LOWER(name) = LOWER(?)",
        (dep_name,),
    ).fetchone()

    if not target_row:
        target_id = f"lib-{uuid.uuid4().hex[:12]}"
        if not dry_run:
            conn.execute(
                """INSERT INTO kg_entities
                   (id, entity_type, name, importance, valid_until, expired_at, created_at, updated_at)
                   VALUES (?, 'library', ?, 3.0, ?, NULL, strftime('%Y-%m-%dT%H:%M:%fZ','now'), strftime('%Y-%m-%dT%H:%M:%fZ','now'))""",
                (target_id, dep_name, valid_until),
            )
        stats["dep_entities_created"] += 1
    else:
        target_id = target_row[0]
        if not dry_run:
            conn.execute(
                """UPDATE kg_entities
                   SET valid_until = ?, expired_at = NULL,
                       updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                   WHERE id = ?""",
                (valid_until, target_id),
            )

    # Check if relation already exists
    existing = conn.execute(
        "SELECT id, properties FROM kg_relations WHERE source_id = ? AND target_id = ? AND relation_type = 'depends_on'",
        (source_id, target_id),
    ).fetchone()

    if existing:
        if not _is_code_intelligence_relation(existing[1]):
            return target_id
        if not dry_run:
            conn.execute(
                """UPDATE kg_relations
                   SET properties = ?, confidence = 0.99, valid_from = COALESCE(valid_from, ?),
                       valid_until = ?, expired_at = NULL
                   WHERE id = ?""",
                (_code_intelligence_properties(existing[1]), observed_at, valid_until, existing[0]),
            )
        return target_id

    rel_id = f"rel-{uuid.uuid4().hex[:12]}"
    props = _code_intelligence_properties()

    if not dry_run:
        conn.execute(
            """INSERT OR IGNORE INTO kg_relations
               (id, source_id, target_id, relation_type, properties, confidence, valid_from, valid_until, expired_at, created_at)
               VALUES (?, ?, ?, 'depends_on', ?, 0.99, ?, ?, NULL, strftime('%Y-%m-%dT%H:%M:%fZ','now'))""",
            (rel_id, source_id, target_id, props, observed_at, valid_until),
        )
    stats["relations_added"] += 1
    logger.info("Added relation: %s --depends_on--> %s", source_name, dep_name)
    return target_id


def _expire_stale_dependency_relations(
    conn: sqlite3.Connection,
    source_id: str,
    observed_targets: set[str],
    stats: dict[str, int],
    dry_run: bool,
    observed_at: str,
) -> None:
    """Expire code_intelligence depends_on facts absent from the current project scan."""
    if dry_run:
        return

    stale_target_ids: set[str] = set()
    rows = conn.execute(
        """
        SELECT id, target_id, properties, expired_at
        FROM kg_relations
        WHERE source_id = ? AND relation_type = 'depends_on'
        """,
        (source_id,),
    ).fetchall()

    for rel_id, target_id, properties, expired_at in rows:
        if target_id in observed_targets or not _is_code_intelligence_relation(properties):
            continue
        stale_target_ids.add(target_id)
        if expired_at is None:
            conn.execute(
                "UPDATE kg_relations SET valid_until = ?, expired_at = ? WHERE id = ?",
                (observed_at, observed_at, rel_id),
            )
            stats["relations_expired"] += 1

    for target_id in stale_target_ids:
        active_depends_on = conn.execute(
            """
            SELECT 1
            FROM kg_current_facts
            WHERE target_id = ?
              AND relation_type = 'depends_on'
            LIMIT 1
            """,
            (target_id,),
        ).fetchone()
        if active_depends_on:
            continue
        result = conn.execute(
            """
            UPDATE kg_entities
            SET valid_until = ?, expired_at = COALESCE(expired_at, ?),
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE id = ? AND entity_type = 'library'
            """,
            (observed_at, observed_at, target_id),
        )
        stats["dep_entities_expired"] += result.rowcount


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
