"""Git-based continual learning helpers for BrainLayer."""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .vector_store import VectorStore

CONVENTIONAL_RE = re.compile(
    r"^(?P<kind>feat|fix|refactor)(?:\((?P<scope>[^)]+)\))?(?P<breaking>!)?:\s*(?P<summary>.+)$"
)
MIGRATION_PATTERNS = [
    re.compile(r"migrate from (?P<old>.+?) to (?P<new>.+)", re.IGNORECASE),
    re.compile(r"replace (?P<old>.+?) with (?P<new>.+)", re.IGNORECASE),
    re.compile(r"switch from (?P<old>.+?) to (?P<new>.+)", re.IGNORECASE),
]
BREAKING_RE = re.compile(r"BREAKING CHANGE:", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedCommit:
    kind: str | None
    scope: str | None
    summary: str
    breaking: bool
    memory_type: str


def get_repos_config_path() -> Path:
    """Return the config path for tracked git repositories."""
    env = os.environ.get("BRAINLAYER_REPOS")
    if env:
        return Path(env)
    return Path.home() / ".brainlayer" / "repos.json"


def ensure_repos_config(path: Path | None = None, default_repos: list[str] | None = None) -> Path:
    """Ensure the repos config exists and is valid JSON."""
    target = path or get_repos_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        payload = {"repos": default_repos or []}
        target.write_text(json.dumps(payload, indent=2) + "\n")
    return target


def load_repos_config(path: Path | None = None) -> list[Path]:
    """Load tracked repos from the user-editable config."""
    target = ensure_repos_config(path)
    payload = json.loads(target.read_text() or "{}")
    repos = payload.get("repos", [])
    return [Path(repo).expanduser() for repo in repos]


def parse_commit_message(message: str) -> ParsedCommit:
    """Classify a commit message using a conventional-commit-first heuristic."""
    message = message.strip()
    first_line = message.splitlines()[0] if message else ""
    conventional = CONVENTIONAL_RE.match(first_line)
    breaking = bool(BREAKING_RE.search(message)) or (bool(conventional) and bool(conventional.group("breaking")))
    if _extract_migration(message):
        return ParsedCommit(
            kind=conventional.group("kind") if conventional else None,
            scope=conventional.group("scope") if conventional else None,
            summary=conventional.group("summary") if conventional else first_line,
            breaking=breaking,
            memory_type="migration",
        )
    if conventional:
        kind = conventional.group("kind")
        memory_type = {"feat": "pattern", "fix": "error", "refactor": "decision"}[kind]
        return ParsedCommit(
            kind=kind,
            scope=conventional.group("scope"),
            summary=conventional.group("summary"),
            breaking=breaking,
            memory_type=memory_type,
        )
    return ParsedCommit(kind=None, scope=None, summary=first_line, breaking=breaking, memory_type="decision")


def compute_cross_repo_importance(base_importance: float, repo_count: int) -> float:
    """Boost importance when the same pattern shows up across many repos."""
    if repo_count < 3:
        return base_importance
    return base_importance + 2.0


def apply_migration_invalidation(
    store: VectorStore,
    repo: str,
    commit_hash: str,
    commit_message: str,
    committed_at: float,
) -> int:
    """Weaken old git-memory pattern references when a migration commit is detected."""
    migration = _extract_migration(commit_message)
    if migration is None:
        return 0

    old_pattern, new_pattern = migration
    invalidation_id = f"migration-{commit_hash}"
    cursor = store.conn.cursor()
    rowcount = (
        cursor.execute(
            """
        UPDATE git_memories
        SET strength = ROUND(strength * 0.3, 6),
            invalidated_by = ?
        WHERE invalidated_by IS NULL
          AND (
                lower(content) LIKE lower(?)
             OR lower(commit_message) LIKE lower(?)
             OR lower(COALESCE(tags, '')) LIKE lower(?)
          )
        """,
            (invalidation_id, f"%{old_pattern}%", f"%{old_pattern}%", f"%{old_pattern}%"),
        )
        .getconnection()
        .changes()
    )
    cursor.execute(
        """
        INSERT OR REPLACE INTO migration_events(
            id, from_pattern, to_pattern, commit_hash, repo, detected_at, confidence, memories_weakened
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (invalidation_id, old_pattern, new_pattern, commit_hash, repo, committed_at, 0.9, rowcount),
    )
    return rowcount


def learn_git_history(store: VectorStore, repos: list[Path], since: str = "30d") -> dict[str, Any]:
    """Ingest recent conventional commits from tracked repositories."""
    learned = 0
    skipped = 0
    invalidations = 0
    for repo in repos:
        for commit in _iter_commits(repo, since):
            cursor = store.conn.cursor()
            existing = cursor.execute(
                "SELECT 1 FROM git_memories WHERE repo = ? AND commit_hash = ? LIMIT 1",
                (repo.name, commit["commit_hash"]),
            ).fetchone()
            if existing:
                skipped += 1
                continue

            parsed = parse_commit_message(commit["message"])
            importance = 7.0 if parsed.memory_type == "migration" else 5.0
            cursor.execute(
                """
                INSERT INTO git_memories(
                    id, content, memory_type, commit_hash, repo, author, committed_at,
                    affected_files, strength, half_life_days, confidence, retrieval_count,
                    invalidated_by, commit_message, tags, importance
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"git-{uuid.uuid4().hex[:16]}",
                    parsed.summary,
                    parsed.memory_type,
                    commit["commit_hash"],
                    repo.name,
                    commit["author"],
                    commit["committed_at"],
                    json.dumps(commit["files"]),
                    1.0,
                    90.0 if parsed.memory_type == "migration" else 30.0,
                    0.7,
                    0,
                    None,
                    commit["message"],
                    json.dumps(_build_tags(parsed, repo.name)),
                    compute_cross_repo_importance(importance, 1),
                ),
            )
            learned += 1
            invalidations += apply_migration_invalidation(
                store=store,
                repo=repo.name,
                commit_hash=commit["commit_hash"],
                commit_message=commit["message"],
                committed_at=commit["committed_at"],
            )
    return {"learned": learned, "skipped": skipped, "invalidations": invalidations}


def _build_tags(parsed: ParsedCommit, repo_name: str) -> list[str]:
    tags = ["git", repo_name]
    if parsed.kind:
        tags.append(parsed.kind)
    if parsed.breaking:
        tags.append("breaking")
    if parsed.memory_type == "migration":
        tags.append("migration")
    return tags


def _extract_migration(message: str) -> tuple[str, str] | None:
    for pattern in MIGRATION_PATTERNS:
        match = pattern.search(message)
        if match:
            return match.group("old").strip(), match.group("new").strip()
    return None


def _iter_commits(repo: Path, since: str) -> list[dict[str, Any]]:
    commits: list[dict[str, Any]] = []
    hashes = subprocess.run(
        ["git", "log", "--no-merges", f"--since={_git_since_arg(since)}", "--format=%H"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    for commit_hash in [line.strip() for line in hashes.stdout.splitlines() if line.strip()]:
        metadata = subprocess.run(
            ["git", "show", "-s", "--format=%an%x1f%at%x1f%s%x1f%b", commit_hash],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        author, committed_at, subject, body = metadata.stdout.split("\x1f", maxsplit=3)
        file_listing = subprocess.run(
            ["git", "show", "--pretty=format:", "--name-only", commit_hash],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        files = [line.strip() for line in file_listing.stdout.splitlines() if line.strip()]
        commits.append(
            {
                "commit_hash": commit_hash,
                "author": author,
                "committed_at": float(committed_at),
                "message": subject if not body.strip() else f"{subject}\n\n{body.strip()}",
                "files": files,
            }
        )
    return commits


def _git_since_arg(since: str) -> str:
    match = re.fullmatch(r"(\d+)d", since)
    if match:
        return f"{match.group(1)} days ago"
    return since
