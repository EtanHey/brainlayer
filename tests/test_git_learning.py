"""Tests for Phase 3 git-based continual learning MVP."""

import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from brainlayer.cli import app
from brainlayer.vector_store import VectorStore

runner = CliRunner()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")


def _commit_file(repo: Path, rel_path: str, content: str, message: str) -> str:
    path = repo / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    _git(repo, "add", rel_path)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def test_git_learning_schema_tables_exist(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)

    cursor = store.conn.cursor()
    tables = {row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}

    assert "git_memories" in tables
    assert "migration_events" in tables
    assert "file_cochanges" in tables

    fts_tables = {row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "git_memories_fts" in fts_tables


def test_parse_commit_message_detects_conventional_and_breaking():
    from brainlayer.git_learning import parse_commit_message

    parsed = parse_commit_message(
        "feat(parser)!: switch from REST API to GraphQL\n\nBREAKING CHANGE: remove old client"
    )

    assert parsed.kind == "feat"
    assert parsed.scope == "parser"
    assert parsed.breaking is True
    assert parsed.memory_type == "migration"


def test_brain_learn_git_seeds_single_repo_without_duplicates(tmp_path, monkeypatch):
    from brainlayer.git_learning import ensure_repos_config

    repo = tmp_path / "repo-one"
    _init_repo(repo)
    _commit_file(repo, "src/app.py", "print('one')\n", "feat: add app bootstrap")
    _commit_file(repo, "src/app.py", "print('two')\n", "fix: handle startup failure")

    db_path = tmp_path / "brainlayer.db"
    config_path = tmp_path / "repos.json"
    ensure_repos_config(config_path, default_repos=[str(repo)])

    monkeypatch.setenv("BRAINLAYER_DB", str(db_path))

    first = runner.invoke(app, ["brain-learn", "--git", "--repos-config", str(config_path), "--since", "30d"])
    assert first.exit_code == 0, first.stdout

    second = runner.invoke(app, ["brain-learn", "--git", "--repos-config", str(config_path), "--since", "30d"])
    assert second.exit_code == 0, second.stdout

    store = VectorStore(db_path)
    cursor = store.conn.cursor()
    count = cursor.execute("SELECT COUNT(*) FROM git_memories").fetchone()[0]
    repos = {row[0] for row in cursor.execute("SELECT DISTINCT repo FROM git_memories")}

    assert count == 2
    assert repos == {repo.name}


def test_migration_commit_weakens_old_pattern_refs_and_logs_event(tmp_path):
    from brainlayer.git_learning import apply_migration_invalidation

    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)
    cursor = store.conn.cursor()
    cursor.execute(
        """
        INSERT INTO git_memories(
            id, content, memory_type, commit_hash, repo, author, committed_at,
            affected_files, strength, half_life_days, confidence, retrieval_count, invalidated_by,
            commit_message, tags, importance
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "gm-old",
            "Use REST API client adapters for new integrations",
            "pattern",
            "abc123",
            "repo-one",
            "Tester",
            1_700_000_000.0,
            json.dumps(["src/api.py"]),
            1.0,
            30.0,
            0.8,
            0,
            None,
            "feat: add REST API client",
            json.dumps(["api", "rest"]),
            5.0,
        ),
    )

    weakened = apply_migration_invalidation(
        store=store,
        repo="repo-one",
        commit_hash="def456",
        commit_message="refactor: migrate from REST API to GraphQL",
        committed_at=1_700_000_100.0,
    )

    row = cursor.execute(
        "SELECT strength, invalidated_by FROM git_memories WHERE id = ?",
        ("gm-old",),
    ).fetchone()
    event = cursor.execute(
        "SELECT from_pattern, to_pattern, memories_weakened FROM migration_events WHERE commit_hash = ?",
        ("def456",),
    ).fetchone()

    assert weakened == 1
    assert row == (0.3, "migration-def456")
    assert event == ("REST API", "GraphQL", 1)


def test_cross_repo_consolidation_boosts_importance():
    from brainlayer.git_learning import compute_cross_repo_importance

    assert compute_cross_repo_importance(base_importance=5.0, repo_count=2) == 5.0
    assert compute_cross_repo_importance(base_importance=5.0, repo_count=3) == 7.0


def test_repos_config_is_user_editable(tmp_path):
    from brainlayer.git_learning import ensure_repos_config, load_repos_config

    config_path = tmp_path / "repos.json"
    ensure_repos_config(config_path, default_repos=["/tmp/repo-a"])

    config_path.write_text(json.dumps({"repos": ["/tmp/repo-b", "/tmp/repo-c"]}, indent=2) + "\n")

    assert load_repos_config(config_path) == [Path("/tmp/repo-b"), Path("/tmp/repo-c")]


def test_git_memories_fts_tracks_inserted_rows(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)
    cursor = store.conn.cursor()
    cursor.execute(
        """
        INSERT INTO git_memories(
            id, content, memory_type, commit_hash, repo, author, committed_at,
            affected_files, strength, half_life_days, confidence, retrieval_count,
            invalidated_by, commit_message, tags, importance
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "gm-fts",
            "Adopt GraphQL for external APIs",
            "migration",
            "ghi789",
            "repo-two",
            "Tester",
            1_700_000_200.0,
            json.dumps(["src/graphql.py"]),
            1.0,
            90.0,
            0.9,
            0,
            None,
            "refactor: migrate from REST API to GraphQL",
            json.dumps(["graphql", "migration"]),
            7.0,
        ),
    )

    row = cursor.execute(
        "SELECT git_memory_id FROM git_memories_fts WHERE git_memories_fts MATCH ?",
        ("GraphQL",),
    ).fetchone()
    assert row == ("gm-fts",)
