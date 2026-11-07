"""#1349: first-calendar policy when the configured runtime DB does not exist yet.

Policy: the calendar `brainlayer index` pass never creates a database. An absent DB is a loud,
actionable rc78 `INDEX_BOOTSTRAP_REQUIRED` -- before the checkpoint sidecar or the writer is built --
and `brainlayer bootstrap-db` is the one explicit creator, which refuses any existing path. Every DB here is a synthetic scratch file under tmp_path.
"""

import os
import threading

import pytest
from typer.testing import CliRunner

import brainlayer.cli as cli
import brainlayer.runtime_store as runtime
import brainlayer.vector_store as storage
from brainlayer.runtime_store import (
    EXPECTED_RUNTIME_SCHEMA_FINGERPRINT,
    ReadonlyStore,
    RuntimeBootstrapRefused,
    bootstrap_runtime_database,
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(tmp_path / "pidfiles"))
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY_PATH", str(tmp_path / "telemetry.jsonl"))
    monkeypatch.delenv("BRAINLAYER_RUNTIME_STORE", raising=False)
    monkeypatch.setattr(storage, "_write_busy_timeout_ms", lambda: 5)
    monkeypatch.setattr(runtime, "_write_busy_timeout_ms", lambda: 5)
    projects = tmp_path / "projects"
    projects.mkdir()
    (projects / "a.jsonl").write_text("{}\n")
    db = tmp_path / "data" / "brainlayer.db"
    monkeypatch.setattr(cli, "get_db_path", lambda: db)
    monkeypatch.setattr("brainlayer.paths.get_db_path", lambda: db)
    monkeypatch.setattr(cli, "_calendar_index_budget_s", lambda: 60)
    monkeypatch.setattr("brainlayer.pipeline.extract.parse_jsonl", lambda p: [])
    return tmp_path, projects, db


def calendar_index(monkeypatch, projects):
    monkeypatch.setenv("BRAINLAYER_LAUNCHD_SERVICE", "index")
    return CliRunner().invoke(cli.app, ["index", str(projects)])


def flat(output):
    # Rich wraps long tmp paths at arbitrary points; compare with all whitespace removed.
    return "".join(output.split())


def test_first_calendar_with_absent_db_refuses_with_bootstrap_required(env, monkeypatch):
    tmp_path, projects, db = env
    db.parent.mkdir()
    result = calendar_index(monkeypatch, projects)
    assert result.exit_code == 78, result.output
    out = flat(result.output)
    assert f"INDEX_BOOTSTRAP_REQUIREDdb={db}" in out
    assert 'run="brainlayerbootstrap-db"' in out
    # Nothing was created: no DB, no checkpoint/backlog sidecar, no WAL.
    assert sorted(p.name for p in db.parent.iterdir()) == []


def test_first_calendar_with_absent_parent_dir_refuses_without_creating_it(env, monkeypatch):
    _tmp, projects, db = env
    result = calendar_index(monkeypatch, projects)
    assert result.exit_code == 78, result.output
    assert not db.parent.exists()


def test_interactive_runtime_index_with_absent_db_keeps_the_writer_refusal(env):
    """Out of #1349's calendar scope: interactive stays fail-closed via the writer, creating nothing."""
    _tmp, projects, db = env
    db.parent.mkdir()
    result = CliRunner().invoke(cli.app, ["index", str(projects)])
    assert result.exit_code == 1, result.output
    assert "runtimedatabasedoesnotexist" in flat(result.output)
    assert sorted(p.name for p in db.parent.iterdir()) == []


def test_dangling_symlink_db_is_not_treated_as_bootstrap_required(env, monkeypatch):
    _tmp, projects, db = env
    db.parent.mkdir()
    db.symlink_to(db.parent / "elsewhere.db")
    result = calendar_index(monkeypatch, projects)
    assert result.exit_code == 1, result.output
    assert "INDEX_DB_UNAVAILABLE" in flat(result.output)
    assert "bootstrap-db" not in result.output
    assert db.is_symlink() and not (db.parent / "elsewhere.db").exists()


def test_bootstrap_then_first_calendar_pass_completes(env, monkeypatch):
    _tmp, projects, db = env
    boot = CliRunner().invoke(cli.app, ["bootstrap-db"])
    assert boot.exit_code == 0, boot.output
    assert f"BOOTSTRAP_CREATEDdb={db}" in flat(boot.output)
    # Only the published DB: no temp file, WAL or sidecar left behind by bootstrap.
    assert sorted(p.name for p in db.parent.iterdir()) == ["brainlayer.db"]
    with ReadonlyStore(db) as store:
        assert store.schema_fingerprint == EXPECTED_RUNTIME_SCHEMA_FINGERPRINT
        assert store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone() == (0,)
    result = calendar_index(monkeypatch, projects)
    assert result.exit_code == 0, result.output
    assert any(db.parent.glob("index-checkpoint-*.json"))


def test_existing_valid_db_index_is_unchanged_and_bootstrap_refuses_it(env, monkeypatch):
    _tmp, projects, db = env
    bootstrap_runtime_database(db)
    before = db.read_bytes()
    boot = CliRunner().invoke(cli.app, ["bootstrap-db"])
    assert boot.exit_code == 1, boot.output
    assert "BOOTSTRAP_REFUSED" in flat(boot.output)
    assert db.read_bytes() == before
    result = calendar_index(monkeypatch, projects)
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("content", [b"", b"not a database at all", "foreign"])
def test_partial_or_foreign_db_stays_fail_closed_and_is_never_overwritten(env, monkeypatch, content):
    _tmp, projects, db = env
    db.parent.mkdir()
    if content == "foreign":
        import apsw

        conn = apsw.Connection(str(db))
        conn.execute("CREATE TABLE chunks(id TEXT PRIMARY KEY)")
        conn.close()
    else:
        db.write_bytes(content)
    before = db.read_bytes()
    with pytest.raises(RuntimeBootstrapRefused):
        bootstrap_runtime_database(db)
    result = calendar_index(monkeypatch, projects)
    assert result.exit_code == 1, result.output
    assert "INDEX_BOOTSTRAP_REQUIRED" not in result.output
    assert db.read_bytes() == before


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file permission bits")
def test_unwritable_db_stays_fail_closed(env, monkeypatch):
    _tmp, projects, db = env
    bootstrap_runtime_database(db)
    db.chmod(0o444)
    try:
        result = calendar_index(monkeypatch, projects)
    finally:
        db.chmod(0o644)
    assert result.exit_code == 1, result.output
    assert "INDEX_BOOTSTRAP_REQUIRED" not in result.output


def test_bootstrap_loses_a_forced_race_without_touching_the_winner(env, monkeypatch):
    """Forced ordering: another creator lands at the target between schema build and publish."""
    _tmp, _projects, db = env
    real_link = os.link

    def racing_link(src, dst):
        db.write_bytes(b"winner")
        return real_link(src, dst)

    monkeypatch.setattr(runtime.os, "link", racing_link)
    with pytest.raises(RuntimeBootstrapRefused, match="appeared during bootstrap"):
        bootstrap_runtime_database(db)
    assert db.read_bytes() == b"winner"
    assert sorted(p.name for p in db.parent.iterdir()) == ["brainlayer.db"]


def test_concurrent_bootstraps_publish_exactly_one_database(env):
    _tmp, _projects, db = env
    db.parent.mkdir()
    gate = threading.Barrier(4)
    outcomes = []

    def run():
        gate.wait()
        try:
            outcomes.append(bootstrap_runtime_database(db))
        except RuntimeBootstrapRefused:
            outcomes.append("refused")

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes.count(EXPECTED_RUNTIME_SCHEMA_FINGERPRINT) == 1, outcomes
    assert outcomes.count("refused") == 3
    assert sorted(p.name for p in db.parent.iterdir()) == ["brainlayer.db"]
    with ReadonlyStore(db) as store:
        assert store.schema_fingerprint == EXPECTED_RUNTIME_SCHEMA_FINGERPRINT
