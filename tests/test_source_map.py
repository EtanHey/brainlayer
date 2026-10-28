"""Synthetic freshness and review-regression fixtures; no live inputs."""

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from brainlayer import source_map as sm


def _clean_git_env():
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, env=_clean_git_env()).strip()


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    repo = tmp_path / "cmuxlayer"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "fixture@example.invalid")
    git(repo, "config", "user.name", "Fixture")
    git(repo, "remote", "add", "origin", "https://github.com/EtanHey/cmuxlayer.git")
    (repo / "source.ts").write_text("export function original() {}")
    (repo / ".gitignore").write_text("ignored/\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "fixture")
    monkeypatch.setattr(sm, "engine_identity", lambda _: {"version": "0.9.96", "epoch": 27})

    def build(python, args, cwd, out):
        if args[0] == "update":
            (out / "graph.json").write_text(
                json.dumps({"nodes": [{"id": "original", "label": "original()"}], "links": []})
            )
        return "original source.ts caller tests"

    monkeypatch.setattr(sm, "run_graphify", build)
    return sm.SourceMap(repo, tmp_path / "state", Path("/fixture/python"))


def ready(a):
    assert a.refresh()["status"] == "queued"
    assert a.refresh(run=True)["status"] == "fresh"
    assert a.doctor()["status"] == "fresh"


@pytest.mark.parametrize("key,value", [("PYTHONHASHSEED", "0"), ("GRAPHIFY_NO_AUTO_REFRESH", "1")])
def test_graphify_environment_preserves_isolation_and_disables_refresh(tmp_path, monkeypatch, key, value):
    process = MagicMock(returncode=0)
    process.__enter__.return_value = process
    process.communicate.return_value = ("", "")
    popen = MagicMock(return_value=process)
    monkeypatch.setattr(sm.subprocess, "Popen", popen)
    sm.run_graphify(Path("/fixture/python"), ["update", "--help"], tmp_path, tmp_path)
    assert popen.call_args.args[0][1:5] == ["-I", "-B", "-m", "graphify"]
    assert popen.call_args.kwargs["env"].get(key) == value


def test_query_contains_process_writes_in_artifact_scratch(adapter, monkeypatch):
    ready(adapter)
    graph = Path(adapter.doctor()["graph"])

    def writing_query(python, args, cwd, out):
        (cwd / "query-write-probe").write_text("owned synthetic mutation")
        return "original source.ts"

    monkeypatch.setattr(sm, "run_graphify", writing_query)
    result = adapter.query("original")
    assert not (adapter.repo / "query-write-probe").exists()
    assert (graph.parent / "query-write-probe").is_file()
    assert result["status"] == "fresh"


@pytest.mark.parametrize("mutation", ["tracked", "untracked", "ignore", "config", "head", "checksum", "missing"])
def test_refuses_changed_evidence(adapter, mutation):
    a = adapter
    ready(a)
    if mutation in ("tracked", "untracked", "ignore"):
        (a.repo / {"tracked": "source.ts", "untracked": "new.ts", "ignore": ".graphifyignore"}[mutation]).write_text(
            "changed"
        )
    elif mutation == "config":
        git(a.repo, "config", "core.excludesFile", str(a.repo / "exclude"))
    elif mutation == "head":
        git(a.repo, "commit", "--allow-empty", "-qm", "new head")
    else:
        graph = Path(a.doctor()["graph"])
        graph.unlink() if mutation == "missing" else graph.write_text("{}")
    assert a.query("original")["status"] in ("stale", "corrupt", "missing")


def test_failure_and_race_preserve_prior_checksum(adapter, monkeypatch):
    a = adapter
    ready(a)
    prior = a.doctor()["checksum"]

    def broken(*args):
        raise RuntimeError("synthetic extraction failure")

    monkeypatch.setattr(sm, "run_graphify", broken)
    a.refresh()
    assert a.refresh(run=True)["status"] == "failed"
    assert a.query("original")["checksum"] == prior
    assert a.query("original")["status"] == "failed"
    failed = (a.state / "current.json").read_bytes()
    assert a.refresh(run=True)["status"] == "failed"
    assert (a.state / "current.json").read_bytes() == failed

    def racing(python, args, cwd, out):
        (out / "graph.json").write_text('{"nodes": [], "links": []}')
        (a.repo / "source.ts").write_text("raced")
        return ""

    monkeypatch.setattr(sm, "run_graphify", racing)
    a.refresh()
    assert a.refresh(run=True)["status"] == "failed"
    assert a.doctor()["checksum"] == prior


def test_worker_without_pending_request_preserves_current_state(adapter, monkeypatch):
    assert adapter.refresh(run=True)["status"] == "missing"
    assert not (adapter.state / "current.json").exists()
    ready(adapter)
    prior = (adapter.state / "current.json").read_bytes()
    monkeypatch.setattr(sm, "run_graphify", lambda *args: pytest.fail("request was already consumed"))
    assert adapter.refresh(run=True)["status"] == "fresh"
    assert (adapter.state / "current.json").read_bytes() == prior
    (adapter.state / "queue.json").unlink()
    assert adapter.refresh(run=True)["status"] == "fresh"
    assert (adapter.state / "current.json").read_bytes() == prior


def test_malformed_node_refuses_instead_of_crashing(adapter):
    ready(adapter)
    current = adapter.doctor()
    graph = Path(current["graph"])
    graph.write_text('{"nodes":[{"label":"orphan()"}],"links":[]}')
    current["checksum"] = sm.digest(graph.read_bytes())
    sm.atomic(adapter.state / "current.json", current)
    assert adapter.query("orphan")["status"] == "refused"


def test_adapter_git_fixture_ignores_inherited_parent_env(tmp_path, monkeypatch, request):
    parent = tmp_path / "parent"
    parent.mkdir()
    git(parent, "init", "-q")
    git(parent, "config", "user.email", "parent@example.invalid")
    git(parent, "config", "user.name", "Synthetic Parent")
    (parent / "sentinel.txt").write_text("parent must stay untouched")
    git(parent, "add", ".")
    git(parent, "commit", "-qm", "parent sentinel")

    def snapshot():
        return {str(p.relative_to(parent)): p.read_bytes() for p in parent.rglob("*") if p.is_file()}

    before = snapshot()
    monkeypatch.setenv("GIT_DIR", str(parent / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(parent))
    monkeypatch.setenv("GIT_INDEX_FILE", str(parent / ".git" / "index"))
    fixture = request.getfixturevalue("adapter")
    assert Path(git(fixture.repo, "rev-parse", "--show-toplevel")) == fixture.repo
    assert git(fixture.repo, "ls-files").splitlines() == [".gitignore", "source.ts"]
    assert git(fixture.repo, "log", "-1", "--format=%s") == "fixture"
    assert snapshot() == before
