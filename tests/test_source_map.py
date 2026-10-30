"""Synthetic freshness and review-regression fixtures; no live inputs."""

import json
import subprocess
from pathlib import Path

import pytest

from brainlayer import source_map as sm


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


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
