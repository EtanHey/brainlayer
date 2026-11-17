"""Only successful push-to-main attestations for the exact release commit count."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40


@pytest.fixture
def gate():
    spec = importlib.util.spec_from_file_location("ratchet_gate", ROOT / "scripts/ci/ratchet_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("defect", [None, "missing", "pending", "failure", "wrong-sha", "branch", "dispatch"])
def test_release_requires_successful_main_push_on_exact_sha(gate, monkeypatch, capsys, defect):
    run = {
        "head_sha": SHA,
        "head_branch": "main",
        "event": "push",
        "status": "completed",
        "conclusion": "success",
        "html_url": "https://github.com/EtanHey/brainlayer/actions/runs/123",
    }
    if defect == "pending":
        run.update(status="in_progress", conclusion=None)
    elif defect == "failure":
        run["conclusion"] = "failure"
    elif defect == "wrong-sha":
        run["head_sha"] = "b" * 40
    elif defect == "branch":
        run["head_branch"] = "feature"
    elif defect == "dispatch":
        run["event"] = "workflow_dispatch"
    calls = []

    def api(endpoint):
        calls.append(endpoint)
        return [{"workflow_runs": []}, {"workflow_runs": [] if defect == "missing" else [run]}]

    monkeypatch.setattr(gate, "api", api)
    if defect:
        reason = defect if defect in {"pending", "failure"} else "missing"
        with pytest.raises(RuntimeError, match=reason):
            gate.verify_release("EtanHey/brainlayer", SHA)
    else:
        gate.verify_release("EtanHey/brainlayer", SHA)
        assert run["html_url"] in capsys.readouterr().out
    assert "ratchet-attest.yml/runs?" in calls[0]
    assert f"head_sha={SHA}" in calls[0]


def test_api_failure_blocks_release(gate, monkeypatch):
    def unavailable(endpoint):
        raise RuntimeError("API unavailable")

    monkeypatch.setattr(gate, "api", unavailable)
    with pytest.raises(RuntimeError, match="API unavailable"):
        gate.verify_release("EtanHey/brainlayer", SHA)


@pytest.mark.parametrize("sha", ["main", "v1.2.3", "a" * 39, "g" * 40])
def test_refuses_non_exact_sha(gate, sha):
    with pytest.raises(RuntimeError, match="full commit SHA"):
        gate.verify_release("EtanHey/brainlayer", sha)
