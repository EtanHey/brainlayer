"""PR selection and exact-batch release custody for the exhaustive crash sweep."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40


@pytest.fixture
def gate():
    spec = importlib.util.spec_from_file_location("crash_sweep_gate", ROOT / "scripts/ci/crash_sweep_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "path",
    [
        *[
            f"src/brainlayer/{name}.py"
            for name in (
                "maintenance_calendar maintenance maintenance_control service_supervisor supervisor_guard "
                "supervisor_child supervisor_jobs supervisor_diagnostics vector_store backup_daily migration_journal "
                "migration_runtime migration_custody launchd_consolidation launchd_revert watcher watcher_bridge"
            ).split()
        ],
        "src/brainlayer/pipeline/nested/stage.py",
        "scripts/launchd/install.sh",
        "tests/test_launchd_revert_crash.py",
        "tests/test_launchd_consolidation_recovery.py",
        "scripts/ci/crash_sweep_gate.py",
        ".github/workflows/ci.yml",
    ],
)
def test_sensitive_paths_require_all_sweeps(gate, path):
    assert gate.required("pull_request", [path], [])


def test_docs_skip_unless_opted_in_and_every_batch_requires_sweeps(gate):
    assert not gate.required("pull_request", ["docs/guide.md"], [])
    assert gate.required("pull_request", ["docs/guide.md"], ["crash-sweep"])
    for event in ("push", "workflow_dispatch"):
        assert gate.required(event, ["docs/guide.md"], [])


@pytest.mark.parametrize(
    "defect", [None, "dispatch", "wrong-sha", "pr", "failed", "skipped-job", "missing-job", "wrong-job-sha"]
)
def test_release_requires_successful_full_sweep_on_exact_sha(gate, monkeypatch, defect):
    run = {
        "id": 123,
        "run_attempt": 2,
        "head_sha": SHA,
        "event": "push",
        "head_branch": "main",
        "conclusion": "success",
    }
    job = {"name": "crash sweeps", "head_sha": SHA, "conclusion": "success", "html_url": "https://github.com/job/123"}
    if defect == "dispatch":
        run["event"] = "workflow_dispatch"
        run["head_branch"] = "batch"
    if defect == "wrong-sha":
        run["head_sha"] = "b" * 40
    elif defect == "pr":
        run["event"] = "pull_request"
    elif defect == "failed":
        run["conclusion"] = "failure"
    elif defect == "skipped-job":
        job["conclusion"] = "skipped"
    elif defect == "wrong-job-sha":
        job["head_sha"] = "b" * 40
    calls = []

    def api(endpoint):
        calls.append(endpoint)
        if "/attempts/" in endpoint:
            return [{"jobs": [] if defect == "missing-job" else [job]}]
        return [{"workflow_runs": [run]}]

    monkeypatch.setattr(gate, "api", api)
    if defect not in (None, "dispatch"):
        with pytest.raises(RuntimeError, match="full crash sweep"):
            gate.verify_release("EtanHey/brainlayer", SHA)
    else:
        gate.verify_release("EtanHey/brainlayer", SHA)
        assert "head_sha=" + SHA in calls[0]
        assert "/runs/123/attempts/2/jobs" in calls[1]


def test_api_failure_blocks_release(gate, monkeypatch):
    def unavailable(endpoint):
        raise RuntimeError("API unavailable")

    monkeypatch.setattr(gate, "api", unavailable)
    with pytest.raises(RuntimeError, match="API unavailable"):
        gate.verify_release("EtanHey/brainlayer", SHA)


def test_workflows_keep_required_context_and_gate_both_publish_paths():
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    assert "workflow_dispatch" in ci[True]  # PyYAML treats YAML 1.1 `on` as True.
    assert {"labeled", "unlabeled"} <= set(ci[True]["pull_request"]["types"])
    changes = ci["jobs"]["changes"]
    assert changes["outputs"]["crash_sweeps"] == "${{ steps.sweeps.outputs.required }}"
    assert any("crash_sweep_gate.py required" in step.get("run", "") for step in changes["steps"])
    crash = ci["jobs"]["crash-sweeps"]
    assert crash["name"] == "crash sweeps"
    assert crash["if"] == "always()"
    assert crash["needs"] == "changes"
    assert "needs.changes.result" in str(crash["steps"][0])
    for step in crash["steps"][2:]:
        assert "needs.changes.outputs.crash_sweeps == 'true'" in step["if"]
    for name in ("publish.yml", "brainbar-release.yml"):
        workflow = yaml.safe_load((ROOT / ".github/workflows" / name).read_text())
        job = next(iter(workflow["jobs"].values()))
        check = job["steps"][1]
        assert "crash_sweep_gate.py verify-release" in check["run"]
        assert "git rev-parse HEAD" in check["run"]
        assert check["env"]["GH_TOKEN"] == "${{ github.token }}"
    hook = (ROOT / ".githooks/pre-push").read_text()
    assert "crash_sweep_gate.py verify-release" in hook


@pytest.mark.parametrize(
    "kind,expected", [("docs", "false"), ("label", "true"), ("rename", "true"), ("bad-base", None)]
)
def test_required_cli_reads_real_diff_and_fails_closed(tmp_path, monkeypatch, kind, expected):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "Fixture")
    sensitive = tmp_path / "src/brainlayer/watcher.py"
    sensitive.parent.mkdir(parents=True)
    sensitive.write_text("original")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    if kind == "rename":
        git("mv", "src/brainlayer/watcher.py", "guide.md")
    else:
        (tmp_path / "guide.md").write_text("docs")
    git("add", ".")
    git("commit", "-qm", "change")
    event = tmp_path / "event.json"
    event.write_text(
        json.dumps(
            {
                "pull_request": {
                    "base": {"sha": "0" * 40 if kind == "bad-base" else base},
                    "head": {"sha": git("rev-parse", "HEAD")},
                    "labels": [{"name": "crash-sweep"}] if kind == "label" else [],
                }
            }
        )
    )
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/ci/crash_sweep_gate.py"), "required"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    if expected is None:
        assert result.returncode != 0 and not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert output.read_text() == f"required={expected}\n"


@pytest.mark.parametrize(
    "changes,value,passed",
    [("success", "true", True), ("success", "false", True), ("failure", "false", False), ("success", "", False)],
)
def test_required_context_rejects_missing_or_failed_policy(monkeypatch, changes, value, passed):
    job = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]["crash-sweeps"]
    monkeypatch.setenv("CHANGES_RESULT", changes)
    monkeypatch.setenv("SWEEPS_REQUIRED", value)
    result = subprocess.run(["bash", "-e", "-o", "pipefail", "-c", job["steps"][0]["run"]])
    assert (result.returncode == 0) == passed


def test_tag_hook_blocks_before_suite_when_full_sweep_missing(tmp_path):
    spec = importlib.util.spec_from_file_location("hook_fixture", ROOT / "tests/test_run_tests_script.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    repo, log = fixture._repo_with_the_pre_push_hook(tmp_path)
    sha = fixture._rev_parse(repo, "v1.1.0")
    result = fixture._run_hook(
        repo, log, f"refs/tags/v1.1.0 {sha} refs/tags/v1.1.0 {'0' * 40}\n", extra_env={"FIXTURE_SWEEP_EXIT": "1"}
    )
    assert result.returncode != 0 and not log.exists()


def test_release_gate_peels_an_annotated_tag_before_checking_receipt(gate, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("hook_fixture", ROOT / "tests/test_run_tests_script.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    repo, _ = fixture._repo_with_annotated_tags(tmp_path)
    tag_object = fixture._rev_parse(repo, "v1.1.0")
    commit = fixture._rev_parse(repo, "v1.1.0^{commit}")
    assert tag_object != commit
    monkeypatch.chdir(repo)
    monkeypatch.setattr(sys, "argv", ["gate", "verify-release", "EtanHey/brainlayer", tag_object, "--tag"])
    checked = []
    monkeypatch.setattr(gate, "verify_release", lambda repo, sha: checked.append(sha))
    gate.main()
    assert checked == [commit]
