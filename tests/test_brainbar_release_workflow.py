from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from yaml.parser import ParserError
from yaml.scanner import ScannerError

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "brainbar-release.yml"


def _load_workflow() -> dict:
    with WORKFLOW_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _events(workflow: dict) -> dict:
    return workflow.get("on") or workflow.get(True) or {}


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _release_job(workflow: dict) -> dict:
    for job in workflow.get("jobs", {}).values():
        if job.get("runs-on") == "macos-15":
            return job
    raise AssertionError("BrainBar release workflow must run a job on macos-15")


def _step_run(step: dict) -> str:
    return str(step.get("run", ""))


def _step_runs_in_brain_bar(step: dict, command: str) -> bool:
    run = _step_run(step)
    working_directory = str(step.get("working-directory", ""))
    return command in run and (working_directory == "brain-bar" or "cd brain-bar" in run)


def _is_release_step(step: dict) -> bool:
    uses = str(step.get("uses", ""))
    run = _step_run(step)
    return "action-gh-release" in uses or "gh release" in run


def _is_no_zip_guardrail(step: dict) -> bool:
    run = _step_run(step)
    return all(
        token in run
        for token in (
            "BrainBar.zip",
            "-s",
            "exit 1",
            "::error",
        )
    )


def _assert_tag_triggered(workflow: dict) -> None:
    push = _events(workflow).get("push", {})
    assert "tags" in push, "BrainBar release workflow must be triggered by pushed tags"
    assert _as_list(push["tags"]), "BrainBar release workflow must define at least one pushed tag pattern"


def _assert_swift_build_contract(job: dict) -> None:
    steps = job.get("steps", [])
    assert any(
        isinstance(step, dict) and str(step.get("uses", "")).startswith("maxim-lobanov/setup-xcode@") for step in steps
    ), "BrainBar release workflow must select Xcode with maxim-lobanov/setup-xcode"
    assert any(
        isinstance(step, dict) and step.get("with", {}).get("xcode-version") == "latest-stable" for step in steps
    ), "BrainBar release workflow must use the latest stable Xcode for Swift 6.x"
    assert any(isinstance(step, dict) and _step_runs_in_brain_bar(step, "swift build -c release") for step in steps), (
        "BrainBar release workflow must run swift build -c release inside brain-bar"
    )


def _assert_no_zip_guardrail_before_release(job: dict) -> None:
    steps = job.get("steps", [])
    guard_indexes = [index for index, step in enumerate(steps) if isinstance(step, dict) and _is_no_zip_guardrail(step)]
    assert guard_indexes, "BrainBar release workflow must fail loudly when BrainBar.zip is missing or empty"

    release_indexes = [index for index, step in enumerate(steps) if isinstance(step, dict) and _is_release_step(step)]
    assert release_indexes, "BrainBar release workflow must attach BrainBar.zip to a GitHub Release"
    assert min(guard_indexes) < min(release_indexes), "BrainBar.zip guardrail must run before the release step"


def _assert_version_stamp(job: dict) -> None:
    version_steps = [
        step for step in job.get("steps", []) if isinstance(step, dict) and "Info.plist" in _step_run(step)
    ]
    assert any("GitCommit" in _step_run(step) for step in version_steps), "Info.plist must be stamped with GitCommit"
    assert any("BuildTimeUTC" in _step_run(step) for step in version_steps), (
        "Info.plist must be stamped with BuildTimeUTC"
    )
    assert any("CFBundleShortVersionString" in _step_run(step) for step in version_steps), (
        "Info.plist release version must come from the pushed tag"
    )


def _assert_release_workflow_contract(workflow: dict) -> None:
    _assert_tag_triggered(workflow)
    job = _release_job(workflow)
    _assert_swift_build_contract(job)
    _assert_version_stamp(job)
    _assert_no_zip_guardrail_before_release(job)


def test_brainbar_release_workflow_is_valid_yaml() -> None:
    try:
        _load_workflow()
    except (ParserError, ScannerError) as exc:
        raise AssertionError(f"{WORKFLOW_PATH} is not valid YAML") from exc


def test_brainbar_release_workflow_contract() -> None:
    _assert_release_workflow_contract(_load_workflow())


def test_release_workflow_without_no_zip_guardrail_fails_contract() -> None:
    workflow = deepcopy(_load_workflow())
    job = _release_job(workflow)
    job["steps"] = [
        step for step in job.get("steps", []) if not (isinstance(step, dict) and _is_no_zip_guardrail(step))
    ]

    with pytest.raises(AssertionError, match="missing or empty"):
        _assert_no_zip_guardrail_before_release(job)
