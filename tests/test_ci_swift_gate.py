from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _workflow_path() -> Path:
    return Path(os.environ.get("BRAINLAYER_CI_WORKFLOW", DEFAULT_WORKFLOW_PATH))


def _load_workflow() -> Mapping:
    with _workflow_path().open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, Sequence) and not isinstance(value, bytes):
        for item in value:
            yield from _strings(item)


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _is_macos_job(job: Mapping) -> bool:
    return any("macos" in item for item in _strings(job.get("runs-on", "")))


def _step_runs_command_in_brain_bar(step: Mapping, command: str) -> bool:
    run = str(step.get("run", ""))
    if command not in run:
        return False
    working_directory = str(step.get("working-directory", ""))
    return working_directory == "brain-bar" or "cd brain-bar" in run


def _find_swift_job(workflow: Mapping) -> tuple[str, Mapping] | None:
    for job_id, job in workflow.get("jobs", {}).items():
        if not _is_macos_job(job):
            continue
        steps = job.get("steps", [])
        if _step_runs_command_in_brain_bar_any_step(steps, "swift build") and _step_runs_command_in_brain_bar_any_step(
            steps, "swift test"
        ):
            return job_id, job
    return None


def _step_runs_command_in_brain_bar_any_step(steps: Sequence, command: str) -> bool:
    return any(isinstance(step, Mapping) and _step_runs_command_in_brain_bar(step, command) for step in steps)


def _contains_brain_bar_path_filter(value) -> bool:
    for item in _strings(value):
        for line in item.splitlines():
            normalized = line.strip().removeprefix("-").strip().strip("'\"")
            if normalized == "brain-bar/**":
                return True
    return False


def _workflow_event_paths_include_brain_bar(workflow: Mapping) -> bool:
    events = workflow.get("on") or workflow.get(True) or {}
    return _contains_brain_bar_path_filter(events)


def _swift_job_is_gated_on_brain_bar_paths(workflow: Mapping, job_id: str, job: Mapping) -> bool:
    if _workflow_event_paths_include_brain_bar(workflow):
        return True

    if _contains_brain_bar_path_filter(job):
        return True

    condition = str(job.get("if", ""))
    for needed_job_id in _as_list(job.get("needs")):
        needed_job = workflow.get("jobs", {}).get(needed_job_id, {})
        references_needed_output = f"needs.{needed_job_id}.outputs." in condition
        if references_needed_output and _contains_brain_bar_path_filter(needed_job):
            return True

    raise AssertionError(f"{job_id!r} is not gated by a brain-bar/** path filter")


def test_macos_swift_job_builds_and_tests_brain_bar_package():
    workflow = _load_workflow()

    swift_job = _find_swift_job(workflow)

    assert swift_job is not None, "CI must include a macOS Swift job that builds and tests brain-bar"


def test_swift_job_runs_only_for_brain_bar_changes():
    workflow = _load_workflow()
    swift_job = _find_swift_job(workflow)
    assert swift_job is not None, "CI must include a macOS Swift job before it can be path-gated"

    job_id, job = swift_job

    assert _swift_job_is_gated_on_brain_bar_paths(workflow, job_id, job)
