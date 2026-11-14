from __future__ import annotations

import importlib.util
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
SWIFT_GATE_CONDITION = "needs.changes.outputs.brain_bar == 'true'"
SWIFT_TEST_WRAPPER = "python3 Tests/BrainBarTests/Scripts/run_swift_tests.py"
SWIFT_RUNNER = "${{ " + SWIFT_GATE_CONDITION + " && 'macos-15' || 'ubuntu-latest' }}"


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
    if command not in run and not (command == "swift test" and SWIFT_TEST_WRAPPER in run):
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


def _find_steps(steps: Sequence, predicate) -> list[Mapping]:
    return [step for step in steps if isinstance(step, Mapping) and predicate(step)]


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

    step_conditions = " ".join(str(step.get("if", "")) for step in job.get("steps", []) if isinstance(step, Mapping))
    condition = " ".join([str(job.get("if", "")), step_conditions])
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


def test_failed_native_job_preserves_only_new_test_crash_reports(tmp_path, monkeypatch):
    workflow = _load_workflow()
    _, job = _find_swift_job(workflow)
    collector = next(step for step in job["steps"] if step.get("name") == "Preserve native test crash reports")
    home, temporary = tmp_path / "home", tmp_path / "runner"
    reports = home / "Library/Logs/DiagnosticReports"
    reports.mkdir(parents=True)
    temporary.mkdir()
    (temporary / "brainbar-test-start").write_text("1000")
    for name, modified in [
        ("xctest-new.ips", 1001),
        ("BrainBarPackageTests-new.crash", 1002),
        ("xctest-old.ips", 999),
        ("unrelated-new.ips", 1003),
    ]:
        report = reports / name
        report.write_text(name)
        os.utime(report, (modified, modified))
    (reports / "xctest-link.ips").symlink_to(reports / "unrelated-new.ips")
    monkeypatch.setenv("RUNNER_TEMP", str(temporary))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    code = collector["run"].split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    exec(compile(code, "native-crash-collector", "exec"), {})
    destination = temporary / "brainbar-native-crashes"
    assert {p.name for p in destination.iterdir()} == {"xctest-new.ips", "BrainBarPackageTests-new.crash"}
    for copied in destination.iterdir():
        assert copied.read_bytes() == (reports / copied.name).read_bytes()
    assert len(list(reports.iterdir())) == 5, "diagnostic collection must not delete any input"
    assert collector["if"] == "failure() && steps.brainbar-tests.outcome == 'failure'"


@pytest.mark.parametrize("swift_exit,leak,expected", [(0, False, 0), (7, False, 7), (0, True, 1)])
def test_swift_wrapper_executes_tests_and_preserves_failure_gates(monkeypatch, swift_exit, leak, expected):
    path = REPO_ROOT / "brain-bar/Tests/BrainBarTests/Scripts/run_swift_tests.py"
    spec = importlib.util.spec_from_file_location("swift_fixture_gate", path)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    calls = []

    def run(command, *, env):
        root = Path(env["BRAINBAR_TEST_TMP_ROOT"])
        assert root.is_absolute() and root.is_dir()
        assert root.name.endswith(".noindex")
        assert env["TMPDIR"] == str(root) + "/"
        assert list(root.glob("brainbar-*")) == []
        calls.append(command)
        if leak:
            (root / "brainbar-unremoved-fixture").write_text("owned test fixture")
        return SimpleNamespace(returncode=swift_exit)

    monkeypatch.setattr(wrapper.subprocess, "run", run)
    monkeypatch.setattr(wrapper.sys, "argv", [str(path), "--filter", "FixtureTest"])
    assert wrapper.main() == expected
    assert calls == [
        ["swift", "test", "--scratch-path", str(REPO_ROOT / ".build.noindex/swift"), "--filter", "FixtureTest"]
    ]


def test_swift_job_runs_only_for_brain_bar_changes():
    workflow = _load_workflow()
    swift_job = _find_swift_job(workflow)
    assert swift_job is not None, "CI must include a macOS Swift job before it can be path-gated"

    job_id, job = swift_job

    assert _swift_job_is_gated_on_brain_bar_paths(workflow, job_id, job)


def test_swift_required_check_always_reports_status():
    workflow = _load_workflow()
    swift_job = _find_swift_job(workflow)
    assert swift_job is not None, "CI must include a macOS Swift job before required-check semantics can be verified"

    _job_id, job = swift_job
    steps = job.get("steps", [])

    assert "if" not in job, "swift required-check candidate must not be skipped at job level"

    gated_steps = _find_steps(steps, lambda step: step.get("if") == SWIFT_GATE_CONDITION)
    assert any(step.get("uses") == "actions/cache@v4" for step in gated_steps), "SwiftPM cache step must be gated"
    assert any(_step_runs_command_in_brain_bar(step, "swift build") for step in gated_steps), (
        "swift build step must be gated"
    )
    assert any(_step_runs_command_in_brain_bar(step, "swift test") for step in gated_steps), (
        "swift test step must be gated"
    )

    skip_steps = _find_steps(
        steps,
        lambda step: (
            "no brain-bar changes" in str(step.get("run", "")).lower()
            and "swift build/test skipped" in str(step.get("run", "")).lower()
        ),
    )
    assert skip_steps, "swift job must report success explicitly when brain-bar did not change"


def test_swift_job_selects_swift_6_xcode_before_version_check():
    workflow = _load_workflow()
    swift_job = _find_swift_job(workflow)
    assert swift_job is not None, "CI must include a macOS Swift job before Xcode selection can be verified"

    _job_id, job = swift_job
    steps = job.get("steps", [])
    xcode_steps = _find_steps(steps, lambda step: str(step.get("uses", "")).startswith("maxim-lobanov/setup-xcode@"))
    assert xcode_steps, "swift job must select an Xcode version with Swift 6 before running SwiftPM"

    xcode_step = xcode_steps[0]
    assert xcode_step.get("if") == SWIFT_GATE_CONDITION
    assert xcode_step.get("with", {}).get("xcode-version") == "latest-stable"

    xcode_index = steps.index(xcode_step)
    swift_version_index = next(index for index, step in enumerate(steps) if step.get("run") == "swift --version")
    assert xcode_index < swift_version_index, "Xcode selection must happen before swift --version"


def test_swift_job_uses_macos_15_only_when_brain_bar_changed():
    """Python-only changes take the skip on Linux instead of queueing for macOS.

    The required check is the job NAME, so it stays "swift (macos-15)". Before this, a
    Python-only PR waited ~3 h in the macOS queue just to echo "skipped" (#1727, 2026-11-09).
    """
    workflow = _load_workflow()
    swift_job = _find_swift_job(workflow)
    assert swift_job is not None, "CI must include a macOS Swift job before runner selection can be verified"

    _job_id, job = swift_job

    assert job.get("runs-on") == SWIFT_RUNNER
    assert job.get("name") == "swift (macos-15)"


def test_linux_skip_path_runs_only_os_neutral_steps():
    workflow = _load_workflow()
    _job_id, job = _find_swift_job(workflow)
    on_skip_path = [
        step.get("uses") or step.get("run")
        for step in job["steps"]
        if step.get("if") in (None, "needs.changes.outputs.brain_bar != 'true'")
    ]
    assert on_skip_path == ["actions/checkout@v4", 'echo "no brain-bar changes — swift build/test skipped"']


def test_swift_required_gate_has_a_bounded_job_timeout():
    workflow = _load_workflow()
    swift_job = _find_swift_job(workflow)
    assert swift_job is not None

    _job_id, job = swift_job

    # 2026-11-13: Test BrainBar took 11m29s (PR #1788, run 43438292244; job 14m30s) and was cut off at 11m39s by the 15m limit (PR #1789, run 43459850246). 25 = ~10m headroom over the last passing job; still a finite hang bound.
    value = job.get("timeout-minutes")
    assert value == 25
    assert 10 <= value <= 30, "Swift job must retain a finite hang-protection budget"


def _filter_matches(path: str, pattern: str) -> bool:
    # The filter's current glob subset: * stays in one component; ** crosses directories.
    expression = re.escape(pattern).replace(r"\*\*/", "(?:.*/)?").replace(r"\*\*", ".*").replace(r"\*", "[^/]*")
    return re.fullmatch(expression, path) is not None


@pytest.mark.parametrize(
    "path,pattern,expected",
    [
        ("a/Menu.swift", "a/*.swift", True),
        ("a/nested/Menu.swift", "a/*.swift", False),
        ("a/Menu.swift", "a/**/*.swift", True),
        ("a/nested/Menu.swift", "a/**/*.swift", True),
        ("a/nested/Menu.swift", "a/**", True),
    ],
)
def test_filter_glob_directory_boundaries(path, pattern, expected):
    assert _filter_matches(path, pattern) == expected


def test_menu_function_filter_covers_dependencies_and_future_menu_sources():

    changes = _load_workflow()["jobs"]["changes"]
    filters = yaml.safe_load(next(s for s in changes["steps"] if s.get("id") == "filter")["with"]["filters"])
    required = [
        *REPO_ROOT.glob("brain-bar/Sources/BrainBar/BrainBarMenu*.swift"),
        *REPO_ROOT.glob("brain-bar/Tests/BrainBarTests/BrainBarMenu*.swift"),
    ]
    # Resolve top-level project types used by the function and its pure test helpers by declaration,
    # so moving a type to an uncovered file fails without a path-list update hiding the move.
    function_files = [
        REPO_ROOT / f"brain-bar/Sources/BrainBar/BrainBarMenu{name}.swift"
        for name in ("Presenter", "Presentation", "Sentences", "Cases")
    ]
    test_files = [
        REPO_ROOT / f"brain-bar/Tests/BrainBarTests/BrainBarMenu{name}.swift"
        for name in ("InvariantTests", "PresentationTests", "EdgeCaptures")
    ]
    identifiers = set(re.findall(r"\b[A-Z]\w*\b", "\n".join(p.read_text() for p in function_files + test_files)))
    definitions = {}
    # BrainBarDaemon is a separate module with same-named types, not imported by these files.
    for source in (REPO_ROOT / "brain-bar/Sources/BrainBar").rglob("*.swift"):
        for name in re.findall(r"^(?:struct|enum|class|protocol|typealias) (\w+)\b", source.read_text(), re.M):
            if name in identifiers:
                definitions.setdefault(name, set()).add(source)
    assert {"BrainBarHealthVerdict", "DriveAuthStatus"} <= definitions.keys()
    required += [path for paths in definitions.values() for path in paths]
    paths = [str(p.relative_to(REPO_ROOT)) for p in required] + [
        "brain-bar/Sources/BrainBar/BrainBarMenuFuture.swift",
        "brain-bar/Tests/BrainBarTests/Fixtures/presentation-cases.json",
        "brain-bar/Tests/BrainBarTests/Fixtures/future-case.json",
        "brain-bar/Package.swift",
        "brain-bar/Package.resolved",
        "brain-bar/Tests/BrainBarTests/Scripts/run_swift_tests.py",
        ".github/workflows/ci.yml",
    ]
    for path in paths:
        assert any(_filter_matches(path, pattern) for pattern in filters["menu_function"]), path
    assert ".github/workflows/ci.yml" in filters["brain_bar"]
    assert "base" not in next(s for s in changes["steps"] if s.get("id") == "filter")["with"]
    events = _load_workflow().get("on") or _load_workflow()[True]
    assert events["schedule"] == [{"cron": "0 3 * * 1"}]
    for output in ("brain_bar", "menu_function"):
        assert changes["outputs"][output] == (
            "${{ github.event_name == 'workflow_dispatch' || github.event_name == 'schedule' || "
            f"steps.filter.outputs.{output} }}}}"
        )


@pytest.mark.parametrize("event", ["pull_request", "push", "workflow_dispatch", "schedule"])
@pytest.mark.parametrize("output", [None, "", "false", "true", "FALSE", "unexpected"])
def test_menu_suite_skip_is_fail_safe(event, output, tmp_path):
    import subprocess

    step = next(s for s in _load_workflow()["jobs"]["swift"]["steps"] if s.get("id") == "brainbar-tests")
    assert step["env"]["MENU_FUNCTION"] == "${{ needs.changes.outputs.menu_function }}"
    assert step["env"]["EVENT_NAME"] == "${{ github.event_name }}"
    # Parse the shell condition rather than GitHub equality, which ignores case.
    parsed = re.search(
        r'if \[\[ "\$EVENT_NAME" (!=|==) "([^"]*)" (&&|\|\|) '
        r'"\$EVENT_NAME" (!=|==) "([^"]*)" (&&|\|\|) '
        r'"\$MENU_FUNCTION" (!=|==) "([^"]*)" \]\]; then',
        step["run"],
    )
    assert parsed, step["run"]
    op1, rhs1, join1, op_schedule, rhs_schedule, join2, op2, rhs2 = parsed.groups()
    left = (event == rhs1) if op1 == "==" else (event != rhs1)
    right = ((output or "") == rhs2) if op2 == "==" else ((output or "") != rhs2)
    scheduled = (event == rhs_schedule) if op_schedule == "==" else (event != rhs_schedule)
    combined = (left and scheduled) if join1 == "&&" else (left or scheduled)
    skip = (combined and right) if join2 == "&&" else (combined or right)
    assert skip == (event not in ("workflow_dispatch", "schedule") and output == "false")
    # Execute the actual shell with a recording Python stand-in: never invoke Swift or a build.
    executable = tmp_path / "python3"
    executable.write_text('#!/bin/sh\nprintf "CALL %s %s %s\\n" "$1" "$2" "$3"\n')
    executable.chmod(0o755)
    (tmp_path / "brain-bar").mkdir()
    result = subprocess.run(
        ["/bin/bash", "-e", "-c", step["run"]],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": str(tmp_path), "MENU_FUNCTION": output or "", "EVENT_NAME": event},
    )
    assert result.returncode == 0, result.stderr
    calls = [line for line in result.stdout.splitlines() if line.startswith("CALL Tests/")]
    runner = "CALL Tests/BrainBarTests/Scripts/run_swift_tests.py"
    assert calls == [f"{runner} --skip BrainBarMenuInvariantTests"] + (
        [] if skip else [f"{runner} --filter BrainBarMenuInvariantTests"]
    )
    assert "running everything except BrainBarMenuInvariantTests" in result.stdout
    if skip:
        assert "menu function unchanged: skipping BrainBarMenuInvariantTests" in result.stdout
    else:
        assert "running BrainBarMenuInvariantTests in its own process" in result.stdout


@pytest.mark.parametrize("first,second", [(0, 0), (7, 0), (0, 9), (7, 9), (9, 7)])
def test_menu_split_runs_both_and_returns_worse_status(tmp_path, first, second):
    import subprocess

    step = next(s for s in _load_workflow()["jobs"]["swift"]["steps"] if s.get("id") == "brainbar-tests")
    executable = tmp_path / "python3"
    executable.write_text(
        "#!/bin/sh\n"
        'case "$2" in\n'
        '  --skip) echo rest; exit "$FIRST_EXIT" ;;\n'
        '  --filter) echo menu; exit "$SECOND_EXIT" ;;\n'
        "esac\n"
    )
    executable.chmod(0o755)
    (tmp_path / "brain-bar").mkdir()
    result = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": str(tmp_path),
            "MENU_FUNCTION": "true",
            "EVENT_NAME": "pull_request",
            "FIRST_EXIT": str(first),
            "SECOND_EXIT": str(second),
        },
    )
    assert [line for line in result.stdout.splitlines() if line in ("rest", "menu")] == ["rest", "menu"]
    assert result.returncode == max(first, second), result.stderr


@pytest.mark.parametrize("leaking_call", [0, 1])
def test_swift_wrapper_split_owns_temp_roots_and_reuses_build(monkeypatch, leaking_call):
    path = REPO_ROOT / "brain-bar/Tests/BrainBarTests/Scripts/run_swift_tests.py"
    spec = importlib.util.spec_from_file_location("swift_split_gate", path)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    roots, commands = [], []

    def run(command, *, env):
        root = Path(env["BRAINBAR_TEST_TMP_ROOT"])
        assert root.is_dir() and not list(root.iterdir())
        assert env["TMPDIR"] == str(root) + "/"
        roots.append(root)
        commands.append(command)
        if len(roots) - 1 == leaking_call:
            (root / "brainbar-leaked-fixture").write_text("owned fixture")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(wrapper.subprocess, "run", run)
    results = []
    for option in ("--skip", "--filter"):
        monkeypatch.setattr(wrapper.sys, "argv", [str(path), option, "BrainBarMenuInvariantTests"])
        results.append(wrapper.main())
    assert results == ([1, 0] if leaking_call == 0 else [0, 1])
    assert roots[0] != roots[1]
    assert all(not root.exists() for root in roots)
    assert (
        commands[0][:4]
        == commands[1][:4]
        == ["swift", "test", "--scratch-path", str(REPO_ROOT / ".build.noindex/swift")]
    )


@pytest.mark.parametrize(
    "arguments",
    [[], ["--skip", "BrainBarMenuInvariantTests"], ["--filter", "FixtureTest", "--skip", "BrainBarMenuInvariantTests"]],
)
def test_swift_wrapper_forwards_skip_and_preserves_default(monkeypatch, arguments):
    path = REPO_ROOT / "brain-bar/Tests/BrainBarTests/Scripts/run_swift_tests.py"
    spec = importlib.util.spec_from_file_location("swift_skip_gate", path)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    calls = []
    monkeypatch.setattr(
        wrapper.subprocess, "run", lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=0)
    )
    monkeypatch.setattr(wrapper.sys, "argv", [str(path), *arguments])
    assert wrapper.main() == 0
    assert calls == [["swift", "test", "--scratch-path", str(REPO_ROOT / ".build.noindex/swift"), *arguments]]
