"""Contract tests for scripts/run_tests.sh."""

import os
import stat
import subprocess
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_tests.sh"
HOOK_PATH = Path(__file__).resolve().parent.parent / ".githooks" / "pre-push"


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _make_stub_bin(tmp_path: Path, *, pytest_exit: int, bun_exit: int | None) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    pytest_log = tmp_path / "pytest.log"
    bun_log = tmp_path / "bun.log"

    _write_executable(
        bin_dir / "pytest",
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'echo "$*" >> "$PYTEST_LOG"',
                f"exit {pytest_exit}",
                "",
            ]
        ),
    )

    if bun_exit is not None:
        _write_executable(
            bin_dir / "bun",
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    'echo "$*" >> "$BUN_LOG"',
                    f"exit {bun_exit}",
                    "",
                ]
            ),
        )

    return pytest_log, bun_log


def _script_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "BRAINLAYER_CHANGED_FILES",
        "BRAINLAYER_CHANGED_FILES_RANGE",
        "BRAINLAYER_PREPUSH",
        "BRAINLAYER_PREPUSH_CACHE_DIR",
        "BRAINLAYER_PREPUSH_SCOPE",
        "BRAINLAYER_PREPUSH_TREE_HASH",
    ):
        env.pop(key, None)
    # An inherited GIT_DIR/GIT_WORK_TREE overrides the cwd, so a script copied OUTSIDE a repo would
    # still find one and the detection-failure path would never be reached.
    for key in [key for key in env if key.startswith("GIT_")]:
        env.pop(key, None)
    return env


def _clean_git_env() -> dict[str, str]:
    # Same guard as tests/test_build_sha.py: an inherited GIT_DIR/GIT_WORK_TREE OVERRIDES `-C`, so
    # these fixture repos would answer for the real checkout and the scope paths would never be
    # exercised as written.
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=_clean_git_env())


def _repo_with_an_empty_head_commit(tmp_path: Path) -> Path:
    """A real repo where `git diff HEAD~1...HEAD` succeeds and names nothing.

    This is the ONLY honest "nothing changed": git ran, git answered, and the answer was empty.
    """
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    _git(repo.parent, "init", "-q", "-b", "main", str(repo))
    _git(repo, "config", "user.email", "fixture@example.com")
    _git(repo, "config", "user.name", "Fixture User")
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "seed.txt")
    _git(repo, "commit", "-qm", "seed")
    _git(repo, "commit", "-qm", "empty", "--allow-empty")
    (repo / "scripts" / "run_tests.sh").write_text(SCRIPT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return repo


def _script_outside_a_repo(tmp_path: Path) -> Path:
    """A copy of the script whose ROOT_DIR is not a git work tree.

    `changed_files()` resolves the changed set from git when `BRAINLAYER_CHANGED_FILES` is unset;
    running the real script from the repo can never exercise the path where that fails.
    """
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    copied = scripts_dir / "run_tests.sh"
    copied.write_text(SCRIPT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return copied


def _rev_parse(repo: Path, rev: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", rev], text=True, env=_clean_git_env()).strip()


def _repo_with_two_tags(tmp_path: Path) -> Path:
    """A repo tagged twice, with one mapped source file changed between the tags.

    A tag push is the one ref that has no branch to diff against, so the fixture has to carry
    real tags: the range this exercises is `previous tag..tag`, not `origin/main...HEAD`.
    """
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    _git(repo.parent, "init", "-q", "-b", "main", str(repo))
    _git(repo, "config", "user.email", "fixture@example.com")
    _git(repo, "config", "user.name", "Fixture User")
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "seed.txt")
    _git(repo, "commit", "-qm", "seed")
    _git(repo, "tag", "v1.0.0")
    watcher = repo / "src" / "brainlayer" / "watcher.py"
    watcher.parent.mkdir(parents=True)
    watcher.write_text("# fixture\n", encoding="utf-8")
    _git(repo, "add", "src/brainlayer/watcher.py")
    _git(repo, "commit", "-qm", "watcher change")
    _git(repo, "tag", "v1.1.0")
    # One more commit AFTER the tag, touching a DIFFERENT mapped module. Without it the fallback
    # `HEAD~1...HEAD` diff names the same file as the tag range, and a test asserting the range
    # was honoured would pass on the fallback -- green for the wrong reason.
    index_new = repo / "src" / "brainlayer" / "index_new.py"
    index_new.write_text("# fixture\n", encoding="utf-8")
    _git(repo, "add", "src/brainlayer/index_new.py")
    _git(repo, "commit", "-qm", "post-tag change")
    (repo / "scripts" / "run_tests.sh").write_text(SCRIPT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return repo


def _install_pre_push_hook(repo: Path, tmp_path: Path) -> Path:
    """The real hook over a run_tests.sh stub that records the scope env it was handed."""
    env_log = tmp_path / "hook-env.log"
    _write_executable(
        repo / "scripts" / "run_tests.sh",
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "{",
                '  echo "SCOPE=${BRAINLAYER_PREPUSH_SCOPE:-<unset>}"',
                '  echo "RANGE=${BRAINLAYER_CHANGED_FILES_RANGE:-<unset>}"',
                '  echo "FILES=${BRAINLAYER_CHANGED_FILES:-<unset>}"',
                '  echo "TAG=${BRAINLAYER_PREPUSH_TAG:-<unset>}"',
                '} >> "$HOOK_ENV_LOG"',
                "exit 0",
                "",
            ]
        ),
    )
    hook = repo / ".githooks" / "pre-push"
    hook.parent.mkdir(parents=True)
    hook.write_text(HOOK_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return env_log


def _repo_with_the_pre_push_hook(tmp_path: Path) -> tuple[Path, Path]:
    repo = _repo_with_two_tags(tmp_path)
    return repo, _install_pre_push_hook(repo, tmp_path)


def _repo_with_a_pre_release_tag_between_releases(tmp_path: Path) -> tuple[Path, Path]:
    """v1.0.0 -> v1.1.0-rc1 -> v1.1.0: the predecessor `--match 'v*'` alone would wrongly pick."""
    repo = _tagged_release_line(tmp_path, middle_tag="v1.1.0-rc1")
    return repo, _install_pre_push_hook(repo, tmp_path)


def _repo_with_annotated_tags(tmp_path: Path) -> tuple[Path, Path]:
    """Both releases as tag OBJECTS, the way `git tag -a` writes a real release."""
    repo = _tagged_release_line(tmp_path, annotated=True)
    return repo, _install_pre_push_hook(repo, tmp_path)


def _tagged_release_line(tmp_path: Path, *, middle_tag: str = "nightly", annotated: bool = False) -> Path:
    """seed(v1.0.0) -> watcher change(middle_tag) -> index change(v1.1.0), scripts/ populated."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    _git(repo.parent, "init", "-q", "-b", "main", str(repo))
    _git(repo, "config", "user.email", "fixture@example.com")
    _git(repo, "config", "user.name", "Fixture User")

    def _tag(name: str) -> None:
        if annotated:
            _git(
                repo,
                "-c",
                "user.email=fixture@example.com",
                "-c",
                "user.name=Fixture User",
                "tag",
                "-a",
                name,
                "-m",
                name,
            )
        else:
            _git(repo, "tag", name)

    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "seed.txt")
    _git(repo, "commit", "-qm", "seed")
    _tag("v1.0.0")
    watcher = repo / "src" / "brainlayer" / "watcher.py"
    watcher.parent.mkdir(parents=True)
    watcher.write_text("# fixture\n", encoding="utf-8")
    _git(repo, "add", "src/brainlayer/watcher.py")
    _git(repo, "commit", "-qm", "watcher change")
    _git(repo, "tag", middle_tag)
    index_new = repo / "src" / "brainlayer" / "index_new.py"
    index_new.write_text("# fixture\n", encoding="utf-8")
    _git(repo, "add", "src/brainlayer/index_new.py")
    _git(repo, "commit", "-qm", "index change")
    _tag("v1.1.0")
    (repo / "scripts" / "run_tests.sh").write_text(SCRIPT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return repo


def _repo_with_a_non_release_tag_between_releases(tmp_path: Path) -> tuple[Path, Path]:
    """v1.0.0 -> `nightly` -> v1.1.0, so the nearest ANY-name tag is not the previous release.

    `git describe --tags --abbrev=0 <tag>^` answers with the nearest reachable tag of any name, so
    an intervening nightly/rc/checkpoint on the release line silently starts the range there and
    the commits between the real previous v* and that tag are never mapped. Fail-open, on a gate.
    """
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    _git(repo.parent, "init", "-q", "-b", "main", str(repo))
    _git(repo, "config", "user.email", "fixture@example.com")
    _git(repo, "config", "user.name", "Fixture User")
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "seed.txt")
    _git(repo, "commit", "-qm", "seed")
    _git(repo, "tag", "v1.0.0")
    watcher = repo / "src" / "brainlayer" / "watcher.py"
    watcher.parent.mkdir(parents=True)
    watcher.write_text("# fixture\n", encoding="utf-8")
    _git(repo, "add", "src/brainlayer/watcher.py")
    _git(repo, "commit", "-qm", "watcher change")
    _git(repo, "tag", "nightly")
    index_new = repo / "src" / "brainlayer" / "index_new.py"
    index_new.write_text("# fixture\n", encoding="utf-8")
    _git(repo, "add", "src/brainlayer/index_new.py")
    _git(repo, "commit", "-qm", "index change")
    _git(repo, "tag", "v1.1.0")
    return repo, _install_pre_push_hook(repo, tmp_path)


def _run_hook(
    repo: Path, env_log: Path, stdin: str, *, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = _script_env()
    env["HOOK_ENV_LOG"] = str(env_log)
    env.update(extra_env or {})
    return subprocess.run(  # noqa: S603 - fixture hook, returncode asserted by callers
        ["bash", str(repo / ".githooks" / "pre-push"), "origin", "git@example.invalid:fixture.git"],
        cwd=repo,
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_run_tests_aggregates_exit_codes_and_keeps_running(tmp_path: Path) -> None:
    test_root = tmp_path / "tests"
    test_root.mkdir()
    (test_root / "fixture.test.ts").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=2, bun_exit=4)

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    result = subprocess.run(["bash", str(SCRIPT_PATH)], capture_output=True, text=True, env=env)

    assert result.returncode == 6
    assert pytest_log.read_text().strip()
    assert bun_log.read_text().strip()


def test_run_tests_skips_bun_when_no_typescript_tests_exist(tmp_path: Path) -> None:
    test_root = tmp_path / "tests"
    test_root.mkdir()

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    result = subprocess.run(["bash", str(SCRIPT_PATH)], capture_output=True, text=True, env=env)

    assert result.returncode == 0
    assert pytest_log.read_text().strip()
    assert not bun_log.exists()


def test_run_tests_executes_regression_shell_scripts(tmp_path: Path) -> None:
    test_root = tmp_path / "tests"
    regression_root = test_root / "regression"
    regression_root.mkdir(parents=True)
    (test_root / "fixture.test.ts").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)
    shell_log = tmp_path / "shell.log"
    _write_executable(
        regression_root / "test_fixture.sh",
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'echo "ran" >> "$SHELL_LOG"',
                "exit 0",
                "",
            ]
        ),
    )

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)
    env["SHELL_LOG"] = str(shell_log)

    result = subprocess.run(["bash", str(SCRIPT_PATH)], capture_output=True, text=True, env=env)

    assert result.returncode == 0
    assert shell_log.read_text().strip() == "ran"


def test_prepush_cache_skips_same_tree_hash_after_success(tmp_path: Path) -> None:
    test_root = tmp_path / "tests"
    test_root.mkdir()
    (test_root / "test_think_recall_integration.py").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["BRAINLAYER_PREPUSH"] = "1"
    env["BRAINLAYER_PREPUSH_SCOPE"] = "full"
    env["BRAINLAYER_PREPUSH_TREE_HASH"] = "tree-same"
    env["BRAINLAYER_PREPUSH_CACHE_DIR"] = str(tmp_path / "cache")
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    first = subprocess.run(["bash", str(SCRIPT_PATH)], capture_output=True, text=True, env=env)
    first_log = pytest_log.read_text()
    second = subprocess.run(["bash", str(SCRIPT_PATH)], capture_output=True, text=True, env=env)

    assert first.returncode == 0
    assert second.returncode == 0
    assert "SKIP: pre-push tree hash tree-same already passed" in second.stdout
    assert pytest_log.read_text() == first_log
    assert (tmp_path / "cache" / "tree-same.full.ok").is_file()


def test_prepush_cache_does_not_skip_after_failure(tmp_path: Path) -> None:
    test_root = tmp_path / "tests"
    test_root.mkdir()

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=2, bun_exit=0)

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["BRAINLAYER_PREPUSH"] = "1"
    env["BRAINLAYER_PREPUSH_SCOPE"] = "full"
    env["BRAINLAYER_PREPUSH_TREE_HASH"] = "tree-fails"
    env["BRAINLAYER_PREPUSH_CACHE_DIR"] = str(tmp_path / "cache")
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    first = subprocess.run(["bash", str(SCRIPT_PATH)], capture_output=True, text=True, env=env)
    second = subprocess.run(["bash", str(SCRIPT_PATH)], capture_output=True, text=True, env=env)

    assert first.returncode != 0
    assert second.returncode != 0
    assert "already passed" not in second.stdout
    assert len(pytest_log.read_text().splitlines()) >= 2


def test_changed_only_scope_maps_changed_source_to_targeted_tests(tmp_path: Path) -> None:
    test_root = tmp_path / "tests"
    test_root.mkdir()
    (test_root / "test_backup_daily.py").write_text("test placeholder\n")
    (test_root / "test_think_recall_integration.py").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["BRAINLAYER_PREPUSH"] = "1"
    env["BRAINLAYER_PREPUSH_SCOPE"] = "changed-only"
    env["BRAINLAYER_CHANGED_FILES"] = "src/brainlayer/backup_daily.py"
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    result = subprocess.run(["bash", str(SCRIPT_PATH)], capture_output=True, text=True, env=env)

    assert result.returncode == 0
    logged = pytest_log.read_text()
    assert str(test_root / "test_backup_daily.py") in logged
    assert f"{test_root}/ -v" not in logged


def test_changed_only_scope_maps_watcher_source_to_jsonl_watcher_tests(tmp_path: Path) -> None:
    test_root = tmp_path / "tests"
    test_root.mkdir()
    (test_root / "test_jsonl_watcher.py").write_text("test placeholder\n")
    (test_root / "test_think_recall_integration.py").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["BRAINLAYER_PREPUSH"] = "1"
    env["BRAINLAYER_PREPUSH_SCOPE"] = "changed-only"
    env["BRAINLAYER_CHANGED_FILES"] = "src/brainlayer/watcher.py"
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    result = subprocess.run(["bash", str(SCRIPT_PATH)], capture_output=True, text=True, env=env)

    assert result.returncode == 0
    logged = pytest_log.read_text()
    assert str(test_root / "test_jsonl_watcher.py") in logged
    assert "falling back to full pytest unit suite" not in result.stdout
    assert f"{test_root}/ -v" not in logged


def test_changed_only_scope_maps_source_class_storage_pipeline_to_safe_tests(tmp_path: Path) -> None:
    test_root = tmp_path / "tests"
    test_root.mkdir()
    expected = (
        "test_source_class.py",
        "test_vector_store_schema_flags.py",
        "test_vector_store_upsert_transactions.py",
        "test_hybrid_search.py",
        "test_vector_store_readonly.py",
        "test_search_trigram_fts.py",
        "test_precompact_chunk_origin.py",
        "test_ingest_t3.py",
        "test_context_pipeline.py",
    )
    for filename in (*expected, "test_think_recall_integration.py"):
        (test_root / filename).write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["BRAINLAYER_PREPUSH"] = "1"
    env["BRAINLAYER_PREPUSH_SCOPE"] = "changed-only"
    env["BRAINLAYER_CHANGED_FILES"] = "\n".join(
        [
            "src/brainlayer/vector_store.py",
            "src/brainlayer/search_repo.py",
            "src/brainlayer/index_new.py",
        ]
    )
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    result = subprocess.run(["bash", str(SCRIPT_PATH)], capture_output=True, text=True, env=env)

    assert result.returncode == 0
    logged = pytest_log.read_text()
    assert "falling back to full pytest unit suite" not in result.stdout
    assert f"{test_root}/ -v" not in logged
    for filename in expected:
        assert str(test_root / filename) in logged


def test_changed_only_scope_falls_back_when_mapped_and_unmapped_sources_change(tmp_path: Path) -> None:
    test_root = tmp_path / "tests"
    test_root.mkdir()
    (test_root / "test_backup_daily.py").write_text("test placeholder\n")
    (test_root / "test_think_recall_integration.py").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["BRAINLAYER_PREPUSH"] = "1"
    env["BRAINLAYER_PREPUSH_SCOPE"] = "changed-only"
    env["BRAINLAYER_CHANGED_FILES"] = "\n".join(
        ["src/brainlayer/backup_daily.py", "src/brainlayer/mcp/search_handler.py"]
    )
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    result = subprocess.run(["bash", str(SCRIPT_PATH)], capture_output=True, text=True, env=env)

    assert result.returncode == 0
    assert "falling back to full pytest unit suite" in result.stdout
    assert f"{test_root}/ -v" in pytest_log.read_text()


def test_changed_only_scope_falls_back_to_full_suite_for_unmapped_source(tmp_path: Path) -> None:
    test_root = tmp_path / "tests"
    test_root.mkdir()
    (test_root / "test_think_recall_integration.py").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["BRAINLAYER_PREPUSH"] = "1"
    env["BRAINLAYER_PREPUSH_SCOPE"] = "changed-only"
    env["BRAINLAYER_CHANGED_FILES"] = "src/brainlayer/mcp/search_handler.py"
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    result = subprocess.run(["bash", str(SCRIPT_PATH)], capture_output=True, text=True, env=env)

    assert result.returncode == 0
    assert "falling back to full pytest unit suite" in result.stdout
    # The escalation names what forced it, so the missing mapping is visible rather than paid for
    # silently on every push.
    assert "WARNING: unmapped: src/brainlayer/mcp/search_handler.py" in result.stdout
    assert f"{test_root}/ -v" in pytest_log.read_text()


def test_changed_only_scope_skips_the_unit_suite_for_a_measured_empty_diff(tmp_path: Path) -> None:
    """A MEASURED empty change set skips — and only a measured one.

    The old escalation was a fail-open the expensive way: it ran the most costly thing this script
    can do exactly when the evidence said there was nothing to run (on the M4 the full suite spawns
    `scripts/reembed_bgem3.py --test`, a 2.5 GB model holding fds on the production DB). The skip
    that replaced it is legitimate ONLY here, where git ran, answered, and named nothing — an empty
    HEAD commit with no origin/main. Detection FAILING takes the fail-closed path instead.
    """
    repo = _repo_with_an_empty_head_commit(tmp_path)
    test_root = repo / "tests"
    test_root.mkdir()
    (test_root / "test_think_recall_integration.py").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["BRAINLAYER_PREPUSH"] = "1"
    env["BRAINLAYER_PREPUSH_SCOPE"] = "changed-only"
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    result = subprocess.run(
        ["bash", str(repo / "scripts" / "run_tests.sh")],
        capture_output=True,
        text=True,
        env=env,
        check=False,  # the script's exit code is what this test asserts on
    )

    assert result.returncode == 0, result.stdout
    assert "WARNING: changed-only scope MEASURED an empty change set" in result.stdout
    assert "falling back to full pytest unit suite" not in result.stdout
    assert "could not be determined" not in result.stdout
    assert f"{test_root}/ -v" not in pytest_log.read_text()


def test_changed_only_scope_falls_back_to_full_suite_for_nested_hook_source(tmp_path: Path) -> None:
    test_root = tmp_path / "tests"
    test_root.mkdir()
    (test_root / "test_think_recall_integration.py").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["BRAINLAYER_PREPUSH"] = "1"
    env["BRAINLAYER_PREPUSH_SCOPE"] = "changed-only"
    env["BRAINLAYER_CHANGED_FILES"] = "src/brainlayer/hooks/indexer.py"
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    result = subprocess.run(["bash", str(SCRIPT_PATH)], capture_output=True, text=True, env=env)

    assert result.returncode == 0
    assert "falling back to full pytest unit suite" in result.stdout
    assert f"{test_root}/ -v" in pytest_log.read_text()


def test_changed_files_env_preserves_paths_with_spaces(tmp_path: Path) -> None:
    test_root = tmp_path / "tests"
    test_root.mkdir()
    spaced_test = test_root / "test_space path.py"
    spaced_test.write_text("test placeholder\n")
    (test_root / "test_think_recall_integration.py").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["BRAINLAYER_PREPUSH"] = "1"
    env["BRAINLAYER_PREPUSH_SCOPE"] = "changed-only"
    env["BRAINLAYER_CHANGED_FILES"] = "tests/test_space path.py"
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    result = subprocess.run(["bash", str(SCRIPT_PATH)], capture_output=True, text=True, env=env)

    assert result.returncode == 0
    assert str(spaced_test) in pytest_log.read_text()


def test_changed_only_scope_runs_nested_pytest_file(tmp_path: Path) -> None:
    test_root = tmp_path / "tests"
    nested_dir = test_root / "eval" / "phoenix_gate"
    nested_dir.mkdir(parents=True)
    nested_test = nested_dir / "test_phoenix_gate.py"
    nested_test.write_text("test placeholder\n")
    (test_root / "test_think_recall_integration.py").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["BRAINLAYER_PREPUSH"] = "1"
    env["BRAINLAYER_PREPUSH_SCOPE"] = "changed-only"
    env["BRAINLAYER_CHANGED_FILES"] = "tests/eval/phoenix_gate/test_phoenix_gate.py"
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    result = subprocess.run(["bash", str(SCRIPT_PATH)], capture_output=True, text=True, env=env)

    assert result.returncode == 0
    logged = pytest_log.read_text()
    assert str(nested_test) in logged
    assert f"{test_root}/ -v" not in logged


def test_changed_only_scope_falls_back_for_excluded_real_db_test_edit(tmp_path: Path) -> None:
    test_root = tmp_path / "tests"
    test_root.mkdir()
    (test_root / "test_vector_store.py").write_text("test placeholder\n")
    (test_root / "test_think_recall_integration.py").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["BRAINLAYER_PREPUSH"] = "1"
    env["BRAINLAYER_PREPUSH_SCOPE"] = "changed-only"
    env["BRAINLAYER_CHANGED_FILES"] = "tests/test_vector_store.py"
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    result = subprocess.run(["bash", str(SCRIPT_PATH)], capture_output=True, text=True, env=env)

    assert result.returncode == 0
    logged = pytest_log.read_text()
    assert "falling back to full pytest unit suite" in result.stdout
    assert f"{test_root}/ -v" in logged
    assert f"--ignore={test_root / 'test_vector_store.py'}" in logged


def test_worker_prepush_excludes_real_db_test_files(tmp_path: Path) -> None:
    test_root = tmp_path / "tests"
    test_root.mkdir()
    (test_root / "test_vector_store.py").write_text("test placeholder\n")
    (test_root / "test_engine.py").write_text("test placeholder\n")
    (test_root / "test_backup_daily.py").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["BRAINLAYER_PREPUSH"] = "1"
    env["BRAINLAYER_PREPUSH_CACHE_DIR"] = str(tmp_path / "cache")
    env["BRAINLAYER_PREPUSH_SCOPE"] = "full"
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    result = subprocess.run(["bash", str(SCRIPT_PATH)], capture_output=True, text=True, env=env)

    assert result.returncode == 0
    logged = pytest_log.read_text()
    assert f"--ignore={test_root / 'test_vector_store.py'}" in logged
    assert f"--ignore={test_root / 'test_engine.py'}" in logged


def test_default_unit_mark_expression_deselects_embedding_model_tests() -> None:
    """Pre-push must never load a real embedding model on Etan's Macs.

    `tests/conftest.py` fails any UNMARKED test that loads one; the marker is the declared escape,
    and this script is where the declaration has to be honoured. CI passes its own `-m` expression
    in `.github/workflows/ci.yml` and still runs them, on a runner that warms the HF cache.
    """
    text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert (
        'UNIT_MARK_EXPR="${BRAINLAYER_PYTEST_MARK_EXPR:-not integration and not live and not embedding_model}"' in text
    )


def test_changed_only_scope_fails_closed_when_git_cannot_name_the_changed_files(tmp_path: Path) -> None:
    """Detection FAILING must not read as "nothing changed".

    `changed_files()` used to swallow a git error into an empty list with rc 0, so a missing
    origin/main or a broken diff produced the same evidence as a genuinely empty change set — and
    the skip below then reported green. That is a fail-OPEN on a gate: the previous behaviour was
    wrong because it was expensive, this one would be wrong because it is silent.
    """
    test_root = tmp_path / "tests"
    test_root.mkdir()
    (test_root / "test_think_recall_integration.py").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)
    script = _script_outside_a_repo(tmp_path)

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["BRAINLAYER_PREPUSH"] = "1"
    env["BRAINLAYER_PREPUSH_SCOPE"] = "changed-only"
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    result = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        env=env,
        check=False,  # a NON-ZERO exit is the assertion; raising here would hide it
    )

    assert result.returncode != 0, result.stdout
    assert "FAIL: changed-only scope could not be determined" in result.stdout
    assert "no origin/main and no HEAD~1" in result.stdout
    # And it did NOT quietly widen to the full suite either.
    assert f"{test_root}/ -v" not in pytest_log.read_text()


def test_changed_only_scope_fails_closed_when_the_env_names_no_paths(tmp_path: Path) -> None:
    """A caller that ASSERTS a scope and names nothing in it has made an error, not a measurement."""
    test_root = tmp_path / "tests"
    test_root.mkdir()
    (test_root / "test_think_recall_integration.py").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["BRAINLAYER_PREPUSH"] = "1"
    env["BRAINLAYER_PREPUSH_SCOPE"] = "changed-only"
    env["BRAINLAYER_CHANGED_FILES"] = "\n"
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        env=env,
        check=False,  # a NON-ZERO exit is the assertion; raising here would hide it
    )

    assert result.returncode != 0, result.stdout
    assert "BRAINLAYER_CHANGED_FILES was set but named no paths" in result.stdout
    assert f"{test_root}/ -v" not in pytest_log.read_text()


def test_changed_only_scope_maps_package_init_to_version_consistency_tests(
    tmp_path: Path,
) -> None:
    """A release bump touches src/brainlayer/__init__.py and nothing else in src/.

    There is no tests/test___init__.py, so the generic src/brainlayer/*.py rule found no
    target and the run escalated to the whole 4,386-test suite -- on every release, for a
    one-line version string. The version bump's real gate is the metadata consistency
    suite, so name it.
    """
    test_root = tmp_path / "tests"
    test_root.mkdir()
    (test_root / "test_version_consistency.py").write_text("test placeholder\n")
    (test_root / "test_build_sha.py").write_text("test placeholder\n")
    (test_root / "test_think_recall_integration.py").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["BRAINLAYER_PREPUSH"] = "1"
    env["BRAINLAYER_PREPUSH_SCOPE"] = "changed-only"
    env["BRAINLAYER_CHANGED_FILES"] = "src/brainlayer/__init__.py"
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    result = subprocess.run(  # noqa: S603 - returncode is asserted below
        ["bash", str(SCRIPT_PATH)], capture_output=True, text=True, env=env, check=False
    )

    assert result.returncode == 0
    logged = pytest_log.read_text()
    # BOTH suites, not just the version one: test_version_consistency.py reads __version__ with
    # `ast` and never imports the package, so alone it would go green while `import brainlayer`
    # was broken. test_build_sha.py is the one that actually imports it.
    assert str(test_root / "test_version_consistency.py") in logged
    assert str(test_root / "test_build_sha.py") in logged
    assert "falling back to full pytest unit suite" not in result.stdout
    assert f"{test_root}/ -v" not in logged


def test_changed_only_scope_escalates_when_the_build_sha_sibling_is_missing(
    tmp_path: Path,
) -> None:
    """A missing sibling must escalate, not silently narrow the gate.

    The __init__.py mapping names two suites because the file has two behaviours. If
    test_build_sha.py is deleted or renamed, mapping to test_version_consistency.py alone
    would go green while `import brainlayer` was broken -- that AST-only suite never imports
    the package. That is the same fail-open, reintroduced through a missing sibling instead
    of an incomplete case list, so a partial mapping must not count as mapped.
    """
    test_root = tmp_path / "tests"
    test_root.mkdir()
    (test_root / "test_version_consistency.py").write_text("test placeholder\n")
    # test_build_sha.py deliberately absent
    (test_root / "test_think_recall_integration.py").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["BRAINLAYER_PREPUSH"] = "1"
    env["BRAINLAYER_PREPUSH_SCOPE"] = "changed-only"
    env["BRAINLAYER_CHANGED_FILES"] = "src/brainlayer/__init__.py"
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    result = subprocess.run(  # noqa: S603 - returncode is asserted below
        ["bash", str(SCRIPT_PATH)], capture_output=True, text=True, env=env, check=False
    )

    assert result.returncode == 0
    assert "falling back to full pytest unit suite" in result.stdout
    assert "WARNING: unmapped: src/brainlayer/__init__.py" in result.stdout
    logged = pytest_log.read_text()
    assert f"{test_root}/ -v" in logged


def test_changed_only_scope_escalates_when_the_version_consistency_sibling_is_missing(
    tmp_path: Path,
) -> None:
    """The mirror of the build-sha case: either sibling missing must escalate.

    The round-1 review on #762 asked for the missing-sibling regression and got the
    test_build_sha.py half of it. Nothing pinned the other direction, so a mapping that
    kept `mapped=1` when only test_build_sha.py existed would still have looked covered:
    the import path gated, the six version sites and the cask-lag reason not gated at all.
    Both suites exist or the file is unmapped.
    """
    test_root = tmp_path / "tests"
    test_root.mkdir()
    (test_root / "test_build_sha.py").write_text("test placeholder\n")
    # test_version_consistency.py deliberately absent
    (test_root / "test_think_recall_integration.py").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["BRAINLAYER_PREPUSH"] = "1"
    env["BRAINLAYER_PREPUSH_SCOPE"] = "changed-only"
    env["BRAINLAYER_CHANGED_FILES"] = "src/brainlayer/__init__.py"
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    result = subprocess.run(  # noqa: S603 - returncode is asserted below
        ["bash", str(SCRIPT_PATH)], capture_output=True, text=True, env=env, check=False
    )

    assert result.returncode == 0
    assert "falling back to full pytest unit suite" in result.stdout
    assert "WARNING: unmapped: src/brainlayer/__init__.py" in result.stdout
    logged = pytest_log.read_text()
    assert f"{test_root}/ -v" in logged


def test_changed_only_scope_escalates_when_both_init_suites_are_missing(
    tmp_path: Path,
) -> None:
    """Neither suite present is the same fail-open, and must escalate too."""
    test_root = tmp_path / "tests"
    test_root.mkdir()
    (test_root / "test_think_recall_integration.py").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["BRAINLAYER_PREPUSH"] = "1"
    env["BRAINLAYER_PREPUSH_SCOPE"] = "changed-only"
    env["BRAINLAYER_CHANGED_FILES"] = "src/brainlayer/__init__.py"
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    result = subprocess.run(  # noqa: S603 - returncode is asserted below
        ["bash", str(SCRIPT_PATH)], capture_output=True, text=True, env=env, check=False
    )

    assert result.returncode == 0
    assert "falling back to full pytest unit suite" in result.stdout
    logged = pytest_log.read_text()
    assert f"{test_root}/ -v" in logged


def test_changed_only_scope_resolves_the_changed_set_from_a_tag_range(tmp_path: Path) -> None:
    """A tag push has no branch diff, so it needs a range -- previous tag..this tag.

    `git push origin v1.5.14` used to arrive with nothing to diff against and run the DEFAULT
    scope: the whole 4,386-test suite, on the M4. A tag's content is exactly one range, so the
    script accepts one.
    """
    repo = _repo_with_two_tags(tmp_path)
    test_root = tmp_path / "tests"
    test_root.mkdir()
    (test_root / "test_jsonl_watcher.py").write_text("test placeholder\n")
    (test_root / "test_source_class.py").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["BRAINLAYER_PREPUSH"] = "1"
    env["BRAINLAYER_PREPUSH_SCOPE"] = "changed-only"
    env["BRAINLAYER_CHANGED_FILES_RANGE"] = "v1.0.0..v1.1.0"
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    result = subprocess.run(  # noqa: S603 - returncode is asserted below
        ["bash", str(repo / "scripts" / "run_tests.sh")], capture_output=True, text=True, env=env, check=False
    )

    assert result.returncode == 0, result.stdout
    logged = pytest_log.read_text()
    assert str(test_root / "test_jsonl_watcher.py") in logged
    # The post-tag commit's module maps here. Seeing it would mean the run fell back to the
    # branch diff and only LOOKED like it honoured the range.
    assert str(test_root / "test_source_class.py") not in logged
    assert f"{test_root}/ -v" not in logged
    assert "falling back to full pytest unit suite" not in result.stdout


def test_changed_only_scope_fails_closed_on_a_range_git_cannot_resolve(tmp_path: Path) -> None:
    """An unresolvable range is a scope that was never established, not an empty one."""
    repo = _repo_with_two_tags(tmp_path)
    test_root = tmp_path / "tests"
    test_root.mkdir()
    (test_root / "test_jsonl_watcher.py").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["BRAINLAYER_PREPUSH"] = "1"
    env["BRAINLAYER_PREPUSH_SCOPE"] = "changed-only"
    env["BRAINLAYER_CHANGED_FILES_RANGE"] = "v9.9.9..v9.9.8"
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    result = subprocess.run(  # noqa: S603 - returncode is asserted below
        ["bash", str(repo / "scripts" / "run_tests.sh")], capture_output=True, text=True, env=env, check=False
    )

    assert result.returncode != 0
    assert "changed-only scope could not be determined" in result.stdout
    assert f"{test_root}/ -v" not in pytest_log.read_text()


def test_changed_only_scope_measures_an_empty_tag_range_instead_of_escalating(tmp_path: Path) -> None:
    """Two tags on one commit is a MEASURED empty set: skip loudly, never escalate."""
    repo = _repo_with_two_tags(tmp_path)
    _git(repo, "tag", "v1.1.1", "v1.1.0")
    test_root = tmp_path / "tests"
    test_root.mkdir()
    (test_root / "test_jsonl_watcher.py").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["BRAINLAYER_PREPUSH"] = "1"
    env["BRAINLAYER_PREPUSH_SCOPE"] = "changed-only"
    env["BRAINLAYER_CHANGED_FILES_RANGE"] = "v1.1.0..v1.1.1"
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    result = subprocess.run(  # noqa: S603 - returncode is asserted below
        ["bash", str(repo / "scripts" / "run_tests.sh")], capture_output=True, text=True, env=env, check=False
    )

    assert result.returncode == 0, result.stdout
    assert "MEASURED an empty change set" in result.stdout
    assert f"{test_root}/ -v" not in pytest_log.read_text()


def test_pre_push_hook_scopes_a_tag_push_to_the_previous_tag(tmp_path: Path) -> None:
    """The refs live on the hook's STDIN, which the hook used to throw away.

    That is the whole reason `git push origin v1.5.14` ran 4,386 tests: nothing downstream
    could know a tag was being pushed, so nothing could name the range.
    """
    repo, env_log = _repo_with_the_pre_push_hook(tmp_path)
    sha = _rev_parse(repo, "v1.1.0")

    result = _run_hook(repo, env_log, f"refs/tags/v1.1.0 {sha} refs/tags/v1.1.0 {'0' * 40}\n")

    assert result.returncode == 0, result.stdout + result.stderr
    handed = env_log.read_text()
    assert "SCOPE=changed-only" in handed
    assert "RANGE=v1.0.0..v1.1.0" in handed
    assert "v1.0.0..v1.1.0" in result.stdout


def test_pre_push_hook_escalates_loudly_when_a_tag_has_no_previous_tag(tmp_path: Path) -> None:
    """No previous tag means no range. Escalate -- but say so, do not narrow silently."""
    repo, env_log = _repo_with_the_pre_push_hook(tmp_path)
    sha = _rev_parse(repo, "v1.0.0")

    result = _run_hook(repo, env_log, f"refs/tags/v1.0.0 {sha} refs/tags/v1.0.0 {'0' * 40}\n")

    assert result.returncode == 0, result.stdout + result.stderr
    handed = env_log.read_text()
    assert "SCOPE=<unset>" in handed
    assert "RANGE=<unset>" in handed
    assert "no previous release tag" in result.stdout


def test_pre_push_hook_leaves_a_branch_push_alone(tmp_path: Path) -> None:
    """A branch push already has a diff. The hook must not touch its scope."""
    repo, env_log = _repo_with_the_pre_push_hook(tmp_path)
    sha = _rev_parse(repo, "HEAD")

    result = _run_hook(repo, env_log, f"refs/heads/main {sha} refs/heads/main {'0' * 40}\n")

    assert result.returncode == 0, result.stdout + result.stderr
    handed = env_log.read_text()
    assert "SCOPE=<unset>" in handed
    assert "RANGE=<unset>" in handed


def test_pre_push_hook_leaves_a_mixed_branch_and_tag_push_alone(tmp_path: Path) -> None:
    """A branch riding along carries changes no tag range covers, so the branch wins."""
    repo, env_log = _repo_with_the_pre_push_hook(tmp_path)
    head = _rev_parse(repo, "HEAD")
    tag = _rev_parse(repo, "v1.1.0")

    result = _run_hook(
        repo,
        env_log,
        f"refs/heads/main {head} refs/heads/main {'0' * 40}\nrefs/tags/v1.1.0 {tag} refs/tags/v1.1.0 {'0' * 40}\n",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    handed = env_log.read_text()
    assert "SCOPE=<unset>" in handed
    assert "RANGE=<unset>" in handed
    # Refusing to narrow is only half the contract; saying WHY is the other half, and this case
    # skipped the messaging block entirely (#775 round-1 review, medium).
    assert "a branch is in this push" in result.stdout


def test_pre_push_hook_ignores_a_tag_deletion(tmp_path: Path) -> None:
    """An all-zero LOCAL sha is a deletion: it ships no content, so there is nothing to scope."""
    repo, env_log = _repo_with_the_pre_push_hook(tmp_path)
    sha = _rev_parse(repo, "v1.1.0")

    result = _run_hook(repo, env_log, f"(delete) {'0' * 40} refs/tags/v1.1.0 {sha}\n")

    assert result.returncode == 0, result.stdout + result.stderr
    handed = env_log.read_text()
    assert "SCOPE=<unset>" in handed
    assert "RANGE=<unset>" in handed


def test_pre_push_hook_respects_an_explicit_changed_set_on_a_tag_push(tmp_path: Path) -> None:
    """An operator who named the scope has already decided. The hook does not overrule them."""
    repo, env_log = _repo_with_the_pre_push_hook(tmp_path)
    sha = _rev_parse(repo, "v1.1.0")

    result = _run_hook(
        repo,
        env_log,
        f"refs/tags/v1.1.0 {sha} refs/tags/v1.1.0 {'0' * 40}\n",
        extra_env={
            "BRAINLAYER_PREPUSH_SCOPE": "changed-only",
            "BRAINLAYER_CHANGED_FILES": "tests/test_jsonl_watcher.py",
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    handed = env_log.read_text()
    assert "FILES=tests/test_jsonl_watcher.py" in handed
    assert "RANGE=<unset>" in handed


def test_pre_push_hook_skips_a_non_release_predecessor_tag(tmp_path: Path) -> None:
    """The predecessor must be a RELEASE tag, or the gate under-scopes itself.

    `git describe --tags --abbrev=0` returns the nearest reachable tag of ANY name. With a
    `nightly` between v1.0.0 and v1.1.0 the range would start at `nightly`, and the commits
    between v1.0.0 and it would never be mapped -- a fail-open on a pre-push regression gate.
    This repo already carries non-v* tags (`pre-rename`, `archive/*`), so the footgun is real.
    """
    repo, env_log = _repo_with_a_non_release_tag_between_releases(tmp_path)
    sha = _rev_parse(repo, "v1.1.0")

    result = _run_hook(repo, env_log, f"refs/tags/v1.1.0 {sha} refs/tags/v1.1.0 {'0' * 40}\n")

    assert result.returncode == 0, result.stdout + result.stderr
    handed = env_log.read_text()
    assert "RANGE=v1.0.0..v1.1.0" in handed
    assert "nightly" not in handed


def test_pre_push_hook_never_narrows_an_explicitly_requested_scope(tmp_path: Path) -> None:
    """`BRAINLAYER_PREPUSH_SCOPE=full git push origin vX` asked for the whole suite. Explicit wins.

    The operator escape only looked at BRAINLAYER_CHANGED_FILES/_RANGE, so an explicit `full` was
    overwritten with `changed-only` plus a range -- defeating an intentional full-suite tag push on
    the very gate this change hardens (#775 round-1 review, high).
    """
    repo, env_log = _repo_with_the_pre_push_hook(tmp_path)
    sha = _rev_parse(repo, "v1.1.0")

    result = _run_hook(
        repo,
        env_log,
        f"refs/tags/v1.1.0 {sha} refs/tags/v1.1.0 {'0' * 40}\n",
        extra_env={"BRAINLAYER_PREPUSH_SCOPE": "full"},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    handed = env_log.read_text()
    assert "SCOPE=full" in handed
    assert "RANGE=<unset>" in handed
    assert "TAG=<unset>" in handed
    assert "set explicitly" in result.stdout


def test_pre_push_hook_attaches_the_range_under_an_explicit_changed_only_scope(tmp_path: Path) -> None:
    """An explicit scope picks the MODE. The range is the DATA, and a tag still needs it.

    My round-1 fix over-corrected: any explicit BRAINLAYER_PREPUSH_SCOPE blocked the range, and
    AGENTS.md documents `BRAINLAYER_PREPUSH_SCOPE=changed-only git push` as the normal worker path.
    Under that env a tag-only push got no range and fell back to `origin/main...HEAD` -- which at a
    release tip matching origin/main is EMPTY, so the MEASURED-empty path skipped the unit suite
    entirely. Fail-open against both the old full-suite tag behaviour and this PR's own contract,
    and the previous version of this test locked it in (#775 round-2 review, high).
    """
    repo, env_log = _repo_with_the_pre_push_hook(tmp_path)
    sha = _rev_parse(repo, "v1.1.0")

    result = _run_hook(
        repo,
        env_log,
        f"refs/tags/v1.1.0 {sha} refs/tags/v1.1.0 {'0' * 40}\n",
        extra_env={"BRAINLAYER_PREPUSH_SCOPE": "changed-only"},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    handed = env_log.read_text()
    assert "SCOPE=changed-only" in handed
    assert "RANGE=v1.0.0..v1.1.0" in handed
    assert "TAG=v1.1.0" in handed


def test_pre_push_hook_refuses_loudly_when_several_tags_ride_one_push(tmp_path: Path) -> None:
    """Two tags have no single range. The PR claims this path is loud; assert that it is."""
    repo, env_log = _repo_with_the_pre_push_hook(tmp_path)
    first = _rev_parse(repo, "v1.0.0")
    second = _rev_parse(repo, "v1.1.0")

    result = _run_hook(
        repo,
        env_log,
        f"refs/tags/v1.0.0 {first} refs/tags/v1.0.0 {'0' * 40}\n"
        f"refs/tags/v1.1.0 {second} refs/tags/v1.1.0 {'0' * 40}\n",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    handed = env_log.read_text()
    assert "SCOPE=<unset>" in handed
    assert "RANGE=<unset>" in handed
    assert "2 tags in one push" in result.stdout


def test_pre_push_hook_leaves_an_unrecognised_explicit_scope_alone(tmp_path: Path) -> None:
    """Only `changed-only` accepts a range. Anything else explicit is the caller's, untouched."""
    repo, env_log = _repo_with_the_pre_push_hook(tmp_path)
    sha = _rev_parse(repo, "v1.1.0")

    result = _run_hook(
        repo,
        env_log,
        f"refs/tags/v1.1.0 {sha} refs/tags/v1.1.0 {'0' * 40}\n",
        extra_env={"BRAINLAYER_PREPUSH_SCOPE": "belt-and-braces"},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    handed = env_log.read_text()
    assert "SCOPE=belt-and-braces" in handed
    assert "RANGE=<unset>" in handed


def test_pre_push_hook_skips_a_pre_release_predecessor_tag(tmp_path: Path) -> None:
    """`--match 'v*'` alone still matches v1.1.0-rc1, which is the same under-scope one name up.

    Measured on git 2.54.0: `describe --tags --abbrev=0 --match 'v*' v1.1.0^` answers `v1.1.0-rc1`;
    adding `--exclude '*-*'` answers `v1.0.0`. Over-scoping from the last full release is safe;
    starting at an rc is not (#775 round-2 review, low).
    """
    repo, env_log = _repo_with_a_pre_release_tag_between_releases(tmp_path)
    sha = _rev_parse(repo, "v1.1.0")

    result = _run_hook(repo, env_log, f"refs/tags/v1.1.0 {sha} refs/tags/v1.1.0 {'0' * 40}\n")

    assert result.returncode == 0, result.stdout + result.stderr
    handed = env_log.read_text()
    assert "RANGE=v1.0.0..v1.1.0" in handed
    assert "rc1" not in handed


def test_pre_push_hook_scopes_an_annotated_tag(tmp_path: Path) -> None:
    """Release tags are usually annotated (`git tag -a`), which is a tag OBJECT, not a ref alias."""
    repo, env_log = _repo_with_annotated_tags(tmp_path)
    sha = _rev_parse(repo, "v1.1.0")

    result = _run_hook(repo, env_log, f"refs/tags/v1.1.0 {sha} refs/tags/v1.1.0 {'0' * 40}\n")

    assert result.returncode == 0, result.stdout + result.stderr
    handed = env_log.read_text()
    assert "SCOPE=changed-only" in handed
    assert "RANGE=v1.0.0..v1.1.0" in handed


def test_changed_only_scope_escalates_an_empty_tag_range_instead_of_skipping(tmp_path: Path) -> None:
    """A RELEASE tag is the one place the full suite is right: it must never skip silently.

    The sibling test above proves a measured-empty range SKIPS loudly for an ordinary caller, which
    is the ratified behaviour. For a tag it is fail-open: before this PR a tag push ran the full
    suite, and a bump whose range maps nothing would now gate nothing at all (#775 round-2 review,
    medium). BRAINLAYER_PREPUSH_TAG is how the hook says "this scope came from a release tag".
    """
    repo = _repo_with_two_tags(tmp_path)
    _git(repo, "tag", "v1.1.1", "v1.1.0")
    test_root = tmp_path / "tests"
    test_root.mkdir()
    (test_root / "test_jsonl_watcher.py").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["BRAINLAYER_PREPUSH"] = "1"
    env["BRAINLAYER_PREPUSH_SCOPE"] = "changed-only"
    env["BRAINLAYER_CHANGED_FILES_RANGE"] = "v1.1.0..v1.1.1"
    env["BRAINLAYER_PREPUSH_TAG"] = "v1.1.1"
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    result = subprocess.run(  # noqa: S603 - returncode is asserted below
        ["bash", str(repo / "scripts" / "run_tests.sh")], capture_output=True, text=True, env=env, check=False
    )

    assert result.returncode == 0, result.stdout
    assert "SKIPPING the pytest unit suite" not in result.stdout
    assert "release tag v1.1.1" in result.stdout
    assert f"{test_root}/ -v" in pytest_log.read_text()


def test_changed_only_scope_escalates_a_tag_range_that_maps_nothing(tmp_path: Path) -> None:
    """A docs-only or changelog-only bump maps no pytest target. For a tag, that means FULL."""
    repo = _repo_with_two_tags(tmp_path)
    # Tag the post-v1.1.0 src commit, so the range under test holds ONLY the changelog. Without
    # this the range also carries src/brainlayer/index_new.py and escalates via the existing
    # unmapped-source path -- green for a different reason than the one being asserted.
    _git(repo, "tag", "v1.1.5")
    (repo / "CHANGELOG.md").write_text("# 1.2.0\n", encoding="utf-8")
    _git(repo, "add", "CHANGELOG.md")
    _git(repo, "commit", "-qm", "changelog only")
    _git(repo, "tag", "v1.2.0")
    test_root = tmp_path / "tests"
    test_root.mkdir()
    (test_root / "test_jsonl_watcher.py").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)

    env = _script_env()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["BRAINLAYER_PREPUSH"] = "1"
    env["BRAINLAYER_PREPUSH_SCOPE"] = "changed-only"
    env["BRAINLAYER_CHANGED_FILES_RANGE"] = "v1.1.5..v1.2.0"
    env["BRAINLAYER_PREPUSH_TAG"] = "v1.2.0"
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    result = subprocess.run(  # noqa: S603 - returncode is asserted below
        ["bash", str(repo / "scripts" / "run_tests.sh")], capture_output=True, text=True, env=env, check=False
    )

    assert result.returncode == 0, result.stdout
    assert "SKIP: changed-only scope found no mapped pytest targets" not in result.stdout
    assert "release tag v1.2.0" in result.stdout
    assert f"{test_root}/ -v" in pytest_log.read_text()
