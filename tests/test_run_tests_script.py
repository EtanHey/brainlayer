"""Contract tests for scripts/run_tests.sh."""

import os
import stat
import subprocess
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_tests.sh"


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

    result = subprocess.run(["bash", str(repo / "scripts" / "run_tests.sh")], capture_output=True, text=True, env=env)

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

    result = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env)

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

    result = subprocess.run(["bash", str(SCRIPT_PATH)], capture_output=True, text=True, env=env)

    assert result.returncode != 0, result.stdout
    assert "BRAINLAYER_CHANGED_FILES was set but named no paths" in result.stdout
    assert f"{test_root}/ -v" not in pytest_log.read_text()
