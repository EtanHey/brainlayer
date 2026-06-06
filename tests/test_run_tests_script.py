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


def test_run_tests_aggregates_exit_codes_and_keeps_running(tmp_path: Path) -> None:
    test_root = tmp_path / "tests"
    test_root.mkdir()
    (test_root / "fixture.test.ts").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=2, bun_exit=4)

    env = os.environ.copy()
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

    env = os.environ.copy()
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

    env = os.environ.copy()
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

    env = os.environ.copy()
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

    env = os.environ.copy()
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

    env = os.environ.copy()
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


def test_changed_only_scope_falls_back_when_mapped_and_unmapped_sources_change(tmp_path: Path) -> None:
    test_root = tmp_path / "tests"
    test_root.mkdir()
    (test_root / "test_backup_daily.py").write_text("test placeholder\n")
    (test_root / "test_think_recall_integration.py").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)

    env = os.environ.copy()
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

    env = os.environ.copy()
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
    assert f"{test_root}/ -v" in pytest_log.read_text()


def test_changed_only_scope_falls_back_to_full_suite_for_empty_diff(tmp_path: Path) -> None:
    test_root = tmp_path / "tests"
    test_root.mkdir()
    (test_root / "test_think_recall_integration.py").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)

    env = os.environ.copy()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["BRAINLAYER_PREPUSH"] = "1"
    env["BRAINLAYER_PREPUSH_SCOPE"] = "changed-only"
    env["BRAINLAYER_CHANGED_FILES"] = "\n"
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    result = subprocess.run(["bash", str(SCRIPT_PATH)], capture_output=True, text=True, env=env)

    assert result.returncode == 0
    assert "changed-only scope found no changed files; falling back to full pytest unit suite" in result.stdout
    assert f"{test_root}/ -v" in pytest_log.read_text()


def test_changed_only_scope_falls_back_to_full_suite_for_nested_hook_source(tmp_path: Path) -> None:
    test_root = tmp_path / "tests"
    test_root.mkdir()
    (test_root / "test_think_recall_integration.py").write_text("test placeholder\n")

    pytest_log, bun_log = _make_stub_bin(tmp_path, pytest_exit=0, bun_exit=0)

    env = os.environ.copy()
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

    env = os.environ.copy()
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

    env = os.environ.copy()
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

    env = os.environ.copy()
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

    env = os.environ.copy()
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["BRAINLAYER_TEST_ROOT"] = str(test_root)
    env["BRAINLAYER_USE_UV"] = "0"
    env["BRAINLAYER_PREPUSH"] = "1"
    env["BRAINLAYER_PREPUSH_SCOPE"] = "full"
    env["PYTEST_LOG"] = str(pytest_log)
    env["BUN_LOG"] = str(bun_log)

    result = subprocess.run(["bash", str(SCRIPT_PATH)], capture_output=True, text=True, env=env)

    assert result.returncode == 0
    logged = pytest_log.read_text()
    assert f"--ignore={test_root / 'test_vector_store.py'}" in logged
    assert f"--ignore={test_root / 'test_engine.py'}" in logged
