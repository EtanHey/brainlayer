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
