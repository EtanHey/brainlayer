"""Spotlight hygiene checks run against owned trees, never the whole disk."""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "scripts/check_stray_apps.py"


@pytest.fixture
def scan_root():
    parent = ROOT / "docs.local"
    parent.mkdir(exist_ok=True)
    # Deliberately indexable fixtures to prove the guard; removed even on assertion failure.
    with tempfile.TemporaryDirectory(prefix="app-hygiene-probe-", dir=parent) as directory:
        yield Path(directory)


def test_scan_rejects_strays_and_ignores_git_and_noindex(scan_root):
    repo = scan_root / "repo"
    scratch = scan_root / "scratch"
    for path in (
        repo / "docs.local/extracted/BrainBar.app",
        repo / ".git/ignored.app",
        repo / "docs.local/artifacts.noindex/release/BrainBar.app",
        scratch / "Applications/BrainBar.app",
    ):
        path.mkdir(parents=True)
    result = subprocess.run([sys.executable, str(CHECK), str(repo), str(scratch)], capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    assert "docs.local/extracted/BrainBar.app" in result.stdout
    assert "scratch/Applications/BrainBar.app" in result.stdout
    assert "ignored.app" not in result.stdout
    assert "artifacts.noindex/release" not in result.stdout


@pytest.mark.parametrize("explicit", [False, True])
def test_pytest_routes_temp_bundles_under_noindex(scan_root, explicit):
    probe = scan_root / "probe"
    probe.mkdir()
    (probe / "pytest.ini").write_text("[pytest]\n")
    (probe / "test_probe.py").write_text(
        "def test_bundle(tmp_path):\n"
        "    assert any(p.endswith('.noindex') for p in tmp_path.parts), str(tmp_path)\n"
        "    (tmp_path / 'Applications/BrainBar.app').mkdir(parents=True)\n"
    )
    command = [sys.executable, "-m", "pytest", "-p", "tests.conftest", "-q", str(probe)]
    if explicit:
        command += ["--basetemp", str(probe / "custom-temp")]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_session_end_scan_fails_for_a_bundle_left_by_teardown(scan_root):
    probe = scan_root / "test_probe.py"
    probe.write_text(
        "from pathlib import Path\n"
        "import pytest\n"
        "@pytest.fixture\n"
        "def leave_bundle():\n"
        "    yield\n"
        "    (Path(__file__).parent / 'BrainBar.app').mkdir()\n"
        "def test_passing(leave_bundle):\n"
        "    pass\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "tests.conftest", "-q", str(probe)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    assert str(scan_root / "BrainBar.app") in result.stdout


def test_build_refuses_an_indexable_output_override(tmp_path, scan_root):
    from tests.test_brainbar_build_app_guards import _prepare_build_repo, _run_build_script

    repo, script = _prepare_build_repo(tmp_path, "repo")
    unsafe_app = scan_root / "BrainBar.app"
    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=tmp_path / "home",
        extra_env={"BRAINBAR_APP_DIR": str(unsafe_app)},
    )
    assert result.returncode == 1
    assert "must live under *.noindex" in result.stderr
    assert not unsafe_app.exists()
