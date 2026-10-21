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


def test_scan_allows_only_build_noindex_and_scans_nested_worktrees(tmp_path):
    repo = tmp_path / "repo"
    allowed = repo / ".build.noindex/build/BrainBar.app"
    allowed.mkdir(parents=True)
    (repo / ".git/ignored.app").mkdir(parents=True)
    rejected = [
        repo / "dist/BrainBar.app",
        repo / "other.noindex/BrainBar.app",
        repo / ".worktrees/child/BrainBar.app",
        repo / ".worktrees/child/docs.local/archive/worktrees-old/BrainBar.app",
    ]
    (repo / ".worktrees/child/.build.noindex/build/BrainBar.app").mkdir(parents=True)
    for path in rejected:
        path.mkdir(parents=True)
    result = subprocess.run([sys.executable, str(CHECK), str(repo)], capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    for path in rejected:
        assert str(path) in result.stdout
    assert str(allowed) not in result.stdout
    assert "ignored.app" not in result.stdout


@pytest.mark.parametrize(
    "archive", ["artifacts.noindex", "archive/worktrees-old", "archive/worktrees-old/.build.noindex"]
)
def test_docs_local_archive_excludes_apps_and_large_binaries_even_under_noindex(tmp_path, archive):
    repo = tmp_path / "repo"
    archived = repo / "docs.local" / archive
    app = archived / "BrainBar.app"
    app.mkdir(parents=True)
    binary = archived / "BrainBarDaemon"
    with binary.open("wb") as stream:
        stream.write(b"\x00Mach-O")
        stream.truncate(1024 * 1024 + 1)
    receipt = archived / "receipt.log"
    receipt.write_text("version=1.5.68 sha=abc cdhash=def\n")
    result = subprocess.run([sys.executable, str(CHECK), str(repo)], capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    assert str(app) in result.stdout
    assert str(binary) in result.stdout
    assert str(receipt) not in result.stdout


def test_binary_threshold_and_scan_boundary(tmp_path):
    repo = tmp_path / "repo"
    docs = repo / "docs.local"
    docs.mkdir(parents=True)
    (docs / "small.bin").write_bytes(b"\x00" * (1024 * 1024))
    (docs / "large.log").write_text("x" * (1024 * 1024 + 1))
    outside = tmp_path / "outside"
    (outside / "BrainBar.app").mkdir(parents=True)
    (repo / "external").symlink_to(outside, target_is_directory=True)
    result = subprocess.run([sys.executable, str(CHECK), str(repo)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("explicit", [None, "custom-temp", "custom.noindex"])
def test_pytest_routes_temp_bundles_under_worktree_build_root(scan_root, explicit):
    probe = scan_root / "probe"
    probe.mkdir()
    (probe / "pytest.ini").write_text("[pytest]\n")
    (probe / "test_probe.py").write_text(
        "def test_bundle(tmp_path):\n"
        f"    assert tmp_path.is_relative_to({str(ROOT / '.build.noindex/pytest')!r}), str(tmp_path)\n"
        "    (tmp_path / 'Applications/BrainBar.app').mkdir(parents=True)\n"
    )
    command = [sys.executable, "-m", "pytest", "-p", "tests.conftest", "-q", str(probe)]
    if explicit:
        command += ["--basetemp", str(probe / explicit)]
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
    assert "must live under" in result.stderr
    assert not unsafe_app.exists()


@pytest.mark.parametrize(
    "output",
    [
        "docs.local/artifacts.noindex/BrainBar.app",
        "other.noindex/BrainBar.app",
        ".build.noindex/../../escaped/missing/BrainBar.app",
    ],
)
def test_build_refuses_other_noindex_output_overrides(tmp_path, output):
    from tests.test_brainbar_build_app_guards import _prepare_build_repo, _run_build_script

    repo, script = _prepare_build_repo(tmp_path, "repo")
    result = _run_build_script(
        repo,
        script,
        canonical_root=repo,
        home=tmp_path / "home",
        extra_env={"BRAINBAR_APP_DIR": str(repo / output)},
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "must live under" in result.stderr
