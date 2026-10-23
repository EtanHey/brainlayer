"""Spotlight hygiene checks run against owned trees, never the whole disk."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "scripts/check_stray_apps.py"


def _hygiene_session(checkout, *, plugin_root=None):
    plugin_root = plugin_root or checkout
    tests = plugin_root / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    (tests / "__init__.py").touch()
    shutil.copy2(ROOT / "tests/conftest.py", tests / "conftest.py")
    scripts = plugin_root / "scripts"
    scripts.mkdir(exist_ok=True)
    shutil.copy2(CHECK, scripts / CHECK.name)
    (checkout / "pytest.ini").write_text("[pytest]\n")
    (checkout / "test_probe.py").write_text("def test_probe():\n    pass\n")
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(plugin_root), str(ROOT / "src"))),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "tests.conftest", "-q", "test_probe.py"],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("location", [".", ".worktrees/active", "import-parent-plugin"])
def test_session_ignores_other_worktrees_but_cli_sweeps_them(tmp_path, location):
    repo = tmp_path / "repo"
    checkout = repo / (".worktrees/active" if location == "import-parent-plugin" else location)
    checkout.mkdir(parents=True)
    sibling = repo / ".worktrees/legacy/docs.local/small.png"
    sibling.parent.mkdir(parents=True)
    sibling.write_bytes(b"\x89PNG\r\n\x1a\n")
    if location != ".":
        parent_artifact = repo / "docs.local/worktree-archive/small.png"
        parent_artifact.parent.mkdir(parents=True)
        parent_artifact.write_bytes(b"\x89PNG\r\n\x1a\n")

    result = _hygiene_session(checkout, plugin_root=repo if location == "import-parent-plugin" else None)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    sweep = subprocess.run([sys.executable, str(CHECK), str(repo)], capture_output=True, text=True)
    assert sweep.returncode == 1, sweep.stdout + sweep.stderr
    assert str(sibling) in sweep.stdout


@pytest.mark.parametrize("import_parent_plugin", [False, True])
def test_session_rejects_own_docs_binary_even_with_a_conftest_from_another_checkout(tmp_path, import_parent_plugin):
    repo = tmp_path / "repo"
    checkout = repo / ".worktrees/active"
    checkout.mkdir(parents=True)
    binary = checkout / "docs.local/small.png"
    binary.parent.mkdir()
    binary.write_bytes(b"\x89PNG\r\n\x1a\n")

    result = _hygiene_session(checkout, plugin_root=repo if import_parent_plugin else None)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    assert str(binary) in result.stdout


def test_root_session_still_rejects_its_own_legacy_archive(tmp_path):
    repo = tmp_path / "repo"
    archive = repo / "docs.local/worktree-archive/legacy.png"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"\x89PNG\r\n\x1a\n")
    os.utime(archive, (1, 1))
    result = _hygiene_session(repo)
    assert result.returncode == 1, result.stdout + result.stderr
    assert str(archive) in result.stdout


@pytest.fixture
def scan_root(monkeypatch):
    # These deliberate docs.local source probes must not emit binary bytecode there.
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
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
@pytest.mark.parametrize("binary_size", [16, 1024 * 1024 + 1])
def test_docs_local_archive_excludes_apps_and_binaries_even_under_noindex(tmp_path, archive, binary_size):
    repo = tmp_path / "repo"
    archived = repo / "docs.local" / archive
    app = archived / "BrainBar.app"
    app.mkdir(parents=True)
    binary = archived / "BrainBarDaemon"
    with binary.open("wb") as stream:
        stream.write(b"\x00Mach-O")
        stream.truncate(binary_size)
    receipt = archived / "receipt.log"
    receipt.write_text("version=1.5.68 sha=abc cdhash=def\n")
    result = subprocess.run([sys.executable, str(CHECK), str(repo)], capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    assert str(app) in result.stdout
    assert str(binary) in result.stdout
    assert str(receipt) not in result.stdout


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("small.png", b"\x89PNG\r\n\x1a\n"),
        ("small.zip", b"PK\x03\x04"),
        ("empty.zip", b""),
        ("small.bin", b"\x00"),
        ("receipt.log", b"valid prefix\x01"),
        ("exactly-one-mib.bin", b"\x00" * (1024 * 1024)),
    ],
    ids=["png", "zip", "empty-zip", "nul", "control-byte", "one-mib"],
)
def test_docs_local_rejects_binaries_of_any_size(tmp_path, name, content):
    docs = tmp_path / "repo/brain-bar/docs.local/qa"
    docs.mkdir(parents=True)
    binary = docs / name
    binary.write_bytes(content)
    result = subprocess.run([sys.executable, str(CHECK), str(tmp_path / "repo")], capture_output=True, text=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert str(binary) in result.stdout


def test_text_receipts_and_scan_boundary(tmp_path):
    repo = tmp_path / "repo"
    docs = repo / "docs.local"
    docs.mkdir(parents=True)
    (docs / "small.log").write_text("SHA=abc\tversion=1.5.68\n\u05e9\u05dc\u05d5\u05dd\n")
    (docs / "large.log").write_text("x" * (1024 * 1024 + 1))
    fixture = repo / "docs/test-artifacts/fixture.png"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(b"\x89PNG\r\n\x1a\n")
    outside = tmp_path / "outside"
    (outside / "BrainBar.app").mkdir(parents=True)
    (repo / "external").symlink_to(outside, target_is_directory=True)
    result = subprocess.run([sys.executable, str(CHECK), str(repo)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("tail", [b"\x00", b"\xff", b"\xe2\x82"])
def test_scan_rejects_binary_data_after_a_large_text_prefix(tmp_path, tail):
    docs = tmp_path / "docs.local"
    docs.mkdir()
    binary = docs / "receipt.log"
    binary.write_bytes(b"x" * (1024 * 1024 + 1) + tail)
    result = subprocess.run([sys.executable, str(CHECK), str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert str(binary) in result.stdout


def test_scan_accepts_utf8_split_across_read_boundaries(tmp_path):
    docs = tmp_path / "docs.local"
    docs.mkdir()
    (docs / "receipt.log").write_bytes(b"x" * 8191 + "\u20ac".encode() + b"x" * (1024 * 1024))
    result = subprocess.run([sys.executable, str(CHECK), str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("link", [".build.noindex", ".build.noindex/pytest", ".build.noindex/pytest/explicit"])
def test_pytest_rejects_symlinked_temp_output_before_writing(tmp_path, link):
    repo = tmp_path / "repo"
    tests = repo / "tests"
    tests.mkdir(parents=True)
    shutil.copy2(ROOT / "tests/conftest.py", tests / "conftest.py")
    scripts = repo / "scripts"
    scripts.mkdir()
    shutil.copy2(CHECK, scripts / CHECK.name)
    (tests / "test_probe.py").write_text("def test_probe(tmp_path):\n    pass\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    output = repo / link
    output.parent.mkdir(parents=True, exist_ok=True)
    output.symlink_to(outside, target_is_directory=True)
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_probe.py"]
    if link.endswith("explicit"):
        command += ["--basetemp", str(output)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(repo), str(ROOT / "src")))}
    result = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "pytest output must stay inside" in result.stdout + result.stderr
    assert list(outside.iterdir()) == []


def test_default_collection_excludes_build_output(tmp_path):
    import tomllib

    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]
    assert "tests" in config.get("testpaths", [])
    assert ".build.noindex" in config.get("norecursedirs", [])
    (tmp_path / "pyproject.toml").write_text((ROOT / "pyproject.toml").read_text())
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_probe.py").write_text("def test_probe():\n    pass\n")
    for build in (tmp_path / ".build.noindex", tests / ".build.noindex"):
        build.mkdir()
        (build / "conftest.py").write_text("raise RuntimeError('build conftest must never load')\n")
        (build / "test_poison.py").write_text("raise RuntimeError('build test must never load')\n")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 test collected" in result.stdout


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


@pytest.mark.parametrize("artifact", ["BrainBar.app", "small.png"])
def test_session_end_scan_fails_for_an_artifact_left_by_teardown(scan_root, artifact):
    probe = scan_root / "test_probe.py"
    probe.write_text(
        "from pathlib import Path\n"
        "import pytest\n"
        "@pytest.fixture\n"
        "def leave_bundle():\n"
        "    yield\n"
        f"    path = Path(__file__).parent / {artifact!r}\n"
        "    path.mkdir() if path.suffix == '.app' else path.write_bytes(b'\\x89PNG\\r\\n\\x1a\\n')\n"
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
    assert str(scan_root / artifact) in result.stdout


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
