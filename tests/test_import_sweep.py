import importlib
import tomllib
from pathlib import Path
from zipfile import ZipFile

import pytest

from brainlayer.import_sweep import ImportTarget, discover_wheel_modules, run_target, stage_hook_targets


def _wheel(tmp_path: Path, names: list[str]) -> Path:
    wheel = tmp_path / "brainlayer-1.5.36-py3-none-any.whl"
    with ZipFile(wheel, "w") as archive:
        for name in names:
            archive.writestr(name, "")
    return wheel


def test_wheel_discovery_includes_every_shipped_python_module_without_exclusions(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        [
            "brainlayer/__init__.py",
            "brainlayer/feature.py",
            "brainlayer/nested/__init__.py",
            "brainlayer/nested/worker.py",
            "brainlayer/native.so",
        ],
    )
    assert [target.name for target in discover_wheel_modules(wheel)] == [
        "brainlayer",
        "brainlayer.feature",
        "brainlayer.nested",
        "brainlayer.nested.worker",
    ]


def test_wheel_discovery_rejects_duplicate_module_paths(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, ["brainlayer/feature.py", "brainlayer/feature/__init__.py"])
    with pytest.raises(ValueError, match="duplicate module in wheel: brainlayer.feature"):
        discover_wheel_modules(wheel)


def test_hook_discovery_includes_every_python_file(tmp_path: Path) -> None:
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "entry.py").write_text("", encoding="utf-8")
    (hooks / "helper.py").write_text("", encoding="utf-8")
    (hooks / "ignored.sh").write_text("", encoding="utf-8")
    staged = stage_hook_targets([hooks], tmp_path / "staged")
    assert [target.name for target in staged] == ["hook:entry.py", "hook:helper.py"]


def test_missing_transitive_dependency_fails_the_gate(tmp_path: Path) -> None:
    hook = tmp_path / "broken-hook.py"
    hook.write_text("import dependency_that_is_not_installed\n", encoding="utf-8")
    result = run_target(ImportTarget.hook(hook), timeout_seconds=2, sandbox_root=tmp_path / "sandbox")
    assert result.status == "failed"
    assert "ModuleNotFoundError: No module named 'dependency_that_is_not_installed'" in result.detail


def test_hung_import_times_out_in_its_own_process(tmp_path: Path) -> None:
    hook = tmp_path / "hung-hook.py"
    hook.write_text("import time\ntime.sleep(10)\n", encoding="utf-8")
    result = run_target(ImportTarget.hook(hook), timeout_seconds=0.05, sandbox_root=tmp_path / "sandbox")
    assert result.status == "timed_out"


def test_required_transitive_runtime_dependencies_are_declared_directly() -> None:
    dependencies = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    assert any(item.startswith("idna") for item in dependencies)
    assert any(item.startswith("deprecation") for item in dependencies)


def test_optional_dependency_loader_preserves_transitive_module_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from brainlayer.clustering import _optional_brain_dependency

    transitive = ModuleNotFoundError("No module named 'broken_transitive'")
    transitive.name = "broken_transitive"

    def broken_import(_name: str):
        raise transitive

    monkeypatch.setattr(importlib, "import_module", broken_import)
    with pytest.raises(ModuleNotFoundError) as raised:
        _optional_brain_dependency("faiss")

    assert raised.value is transitive
