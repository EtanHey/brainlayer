"""Regression tests for fail-closed pytest runtime path isolation."""

import os
from pathlib import Path

import pytest


def test_unit_runtime_paths_are_isolated_under_tmp_path(tmp_path: Path) -> None:
    from brainlayer.drain import _default_drain_health_path, _default_log_path
    from brainlayer.paths import get_db_path
    from brainlayer.queue_io import get_queue_dir

    assert os.environ["BRAINLAYER_TEST_PATH_PROVENANCE"] == "pytest"
    isolated_home = Path.home()
    protected_home = Path(os.environ["BRAINLAYER_TEST_PROTECTED_HOME"])
    assert isolated_home != protected_home
    assert get_db_path().is_relative_to(isolated_home)
    assert get_queue_dir().is_relative_to(isolated_home)
    assert _default_log_path().is_relative_to(isolated_home)
    assert _default_drain_health_path().is_relative_to(isolated_home)


def test_pytest_path_guard_rejects_production_brainlayer_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from brainlayer.paths import _guard_test_runtime_path

    protected_home = tmp_path / "captured-real-home"
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_example.py::test_guard (call)")
    monkeypatch.setenv("BRAINLAYER_TEST_PATH_PROVENANCE", "pytest")
    monkeypatch.setenv("BRAINLAYER_TEST_PROTECTED_HOME", str(protected_home))

    for path in (
        protected_home / ".brainlayer" / "queue",
        protected_home / ".brainlayer" / "logs" / "drain.log",
        protected_home / ".local" / "share" / "brainlayer" / "brainlayer.db",
    ):
        with pytest.raises(RuntimeError, match="production BrainLayer path"):
            _guard_test_runtime_path(path, source="regression test")

    safe_path = tmp_path / "isolated" / "brainlayer.db"
    assert _guard_test_runtime_path(safe_path, source="regression test") == safe_path


def test_pytest_path_guard_requires_provenance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from brainlayer.paths import _guard_test_runtime_path

    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_example.py::test_guard (call)")
    monkeypatch.delenv("BRAINLAYER_TEST_PATH_PROVENANCE", raising=False)

    with pytest.raises(RuntimeError, match="provenance"):
        _guard_test_runtime_path(tmp_path / "otherwise-safe.db", source="regression test")


def test_index_defaults_resolve_database_path_at_call_time(tmp_path: Path) -> None:
    import inspect

    from brainlayer import index_new

    index_default = inspect.signature(index_new.index_chunks_to_sqlite).parameters["db_path"].default
    stats_default = inspect.signature(index_new.get_stats).parameters["db_path"].default

    assert index_default is None
    assert stats_default is None
