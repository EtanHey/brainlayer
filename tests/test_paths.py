"""Tests for brainlayer.paths — DB path resolution."""

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

import brainlayer.paths as brainlayer_paths
from brainlayer.paths import get_db_path, resolve_db_path


class TestGetDbPath:
    """Test DB path resolution order."""

    def test_env_var_override(self, tmp_path):
        """BRAINLAYER_DB env var takes highest priority."""
        db_path = tmp_path / "missing-parent" / "custom.db"
        with patch.dict(os.environ, {"BRAINLAYER_DB": str(db_path)}):
            assert get_db_path() == db_path
            assert not db_path.parent.exists()

    def test_env_var_override_expands_home(self, tmp_path, monkeypatch):
        """Runtime resolution matches the Spotlight preflight for tilde paths."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("BRAINLAYER_DB", "~/brainlayer-data/brainlayer.db")

        assert resolve_db_path() == tmp_path / "brainlayer-data" / "brainlayer.db"

    def test_canonical_path_fresh_install(self, tmp_path, monkeypatch):
        """Canonical path used when no DB exists yet."""
        canonical = tmp_path / "brainlayer" / "brainlayer.db"
        with patch("brainlayer.paths._CANONICAL_DB_PATH", canonical):
            monkeypatch.delenv("BRAINLAYER_DB", raising=False)
            result = get_db_path()
            assert result == canonical
            assert canonical.parent.exists()  # Parent dir created

    def test_resolve_db_path_does_not_create_parent(self, tmp_path, monkeypatch):
        canonical = tmp_path / "brainlayer" / "brainlayer.db"
        with patch("brainlayer.paths._CANONICAL_DB_PATH", canonical):
            monkeypatch.delenv("BRAINLAYER_DB", raising=False)

            assert resolve_db_path() == canonical
            assert not canonical.parent.exists()

    @pytest.mark.integration
    def test_real_db_exists(self):
        """The real production DB exists at the resolved path."""
        from brainlayer.paths import DEFAULT_DB_PATH

        assert DEFAULT_DB_PATH.exists(), f"DB not found at {DEFAULT_DB_PATH}"
        assert DEFAULT_DB_PATH.stat().st_size > 1_000_000, "DB too small — might be empty"


def test_spotlight_exclusion_accepts_marker_on_directory(tmp_path):
    marker_name = ".metadata_never_index"
    (tmp_path / marker_name).touch()

    assert brainlayer_paths.is_spotlight_excluded(tmp_path)


def test_spotlight_exclusion_accepts_marker_on_ancestor(tmp_path):
    marker_name = ".metadata_never_index"
    (tmp_path / marker_name).touch()
    child = tmp_path / "logs" / "nested"
    child.mkdir(parents=True)

    assert brainlayer_paths.is_spotlight_excluded(child)


def test_spotlight_exclusion_rejects_unmarked_path(tmp_path):
    child = tmp_path / "queue"
    child.mkdir()

    assert not brainlayer_paths.is_spotlight_excluded(child)


def test_installer_bootstrap_python_can_load_paths_and_resolve_t3_health(tmp_path):
    """macOS installer preflight can use its standard-library-only Python 3.9."""
    bootstrap_python = Path("/usr/bin/python3")
    if not bootstrap_python.exists():
        pytest.skip("installer bootstrap Python unavailable")
    db_path = tmp_path / "uncreated" / "copy.db"
    env = dict(os.environ)
    env.pop("BRAINLAYER_T3_INGEST_HEALTH_PATH", None)
    result = subprocess.run(
        [
            str(bootstrap_python),
            "-c",
            "import runpy,sys; from pathlib import Path; "
            "paths=runpy.run_path(sys.argv[1]); print(paths['resolve_t3_health_path'](Path(sys.argv[2])))",
            str(Path(brainlayer_paths.__file__)),
            str(db_path),
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == str(db_path.parent / "t3-health.json")
    assert not db_path.parent.exists()


@pytest.mark.parametrize("explicit,env_filename", [(None, ""), ("", ""), ("", "override-health.json")])
def test_empty_t3_health_override_preserves_producer_consumer_snapshot(tmp_path, monkeypatch, explicit, env_filename):
    """Blank optional overrides still produce a snapshot the consumer can read."""
    from brainlayer.health_check import HealthCheckConfig, _load_json
    from brainlayer.ingest.t3 import T3Reader

    destination = tmp_path / "uncreated" / "copy.db"
    state = tmp_path / "unused-source.sqlite"
    env_path = tmp_path / env_filename if env_filename else None
    monkeypatch.setenv("BRAINLAYER_DB", str(destination))
    monkeypatch.setenv("BRAINLAYER_T3_INGEST_HEALTH_PATH", str(env_path) if env_path else "")
    health_path = brainlayer_paths.resolve_t3_health_path(destination, explicit)
    config = HealthCheckConfig(db_path=destination, t3_health_path=explicit)
    expected = env_path or destination.parent / "t3-health.json"
    assert health_path == config.t3_health_path == expected

    reader = T3Reader(state_db_path=state, health_path=health_path)
    reader._write_health(alerting=True, alert_reasons=["indexing_failure"])
    payload = _load_json(config.t3_health_path)
    assert payload["alerting"] is True
    assert payload["alert_reasons"] == ["indexing_failure"]
    assert not destination.exists() and not state.exists()
