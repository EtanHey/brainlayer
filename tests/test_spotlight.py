"""Setup preflight for private JSONL staging; all paths are disposable fixtures."""

import os
import stat
from pathlib import Path

import pytest

STAGING_KEY = "BRAINLAYER_JSONL_BACKUP_STAGING_DIR"
MARKER = ".metadata_never_index"


def _snapshot(root: Path) -> dict:
    result = {}
    for path in (root, *root.rglob("*")):
        info = path.lstat()
        payload = os.readlink(path) if path.is_symlink() else path.read_bytes() if path.is_file() else None
        result[str(path.relative_to(root))] = (info.st_mode, info.st_uid, info.st_gid, payload)
    return result


@pytest.fixture
def staging_layout(tmp_path, monkeypatch):
    import brainlayer.setup as setup

    data = tmp_path / "brainlayer-data"
    data.mkdir(mode=0o700)
    (data / MARKER).write_bytes(b"existing marker\n")
    (data / "brainlayer.db").write_bytes(b"fixture bytes, never opened as a database\n")
    env = tmp_path / "selected.env"
    monkeypatch.delenv(STAGING_KEY, raising=False)
    monkeypatch.setattr(setup, "get_canonical_db_path", lambda: data / "brainlayer.db")

    def preflight(value=None):
        env.write_text(
            "BRAINLAYER_JSONL_BACKUP_RETENTION=0\nBRAINLAYER_JSONL_BACKUP_ICLOUD_DIR=\n"
            + (f"{STAGING_KEY}={value}\n" if value is not None else ""),
            encoding="utf-8",
        )
        before = _snapshot(tmp_path)
        try:
            result = setup.ensure_spotlight_excluded_layout(
                env_file=env,
                runtime_dir=tmp_path / "runtime",
                launchd_log_dir=tmp_path / "logs",
                counter_dir=tmp_path / "counter",
            )
        except Exception:
            assert _snapshot(tmp_path) == before
            raise
        assert _snapshot(tmp_path)[env.name] == before[env.name]
        return result

    return data, env, preflight


def test_setup_accepts_marker_covered_private_staging(staging_layout):
    data, env, preflight = staging_layout
    stage = data / "jsonl-backups" / "recovery-fixture"
    stage.mkdir(parents=True, mode=0o700)
    before = _snapshot(data)

    assert preflight(str(stage))[0] == data
    assert preflight(str(stage))[0] == data

    after = _snapshot(data)
    assert all(after[path] == value for path, value in before.items())
    assert stat.S_IMODE(stage.stat().st_mode) == 0o700
    assert list(stage.iterdir()) == []
    assert "BRAINLAYER_JSONL_BACKUP_RETENTION=0\nBRAINLAYER_JSONL_BACKUP_ICLOUD_DIR=\n" in env.read_text()


@pytest.mark.parametrize("value", [None, "", "fresh-leaf"])
def test_setup_staging_defaults_and_missing_covered_leaf(staging_layout, value):
    data, _, preflight = staging_layout
    stage = data / "jsonl-backups" / "new" / "private-leaf"
    assert preflight(str(stage) if value == "fresh-leaf" else value)[0] == data
    assert (data / "jsonl-backups").is_dir()
    assert not stage.exists()


@pytest.mark.parametrize("configured", [None, "", "covered"])
@pytest.mark.parametrize("process", ["covered", "outside", ""])
def test_setup_validates_process_staging_precedence(staging_layout, monkeypatch, tmp_path, configured, process):
    data, _, preflight = staging_layout
    file_value = str(data / "jsonl-backups" / "configured") if configured == "covered" else configured
    process_value = {"covered": str(data / "jsonl-backups" / "process"), "outside": str(tmp_path / "outside"), "": ""}[
        process
    ]
    monkeypatch.setenv(STAGING_KEY, process_value)
    if process == "outside":
        with pytest.raises(RuntimeError, match=STAGING_KEY):
            preflight(file_value)
    else:
        assert preflight(file_value)[0] == data
    assert os.environ[STAGING_KEY] == process_value


def test_process_staging_does_not_waive_unsafe_configured_path(staging_layout, monkeypatch, tmp_path):
    data, _, preflight = staging_layout
    monkeypatch.setenv(STAGING_KEY, str(data / "jsonl-backups" / "covered"))
    with pytest.raises(RuntimeError, match=STAGING_KEY):
        preflight(str(tmp_path / "outside"))


@pytest.mark.parametrize(
    "invalid",
    [
        "outside",
        "relative",
        "tilde",
        "backup-root",
        "data-root",
        "escape",
        "leaf-file",
        "ancestor-file",
        "leaf-symlink",
        "ancestor-symlink",
        "internal-symlink",
        "dangling-symlink",
        "writable-ancestor",
        "public-leaf",
        "unmarked-root",
        "marker-symlink",
        "marker-directory",
        "writable-marker",
        "mount",
    ],
)
def test_setup_staging_refuses_unsafe_paths_without_mutation(staging_layout, tmp_path, monkeypatch, invalid):
    import brainlayer.setup as setup

    data, _, preflight = staging_layout
    backups = data / "jsonl-backups"
    stage = backups / "private-leaf"
    if invalid == "outside":
        stage = tmp_path / "outside"
    elif invalid == "relative":
        stage = Path("brainlayer-data/jsonl-backups/private-leaf")
    elif invalid == "tilde":
        stage = Path("~/brainlayer-data/jsonl-backups/private-leaf")
    elif invalid == "backup-root":
        stage = backups
    elif invalid == "data-root":
        stage = data
    elif invalid == "escape":
        stage = backups / ".." / "escaped"
    elif invalid in {"leaf-file", "leaf-symlink", "public-leaf"}:
        backups.mkdir()
        if invalid == "leaf-file":
            stage.write_bytes(b"preserve this file\n")
        elif invalid == "leaf-symlink":
            stage.symlink_to(data, target_is_directory=True)
        else:
            stage.mkdir(mode=0o755)
    elif invalid == "ancestor-file":
        backups.write_bytes(b"preserve ancestor bytes\n")
    elif invalid in {"ancestor-symlink", "internal-symlink", "dangling-symlink"}:
        target = data / "internal" if invalid == "internal-symlink" else tmp_path / "external"
        if invalid != "dangling-symlink":
            target.mkdir()
        backups.symlink_to(target, target_is_directory=True)
    elif invalid == "writable-ancestor":
        backups.mkdir(mode=0o777)
        backups.chmod(0o777)
    elif invalid in {"unmarked-root", "marker-symlink", "marker-directory", "writable-marker"}:
        marker = data / MARKER
        if invalid == "writable-marker":
            marker.chmod(0o666)
        else:
            marker.unlink()
            if invalid == "marker-symlink":
                marker.symlink_to(data / "brainlayer.db")
            elif invalid == "marker-directory":
                marker.mkdir()
    elif invalid == "mount":
        backups.mkdir()
        monkeypatch.setattr(setup.os.path, "ismount", lambda path: Path(path) == backups)

    with pytest.raises(RuntimeError):
        preflight(str(stage))


@pytest.mark.parametrize("ancestor", ["root", "parent"])
def test_setup_staging_checks_unsafe_ancestors(staging_layout, ancestor):
    data, _, preflight = staging_layout
    (data if ancestor == "root" else data.parent).chmod(0o777)
    with pytest.raises(RuntimeError, match="unsafe ancestor"):
        preflight(str(data / "jsonl-backups" / "leaf"))


def test_setup_staging_keeps_later_root_migration_refusal(staging_layout, tmp_path):
    data, _, preflight = staging_layout
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "keep.jsonl").write_bytes(b"pending fixture\n")
    with pytest.raises(RuntimeError, match="migration runbook"):
        preflight(str(data / "jsonl-backups" / "leaf"))


@pytest.mark.parametrize(
    "key",
    [
        "BRAINLAYER_BACKUP_LOG_PATH",
        "BRAINLAYER_BACKUP_STAGING_DIR",
        "BRAINLAYER_DRAIN_HEALTH_PATH",
        "BRAINLAYER_DRAIN_LOG_PATH",
        "BRAINLAYER_ENRICH_COST_DIR",
        "BRAINLAYER_JSONL_BACKUP_LOG_PATH",
        "BRAINLAYER_JSONL_BACKUP_STATE_PATH",
        "BRAINLAYER_LOG_DIR",
        "BRAINLAYER_OFFSETS_PATH",
        "BRAINLAYER_P0_COUNTER_DIR",
        "BRAINLAYER_QUEUE_DIR",
        "BRAINLAYER_WATCHER_HEALTH_PATH",
        "BRAINLAYER_WATCHER_OFFSETS_PATH",
        "BRAINLAYER_WATCHER_QUARANTINE_DIR",
        "BRAINLAYER_WRITER_HEARTBEAT_DIR",
        "BRAINLAYER_WRITER_TELEMETRY_PATH",
    ],
)
def test_setup_staging_still_rejects_other_runtime_overrides(staging_layout, key):
    from brainlayer import spotlight

    data, env, _ = staging_layout
    env.write_text(f"{key}={data / 'uncovered'}\n{STAGING_KEY}={data / 'jsonl-backups' / 'leaf'}\n")
    before = _snapshot(data.parent)
    with pytest.raises(RuntimeError, match=key):
        spotlight.ensure_spotlight_excluded_layout(data_dir=data, env_file=env)
    assert _snapshot(data.parent) == before


def test_setup_staging_process_without_env_file(staging_layout, monkeypatch, tmp_path):
    import brainlayer.setup as setup

    data, _, _ = staging_layout
    monkeypatch.setenv(STAGING_KEY, str(data / "jsonl-backups" / "leaf"))
    roots = setup.ensure_spotlight_excluded_layout(
        data_dir=data,
        runtime_dir=tmp_path / "runtime",
        launchd_log_dir=tmp_path / "logs",
        counter_dir=tmp_path / "counter",
    )
    assert roots[0] == data
    assert all((root / MARKER).is_file() for root in roots)


def test_setup_staging_under_selected_db_root(tmp_path, monkeypatch):
    import brainlayer.setup as setup

    active, canonical = tmp_path / "brainlayer-selected", tmp_path / "brainlayer-canonical"
    active.mkdir(mode=0o700)
    (active / MARKER).write_bytes(b"existing marker\n")
    stage = active / "jsonl-backups" / "recovery-fixture"
    stage.mkdir(parents=True, mode=0o700)
    env = tmp_path / "selected.env"
    env.write_text(f"BRAINLAYER_DB={active / 'brainlayer.db'}\n{STAGING_KEY}={stage}\n")
    monkeypatch.setattr(setup, "get_canonical_db_path", lambda: canonical / "brainlayer.db")
    roots = setup.ensure_spotlight_excluded_layout(
        env_file=env,
        runtime_dir=tmp_path / "runtime",
        launchd_log_dir=tmp_path / "logs",
        counter_dir=tmp_path / "counter",
    )
    assert roots[:2] == (active, canonical)
    assert all((root / MARKER).is_file() for root in roots)
    assert list(stage.iterdir()) == []
