from __future__ import annotations

import json
import os
import plistlib
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from brainlayer.vector_store import VectorStore

NOW = datetime(2026, 6, 22, 12, 0, tzinfo=UTC)


def _loaded_launchctl(args: list[str]) -> SimpleNamespace:
    assert args[:2] == ["launchctl", "print"]
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def _clean_git_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        env=_clean_git_env(),
        check=True,
    )
    return result.stdout.strip()


def _git_repo_with_two_commits(path: Path) -> tuple[str, str]:
    path.mkdir()
    _run_git(path, "init")
    _run_git(path, "config", "user.email", "brainlayer-tests@example.com")
    _run_git(path, "config", "user.name", "BrainLayer Tests")
    source = path / "src" / "brainlayer"
    source.mkdir(parents=True)
    marker = source / "deploy_marker.py"
    marker.write_text("VERSION = 'old'\n", encoding="utf-8")
    _run_git(path, "add", ".")
    _run_git(path, "commit", "-m", "old launch commit")
    old_commit = _run_git(path, "rev-parse", "HEAD")
    marker.write_text("VERSION = 'new'\n", encoding="utf-8")
    _run_git(path, "add", ".")
    _run_git(path, "commit", "-m", "new deployed commit")
    head_commit = _run_git(path, "rev-parse", "HEAD")
    return old_commit, head_commit


def _git_repo_with_diverged_commits(path: Path) -> tuple[str, str]:
    path.mkdir()
    _run_git(path, "init", "-b", "main")
    _run_git(path, "config", "user.email", "brainlayer-tests@example.com")
    _run_git(path, "config", "user.name", "BrainLayer Tests")
    marker = path / "marker.txt"
    marker.write_text("base\n", encoding="utf-8")
    _run_git(path, "add", ".")
    _run_git(path, "commit", "-m", "base commit")
    _run_git(path, "checkout", "-b", "daemon-launch")
    marker.write_text("daemon launch\n", encoding="utf-8")
    _run_git(path, "add", ".")
    _run_git(path, "commit", "-m", "daemon launch commit")
    launch_commit = _run_git(path, "rev-parse", "HEAD")
    _run_git(path, "checkout", "main")
    marker.write_text("deployed head\n", encoding="utf-8")
    _run_git(path, "add", ".")
    _run_git(path, "commit", "-m", "deployed head commit")
    deployed_commit = _run_git(path, "rev-parse", "HEAD")
    return launch_commit, deployed_commit


def _write_daemon_provenance(
    provenance_dir: Path,
    *,
    label: str,
    repo_root: Path,
    launch_commit: str,
) -> None:
    provenance_dir.mkdir(parents=True, exist_ok=True)
    (provenance_dir / f"{label}.json").write_text(
        json.dumps(
            {
                "label": label,
                "repo_root": str(repo_root),
                "launch_commit": launch_commit,
                "launched_at": "2026-06-22T10:00:00+00:00",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_launchd_install_and_verify_returns_true_when_label_is_loaded(tmp_path):
    from brainlayer.launchd_primitive import install_and_verify_launchagent

    plist_path = tmp_path / "com.example.loaded.plist"
    plist_path.write_text("<plist />", encoding="utf-8")
    commands: list[list[str]] = []

    def command_runner(args: list[str]) -> SimpleNamespace:
        commands.append(args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    assert install_and_verify_launchagent(
        "com.example.loaded",
        plist_path,
        command_runner=command_runner,
    )
    assert ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist_path)] in commands
    assert ["launchctl", "print", f"gui/{os.getuid()}/com.example.loaded"] in commands


def test_launchd_install_and_verify_raises_when_bootstrap_does_not_load_label(tmp_path):
    from brainlayer.launchd_primitive import LaunchdLabelNotLoadedError, install_and_verify_launchagent

    plist_path = tmp_path / "com.example.not-loaded.plist"
    plist_path.write_text("<plist />", encoding="utf-8")
    commands: list[list[str]] = []

    def command_runner(args: list[str]) -> SimpleNamespace:
        commands.append(args)
        if args[:2] == ["launchctl", "print"]:
            return SimpleNamespace(returncode=113, stdout="", stderr="Could not find service")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    try:
        install_and_verify_launchagent(
            "com.example.not-loaded",
            plist_path,
            command_runner=command_runner,
        )
    except LaunchdLabelNotLoadedError as exc:
        assert exc.label == "com.example.not-loaded"
        assert exc.plist_path == plist_path
        assert "not loaded after bootstrap" in str(exc)
    else:
        raise AssertionError("install_and_verify_launchagent returned success for an unloaded label")

    assert ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist_path)] in commands
    assert ["launchctl", "print", f"gui/{os.getuid()}/com.example.not-loaded"] in commands


def test_launchd_install_and_verify_rejects_malformed_command_runner_result(tmp_path):
    from brainlayer.launchd_primitive import LaunchdCommandError, install_and_verify_launchagent

    plist_path = tmp_path / "com.example.malformed.plist"
    plist_path.write_text("<plist />", encoding="utf-8")

    try:
        install_and_verify_launchagent(
            "com.example.malformed",
            plist_path,
            command_runner=lambda _args: object(),
        )
    except LaunchdCommandError as exc:
        assert exc.reason == "launchctl_command_failed"
        assert exc.returncode == 1
    else:
        raise AssertionError("malformed command_runner result was treated as launchctl success")


def test_launchd_install_and_verify_accepts_loaded_label_after_bootstrap_already_loaded(tmp_path):
    from brainlayer.launchd_primitive import install_and_verify_launchagent

    plist_path = tmp_path / "com.example.race.plist"
    plist_path.write_text("<plist />", encoding="utf-8")

    def command_runner(args: list[str]) -> SimpleNamespace:
        if args[:2] == ["launchctl", "bootstrap"]:
            return SimpleNamespace(returncode=37, stdout="", stderr="service already bootstrapped")
        if args[:2] == ["launchctl", "print"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    assert install_and_verify_launchagent(
        "com.example.race",
        plist_path,
        command_runner=command_runner,
    )


def test_launchd_label_loaded_does_not_use_unscoped_list_fallback():
    from brainlayer.launchd_primitive import is_launchd_label_loaded

    commands: list[list[str]] = []

    def command_runner(args: list[str]) -> SimpleNamespace:
        commands.append(args)
        if args[:2] == ["launchctl", "list"]:
            return SimpleNamespace(returncode=0, stdout="com.example.ambiguous\n", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="unexpected launchctl failure")

    assert is_launchd_label_loaded("com.example.ambiguous", command_runner=command_runner) is None
    assert ["launchctl", "list", "com.example.ambiguous"] not in commands


def _hotlane_ps() -> str:
    return "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py --interval 1 --backlog-batch 128\n"


def _insert_chunk(
    store: VectorStore,
    chunk_id: str,
    *,
    content: str | None = None,
    created_at: str = "2026-06-22T10:00:00+00:00",
) -> None:
    text = content or f"doctor fixture content {chunk_id}"
    cursor = store.conn.cursor()
    cursor.execute(
        """
        INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type,
            value_type, char_count, source, created_at, enriched_at, enrich_status,
            summary, tags, importance, chunk_origin, seen_count, last_seen_at,
            content_class
        ) VALUES (?, ?, '{}', 'doctor-fixture.jsonl', 'doctor-fixture', 'note',
            'HIGH', ?, 'test', ?, NULL, NULL,
            NULL, NULL, NULL, 'raw', 1, ?,
            'human-authored')
        """,
        (chunk_id, text, len(text), created_at, created_at),
    )


def _insert_vector(store: VectorStore, chunk_id: str, first: float) -> None:
    store._upsert_chunk_vector(store.conn.cursor(), chunk_id, [first] + [0.0] * 1023)


def _build_db(path: Path, *, missing_recent_vector: bool = False, fts_desync: bool = False) -> None:
    store = VectorStore(path)
    try:
        _insert_chunk(store, "doctor-healthy-1", content="doctor alpha healthy searchable content")
        _insert_chunk(store, "doctor-healthy-2", content="doctor beta healthy searchable content")
        _insert_vector(store, "doctor-healthy-1", 1.0)
        _insert_vector(store, "doctor-healthy-2", 2.0)
        if missing_recent_vector:
            _insert_chunk(
                store,
                "doctor-recent-missing-vector",
                content="doctor recent chunk with no vector should fail the D2 gate",
            )
        if fts_desync:
            store.conn.cursor().execute("DELETE FROM chunks_fts WHERE chunk_id = 'doctor-healthy-1'")
        store.conn.cursor().execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        store.close()


def _add_enrichment_backlog(path: Path) -> None:
    store = VectorStore(path)
    try:
        _insert_chunk(
            store,
            "doctor-enrichment-backlog",
            content="doctor enrichment backlog content long enough to require realtime Gemini enrichment",
        )
        _insert_vector(store, "doctor-enrichment-backlog", 3.0)
        store.conn.cursor().execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        store.close()


def _write_drain_health(path: Path, *, updated_at: datetime, drained_total: int = 0) -> None:
    path.write_text(
        json.dumps(
            {
                "drain_cycles": 3,
                "drained_total": drained_total,
                "updated_at": updated_at.isoformat(),
            }
        ),
        encoding="utf-8",
    )


def _write_daily_cap_reached(cost_dir: Path, *, now: datetime, spent_usd: float = 5.0) -> None:
    cost_dir.mkdir(parents=True, exist_ok=True)
    (cost_dir / "enrich-daily-cost.json").write_text(
        json.dumps({"date": now.astimezone().date().isoformat(), "spent_usd": spent_usd}),
        encoding="utf-8",
    )


def _write_corrupt_daily_cap_counter(cost_dir: Path) -> None:
    cost_dir.mkdir(parents=True, exist_ok=True)
    (cost_dir / "enrich-daily-cost.json").write_text('{"date": ', encoding="utf-8")


def _write_invalid_spent_daily_cap_counter(cost_dir: Path, *, now: datetime, spent_usd: object = "oops") -> None:
    cost_dir.mkdir(parents=True, exist_ok=True)
    (cost_dir / "enrich-daily-cost.json").write_text(
        json.dumps({"date": now.astimezone().date().isoformat(), "spent_usd": spent_usd}),
        encoding="utf-8",
    )


def _doctor_config(tmp_path: Path, db_path: Path):
    from brainlayer.doctor import DoctorConfig

    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    watcher_health_path = tmp_path / "watcher-health.json"
    drain_health_path = tmp_path / "drain-health.json"
    watcher_health_path.write_text(json.dumps({"poll_count": 12}), encoding="utf-8")
    drain_health_path.write_text(json.dumps({"drained_total": 34}), encoding="utf-8")
    return DoctorConfig(
        db_path=db_path,
        queue_dir=queue_dir,
        watcher_health_path=watcher_health_path,
        drain_health_path=drain_health_path,
        queue_movement_sample_seconds=0,
    )


def test_run_doctor_exits_zero_on_healthy_fixture(tmp_path):
    from brainlayer.doctor import run_doctor

    db_path = tmp_path / "healthy.db"
    _build_db(db_path)

    result = run_doctor(
        _doctor_config(tmp_path, db_path),
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    assert result.exit_code == 0
    assert result.ok is True
    assert not [issue for issue in result.issues if issue.severity == "fatal"]


def test_roundtrip_probe_ignores_projectless_keyword_competitors(tmp_path):
    from brainlayer.doctor import _roundtrip_probe

    db_path = tmp_path / "roundtrip-projectless-competitor.db"
    store = VectorStore(db_path)
    try:
        competitor_id = "projectless-keyword-competitor"
        content = "no keyword match for brainlayer doctor probe projectless competing memory"
        created_at = NOW.isoformat()
        cursor = store.conn.cursor()
        cursor.execute(
            """
            INSERT INTO chunks (
                id, content, metadata, source_file, project, content_type,
                value_type, char_count, source, tags, summary, created_at,
                enriched_at, enrich_status, chunk_origin, seen_count, last_seen_at,
                content_class
            ) VALUES (?, ?, '{}', 'brainbar-store', NULL, 'user_message',
                'HIGH', ?, 'mcp', ?, ?, ?, NULL, NULL, 'raw', 1, ?,
                'operational')
            """,
            (
                competitor_id,
                content,
                len(content),
                json.dumps(["doctor-probe"]),
                content,
                created_at,
                created_at,
            ),
        )
        store._upsert_chunk_vector(cursor, competitor_id, [1.0] + [0.0] * 1023)
        store.conn.cursor().execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        store.close()

    ok, _latency, reason = _roundtrip_probe(db_path, timeout_seconds=2.0)

    assert ok is True, reason


def test_roundtrip_probe_reports_writer_conflict_without_traceback(tmp_path, monkeypatch):
    from brainlayer import doctor

    class BusyVectorStore:
        def __init__(self, _db_path):
            raise RuntimeError("another writer is using brainlayer.db")

    monkeypatch.setattr(doctor, "VectorStore", BusyVectorStore)

    ok, _latency, reason = doctor._roundtrip_probe(tmp_path / "busy.db", timeout_seconds=0.1)

    assert ok is False
    assert "another writer is using brainlayer.db" in reason


def test_roundtrip_probe_retries_transient_writer_conflict(tmp_path, monkeypatch):
    from brainlayer import doctor

    db_path = tmp_path / "roundtrip-transient-writer.db"
    _build_db(db_path)
    real_vector_store = doctor.VectorStore
    attempts = 0

    class TransientBusyVectorStore(real_vector_store):
        def __init__(self, db_path):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("another writer is using brainlayer.db (pid 123)")
            super().__init__(db_path)

    monkeypatch.setattr(doctor, "VectorStore", TransientBusyVectorStore)

    ok, _latency, reason = doctor._roundtrip_probe(db_path, timeout_seconds=2.0)

    assert ok is True, reason
    assert attempts == 2


def test_roundtrip_probe_attempts_search_after_slow_setup(tmp_path, monkeypatch):
    from brainlayer import doctor

    class FakeCursor:
        def __init__(self, store):
            self.store = store

        def execute(self, sql, params=()):
            if "INSERT INTO chunks" in sql:
                self.store.chunk_id = params[0]
            return self

    class FakeConnection:
        def __init__(self, store):
            self.store = store

        def cursor(self):
            return FakeCursor(self.store)

    class SlowSetupStore:
        def __init__(self, _db_path):
            self.conn = FakeConnection(self)
            self.chunk_id = None

        def _upsert_chunk_vector(self, _cursor, _chunk_id, _embedding):
            return None

        def hybrid_search(self, **_kwargs):
            return {"ids": [[self.chunk_id]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        def close(self):
            return None

    ticks = iter([0.0, 3.0, 3.1])
    monkeypatch.setattr(doctor, "VectorStore", SlowSetupStore)
    monkeypatch.setattr(doctor.time, "monotonic", lambda: next(ticks, 3.1))

    ok, _latency, reason = doctor._roundtrip_probe(tmp_path / "slow-setup.db", timeout_seconds=2.0)

    assert ok is True, reason


def test_run_doctor_stays_silent_when_loaded_daemon_launch_commit_matches_head(tmp_path):
    from brainlayer.doctor import run_doctor

    db_path = tmp_path / "deploy-drift-current.db"
    repo_root = tmp_path / "repo-current"
    _build_db(db_path)
    _old_commit, head_commit = _git_repo_with_two_commits(repo_root)
    provenance_dir = tmp_path / "daemon-provenance"
    _write_daemon_provenance(
        provenance_dir,
        label="com.brainlayer.drain",
        repo_root=repo_root,
        launch_commit=head_commit,
    )
    config = _doctor_config(tmp_path, db_path)
    config.deploy_provenance_dir = provenance_dir
    config.deploy_drift_labels = ("com.brainlayer.drain",)

    result = run_doctor(
        config,
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    assert result.ok is True
    assert not [issue for issue in result.issues if issue.code == "deploy_drift"]


def test_run_doctor_raises_alarm_when_loaded_daemon_launch_commit_is_older_than_head(tmp_path):
    from brainlayer.alarm import BrainLayerAlarm
    from brainlayer.doctor import run_doctor

    db_path = tmp_path / "deploy-drift-stale.db"
    repo_root = tmp_path / "repo-stale"
    _build_db(db_path)
    old_commit, head_commit = _git_repo_with_two_commits(repo_root)
    provenance_dir = tmp_path / "daemon-provenance"
    _write_daemon_provenance(
        provenance_dir,
        label="com.brainlayer.drain",
        repo_root=repo_root,
        launch_commit=old_commit,
    )
    config = _doctor_config(tmp_path, db_path)
    config.deploy_provenance_dir = provenance_dir
    config.deploy_drift_labels = ("com.brainlayer.drain",)

    with pytest.raises(BrainLayerAlarm) as alarm:
        run_doctor(
            config,
            ps_output_fn=_hotlane_ps,
            command_runner=_loaded_launchctl,
            now_fn=lambda: NOW,
        )

    assert alarm.value.code == "deploy_drift"
    assert alarm.value.message == "daemon com.brainlayer.drain running stale code, redeploy needed"
    assert alarm.value.context["label"] == "com.brainlayer.drain"
    assert alarm.value.context["launch_commit"] == old_commit
    assert alarm.value.context["deployed_commit"] == head_commit


def test_deploy_drift_git_shellouts_ignore_inherited_git_env(tmp_path, monkeypatch):
    from brainlayer.alarm import BrainLayerAlarm
    from brainlayer.doctor import run_doctor

    parent_repo = tmp_path / "parent-repo"
    _old_parent, _head_parent = _git_repo_with_two_commits(parent_repo)
    repo_root = tmp_path / "repo-stale-with-poisoned-env"
    db_path = tmp_path / "deploy-drift-poisoned-env.db"
    _build_db(db_path)
    old_commit, head_commit = _git_repo_with_two_commits(repo_root)
    provenance_dir = tmp_path / "daemon-provenance"
    _write_daemon_provenance(
        provenance_dir,
        label="com.brainlayer.drain",
        repo_root=repo_root,
        launch_commit=old_commit,
    )
    config = _doctor_config(tmp_path, db_path)
    config.deploy_provenance_dir = provenance_dir
    config.deploy_drift_labels = ("com.brainlayer.drain",)
    monkeypatch.setenv("GIT_DIR", str(parent_repo / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(parent_repo))

    with pytest.raises(BrainLayerAlarm) as alarm:
        run_doctor(
            config,
            ps_output_fn=_hotlane_ps,
            command_runner=_loaded_launchctl,
            now_fn=lambda: NOW,
        )

    assert alarm.value.context["launch_commit"] == old_commit
    assert alarm.value.context["deployed_commit"] == head_commit


def test_run_doctor_raises_alarm_when_loaded_daemon_launch_commit_diverged_from_head(tmp_path):
    from brainlayer.alarm import BrainLayerAlarm
    from brainlayer.doctor import run_doctor

    db_path = tmp_path / "deploy-drift-diverged.db"
    repo_root = tmp_path / "repo-diverged"
    _build_db(db_path)
    launch_commit, deployed_commit = _git_repo_with_diverged_commits(repo_root)
    provenance_dir = tmp_path / "daemon-provenance"
    _write_daemon_provenance(
        provenance_dir,
        label="com.brainlayer.drain",
        repo_root=repo_root,
        launch_commit=launch_commit,
    )
    config = _doctor_config(tmp_path, db_path)
    config.deploy_provenance_dir = provenance_dir
    config.deploy_drift_labels = ("com.brainlayer.drain",)

    with pytest.raises(BrainLayerAlarm) as alarm:
        run_doctor(
            config,
            ps_output_fn=_hotlane_ps,
            command_runner=_loaded_launchctl,
            now_fn=lambda: NOW,
        )

    assert alarm.value.code == "deploy_drift"
    assert alarm.value.context["drift_status"] == "diverged"
    assert alarm.value.context["launch_commit"] == launch_commit
    assert alarm.value.context["deployed_commit"] == deployed_commit


def test_brainbar_changed_for_deploy_detects_brainbar_changes_since_launch(tmp_path):
    from brainlayer.deploy_drift import brainbar_changed_for_deploy

    repo_root = tmp_path / "repo-brainbar"
    old_commit, _head_commit = _git_repo_with_two_commits(repo_root)
    brainbar_file = repo_root / "brain-bar" / "Sources" / "BrainBar" / "Marker.swift"
    brainbar_file.parent.mkdir(parents=True)
    brainbar_file.write_text("let marker = 1\n", encoding="utf-8")
    _run_git(repo_root, "add", ".")
    _run_git(repo_root, "commit", "-m", "brainbar source changed")
    provenance_dir = tmp_path / "daemon-provenance"
    _write_daemon_provenance(
        provenance_dir,
        label="com.brainlayer.watch",
        repo_root=repo_root,
        launch_commit=old_commit,
    )

    assert brainbar_changed_for_deploy(provenance_dir, repo_root=repo_root) is True


def test_brainbar_changed_for_deploy_ignores_non_brainbar_changes_since_launch(tmp_path):
    from brainlayer.deploy_drift import brainbar_changed_for_deploy

    repo_root = tmp_path / "repo-non-brainbar"
    old_commit, _head_commit = _git_repo_with_two_commits(repo_root)
    marker = repo_root / "src" / "brainlayer" / "deploy_marker.py"
    marker.write_text("VERSION = 'after-deploy'\n", encoding="utf-8")
    _run_git(repo_root, "add", ".")
    _run_git(repo_root, "commit", "-m", "python source changed")
    provenance_dir = tmp_path / "daemon-provenance"
    _write_daemon_provenance(
        provenance_dir,
        label="com.brainlayer.watch",
        repo_root=repo_root,
        launch_commit=old_commit,
    )

    assert brainbar_changed_for_deploy(provenance_dir, repo_root=repo_root) is False


def test_record_deploy_provenance_requires_repo_root_from_launchd_plist(tmp_path):
    from brainlayer.deploy_drift import DeployProvenanceError, record_deploy_provenance_for_label

    plist_path = tmp_path / "com.example.no-repo.plist"
    with plist_path.open("wb") as handle:
        plistlib.dump({"Label": "com.example.no-repo", "ProgramArguments": ["/bin/echo", "hello"]}, handle)
    provenance_dir = tmp_path / "daemon-provenance"

    with pytest.raises(DeployProvenanceError) as exc:
        record_deploy_provenance_for_label(
            label="com.example.no-repo",
            plist_path=plist_path,
            provenance_dir=provenance_dir,
        )

    assert exc.value.label == "com.example.no-repo"
    assert not (provenance_dir / "com.example.no-repo.json").exists()


def test_run_doctor_exits_nonzero_for_recent_unvectored_chunk(tmp_path):
    from brainlayer.doctor import run_doctor

    db_path = tmp_path / "missing-vector.db"
    _build_db(db_path, missing_recent_vector=True)

    result = run_doctor(
        _doctor_config(tmp_path, db_path),
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    assert result.exit_code == 1
    assert result.ok is False
    assert any(issue.code == "recent_unvectored_chunks" for issue in result.issues)


def test_run_doctor_fails_loudly_when_required_launchd_label_is_not_loaded(tmp_path):
    from brainlayer.doctor import run_doctor

    db_path = tmp_path / "launchd-not-loaded.db"
    _build_db(db_path)

    def command_runner(args: list[str]) -> SimpleNamespace:
        if args == ["launchctl", "print", f"gui/{os.getuid()}/com.brainlayer.watch"]:
            return SimpleNamespace(returncode=113, stdout="", stderr="Could not find service")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    result = run_doctor(
        _doctor_config(tmp_path, db_path),
        ps_output_fn=_hotlane_ps,
        command_runner=command_runner,
        now_fn=lambda: NOW,
    )

    assert result.exit_code == 1
    issue = next(issue for issue in result.issues if issue.code == "watch_unloaded")
    assert issue.severity == "fatal"
    assert issue.details["label"] == "com.brainlayer.watch"
    assert issue.details["target"] == f"gui/{os.getuid()}/com.brainlayer.watch"
    assert issue.details["reason"] == "not_loaded"


def test_run_doctor_reports_fts_desync_without_rebuilding(tmp_path):
    from brainlayer.doctor import run_doctor

    db_path = tmp_path / "fts-desync.db"
    _build_db(db_path, fts_desync=True)

    result = run_doctor(
        _doctor_config(tmp_path, db_path),
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    assert result.exit_code == 1
    assert any(issue.code == "fts5_desync" for issue in result.issues)

    store = VectorStore(db_path)
    try:
        assert store.conn.cursor().execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 1
    finally:
        store.close()


def test_run_doctor_exits_nonzero_when_queue_backlog_is_not_moving(tmp_path):
    from brainlayer.doctor import run_doctor

    db_path = tmp_path / "queue-stalled.db"
    _build_db(db_path)
    config = _doctor_config(tmp_path, db_path)
    (config.queue_dir / "pending.jsonl").write_text('{"kind":"watcher_chunk"}\n', encoding="utf-8")

    result = run_doctor(
        config,
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    assert result.exit_code == 1
    assert any(issue.code == "queue_not_moving_with_backlog" for issue in result.issues)


def test_run_doctor_fails_loudly_when_loaded_drain_heartbeat_stale_with_backlog_without_quota(tmp_path, monkeypatch):
    from brainlayer.doctor import run_doctor

    monkeypatch.delenv("BRAINLAYER_ENRICH_COST_DIR", raising=False)
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    db_path = tmp_path / "drain-liveness-stalled.db"
    _build_db(db_path)
    _add_enrichment_backlog(db_path)
    config = _doctor_config(tmp_path, db_path)
    _write_drain_health(config.drain_health_path, updated_at=NOW - timedelta(minutes=10))

    result = run_doctor(
        config,
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    issue = next(issue for issue in result.issues if issue.code == "drain_liveness_stalled")
    assert result.exit_code == 1
    assert issue.severity == "fatal"
    assert "DRAIN_LIVENESS_STALLED" in issue.message
    assert issue.details["backlog_count"] == 1
    assert issue.details["queue_count"] == 0
    assert issue.details["enrichment_backlog"] == 1
    assert issue.details["drain_label"] == "com.brainlayer.drain"


def test_run_doctor_does_not_fail_drain_liveness_when_heartbeat_is_advancing(tmp_path, monkeypatch):
    from brainlayer.doctor import run_doctor

    monkeypatch.delenv("BRAINLAYER_ENRICH_COST_DIR", raising=False)
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    db_path = tmp_path / "drain-liveness-moving.db"
    _build_db(db_path)
    _add_enrichment_backlog(db_path)
    config = _doctor_config(tmp_path, db_path)
    _write_drain_health(config.drain_health_path, updated_at=NOW - timedelta(seconds=30))

    result = run_doctor(
        config,
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    assert result.exit_code == 0
    assert result.ok is True
    assert not [issue for issue in result.issues if issue.code == "drain_liveness_stalled"]
    assert any(issue.code == "enrichment_idle_with_backlog" for issue in result.issues)


def test_run_doctor_suppresses_enrichment_only_stale_liveness_when_drain_cycles_advance(tmp_path, monkeypatch):
    import brainlayer.doctor as doctor

    monkeypatch.delenv("BRAINLAYER_ENRICH_COST_DIR", raising=False)
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    db_path = tmp_path / "drain-liveness-enrichment-cycles-advance.db"
    _build_db(db_path)
    _add_enrichment_backlog(db_path)
    config = _doctor_config(tmp_path, db_path)
    stale_updated_at = (NOW - timedelta(minutes=10)).isoformat()
    drain_reads = 0
    original_load_json = doctor._load_json

    def fake_load_json(path: Path) -> dict:
        nonlocal drain_reads
        if path == config.drain_health_path:
            drain_reads += 1
            return {
                "drain_cycles": 20 + drain_reads,
                "drained_total": 40,
                "updated_at": stale_updated_at,
            }
        return original_load_json(path)

    monkeypatch.setattr(doctor, "_load_json", fake_load_json)

    result = doctor.run_doctor(
        config,
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    assert result.exit_code == 0
    assert result.ok is True
    assert drain_reads == 2
    assert any(issue.code == "enrichment_idle_with_backlog" for issue in result.issues)
    assert not [issue for issue in result.issues if issue.code == "drain_liveness_stalled"]


def test_run_doctor_keeps_loaded_but_idle_warning_only_when_daily_cap_blocks_enrichment(tmp_path, monkeypatch):
    from brainlayer.doctor import run_doctor

    monkeypatch.delenv("BRAINLAYER_ENRICH_COST_DIR", raising=False)
    canonical_dir = tmp_path / "canonical"
    canonical_dir.mkdir()
    monkeypatch.setenv("BRAINLAYER_DB", str(canonical_dir / "brainlayer.db"))
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    db_path = tmp_path / "drain-liveness-quota.db"
    _write_daily_cap_reached(db_path.parent, now=NOW)
    _build_db(db_path)
    _add_enrichment_backlog(db_path)
    config = _doctor_config(tmp_path, db_path)
    _write_drain_health(config.drain_health_path, updated_at=NOW - timedelta(minutes=10))

    result = run_doctor(
        config,
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    assert result.exit_code == 0
    assert result.ok is True
    assert not [issue for issue in result.issues if issue.severity == "fatal"]
    assert any(issue.code == "enrichment_idle_with_backlog" for issue in result.issues)
    assert any(issue.code == "drain_liveness_quota_blocked" for issue in result.issues)


def test_run_doctor_honors_enrich_cost_dir_for_drain_liveness_quota(tmp_path, monkeypatch):
    from brainlayer.doctor import run_doctor

    cost_dir = tmp_path / "configured-cost-dir"
    monkeypatch.setenv("BRAINLAYER_ENRICH_COST_DIR", str(cost_dir))
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    db_path = tmp_path / "drain-liveness-env-quota.db"
    _write_daily_cap_reached(cost_dir, now=NOW)
    _build_db(db_path)
    _add_enrichment_backlog(db_path)
    config = _doctor_config(tmp_path, db_path)
    _write_drain_health(config.drain_health_path, updated_at=NOW - timedelta(minutes=10))

    result = run_doctor(
        config,
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    assert result.exit_code == 0
    assert result.ok is True
    assert not [issue for issue in result.issues if issue.severity == "fatal"]
    assert any(issue.code == "enrichment_idle_with_backlog" for issue in result.issues)
    assert any(issue.code == "drain_liveness_quota_blocked" for issue in result.issues)
    assert not [issue for issue in result.issues if issue.code == "drain_liveness_stalled"]


def test_run_doctor_fails_loudly_when_drain_liveness_quota_counter_is_corrupt(tmp_path, monkeypatch):
    from brainlayer.doctor import run_doctor

    monkeypatch.delenv("BRAINLAYER_ENRICH_COST_DIR", raising=False)
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    db_path = tmp_path / "drain-liveness-corrupt-quota.db"
    _write_corrupt_daily_cap_counter(db_path.parent)
    _build_db(db_path)
    _add_enrichment_backlog(db_path)
    config = _doctor_config(tmp_path, db_path)
    _write_drain_health(config.drain_health_path, updated_at=NOW - timedelta(minutes=10))

    result = run_doctor(
        config,
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    issue = next(issue for issue in result.issues if issue.code == "drain_liveness_stalled")
    assert result.exit_code == 1
    assert result.ok is False
    assert issue.severity == "fatal"
    assert "DRAIN_LIVENESS_STALLED" in issue.message
    assert not [issue for issue in result.issues if issue.code == "drain_liveness_quota_blocked"]
    assert not [issue for issue in result.issues if issue.code == "drain_liveness_blocker_unknown"]


@pytest.mark.parametrize("invalid_spend", ["oops", "nan", "inf", "-1"])
def test_run_doctor_fails_loudly_when_drain_liveness_quota_counter_has_invalid_spend(
    tmp_path, monkeypatch, invalid_spend
):
    from brainlayer.doctor import run_doctor

    monkeypatch.delenv("BRAINLAYER_ENRICH_COST_DIR", raising=False)
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    db_path = tmp_path / "drain-liveness-invalid-counter.db"
    _write_invalid_spent_daily_cap_counter(db_path.parent, now=NOW, spent_usd=invalid_spend)
    _build_db(db_path)
    _add_enrichment_backlog(db_path)
    config = _doctor_config(tmp_path, db_path)
    _write_drain_health(config.drain_health_path, updated_at=NOW - timedelta(minutes=10))

    result = run_doctor(
        config,
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    issue = next(issue for issue in result.issues if issue.code == "drain_liveness_stalled")
    assert result.exit_code == 1
    assert result.ok is False
    assert issue.severity == "fatal"
    assert "DRAIN_LIVENESS_STALLED" in issue.message
    assert not [issue for issue in result.issues if issue.code == "drain_liveness_quota_blocked"]
    assert not [issue for issue in result.issues if issue.code == "drain_liveness_blocker_unknown"]


def test_run_doctor_does_not_let_enrichment_quota_mask_durable_queue_liveness(tmp_path, monkeypatch):
    from brainlayer.doctor import run_doctor

    monkeypatch.delenv("BRAINLAYER_ENRICH_COST_DIR", raising=False)
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    db_path = tmp_path / "drain-liveness-queue-quota.db"
    _write_daily_cap_reached(db_path.parent, now=NOW)
    _build_db(db_path)
    config = _doctor_config(tmp_path, db_path)
    (config.queue_dir / "pending.jsonl").write_text('{"kind":"watcher_chunk"}\n', encoding="utf-8")
    _write_drain_health(config.drain_health_path, updated_at=NOW - timedelta(minutes=10))

    result = run_doctor(
        config,
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    issue = next(issue for issue in result.issues if issue.code == "drain_liveness_stalled")
    assert result.exit_code == 1
    assert issue.severity == "fatal"
    assert issue.details["backlog_count"] == 1
    assert issue.details["queue_count"] == 1
    assert issue.details["enrichment_backlog"] == 0
    assert not [issue for issue in result.issues if issue.code == "drain_liveness_quota_blocked"]


def test_run_doctor_skips_drain_liveness_when_drain_label_is_disabled(tmp_path, monkeypatch):
    from brainlayer.doctor import run_doctor

    monkeypatch.delenv("BRAINLAYER_ENRICH_COST_DIR", raising=False)
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    db_path = tmp_path / "drain-liveness-disabled-label.db"
    _build_db(db_path)
    _add_enrichment_backlog(db_path)
    config = _doctor_config(tmp_path, db_path)
    config.drain_label = ""
    _write_drain_health(config.drain_health_path, updated_at=NOW - timedelta(minutes=10))

    result = run_doctor(
        config,
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    assert result.exit_code == 0
    assert result.ok is True
    assert any(issue.code == "enrichment_idle_with_backlog" for issue in result.issues)
    assert not [issue for issue in result.issues if issue.code.startswith("drain_liveness")]


def test_run_doctor_suppresses_stale_liveness_when_drain_counter_advances(tmp_path, monkeypatch):
    import brainlayer.doctor as doctor

    monkeypatch.delenv("BRAINLAYER_ENRICH_COST_DIR", raising=False)
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    db_path = tmp_path / "drain-liveness-counter-advances.db"
    _build_db(db_path)
    config = _doctor_config(tmp_path, db_path)
    (config.queue_dir / "pending.jsonl").write_text('{"kind":"watcher_chunk"}\n', encoding="utf-8")
    stale_updated_at = (NOW - timedelta(minutes=10)).isoformat()
    drain_reads = 0
    original_load_json = doctor._load_json

    def fake_load_json(path: Path) -> dict:
        nonlocal drain_reads
        if path == config.drain_health_path:
            drain_reads += 1
            return {
                "drain_cycles": 3 + drain_reads,
                "drained_total": 40 + drain_reads,
                "updated_at": stale_updated_at,
            }
        return original_load_json(path)

    monkeypatch.setattr(doctor, "_load_json", fake_load_json)

    result = doctor.run_doctor(
        config,
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    assert result.exit_code == 0
    assert result.ok is True
    assert drain_reads == 2
    assert not [issue for issue in result.issues if issue.code == "drain_liveness_stalled"]
    assert not [issue for issue in result.issues if issue.code == "queue_not_moving_with_backlog"]


def test_run_doctor_suppresses_stale_liveness_but_fails_queue_when_only_drain_cycles_advance(
    tmp_path,
    monkeypatch,
):
    import brainlayer.doctor as doctor

    monkeypatch.delenv("BRAINLAYER_ENRICH_COST_DIR", raising=False)
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    db_path = tmp_path / "drain-liveness-cycles-advance.db"
    _build_db(db_path)
    config = _doctor_config(tmp_path, db_path)
    (config.queue_dir / "pending.jsonl").write_text('{"kind":"watcher_chunk"}\n', encoding="utf-8")
    stale_updated_at = (NOW - timedelta(minutes=10)).isoformat()
    drain_reads = 0
    original_load_json = doctor._load_json

    def fake_load_json(path: Path) -> dict:
        nonlocal drain_reads
        if path == config.drain_health_path:
            drain_reads += 1
            return {
                "drain_cycles": 20 + drain_reads,
                "drained_total": 40,
                "updated_at": stale_updated_at,
            }
        return original_load_json(path)

    monkeypatch.setattr(doctor, "_load_json", fake_load_json)

    result = doctor.run_doctor(
        config,
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    assert result.exit_code == 1
    assert result.ok is False
    assert drain_reads == 2
    assert not [issue for issue in result.issues if issue.code == "drain_liveness_stalled"]
    assert any(issue.code == "queue_not_moving_with_backlog" for issue in result.issues)


def test_run_doctor_uses_sample_time_for_post_sample_drain_liveness(tmp_path, monkeypatch):
    import brainlayer.doctor as doctor

    monkeypatch.delenv("BRAINLAYER_ENRICH_COST_DIR", raising=False)
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    db_path = tmp_path / "drain-liveness-sample-time.db"
    _build_db(db_path)
    config = _doctor_config(tmp_path, db_path)
    config.drain_liveness_stale_seconds = 300
    (config.queue_dir / "pending.jsonl").write_text('{"kind":"watcher_chunk"}\n', encoding="utf-8")
    drain_reads = 0
    now_reads = 0
    original_load_json = doctor._load_json

    def fake_load_json(path: Path) -> dict:
        nonlocal drain_reads
        if path == config.drain_health_path:
            drain_reads += 1
            return {
                "drain_cycles": 20,
                "drained_total": 40,
                "updated_at": (NOW - timedelta(seconds=299)).isoformat(),
            }
        return original_load_json(path)

    def fake_now() -> datetime:
        nonlocal now_reads
        now_reads += 1
        return NOW if now_reads == 1 else NOW + timedelta(seconds=2)

    monkeypatch.setattr(doctor, "_load_json", fake_load_json)

    result = doctor.run_doctor(
        config,
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=fake_now,
    )

    issue = next(issue for issue in result.issues if issue.code == "drain_liveness_stalled")
    assert result.exit_code == 1
    assert drain_reads == 2
    assert now_reads == 2
    assert issue.details["heartbeat_age_seconds"] == 301.0


def test_run_doctor_suppresses_stale_liveness_when_sampled_heartbeat_is_fresh(tmp_path, monkeypatch):
    import brainlayer.doctor as doctor

    monkeypatch.delenv("BRAINLAYER_ENRICH_COST_DIR", raising=False)
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    db_path = tmp_path / "drain-liveness-sampled-fresh.db"
    _build_db(db_path)
    config = _doctor_config(tmp_path, db_path)
    (config.queue_dir / "pending.jsonl").write_text('{"kind":"watcher_chunk"}\n', encoding="utf-8")
    drain_reads = 0
    watcher_reads = 0
    original_load_json = doctor._load_json

    def fake_load_json(path: Path) -> dict:
        nonlocal drain_reads, watcher_reads
        if path == config.drain_health_path:
            drain_reads += 1
            return {
                "drain_cycles": 20,
                "drained_total": 40,
                "updated_at": (NOW - timedelta(minutes=10)).isoformat() if drain_reads == 1 else NOW.isoformat(),
            }
        if path == config.watcher_health_path:
            watcher_reads += 1
            return {"poll_count": 12 + watcher_reads}
        return original_load_json(path)

    monkeypatch.setattr(doctor, "_load_json", fake_load_json)

    result = doctor.run_doctor(
        config,
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    assert result.exit_code == 0
    assert result.ok is True
    assert drain_reads == 2
    assert watcher_reads == 2
    assert not [issue for issue in result.issues if issue.code == "drain_liveness_stalled"]
    assert not [issue for issue in result.issues if issue.code == "queue_not_moving_with_backlog"]


def test_doctor_cli_is_registered_with_json_and_db_options():
    from typer.main import get_command

    from brainlayer.cli import app

    result = CliRunner().invoke(app, ["doctor", "--help"])

    assert result.exit_code == 0
    doctor_command = get_command(app).commands["doctor"]
    option_decls = {opt for param in doctor_command.params for opt in getattr(param, "opts", [])}
    assert "--json" in option_decls
    assert "--db" in option_decls


def test_doctor_cli_json_uses_injected_runner_and_db_option(tmp_path, monkeypatch):
    import brainlayer.cli as cli
    from brainlayer.cli import app

    db_path = tmp_path / "cli-fixture.db"
    seen: dict[str, Path] = {}

    class FakeResult:
        ok = True
        exit_code = 0
        chunk_count = 2
        recent_unvectored_chunks = 0
        queue_count = 0
        issues = []

        def to_dict(self) -> dict:
            return {"ok": self.ok, "exit_code": self.exit_code, "db_path": str(seen["db_path"])}

    def fake_run(config):
        seen["db_path"] = config.db_path
        return FakeResult()

    monkeypatch.setattr(cli, "_run_doctor_cli", fake_run)

    result = CliRunner().invoke(app, ["doctor", "--json", "--db", str(db_path)])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["db_path"] == str(db_path)
