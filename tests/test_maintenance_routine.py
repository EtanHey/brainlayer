from __future__ import annotations

import datetime as dt
import json
import plistlib
from pathlib import Path

import apsw
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def _create_enrichment_db(path: Path) -> None:
    conn = apsw.Connection(str(path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                content TEXT,
                summary TEXT,
                enriched_at TEXT,
                enrich_status TEXT,
                content_hash TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO chunks (id, content, summary, enriched_at, enrich_status, content_hash)
            VALUES
                ('already-done', 'same content', 'old summary', '2026-05-30T00:00:00Z', 'success', 'hash-ok'),
                ('needs-update', 'same content', NULL, NULL, NULL, 'hash-ok'),
                ('hash-moved', 'new content', 'old summary', '2026-05-30T00:00:00Z', 'success', 'hash-new')
            """
        )
    finally:
        conn.close()


def _config(tmp_path: Path, *, now: dt.datetime):
    from brainlayer.maintenance import MaintenanceConfig

    return MaintenanceConfig(
        db_path=tmp_path / "brainlayer.db",
        queue_dir=tmp_path / "queue",
        quarantine_root=tmp_path / "quarantine",
        log_path=tmp_path / "maintenance.log",
        repo_root=REPO_ROOT,
        now_fn=lambda: now,
        quiet_window_start_hour=4,
        quiet_window_duration_minutes=120,
        idle_sample_seconds=0,
        recent_write_grace_seconds=0,
    )


def test_off_window_gate_aborts_before_touching_queue(tmp_path, monkeypatch):
    from brainlayer import maintenance

    config = _config(tmp_path, now=dt.datetime(2026, 5, 30, 2, 30, tzinfo=dt.timezone.utc))
    config.queue_dir.mkdir(parents=True)
    queued = config.queue_dir / "enrichment-stale.jsonl"
    queued.write_text("{}\n", encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr(maintenance, "collect_lsof_entries", lambda _paths: [])
    monkeypatch.setattr(maintenance, "run_command", lambda args, **_kwargs: commands.append(args))

    with pytest.raises(maintenance.MaintenanceAbort, match="outside quiet window"):
        maintenance.run_maintenance("light", config=config, dry_run=True)

    assert queued.exists()
    assert commands == []


def test_lsof_gate_aborts_on_unexpected_writer(tmp_path, monkeypatch):
    from brainlayer import maintenance

    config = _config(tmp_path, now=dt.datetime(2026, 5, 30, 4, 5, tzinfo=dt.timezone.utc))
    config.db_path.write_bytes(b"db")
    config.queue_dir.mkdir(parents=True)
    monkeypatch.setattr(
        maintenance,
        "collect_lsof_entries",
        lambda _paths: [
            maintenance.LsofEntry(pid=4242, command="python3", fd="9u", path=str(config.db_path)),
        ],
    )

    with pytest.raises(maintenance.MaintenanceAbort, match="unexpected writer"):
        maintenance.run_maintenance("light", config=config, dry_run=True)


def test_recent_queue_activity_abort_message_reports_count(tmp_path, monkeypatch):
    from brainlayer import maintenance

    config = _config(tmp_path, now=dt.datetime(2026, 5, 30, 4, 5, tzinfo=dt.timezone.utc))
    config.recent_write_grace_seconds = 300
    config.queue_dir.mkdir(parents=True)
    (config.queue_dir / "a.jsonl").write_text("{}\n", encoding="utf-8")
    (config.queue_dir / "z.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(maintenance, "collect_lsof_entries", lambda _paths: [])

    with pytest.raises(maintenance.MaintenanceAbort, match=r"2 file\(s\) modified recently"):
        maintenance.run_maintenance("light", config=config, dry_run=True)


def test_dry_run_runs_gates_and_reports_without_moving_queue_files(tmp_path, monkeypatch):
    from brainlayer import maintenance

    config = _config(tmp_path, now=dt.datetime(2026, 5, 30, 4, 5, tzinfo=dt.timezone.utc))
    _create_enrichment_db(config.db_path)
    stale = config.queue_dir / "enrichment-stale.jsonl"
    _write_jsonl(
        stale,
        [
            {
                "kind": "enrichment_update",
                "chunk_id": "already-done",
                "content_hash": "hash-ok",
                "enrichment": {"summary": "new summary"},
            }
        ],
    )
    monkeypatch.setattr(maintenance, "collect_lsof_entries", lambda _paths: [])

    result = maintenance.run_maintenance("light", config=config, dry_run=True)

    assert result.dry_run is True
    assert result.stale_queue.quarantined_files == 0
    assert result.stale_queue.candidate_files == 1
    assert stale.exists()
    assert not config.quarantine_root.exists()


def test_stale_queue_quarantine_moves_only_already_enriched_matching_hash(tmp_path):
    from brainlayer.maintenance import quarantine_stale_queue_files

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    quarantine_root = tmp_path / "quarantine"
    _create_enrichment_db(db_path)
    stale = queue_dir / "enrichment-stale.jsonl"
    needs_update = queue_dir / "enrichment-needs-update.jsonl"
    mismatch = queue_dir / "enrichment-mismatch.jsonl"
    missing = queue_dir / "enrichment-missing.jsonl"
    _write_jsonl(
        stale,
        [
            {
                "kind": "enrichment_update",
                "chunk_id": "already-done",
                "content_hash": "hash-ok",
                "enrichment": {"summary": "new summary"},
            }
        ],
    )
    _write_jsonl(
        needs_update,
        [
            {
                "kind": "enrichment_update",
                "chunk_id": "needs-update",
                "content_hash": "hash-ok",
                "enrichment": {"summary": "real update"},
            }
        ],
    )
    _write_jsonl(
        mismatch,
        [
            {
                "kind": "enrichment_update",
                "chunk_id": "hash-moved",
                "content_hash": "hash-old",
                "enrichment": {"summary": "stale hash mismatch"},
            }
        ],
    )
    _write_jsonl(
        missing,
        [
            {
                "kind": "enrichment_update",
                "chunk_id": "missing",
                "content_hash": "hash-ok",
                "enrichment": {"summary": "missing chunk"},
            }
        ],
    )

    result = quarantine_stale_queue_files(
        db_path=db_path,
        queue_dir=queue_dir,
        quarantine_root=quarantine_root,
        dry_run=False,
        now=dt.datetime(2026, 5, 30, 4, 10, tzinfo=dt.timezone.utc),
    )

    assert result.scanned_files == 4
    assert result.candidate_files == 1
    assert result.quarantined_files == 1
    assert not stale.exists()
    assert needs_update.exists()
    assert mismatch.exists()
    assert missing.exists()
    quarantined = list(quarantine_root.rglob("enrichment-stale.jsonl"))
    assert len(quarantined) == 1


def test_stale_queue_quarantine_preserves_entity_bearing_redundant_enrichment(tmp_path):
    from brainlayer.maintenance import quarantine_stale_queue_files

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    quarantine_root = tmp_path / "quarantine"
    _create_enrichment_db(db_path)
    provenance_only = queue_dir / "enrichment-provenance-only.jsonl"
    _write_jsonl(
        provenance_only,
        [
            {
                "kind": "enrichment_update",
                "chunk_id": "already-done",
                "content_hash": "hash-ok",
                "enrichment": {"summary": "redundant summary"},
                "entities": [{"name": "controlLayer"}],
            }
        ],
    )

    result = quarantine_stale_queue_files(
        db_path=db_path,
        queue_dir=queue_dir,
        quarantine_root=quarantine_root,
        dry_run=False,
        now=dt.datetime(2026, 5, 30, 4, 10, tzinfo=dt.timezone.utc),
    )

    assert result.scanned_files == 1
    assert result.candidate_files == 0
    assert result.quarantined_files == 0
    assert provenance_only.exists()
    assert not quarantine_root.exists()


def test_stale_queue_quarantine_preserves_empty_entity_redundant_enrichment(tmp_path):
    from brainlayer.maintenance import quarantine_stale_queue_files

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    quarantine_root = tmp_path / "quarantine"
    _create_enrichment_db(db_path)
    provenance_empty = queue_dir / "enrichment-empty-entities.jsonl"
    _write_jsonl(
        provenance_empty,
        [
            {
                "kind": "enrichment_update",
                "chunk_id": "already-done",
                "content_hash": "hash-ok",
                "enrichment": {"summary": "redundant summary"},
                "entities": [],
            }
        ],
    )

    result = quarantine_stale_queue_files(
        db_path=db_path,
        queue_dir=queue_dir,
        quarantine_root=quarantine_root,
        dry_run=False,
        now=dt.datetime(2026, 5, 30, 4, 10, tzinfo=dt.timezone.utc),
    )

    assert result.scanned_files == 1
    assert result.candidate_files == 0
    assert result.quarantined_files == 0
    assert provenance_empty.exists()
    assert not quarantine_root.exists()


def test_stale_queue_quarantine_preserves_provenance_class_redundant_enrichment(tmp_path):
    from brainlayer.maintenance import quarantine_stale_queue_files

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    quarantine_root = tmp_path / "quarantine"
    _create_enrichment_db(db_path)
    provenance_only = queue_dir / "enrichment-provenance-class.jsonl"
    _write_jsonl(
        provenance_only,
        [
            {
                "kind": "enrichment_update",
                "chunk_id": "already-done",
                "content_hash": "hash-ok",
                "enrichment": {"summary": "redundant summary"},
                "provenance_class": "RAW-ETAN-DIRECT",
            }
        ],
    )

    result = quarantine_stale_queue_files(
        db_path=db_path,
        queue_dir=queue_dir,
        quarantine_root=quarantine_root,
        dry_run=False,
        now=dt.datetime(2026, 5, 30, 4, 10, tzinfo=dt.timezone.utc),
    )

    assert result.scanned_files == 1
    assert result.candidate_files == 0
    assert result.quarantined_files == 0
    assert provenance_only.exists()
    assert not quarantine_root.exists()


def test_maintenance_launchd_plists_and_installer_wiring():
    nightly_path = REPO_ROOT / "scripts/launchd/com.brainlayer.maintenance-nightly.plist"
    weekly_path = REPO_ROOT / "scripts/launchd/com.brainlayer.maintenance-weekly.plist"

    with nightly_path.open("rb") as handle:
        nightly = plistlib.load(handle)
    with weekly_path.open("rb") as handle:
        weekly = plistlib.load(handle)

    assert nightly["Label"] == "com.brainlayer.maintenance-nightly"
    assert weekly["Label"] == "com.brainlayer.maintenance-weekly"
    assert "--light" in nightly["ProgramArguments"]
    assert "--full" in weekly["ProgramArguments"]
    assert nightly["StartCalendarInterval"] == {"Hour": 4, "Minute": 0}
    assert weekly["StartCalendarInterval"] == {"Weekday": 0, "Hour": 4, "Minute": 0}
    for plist in (nightly, weekly):
        assert plist["Nice"] == 10
        assert plist["ProcessType"] == "Background"
        assert plist["EnvironmentVariables"]["BRAINLAYER_REPO_ROOT"] == "__BRAINLAYER_DIR__"

    install = (REPO_ROOT / "scripts/launchd/install.sh").read_text(encoding="utf-8")
    assert "maintenance-nightly" in install
    assert "maintenance-weekly" in install


def test_enrichment_template_flex_validation_parses_active_env_lines(tmp_path):
    from brainlayer.maintenance import _verify_enrichment_template_flex_backend

    launchd_dir = tmp_path / "scripts" / "launchd"
    launchd_dir.mkdir(parents=True)
    template = launchd_dir / "brainlayer.env.example"
    template.write_text(
        "\n".join(
            [
                "# BRAINLAYER_GEMINI_SERVICE_TIER=standard",
                "export BRAINLAYER_GEMINI_SERVICE_TIER = 'FLEX'",
            ]
        ),
        encoding="utf-8",
    )

    _verify_enrichment_template_flex_backend(tmp_path)


def test_enrichment_template_flex_validation_uses_last_active_assignment(tmp_path):
    from brainlayer.maintenance import _verify_enrichment_template_flex_backend

    launchd_dir = tmp_path / "scripts" / "launchd"
    launchd_dir.mkdir(parents=True)
    template = launchd_dir / "brainlayer.env.example"
    template.write_text(
        "\n".join(
            [
                "BRAINLAYER_GEMINI_SERVICE_TIER=standard",
                "BRAINLAYER_GEMINI_SERVICE_TIER=flex",
            ]
        ),
        encoding="utf-8",
    )

    _verify_enrichment_template_flex_backend(tmp_path)


def test_enrichment_template_flex_validation_rejects_missing_or_commented_flex(tmp_path):
    from brainlayer.maintenance import MaintenanceAbort, _verify_enrichment_template_flex_backend

    with pytest.raises(MaintenanceAbort, match="env template not found"):
        _verify_enrichment_template_flex_backend(tmp_path)

    launchd_dir = tmp_path / "scripts" / "launchd"
    launchd_dir.mkdir(parents=True)
    template = launchd_dir / "brainlayer.env.example"
    template.write_text("# BRAINLAYER_GEMINI_SERVICE_TIER=flex\n", encoding="utf-8")

    with pytest.raises(MaintenanceAbort, match="no longer uses Gemini Flex"):
        _verify_enrichment_template_flex_backend(tmp_path)
