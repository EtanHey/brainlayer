"""Real producer JSON shared with the Swift decoder tests; synthetic receipts only."""

import json
from datetime import UTC, datetime
from pathlib import Path

import jsonschema

from brainlayer.observability_backup import build_backups_section

FIXTURES = Path(__file__).parent / "fixtures/observability"
WIRE = Path(__file__).parents[1] / "brain-bar/Tests/BrainBarTests/Fixtures/neutral-backup-states-v2.json"
NOW = datetime(2026, 9, 13, 12, tzinfo=UTC)


def producer_documents(root: Path) -> list[dict]:
    root.mkdir(parents=True, exist_ok=True)
    transcript = root / "jsonl-backup.log"
    daily = root / "backup-daily.log"
    launchd = root / "launchd.txt"
    launchd.write_text("service = com.brainlayer.jsonl-backup\n")
    env = {
        "BRAINLAYER_DB": str(root / "unused.db"),
        "BRAINLAYER_OBSERVABILITY_JSONL_BACKUP_LOG": str(transcript),
        "BRAINLAYER_OBSERVABILITY_BACKUP_DAILY_LOG": str(daily),
        "BRAINLAYER_OBSERVABILITY_LAUNCHD_OUTPUT": str(launchd),
        "BRAINLAYER_OBSERVABILITY_DISABLED_DIR": str(root / "absent-disabled"),
        "BRAINLAYER_JOB_ALERT_PATH": str(root / "absent-alerts.json"),
    }
    prior = {
        "backup_log_provenance": "real",
        "snapshot_created_at": "2026-09-12T12:00:00Z",
        "snapshot": "/synthetic/previous.db.gz",
        "drive_file": {"name": "previous.db.gz"},
        "verified": True,
        "uploaded": True,
        "drive_md5_match": True,
    }
    cases = []
    for name in (
        "idle",
        "idle_old_upload",
        "pending",
        "in_progress",
        "failed",
        "abandoned",
        "verified",
        "superseded",
        "legacy",
        "stale_offsite",
        "m1_weekly_alert",
        "verified_bundle",
    ):
        attempts = [{"attempted_at": "2026-09-13T10:00:00Z", "status": "no-op", "verified": True, "uploaded": False}]
        if name == "idle_old_upload":
            attempts.insert(
                0,
                {
                    "attempted_at": "2026-08-01T12:00:00Z",
                    "status": "uploaded",
                    "verified": True,
                    "uploaded": True,
                    "archive_id": "synthetic-old-archive",
                },
            )
        if name == "verified_bundle":
            attempts = [
                {
                    "attempted_at": "2026-09-13T10:00:00Z",
                    "status": "uploaded",
                    "verified": True,
                    "uploaded": True,
                    "archive_id": "synthetic-current-archive",
                }
            ]
        alert = Path(env["BRAINLAYER_JOB_ALERT_PATH"])
        alert.write_text(
            json.dumps(
                {"maintenance-weekly": "Weekly maintenance missed its quiet window"}
                if name == "m1_weekly_alert"
                else {}
            )
        )
        transcript.write_text("".join(json.dumps(row) + "\n" for row in attempts))
        receipts = [dict(prior)]
        if name == "stale_offsite":
            receipts[0]["snapshot_created_at"] = "2026-08-14T12:00:00Z"
        if name not in {"idle", "idle_old_upload", "legacy", "m1_weekly_alert"}:
            local = {
                "backup_log_provenance": "real",
                "snapshot_created_at": "2026-09-13T10:00:00Z",
                "snapshot": "/synthetic/local.db.gz",
                "local_verified": name != "abandoned",
                "uploaded": name in {"verified", "verified_bundle"},
                "verified": name in {"verified", "verified_bundle"},
                "upload_state": "in_progress"
                if name == "stale_offsite"
                else "verified"
                if name == "verified_bundle"
                else name,
                "local_integrity": "ok",
                "local_integrity_pragma": "integrity_check",
            }
            if name in {"verified", "verified_bundle"}:
                local["drive_file"] = {"name": "local.db.gz"}
            if name in {"failed", "abandoned"}:
                local.update(error_type="SyntheticUploadFailure", error="Synthetic Drive rejected upload")
            receipts.append(local)
        daily.write_text("".join(json.dumps(row) + "\n" for row in receipts))

        def record(path, **kwargs):
            return {
                "path": f"synthetic/{path.name}",
                "status": kwargs.get("status", "read"),
                "mtime": None,
                "rows_or_bytes": None,
                "sha256_first_64kb": None,
                "skipped_lines": 0,
            }

        backups = build_backups_section(env=env, record_input=record, now=NOW)
        if name == "legacy":
            backups.pop("transcript_attempt", None)  # Replay the exact old optional-field absence.
        document = json.loads((FIXTURES / "golden/healthy-dev.json").read_text())
        document.update(
            schema_version=2,
            generated_at="2026-09-13T12:00:00Z",
            backups=backups,
            transport={"state": "unmeasurable", "reason": "Synthetic fixture; no live transport sampled", "inputs": []},
        )
        cases.append({"name": name, "document": document})
    return cases


def test_swift_wire_fixture_matches_real_producer_and_v2_schema(tmp_path):
    actual = producer_documents(tmp_path)
    assert actual == json.loads(WIRE.read_text())
    schema = json.loads((FIXTURES / "observability-schema.v2.json").read_text())
    for case in actual:
        jsonschema.validate(case["document"], schema)
