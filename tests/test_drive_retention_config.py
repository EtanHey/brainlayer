"""Configured read-only Drive selection and its optional observability contract."""

import datetime as dt
import json
import os
import time
from pathlib import Path
from unittest.mock import Mock

import jsonschema
import pytest

from brainlayer import backup_daily as backup
from brainlayer import backup_retention as policy
from brainlayer import observability_surface
from brainlayer.backup_retention_invariant import inspect_backup_daily_retention_invariant
from brainlayer.backup_retention_observation import drive_retention_observation
from tests.test_backup_local_removal import _live, _old_receipt, _write
from tests.test_backup_local_removal import removal_lane as _removal_lane

removal_lane = _removal_lane

NOW = dt.datetime(2026, 11, 23, tzinfo=dt.UTC)
SCHEMA = Path(__file__).parent / "fixtures/observability/observability-schema.v2.json"


@pytest.fixture(autouse=True)
def default_drive_config(monkeypatch):
    monkeypatch.delenv("BRAINLAYER_BACKUP_DRIVE_DAILY_KEEP", raising=False)
    monkeypatch.delenv("BRAINLAYER_BACKUP_DRIVE_MONTHLY_KEEP", raising=False)


def item(day, *, machine="m4", name=None):
    return {
        "id": f"{machine}-{name or day}",
        "name": name or f"{day}.db.zst",
        "size": "10",
        "md5Checksum": "a" * 32,
        "appProperties": {"brainlayer_machine": machine},
        "parents": ["folder"],
    }


def listing():
    return (
        [item(f"2026-11-{day:02}") for day in range(1, 15)]
        + [
            item(day)
            for day in (
                "2026-10-01",
                "2026-10-31",
                "2026-09-01",
                "2026-09-30",
                "2026-08-01",
                "2026-08-31",
                "2026-07-31",
            )
        ]
        + [item("one", name="before-migration.db"), item("two", name="weekly-pin.db"), item("2026-10-31", machine="m1")]
    )


def proofs(files):
    return [
        {
            "uploaded": True,
            "verified": True,
            "machine_id": row["appProperties"]["brainlayer_machine"],
            "local_md5": row.get("md5Checksum"),
            "drive_file": dict(row),
        }
        for row in files
    ]


def report(*, rule=None, machine="m4", files=None):
    files = listing() if files is None else files
    return policy.drive_retention_report(
        files, machine_id=machine, folder_id="folder", as_of=NOW.date(), rule=rule, receipts=proofs(files)
    )


def observation(*, machine="m4", rule=None):
    return {
        "machine_id": machine,
        "state": "measured",
        "selected_at": "2026-11-22T00:00:00Z",
        "error": None,
        **report(machine=machine, rule=rule)["configured"],
    }


def record(value):
    return {"backup_log_provenance": "real", "machine_id": value["machine_id"], "drive_retention_selection": value}


def projected(records, env=None):
    result = drive_retention_observation(
        records,
        env={} if env is None else env,
        now=NOW,
        machine_id="m4",
        writes_enabled=backup._drive_retention_enabled(),
    )
    schema = json.loads(SCHEMA.read_text())
    jsonschema.validate(result, {"$defs": schema["$defs"], "$ref": "#/$defs/drive_retention"})
    return result


def test_default_config_and_previous_full_month_representatives(monkeypatch):
    monkeypatch.delenv("BRAINLAYER_BACKUP_DRIVE_DAILY_KEEP", raising=False)
    monkeypatch.delenv("BRAINLAYER_BACKUP_DRIVE_MONTHLY_KEEP", raising=False)
    selected = report()["configured"]
    assert selected["rule"]["daily_keep"] == 7 and selected["rule"]["monthly_keep"] == 3
    assert selected["rule"]["per_machine"] is True
    assert selected["on_drive"] == {"count": 21, "bytes": 210, "oldest": "2026-07-31", "newest": "2026-11-14"}
    assert selected["named_one_offs_count"] == 2
    assert selected["kept"]["count"] == 10 and selected["would_remove"]["count"] == 11
    assert [row["date"] for row in selected["monthly_representatives"]] == ["2026-10-31", "2026-09-30", "2026-08-31"]
    assert report(machine="m1")["configured"]["on_drive"]["count"] == 1


def test_runtime_config_drives_actual_selection(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_BACKUP_DRIVE_DAILY_KEEP", "2")
    monkeypatch.setenv("BRAINLAYER_BACKUP_DRIVE_MONTHLY_KEEP", "1")
    assert report()["configured"]["kept"]["count"] == 3
    assert report()["configured"]["would_remove"]["count"] == 18


def test_real_dated_pin_and_unverified_daily_are_never_candidates():
    files = [item(f"2026-11-{day:02}") for day in range(1, 11)]
    selected = policy.drive_retention_report(files, machine_id="m4", folder_id="folder", as_of=NOW.date())["configured"]
    assert selected["would_remove"]["count"] == 0
    selected = policy.drive_retention_report(
        files,
        machine_id="m4",
        folder_id="folder",
        as_of=NOW.date(),
        receipts=proofs([row for row in files if row["name"] != "2026-11-02.db.zst"]),
        protected_names={"2026-11-01.db.zst"},
    )["configured"]
    assert selected["would_remove"] == {"count": 1, "bytes": 10, "oldest": "2026-11-03", "newest": "2026-11-03"}
    assert selected["kept"]["count"] == 9


def test_gap_month_does_not_backfill_an_older_month():
    files = [item(f"2026-11-{day:02}", machine="m1") for day in range(1, 11)]
    files += [item(day, machine="m1") for day in ("2026-10-15", "2026-08-20", "2026-07-31")]
    selected = report(machine="m1", files=files)["configured"]
    assert [row["date"] for row in selected["monthly_representatives"]] == ["2026-10-15", "2026-08-20"]
    assert selected["kept"]["count"] == 9


def test_bad_drive_config_does_not_pause_local_cleanup(removal_lane, monkeypatch):
    target = _old_receipt(removal_lane)
    _write(removal_lane, [target])
    monkeypatch.setenv("BRAINLAYER_BACKUP_DRIVE_DAILY_KEEP", "invalid")
    result = backup.run_backup(**removal_lane)
    assert result["local_retention_inventory"] == "measured"
    assert not Path(target["snapshot"]).exists()
    selected = result["drive_retention_selection"]
    assert selected["state"] == "unmeasurable" and selected["selected_at"]
    assert "DAILY_KEEP" in selected["error"]
    assert projected([record(selected)], {"BRAINLAYER_BACKUP_DRIVE_DAILY_KEEP": "invalid"})["config_error"]


def test_months_cross_year_boundary_and_overlap_is_not_double_counted():
    files = [item(day) for day in ("2026-01-01", "2025-12-31", "2025-11-30", "2025-10-31", "2025-10-01")]
    selected = policy.drive_retention_report(files, machine_id="m4", folder_id="folder", as_of=dt.date(2026, 1, 2))[
        "configured"
    ]
    assert [row["date"] for row in selected["monthly_representatives"]] == ["2025-12-31", "2025-11-30", "2025-10-31"]
    assert selected["kept"]["count"] == 5


@pytest.mark.parametrize(
    "key",
    [
        "BRAINLAYER_BACKUP_DRIVE_RETENTION",
        "BRAINLAYER_JSONL_BACKUP_RETENTION",
        "BRAINLAYER_BACKUP_DRIVE_WRITES_ENABLED",
        "BRAINLAYER_BACKUP_DRIVE_RETENTION_ENABLED",
    ],
)
def test_no_environment_opt_in_enables_drive_writes(monkeypatch, key):
    monkeypatch.setenv(key, "1")
    assert backup._drive_retention_enabled() is False
    assert projected([record(observation())])["writes_enabled"] is False


@pytest.mark.parametrize(
    "env",
    [
        {"BRAINLAYER_BACKUP_DRIVE_DAILY_KEEP": "0"},
        {"BRAINLAYER_BACKUP_DRIVE_MONTHLY_KEEP": "-1"},
        {"BRAINLAYER_BACKUP_DRIVE_MONTHLY_KEEP": "121"},
    ],
)
def test_bad_config_is_visible_with_no_candidate_counts(env):
    result = projected([record(observation())], env)
    assert result["rule"] is None and result["config_error"]
    assert result["machines"][0]["state"] == "unmeasurable"
    assert result["machines"][0]["would_remove"] is None


def test_each_machine_has_its_own_last_selection_and_current_config():
    data = projected([record(observation()), record(observation(machine="m1"))])
    assert [machine["machine_id"] for machine in data["machines"]] == ["m1", "m4"]
    assert [machine["on_drive"]["count"] for machine in data["machines"]] == [1, 21]
    stale = projected([record(observation())], {"BRAINLAYER_BACKUP_DRIVE_DAILY_KEEP": "2"})
    assert stale["rule"]["daily_keep"] == 2
    assert stale["machines"][0]["state"] == "stale" and stale["machines"][0]["would_remove"] is None


def test_latest_failure_replaces_success_without_stale_counts():
    failure = policy.empty_drive_selection(
        "m4", state="unmeasurable", selected_at="2026-11-22T12:00:00Z", error="listing unavailable"
    )
    selected = projected([record(observation()), record(failure)])["machines"][0]
    assert selected["selected_at"] == "2026-11-22T12:00:00+00:00"
    assert selected["state"] == "unmeasurable" and selected["error"] == "listing unavailable"
    assert selected["on_drive"] is None and selected["would_remove"] is None
    assert projected([])["machines"][0]["state"] == "not_run"


def test_month_rollover_withholds_old_candidate_counts():
    data = drive_retention_observation(
        [record(observation())],
        env={},
        now=dt.datetime(2026, 12, 1, tzinfo=dt.UTC),
        machine_id="m4",
        writes_enabled=False,
    )
    selected = data["machines"][0]
    assert selected["state"] == "stale" and selected["would_remove"] is None
    assert "Monthly retention window" in selected["error"]


def test_month_window_uses_same_local_calendar_as_backup_selection(monkeypatch):
    try:
        with monkeypatch.context() as patch:
            patch.setenv("TZ", "Etc/GMT-2")
            time.tzset()
            selected = {
                "machine_id": "m4",
                "state": "measured",
                "selected_at": "2026-11-30T23:00:00Z",
                "error": None,
                **policy.drive_retention_report(
                    listing(), machine_id="m4", folder_id="folder", as_of=dt.date(2026, 12, 1)
                )["configured"],
            }
            data = drive_retention_observation(
                [record(selected)],
                env={},
                now=dt.datetime(2026, 11, 30, 23, 30, tzinfo=dt.UTC),
                machine_id="m4",
                writes_enabled=False,
            )
            assert data["machines"][0]["state"] == "measured"
    finally:
        time.tzset()


@pytest.mark.parametrize(
    "key,value",
    [("selected_at", "2999-01-01T00:00:00Z"), ("selected_at", None), ("on_drive", {}), ("named_one_offs_count", -1)],
)
def test_invalid_receipt_never_becomes_measured(key, value):
    row = observation()
    row[key] = value
    selected = projected([record(row)])["machines"][0]
    assert selected["state"] == "unmeasurable" and selected["error"]
    assert selected["kept"] is None


def test_inconsistent_recorded_totals_are_unmeasurable():
    row = observation()
    row["would_remove"]["count"] += 1
    selected = projected([record(row)])["machines"][0]
    assert selected["state"] == "unmeasurable"
    assert selected["would_remove"] is None


def test_report_deduplicates_pages_and_refuses_unmeasurable_daily_size():
    assert report(files=[item("2026-11-01")] * 2)["configured"]["on_drive"]["count"] == 1
    row = item("2026-11-01")
    del row["size"]
    with pytest.raises(ValueError, match="measurable size"):
        report(files=[row])


def test_run_records_configured_selection_using_write_refusing_service(removal_lane, monkeypatch):
    for flag in (
        "BRAINLAYER_BACKUP_DRIVE_RETENTION",
        "BRAINLAYER_JSONL_BACKUP_RETENTION",
        "BRAINLAYER_BACKUP_DRIVE_WRITES_ENABLED",
    ):
        monkeypatch.setenv(flag, "1")
    _write(removal_lane, [_old_receipt(removal_lane)])
    service = Mock()
    resource = service.files.return_value
    resource.list.return_value.execute.side_effect = lambda: {"files": _live(removal_lane)}
    for verb in ("update", "delete", "patch", "emptyTrash", "create", "copy"):
        getattr(resource, verb).side_effect = AssertionError("Drive write has no authority")
    monkeypatch.setattr(backup, "build_drive_service", lambda *_: service)
    from tests.test_backup_local_removal import REAL_LIST_DRIVE_BACKUP_FILES

    monkeypatch.setattr(backup, "_list_drive_backup_files", REAL_LIST_DRIVE_BACKUP_FILES)
    monkeypatch.setenv("BRAINLAYER_BACKUP_DRIVE_DAILY_KEEP", "1")
    monkeypatch.setenv("BRAINLAYER_BACKUP_DRIVE_MONTHLY_KEEP", "0")
    result = backup.run_backup(**removal_lane)
    assert result["drive_retention"] == "disabled"
    selected = result["drive_retention_selection"]
    assert selected["state"] == "measured" and selected["selected_at"]
    assert selected["rule"]["daily_keep"] == 1
    assert selected["kept"]["count"] == 1 and selected["would_remove"]["count"] >= 1
    assert json.loads(removal_lane["log_path"].read_text().splitlines()[-1])["drive_retention_selection"] == selected
    for verb in ("update", "delete", "patch", "emptyTrash", "create", "copy"):
        getattr(resource, verb).assert_not_called()


def test_listing_failure_has_timestamp_and_durable_error(removal_lane, monkeypatch):
    _write(removal_lane, [_old_receipt(removal_lane)])
    monkeypatch.setattr(backup, "_list_drive_backup_files", Mock(side_effect=OSError("cannot list")))
    result = backup.run_backup(**removal_lane)
    selected = result["drive_retention_selection"]
    assert selected["state"] == "unmeasurable" and selected["selected_at"]
    assert "cannot list" in selected["error"] and selected["on_drive"] is None
    assert json.loads(removal_lane["log_path"].read_text().splitlines()[-1])["drive_retention_selection"] == selected


def test_run_selection_receives_the_real_weekly_pin(removal_lane, monkeypatch):
    target = _old_receipt(removal_lane)
    _write(removal_lane, [target])
    (removal_lane["staging_dir"] / ".weekly-backup-pin.json").write_text(
        json.dumps({"date": "2026-10-21", "snapshot": target["snapshot"]})
    )
    monkeypatch.setenv("BRAINLAYER_BACKUP_DRIVE_DAILY_KEEP", "1")
    monkeypatch.setenv("BRAINLAYER_BACKUP_DRIVE_MONTHLY_KEEP", "0")
    selected = backup.run_backup(**removal_lane)["drive_retention_selection"]
    assert selected["state"] == "measured" and selected["selected_at"]
    assert selected["would_remove"]["oldest"] != "2026-10-21"
    assert Path(target["snapshot"]).exists()


def test_settings_config_survives_unmeasurable_backup_health_in_real_document(tmp_path, monkeypatch):
    monkeypatch.setattr(observability_surface, "_bounded_db_sections", observability_surface._collect_db_sections)
    env = {
        "BRAINLAYER_DB": str(tmp_path / "missing.db"),
        "BRAINLAYER_MACHINE_ID": "m4",
        "BRAINLAYER_OBSERVABILITY_LAUNCHD_OUTPUT": str(tmp_path / "missing-launchd.txt"),
    }
    document, _ = observability_surface.build_document(env=env)
    assert document["backups"]["state"] == "unmeasurable"
    assert document["backups"]["drive_retention"]["rule"]["monthly_keep"] == 3
    assert document["backups"]["drive_retention"]["machines"][0]["state"] == "not_run"
    jsonschema.validate(document, json.loads(SCHEMA.read_text()))


def test_committed_wire_fixture_matches_real_document_without_drive_access(tmp_path, monkeypatch):
    monkeypatch.setattr(observability_surface, "_bounded_db_sections", observability_surface._collect_db_sections)
    monkeypatch.setattr(backup, "build_drive_service", lambda *_: pytest.fail("observability contacted Drive"))
    log = tmp_path / "daily.jsonl"
    log.write_text("".join(json.dumps(record(observation(machine=machine))) + "\n" for machine in ("m4", "m1")))
    os.utime(log, (NOW.timestamp(), NOW.timestamp()))
    env = {
        "BRAINLAYER_DB": str(tmp_path / "missing.db"),
        "BRAINLAYER_MACHINE_ID": "m4",
        "BRAINLAYER_OBSERVABILITY_NOW": NOW.isoformat(),
        "BRAINLAYER_OBSERVABILITY_BACKUP_DAILY_LOG": str(log),
        "BRAINLAYER_OBSERVABILITY_LAUNCHD_OUTPUT": str(tmp_path / "missing-launchd.txt"),
    }
    document, _ = observability_surface.build_document(env=env)
    assert document["backups"]["drive_retention"] == json.loads(
        SCHEMA.with_name("drive-retention-config.v2.json").read_text()
    )
    jsonschema.validate(document, json.loads(SCHEMA.read_text()))


def test_no_drive_writer_reachable_in_policy_or_projection():
    from brainlayer import backup_retention_observation

    for module in (policy, backup_retention_observation):
        source = Path(module.__file__).read_text()
        assert inspect_backup_daily_retention_invariant(source, policy_module=True) == []
        assert inspect_backup_daily_retention_invariant(
            source + "\ndef wrong(resource):\n    resource.update(fileId='x')\n", policy_module=True
        )
