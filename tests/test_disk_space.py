import json
from datetime import datetime
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner


def test_disk_cli_json_contract(tmp_path, monkeypatch):
    from brainlayer.cli import app

    # Keep the red test executable before the shared module/command exists.
    monkeypatch.setattr("shutil.disk_usage", lambda _path: SimpleNamespace(free=13_500_000_000))
    result = CliRunner().invoke(app, ["disk", "--path", str(tmp_path), "--json"])
    assert result.exit_code == 0, result.output
    value = json.loads(result.output)
    assert set(value) == {"available_bytes", "free_now_bytes", "purgeable_bytes", "source", "measured_at"}
    assert value["free_now_bytes"] == 13_500_000_000
    assert value["available_bytes"] == value["free_now_bytes"] + value["purgeable_bytes"]
    datetime.fromisoformat(value["measured_at"])


@pytest.mark.parametrize("important,expected", [(131_300_000_000, 131_300_000_000), (None, 13_500_000_000)])
def test_macos_disk_truth(tmp_path, monkeypatch, important, expected):
    from brainlayer import disk_space

    monkeypatch.setattr(disk_space.sys, "platform", "darwin")
    monkeypatch.setattr(disk_space.shutil, "disk_usage", lambda _path: SimpleNamespace(free=13_500_000_000))
    monkeypatch.setattr(disk_space, "_important_usage_capacity_bytes", lambda _path: important)
    value = disk_space.measure_disk_space(tmp_path)
    assert value.available_bytes == expected
    assert value.free_now_bytes == 13_500_000_000
    assert value.purgeable_bytes == expected - value.free_now_bytes
    assert value.source == ("macos_important_usage" if important else "statfs_macos_available_unavailable")


def test_non_macos_states_no_purgeable_space(tmp_path, monkeypatch):
    from brainlayer import disk_space

    monkeypatch.setattr(disk_space.sys, "platform", "linux")
    monkeypatch.setattr(disk_space.shutil, "disk_usage", lambda _path: SimpleNamespace(free=123))
    monkeypatch.setattr(disk_space, "_important_usage_capacity_bytes", lambda _path: pytest.fail("macOS only"))
    value = disk_space.measure_disk_space(tmp_path)
    assert (value.available_bytes, value.free_now_bytes, value.purgeable_bytes) == (123, 123, 0)
    assert value.source == "statfs_non_macos"


@pytest.mark.parametrize("final_free", [34_000_000_000, 20_800_000_000])
def test_backup_escalates_purge_and_keeps_raw_gate(tmp_path, monkeypatch, final_free):
    from brainlayer import backup_daily, disk_space
    from tests.test_backup_daily import _create_source_db

    source = tmp_path / "source.db"
    _create_source_db(source, chunk_count=2)
    db_size = 15_500_000_000
    monkeypatch.setattr(backup_daily, "_database_logical_size_bytes", lambda _path: db_size)
    required = backup_daily._peak_backup_required_bytes(db_size, db_size, full=False)
    free = [13_500_000_000]
    measured = []
    requests = []
    monkeypatch.setattr(disk_space.sys, "platform", "darwin")
    monkeypatch.setattr(disk_space, "_important_usage_capacity_bytes", lambda _path: 131_300_000_000)

    def raw(_path):
        measured.append(free[0])
        return SimpleNamespace(free=free[0])

    def purge(_path, amount, urgency):
        requests.append((amount, urgency))
        free[0] = 18_000_000_000 if urgency == 3 else final_free

    monkeypatch.setattr(disk_space.shutil, "disk_usage", raw)
    monkeypatch.setattr(backup_daily, "_request_macos_purge", purge)
    monkeypatch.setattr(backup_daily.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(backup_daily, "_brainbar_writer_started_at", lambda _path: None)
    monkeypatch.setattr(backup_daily, "_should_run_full_verify", lambda _date: False)
    monkeypatch.setattr(
        backup_daily,
        "request_brainbar_vacuum_into",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("writer reached")),
    )
    with pytest.raises(RuntimeError) as error:
        backup_daily.create_sqlite_backup_artifact(source, tmp_path / "out", date_stamp="2026-05-14")
    assert requests == [
        (required - 13_500_000_000 + backup_daily.COPY_CHUNK_BYTES, 3),
        (required - 18_000_000_000 + backup_daily.COPY_CHUNK_BYTES, 4),
    ]
    assert 18_000_000_000 in measured and final_free in measured
    if final_free >= required:
        assert "writer reached" in str(error.value)
    else:
        assert "needs 33.1 GB; macOS Available 131.3 GB = 20.8 GB free now + 110.5 GB purgeable" in str(error.value)
        assert "purge freed 7.3 GB" in str(error.value)
