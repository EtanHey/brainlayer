import fcntl

import pytest

from brainlayer import backup_daily as backup


@pytest.mark.parametrize("release", [True, False])
def test_lock_admission_waits_for_brief_probe_but_is_bounded(tmp_path, monkeypatch, release):
    clock = [0.0]
    sleeps = []
    monkeypatch.setattr(backup.time, "monotonic", lambda: clock[0])
    holder = (tmp_path / ".backup.lock").open("a+b")
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds
        if release:
            fcntl.flock(holder, fcntl.LOCK_UN)

    monkeypatch.setattr(backup.time, "sleep", sleep)

    @backup._serialized_backup_run
    def run(staging_dir, log_path, db_path):
        return {"admitted": True}

    try:
        if release:
            assert run(tmp_path, tmp_path / "log", tmp_path / "db") == {"admitted": True}
        else:
            with pytest.raises(backup.BackupAlreadyRunningError):
                run(tmp_path, tmp_path / "log", tmp_path / "db")
            assert 2 <= clock[0] <= 5
        assert sleeps
    finally:
        holder.close()
