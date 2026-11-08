import fcntl
import threading

import pytest

from brainlayer import backup_daily as backup


@pytest.mark.parametrize("release", [True, False])
def test_lock_admission_waits_for_brief_probe_but_is_bounded(tmp_path, monkeypatch, release):
    clock = [0.0]
    sleeps = []
    holder = (tmp_path / ".backup.lock").open("a+b")
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)

    contender = threading.current_thread()
    real_time = backup.time

    def contender_sleep(seconds):
        # Patching backup.time.* would replace the process-wide time module, so any other
        # thread's sleep advanced this clock (CI: 11279.7 <= 5). Count only this thread's wait.
        if threading.current_thread() is not contender:
            return real_time.sleep(seconds)
        sleeps.append(seconds)
        clock[0] += seconds
        if release:
            fcntl.flock(holder, fcntl.LOCK_UN)

    class _ModuleClock:
        """A backup_daily-local stand-in for ``time``; other modules keep the real one."""

        def __getattr__(self, name):
            return getattr(real_time, name)

        @staticmethod
        def monotonic():
            return clock[0]

        @staticmethod
        def sleep(seconds):
            return contender_sleep(seconds)

    monkeypatch.setattr(backup, "time", _ModuleClock())

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
