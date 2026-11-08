import signal

from brainlayer import backup_daily as backup


def test_supervised_child_has_no_second_alarm_during_failure_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv(backup.BACKUP_SUPERVISED_CHILD_ENV, "1")
    monkeypatch.setattr(backup, "get_db_path", lambda: tmp_path / "db")
    timers = []
    monkeypatch.setattr(backup.signal, "setitimer", lambda *args: timers.append(args))
    receipts = []

    def run(**kwargs):
        try:
            backup._raise_backup_timeout(signal.SIGALRM, None)
        finally:
            assert not timers, "a second alarm can interrupt this final receipt"
            receipts.append("final failure")

    monkeypatch.setattr(backup, "run_backup", run)
    assert backup._run_backup_process(3) == 124
    assert receipts == ["final failure"]
    assert timers == []
