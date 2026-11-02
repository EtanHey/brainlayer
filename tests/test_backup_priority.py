"""Backup children lift inherited throttling without affecting maintenance."""

import ctypes
from unittest.mock import Mock, call

from brainlayer import backup_daily as backup
from brainlayer import supervisor_child as policy
from tests.test_weekly_local_gate import local_lane  # noqa: F401  (pytest fixture)


def test_backup_resets_and_reads_policy(monkeypatch):
    monkeypatch.setattr(policy.sys, "platform", "darwin")
    library = Mock()
    library.setiopolicy_np.return_value = 0
    library.getiopolicy_np.return_value = 0
    monkeypatch.setattr(ctypes, "CDLL", Mock(return_value=library))
    setter = Mock()
    monkeypatch.setattr(policy.os, "setpriority", setter)
    monkeypatch.setattr(policy.os, "getpriority", Mock(side_effect=[0, 10]))
    result = policy.backup_io_policy()
    assert setter.call_args_list == [call(4, 0, 0)]
    library.setiopolicy_np.assert_called_once_with(0, 0, 0)
    library.getiopolicy_np.assert_called_once_with(0, 0)
    assert result == {"io_policy": "default", "io_policy_errors": [], "iopol": 0, "darwin_background": 0, "nice": 10}


def test_non_darwin_keeps_inherited_policy(monkeypatch):
    monkeypatch.setattr(policy.sys, "platform", "linux")
    setter = Mock()
    monkeypatch.setattr(policy.os, "setpriority", setter)
    assert policy.backup_io_policy()["io_policy"] == "unsupported"
    setter.assert_not_called()


def test_permission_failure_is_nonfatal_and_measured(monkeypatch):
    monkeypatch.setattr(policy.sys, "platform", "darwin")
    monkeypatch.setattr(policy.os, "setpriority", Mock(side_effect=PermissionError("denied")))
    monkeypatch.setattr(policy.os, "getpriority", Mock(return_value=0x1000))
    library = Mock()
    library.setiopolicy_np.return_value = -1
    library.getiopolicy_np.return_value = 3
    monkeypatch.setattr(ctypes, "CDLL", Mock(return_value=library))
    result = policy.backup_io_policy()
    assert result["io_policy"] == "throttle"
    assert len(result["io_policy_errors"]) == 2


def test_backup_child_passes_effective_policy_to_receipt(monkeypatch, tmp_path):
    from brainlayer import backup_daily

    effective = {"io_policy": "default", "nice": 5}
    reset = Mock(return_value=effective)
    monkeypatch.setattr(policy, "backup_io_policy", reset)
    monkeypatch.setattr(backup_daily, "get_db_path", lambda: tmp_path / "fixture.db")
    run = Mock(return_value={"verified": False})
    monkeypatch.setattr(backup_daily, "run_backup", run)
    monkeypatch.setenv("BRAINLAYER_BACKUP_PHASE", "snapshot")
    assert backup_daily._run_backup_process(30) == 1
    reset.assert_called_once_with()
    assert run.call_args.kwargs["io_policy"] == effective
    assert run.call_args.kwargs["phase"] == "snapshot"


def test_background_reset_denied_cannot_report_default(monkeypatch):
    monkeypatch.setattr(policy.sys, "platform", "darwin")

    def setpriority(which, who, value):
        if which == 4:
            raise PermissionError("denied")
        raise AssertionError("backup must not change nice")

    monkeypatch.setattr(policy.os, "setpriority", setpriority)
    monkeypatch.setattr(policy.os, "getpriority", lambda which, who: 1 if which == 4 else 10)
    library = Mock()
    library.setiopolicy_np.return_value = 0
    library.getiopolicy_np.return_value = 0
    monkeypatch.setattr(ctypes, "CDLL", Mock(return_value=library))
    result = policy.backup_io_policy()
    assert result["io_policy"] == "background"
    assert result["iopol"] == 0
    assert result["darwin_background"] == 1
    assert len(result["io_policy_errors"]) == 1


def test_maintenance_policy_keeps_background(monkeypatch):
    monkeypatch.setattr(policy.sys, "platform", "darwin")
    reset = Mock(side_effect=AssertionError("maintenance must not reset backup policy"))
    monkeypatch.setattr(policy, "backup_io_policy", reset)
    setter = Mock()
    monkeypatch.setattr(policy.os, "setpriority", setter)
    disk = Mock()
    monkeypatch.setattr(policy, "set_disk_io_policy", disk)
    policy.apply_policy({"Nice": 10, "ProcessType": "Background", "LowPriorityIO": True})
    assert setter.call_args_list == [call(policy.os.PRIO_PROCESS, 0, 10), call(4, 0, 0x1000)]
    disk.assert_called_once_with(3)
    reset.assert_not_called()


def test_same_day_snapshot_reuse_records_this_runs_policy(local_lane):  # noqa: F811  (imported fixture)
    """CodeRabbit #1597: a reused receipt must not carry an earlier run's process policy."""
    throttled = {"io_policy": "background", "iopol": 4, "darwin_background": 1, "nice": 10}
    first = backup.run_backup(**local_lane, phase="snapshot", io_policy=throttled)
    assert first["darwin_background"] == 1

    current = {"io_policy": "default", "iopol": 0, "darwin_background": 0, "nice": 0}
    second = backup.run_backup(**local_lane, phase="snapshot", io_policy=current)
    assert second["snapshot"] == first["snapshot"], "same-day snapshot must be reused"
    assert {k: second[k] for k in current} == current
    assert "io_policy_errors" not in second
