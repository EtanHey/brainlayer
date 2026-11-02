"""Backup children lift inherited throttling without affecting maintenance."""

import ctypes
from unittest.mock import Mock, call

from brainlayer import supervisor_child as policy


def test_backup_resets_and_reads_policy(monkeypatch):
    monkeypatch.setattr(policy.sys, "platform", "darwin")
    library = Mock()
    library.setiopolicy_np.return_value = 0
    library.getiopolicy_np.return_value = 0
    monkeypatch.setattr(ctypes, "CDLL", Mock(return_value=library))
    setter = Mock()
    monkeypatch.setattr(policy.os, "setpriority", setter)
    monkeypatch.setattr(policy.os, "getpriority", Mock(side_effect=[0, 5]))
    result = policy.backup_io_policy()
    assert setter.call_args_list == [call(4, 0, 0), call(policy.os.PRIO_PROCESS, 0, 5)]
    library.setiopolicy_np.assert_called_once_with(0, 0, 0)
    library.getiopolicy_np.assert_called_once_with(0, 0)
    assert result == {"io_policy": "default", "io_policy_errors": [], "darwin_background": 0, "nice": 5}


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
    assert len(result["io_policy_errors"]) == 3


def test_backup_child_passes_effective_policy_to_receipt(monkeypatch, tmp_path):
    from brainlayer import backup_daily

    effective = {"io_policy": "default", "nice": 5}
    reset = Mock(return_value=effective)
    monkeypatch.setattr(policy, "backup_io_policy", reset)
    monkeypatch.setattr(backup_daily, "get_db_path", lambda: tmp_path / "fixture.db")
    run = Mock(return_value={"verified": False})
    monkeypatch.setattr(backup_daily, "run_backup", run)
    monkeypatch.setenv("BRAINLAYER_BACKUP_PHASE", "snapshot")
    assert backup_daily._run_backup_process(30) == 0
    reset.assert_called_once_with()
    assert run.call_args.kwargs["io_policy"] == effective
    assert run.call_args.kwargs["phase"] == "snapshot"
