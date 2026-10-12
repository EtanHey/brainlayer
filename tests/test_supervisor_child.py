"""Real private child resource and environment acceptance."""

import json
import os
import sys

import pytest

from brainlayer.service_supervisor import Component


def test_child_environment_does_not_inherit_supervisor_denylist(tmp_path, monkeypatch):
    from brainlayer.supervisor_guard import gated_spawn

    monkeypatch.setenv("BRAINLAYER_INGEST_DENYLIST", "must-not-leak")
    log = tmp_path / "env"
    child = Component((sys.executable, "-c", "import os;print(os.environ.get('BRAINLAYER_INGEST_DENYLIST','unset'))"))
    from brainlayer.service_supervisor import child_environment

    with log.open("wb") as output:
        process, gate = gated_spawn(child.argv, child_environment(child.env), output, None)
    os.write(gate, b"R")
    os.close(gate)
    assert process.wait(timeout=3) == 0 and log.read_text().strip() == "unset"


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin child policy requires macOS")
def test_real_child_has_file_limit_nice_background_and_io_policy(tmp_path):
    from brainlayer.supervisor_guard import gated_spawn

    policy = dict(SoftResourceLimits={"NumberOfFiles": 4096}, Nice=10, LowPriorityIO=True, ProcessType="Background")
    code = "import os,resource,ctypes,json;print(json.dumps([resource.getrlimit(resource.RLIMIT_NOFILE)[0],os.getpriority(0,0),os.getpriority(4,0),ctypes.CDLL(None).getiopolicy_np(0,0)]))"
    log = tmp_path / "policy"
    with log.open("wb") as output:
        process, gate = gated_spawn(
            (sys.executable, "-c", code), {"PATH": "/usr/bin:/bin"}, output, None, policy=policy
        )
    os.write(gate, b"R")
    os.close(gate)
    assert process.wait(timeout=5) == 0
    assert json.loads(log.read_text()) == [4096, 10, 1, 3]  # Darwin getter reports background as 1.


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin background inheritance requires macOS")
def test_standard_policy_clears_inherited_background_in_private_child():
    import subprocess

    from brainlayer import supervisor_child

    code = "import os,runpy,sys; apply=runpy.run_path(sys.argv[1])['apply_policy']; apply({'ProcessType':'Background','LowPriorityIO':True}); apply({'ProcessType':'Standard','LowPriorityIO':False});print(os.getpriority(4,0))"
    result = subprocess.run(
        (sys.executable, "-c", code, supervisor_child.__file__), capture_output=True, text=True, timeout=5
    )
    assert result.returncode == 0 and result.stdout.strip() == "0", result.stderr
