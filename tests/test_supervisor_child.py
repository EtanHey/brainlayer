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
    code = "import os,resource,ctypes,json;lib=ctypes.CDLL(None);values=[resource.getrlimit(resource.RLIMIT_NOFILE)[0],os.getpriority(0,0),os.getpriority(4,0),lib.getiopolicy_np(0,0)];os.setpriority(4,0,0);lib.setiopolicy_np(0,0,0);print(json.dumps(values))"
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
@pytest.mark.parametrize(
    "policy,expected",
    [({"ProcessType": "Standard", "LowPriorityIO": False}, [0, 0]), ({"LowPriorityIO": False}, [1, 0])],
)
def test_explicit_policy_clears_inherited_io_in_private_child(policy, expected):
    import subprocess

    from brainlayer import supervisor_child

    # Measure before normalizing this finite probe's scheduling priority for cleanup.
    code = "import os,runpy,sys,json,ctypes; apply=runpy.run_path(sys.argv[1])['apply_policy']; apply({'ProcessType':'Background','LowPriorityIO':True}); apply(json.loads(sys.argv[2]));values=[os.getpriority(4,0),ctypes.CDLL(None).getiopolicy_np(0,0)];apply({'ProcessType':'Standard','LowPriorityIO':False});print(json.dumps(values))"
    result = subprocess.run(
        (sys.executable, "-c", code, supervisor_child.__file__, json.dumps(policy)),
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected


@pytest.mark.parametrize("cold_imports", [False, True])
def test_gate_preserves_calendar_fd_with_cwd_and_policy(tmp_path, monkeypatch, cold_imports):
    import shlex

    from brainlayer.supervisor_guard import gated_spawn

    python = sys.executable
    if cold_imports:
        wrapper = tmp_path / "cold-python"
        wrapper.write_text(f'#!/bin/sh\nexec {shlex.quote(python)} -S "$@"\n')
        wrapper.chmod(0o700)
        monkeypatch.setattr(sys, "executable", str(wrapper))
    read, write = os.pipe()
    process = None
    (tmp_path / "configured_module.py").write_text("VALUE=17\n")
    code = "import os,resource,sys,configured_module;os.write(int(sys.argv[1]),f'{resource.getrlimit(resource.RLIMIT_NOFILE)[0]}:{os.environ[\"PROOF\"]}:{configured_module.VALUE}'.encode());open('proof','w').write('cwd')"
    try:
        process, gate = gated_spawn(
            (python, "-c", code, str(write)),
            {"PYTHONPATH": str(tmp_path), "PROOF": "kept"},
            None,
            None,
            cwd=tmp_path,
            policy={"SoftResourceLimits": {"NumberOfFiles": 4096}},
            extra_fds=(write,),
        )
        os.close(write)
        os.write(gate, b"R")
        os.close(gate)
        assert process.wait(timeout=3) == 0
        assert os.read(read, 100) == b"4096:kept:17" and (tmp_path / "proof").read_text() == "cwd"
    finally:
        os.close(read)
        if process and process.poll() is None:
            process.kill()
            process.wait(timeout=3)
