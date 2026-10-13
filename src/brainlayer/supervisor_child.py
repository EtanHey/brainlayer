"""Stdlib-only gated child: establish resource policy before exec, after custody."""

import ctypes
import json
import os
import resource
import sys


def apply_policy(policy):
    if "SoftResourceLimits" in policy:
        files = policy["SoftResourceLimits"].get("NumberOfFiles")
        if files is not None:
            resource.setrlimit(resource.RLIMIT_NOFILE, (files, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
    if "Nice" in policy:
        os.setpriority(os.PRIO_PROCESS, 0, policy["Nice"])
    background, low_io = policy.get("ProcessType") == "Background", policy.get("LowPriorityIO", False)
    if background or low_io or sys.platform == "darwin" and ("ProcessType" in policy or "LowPriorityIO" in policy):
        if sys.platform != "darwin":
            raise RuntimeError("Darwin ProcessType/LowPriorityIO unavailable on this platform")
        # sys/resource.h: PRIO_DARWIN_PROCESS=4, PRIO_DARWIN_BG=0x1000.
        if "ProcessType" in policy:
            os.setpriority(4, 0, 0x1000 if background else 0)
        if "LowPriorityIO" in policy:
            library = ctypes.CDLL(None, use_errno=True)
            # IOPOL_TYPE_DISK=0, IOPOL_SCOPE_PROCESS=0, IOPOL_THROTTLE=3.
            if library.setiopolicy_np(0, 0, 3 if low_io else 0) == -1:
                raise OSError(ctypes.get_errno(), "setiopolicy_np failed")


if __name__ == "__main__":
    fd = int(sys.argv[1])
    token = os.read(fd, 1)
    os.close(fd)
    if token != b"R":
        sys.exit(1)
    apply_policy(json.loads(sys.argv[2]))
    os.execvpe(sys.argv[3], sys.argv[3:], os.environ)
