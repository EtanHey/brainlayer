"""Stdlib-only gated child: establish resource policy before exec, after custody."""

import ctypes
import json
import os
import resource
import sys


def set_disk_io_policy(value):
    """Set this process's Darwin disk policy (DEFAULT=0, THROTTLE=3)."""
    library = ctypes.CDLL(None, use_errno=True)
    if library.setiopolicy_np(0, 0, value) == -1:
        raise OSError(ctypes.get_errno(), "setiopolicy_np failed")


def backup_io_policy():
    """Best-effort reset in the dedicated backup child; return measured policy."""
    result = {"io_policy": "unknown", "io_policy_errors": []}
    if sys.platform != "darwin":
        result["io_policy"] = "unsupported"
        print("Backup I/O policy: unsupported platform; keeping inherited policy", flush=True)
        return result
    for name, operation in (
        ("darwin_background", lambda: os.setpriority(4, 0, 0)),
        ("disk", lambda: set_disk_io_policy(0)),
        ("nice", lambda: os.setpriority(os.PRIO_PROCESS, 0, 5)),
    ):
        try:
            operation()
        except (OSError, AttributeError) as exc:
            result["io_policy_errors"].append(f"{name}: {exc}")
    try:
        library = ctypes.CDLL(None, use_errno=True)
        value = library.getiopolicy_np(0, 0)
        if value == -1:
            raise OSError(ctypes.get_errno(), "getiopolicy_np failed")
        result["io_policy"] = {0: "default", 3: "throttle"}.get(value, str(value))
        result["darwin_background"] = os.getpriority(4, 0)
        result["nice"] = os.getpriority(os.PRIO_PROCESS, 0)
    except (OSError, AttributeError) as exc:
        result["io_policy_errors"].append(f"readback: {exc}")
    print("Backup effective policy: " + json.dumps(result, sort_keys=True), flush=True)
    return result


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
            set_disk_io_policy(3 if low_io else 0)


if __name__ == "__main__":
    fd = int(sys.argv[1])
    token = os.read(fd, 1)
    os.close(fd)
    if token != b"R":
        sys.exit(1)
    apply_policy(json.loads(sys.argv[2]))
    os.execvpe(sys.argv[3], sys.argv[3:], os.environ)
