"""Stdlib-only gated child: establish resource policy before exec, after custody."""

import ctypes
import json
import os
import resource
import sys


class _ProcThreadInfo(ctypes.Structure):
    """Darwin proc_threadinfo fields used to read the live scheduler priority."""

    _fields_ = [
        ("user_time", ctypes.c_uint64),
        ("system_time", ctypes.c_uint64),
        ("cpu_usage", ctypes.c_int32),
        ("policy", ctypes.c_int32),
        ("run_state", ctypes.c_int32),
        ("flags", ctypes.c_int32),
        ("sleep_time", ctypes.c_int32),
        ("curpri", ctypes.c_int32),
        ("priority", ctypes.c_int32),
        ("maxpriority", ctypes.c_int32),
        ("name", ctypes.c_char * 64),
    ]


def backup_upload_thread_policy():
    """Measure the socket-owning thread's effective priority and requested QoS."""
    result = {
        "thread_curpri": None,
        "thread_qos_class": None,
        "thread_qos_relative_priority": None,
        "thread_policy_errors": [],
    }
    if sys.platform != "darwin":
        result["thread_policy_errors"].append("unsupported")
        return result

    try:
        lib = ctypes.CDLL(None, use_errno=True)
        lib.pthread_self.restype = ctypes.c_void_p
        lib.pthread_get_qos_class_np.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.pthread_get_qos_class_np.restype = ctypes.c_int
        qos_class, relative_priority = ctypes.c_uint32(), ctypes.c_int()
        rc = lib.pthread_get_qos_class_np(lib.pthread_self(), ctypes.byref(qos_class), ctypes.byref(relative_priority))
        if rc:
            raise OSError(rc, "pthread_get_qos_class_np")
        result["thread_qos_class"] = qos_class.value
        result["thread_qos_relative_priority"] = relative_priority.value
    except Exception as exc:
        result["thread_policy_errors"].append(f"qos: {exc}")

    try:
        lib = ctypes.CDLL(None, use_errno=True)
        thread_id = ctypes.c_uint64()
        lib.pthread_threadid_np.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
        lib.pthread_threadid_np.restype = ctypes.c_int
        rc = lib.pthread_threadid_np(None, ctypes.byref(thread_id))
        if rc:
            raise OSError(rc, "pthread_threadid_np")
        info = _ProcThreadInfo()
        # PROC_PIDTHREADID64INFO returns pth_curpri, matching `ps -M`'s PRI.
        lib.proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.proc_pidinfo.restype = ctypes.c_int
        count = lib.proc_pidinfo(os.getpid(), 15, thread_id.value, ctypes.byref(info), ctypes.sizeof(info))
        if count != ctypes.sizeof(info):
            raise OSError(ctypes.get_errno(), f"proc_pidinfo returned {count} bytes")
        result["thread_curpri"] = info.curpri
    except Exception as exc:
        result["thread_policy_errors"].append(f"priority: {exc}")
    return result


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
        result["iopol"] = value
        # getpriority returns 0 (foreground) or 1 (background), not the setter's 0x1000.
        background = os.getpriority(4, 0)
        result["darwin_background"] = background
        result["io_policy"] = (
            "throttle" if value == 3 else "background" if background != 0 else "default" if value == 0 else str(value)
        )
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
