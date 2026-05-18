"""Parent-death watcher for BrainLayer Python sidecars on macOS/BSD."""

from __future__ import annotations

import errno
import logging
import os
import select
import threading

logger = logging.getLogger(__name__)


def install_parent_death_watcher() -> bool:
    """Exit this process when its current parent exits.

    macOS has no Linux-style PR_SET_PDEATHSIG. For BrainBar-spawned Python
    sidecars, EVFILT_PROC/NOTE_EXIT gives us the same zero-polling behavior.
    Returns False on platforms without kqueue or when already reparented.
    """
    if os.environ.get("BRAINLAYER_DISABLE_PARENT_DEATH_WATCH") == "1":
        return False

    kqueue_factory = getattr(select, "kqueue", None)
    kevent_factory = getattr(select, "kevent", None)
    if kqueue_factory is None or kevent_factory is None:
        return False

    parent_pid = os.getppid()
    if parent_pid <= 1:
        return False

    try:
        kq = kqueue_factory()
        event = kevent_factory(
            parent_pid,
            filter=select.KQ_FILTER_PROC,
            flags=select.KQ_EV_ADD,
            fflags=select.KQ_NOTE_EXIT,
        )
        kq.control([event], 0, 0)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            os._exit(0)
        logger.debug("parent death watcher not installed for pid %s: %s", parent_pid, exc)
        return False

    def _wait_for_parent_exit() -> None:
        try:
            kq.control([], 1, None)
        finally:
            os._exit(0)

    thread = threading.Thread(target=_wait_for_parent_exit, name="brainlayer-parent-death", daemon=True)
    thread.start()
    return True
