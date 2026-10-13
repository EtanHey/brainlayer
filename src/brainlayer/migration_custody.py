"""Private custody and exclusive publication for installer journals."""

import fcntl
import os
import stat
import uuid
from contextlib import contextmanager


def write_new_file(path, content, mode=0o600):
    """Publish complete bytes without replacing any existing file or symlink."""
    _write_exclusive(path, content, mode, private_parent=True)


def write_launch_agent(path, content):
    """Publish only a consolidated launcher/marker in the owned LaunchAgents dir."""
    validate_launch_agent(path)
    _write_exclusive(path, content, 0o644, private_parent=False)


def validate_launch_agent(path):
    """Validate a known target before staging or publishing its bytes."""
    from .consolidation_plan import NEW

    if (
        path.parent.name != "LaunchAgents"
        or path.parent.parent.name != "Library"
        or path.name not in {label + suffix for label in NEW for suffix in (".plist", ".plist.loaded-keg")}
    ):
        raise ValueError("refusing non-consolidated LaunchAgent publication")


def _write_exclusive(path, content, mode, *, private_parent):
    parent = path.parent
    if parent.resolve() != parent.absolute():
        raise ValueError("journal directory must not traverse symlinks")
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(directory)
        if info.st_uid != os.getuid() or (
            stat.S_IMODE(info.st_mode) != 0o700 if private_parent else info.st_mode & 0o022
        ):
            policy = "mode 0700" if private_parent else "not group/world writable"
            raise ValueError(f"publication directory must be owned and {policy}")
        name = ".migration-" + uuid.uuid4().hex
        fd = os.open(name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode, dir_fd=directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                os.fchmod(stream.fileno(), mode)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(name, path.name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
        except BaseException as failure:
            try:
                os.unlink(name, dir_fd=directory)
                os.fsync(directory)
            except OSError as cleanup:
                failure.add_note(f"temporary migration cleanup failed: errno {cleanup.errno}")
            raise
        else:
            os.unlink(name, dir_fd=directory)
            os.fsync(directory)
        current = parent.stat(follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
            raise RuntimeError("journal directory changed during publication")
    finally:
        os.close(directory)


@contextmanager
def migration_lock(home):
    """Hold owner-only transaction custody, refusing a concurrent installer."""
    agents, control = home / "Library/LaunchAgents", home / ".local/share/brainlayer/supervisor"
    if agents.resolve() != agents.absolute() or control.resolve() != control.absolute():
        raise ValueError("migration directories must not traverse symlinks")
    control.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = control.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077 or not stat.S_ISDIR(info.st_mode):
        raise ValueError("migration custody must be owner-only")
    fd = os.open(control / "migration.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1 or info.st_mode & 0o077:
            raise ValueError("migration lock must be owner-only and regular")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("migration busy") from exc
        yield control
    finally:
        os.close(fd)
