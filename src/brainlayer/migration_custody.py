"""Private custody and exclusive publication for installer journals."""

import fcntl
import os
import stat
import uuid
from contextlib import contextmanager


def write_new_file(path, content, mode=0o600):
    """Publish complete bytes without replacing any existing file or symlink."""
    parent = path.parent
    if parent.resolve() != parent.absolute():
        raise ValueError("journal directory must not traverse symlinks")
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(directory)
        if info.st_uid != os.getuid():
            raise ValueError("journal directory must be owned")
        name = ".migration-" + uuid.uuid4().hex
        fd = os.open(name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode, dir_fd=directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                os.fchmod(stream.fileno(), mode)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(name, path.name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
        finally:
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
