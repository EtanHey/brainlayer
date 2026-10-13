"""Checked replacement keeps the prior journal or marker in private custody."""

import os
import stat
import uuid

from .consolidation_plan import read_owned
from .migration_custody import write_new_file


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def archive_replace(path, expected, content, archive, publish):
    """Archive old bytes, then publish exclusively; never replace a new arrival."""
    for directory, private in ((path.parent, False), (archive, True)):
        if directory.resolve() != directory.absolute():
            raise ValueError("journal/archive directory must not traverse symlinks")
        info = directory.stat(follow_symlinks=False)
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or (stat.S_IMODE(info.st_mode) != 0o700 if private else info.st_mode & 0o022)
        ):
            raise ValueError("journal/archive directory custody refused")
    if read_owned(path) != expected:
        raise RuntimeError(f"migration file changed: {path}")
    backup = archive / (path.name + "." + uuid.uuid4().hex)
    write_new_file(backup, b"")
    moved = False
    try:
        path.rename(backup)
        moved = True
        sync_directory(archive)
        sync_directory(path.parent)
        if read_owned(backup) != expected:
            raise RuntimeError(f"migration file changed while archiving: {backup}")
        publish(path, content)
    except BaseException as failure:
        try:
            if moved and not os.path.lexists(path):
                read_owned(backup)  # Refuse a linked/non-owned custody substitution.
                os.link(backup, path)  # No overwrite, including a concurrent arrival.
                backup.unlink()
                sync_directory(path.parent)
                sync_directory(archive)
        except OSError as restore:
            failure.add_note(f"journal restoration failed: errno {restore.errno}; custody: {backup}")
        except ValueError as restore:
            failure.add_note(f"journal restoration refused: {restore}; custody: {backup}")
        raise
