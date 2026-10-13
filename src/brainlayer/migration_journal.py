"""Checked replacement keeps the prior journal or marker in private custody."""

import os
import re
import stat
import time
import uuid

from .consolidation_plan import read_owned
from .migration_custody import _write_exclusive, validate_launch_agent, write_launch_agent, write_new_file


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


_STAGING = re.compile(r"\.migration-(?:[a-f0-9]{32}|[0-9]{8}T[0-9]{6}Z-[0-9]{20}-[a-f0-9]{32})")


def _validate_directories(path, archive):
    if _STAGING.fullmatch(path.name):
        raise ValueError("live journal uses reserved staging namespace")
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


def recover_staging(path, archive):
    """Under migration_lock, remove owned staging; refuse ambiguous custody."""
    _validate_directories(path, archive)
    for parent in (path.parent, archive):
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            backup_name = re.compile(r"[0-9]{8}T[0-9]{6}Z-[0-9]{20}-[a-f0-9]{32}-" + re.escape(path.name))
            entries = {
                name: os.stat(name, dir_fd=directory, follow_symlinks=False)
                for name in os.listdir(directory)
                if _STAGING.fullmatch(name) or backup_name.fullmatch(name)
            }
            candidates = {name: info for name, info in entries.items() if _STAGING.fullmatch(name)}
            for name, info in candidates.items():
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) not in (0o600, 0o644)
                    or info.st_nlink not in (1, 2)
                ):
                    raise ValueError("staging custody refused")
                if info.st_nlink == 2:
                    peers = [
                        other
                        for other, peer in entries.items()
                        if other != name
                        and (peer.st_dev, peer.st_ino) == (info.st_dev, info.st_ino)
                        and (
                            _STAGING.fullmatch(other)
                            or re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[0-9]{20}-[a-f0-9]{32}-" + re.escape(path.name), other)
                        )
                    ]
                    if len(peers) != 1:
                        raise ValueError("staging hardlink custody refused")
            for name, info in candidates.items():
                current = os.stat(name, dir_fd=directory, follow_symlinks=False)
                if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
                    raise RuntimeError("staging changed during recovery")
                os.unlink(name, dir_fd=directory)
            if candidates:
                os.fsync(directory)
        finally:
            os.close(directory)


def archive_replace(path, expected, content, archive, publish):
    """Copy the preimage, then atomically replace the live name under migration_lock."""
    if publish is write_launch_agent:
        validate_launch_agent(path)
    _validate_directories(path, archive)
    recover_staging(path, archive)
    if read_owned(path) != expected:
        raise RuntimeError(f"migration file changed: {path}")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    suffix = f"{stamp}-{time.monotonic_ns():020d}-{uuid.uuid4().hex}"
    backup = archive / (suffix + "-" + path.name)
    write_new_file(backup, expected)
    stage = path.parent / (".migration-" + suffix)
    try:
        if publish is write_launch_agent:
            _write_exclusive(stage, content, 0o644, private_parent=False)
        else:
            publish(stage, content)
        if read_owned(stage) != content:
            raise RuntimeError("staged journal bytes changed")
        if read_owned(path) != expected:
            raise RuntimeError(f"migration file changed before atomic replace: {path}")
        os.replace(stage, path)
        sync_directory(path.parent)
    except BaseException as failure:
        try:
            if os.path.lexists(stage) and read_owned(stage) == content:
                stage.unlink()
                sync_directory(path.parent)
        except (OSError, ValueError) as cleanup:
            failure.add_note(f"staged journal cleanup refused: {cleanup}; custody: {stage}")
        raise
