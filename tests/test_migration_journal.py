"""Checked journal/marker updates retain old bytes in private archive custody."""

import errno

import pytest

from brainlayer.migration_custody import write_new_file


def inputs(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir(mode=0o700)
    source = tmp_path / "journal"
    source.write_bytes(b"old")
    source.chmod(0o600)
    return source, archive


def test_update_keeps_original_bytes_and_publishes_new_exclusively(tmp_path):
    from brainlayer.migration_journal import archive_replace

    source, archive = inputs(tmp_path)
    original_inode = source.stat().st_ino
    archive_replace(source, b"old", b"new", archive, write_new_file)
    assert source.read_bytes() == b"new"
    assert [p.read_bytes() for p in archive.iterdir()] == [b"old"]
    assert source.stat().st_mode & 0o077 == 0
    backup = next(archive.iterdir())
    assert backup.stat().st_nlink == 1 and backup.stat().st_ino != original_inode


def test_changed_source_refused_before_any_archive_move(tmp_path):
    from brainlayer.migration_journal import archive_replace

    source, archive = inputs(tmp_path)
    source.write_bytes(b"foreign")
    with pytest.raises(RuntimeError, match="changed"):
        archive_replace(source, b"old", b"new", archive, write_new_file)
    assert source.read_bytes() == b"foreign" and not list(archive.iterdir())


def test_publish_failure_restores_original_without_replacing_arrivals(tmp_path):
    from brainlayer.migration_journal import archive_replace

    source, archive = inputs(tmp_path)

    def fail(path, content):
        raise OSError(errno.EIO, "publish")

    with pytest.raises(OSError, match="publish"):
        archive_replace(source, b"old", b"new", archive, fail)
    assert source.read_bytes() == b"old"


def test_foreign_arrival_is_preserved_with_original_in_archive(tmp_path):
    from brainlayer.migration_journal import archive_replace

    source, archive = inputs(tmp_path)

    def arrive(path, content):
        source.write_bytes(b"foreign arrival")
        raise FileExistsError(path)

    with pytest.raises(FileExistsError):
        archive_replace(source, b"old", b"new", archive, arrive)
    assert source.read_bytes() == b"foreign arrival"
    assert [p.read_bytes() for p in archive.iterdir()] == [b"old"]


def test_post_replace_sync_failure_keeps_new_live_bytes_and_complete_preimage(tmp_path, monkeypatch):
    from brainlayer import migration_journal

    source, archive = inputs(tmp_path)

    def fail(directory):
        raise OSError(errno.EIO, "directory sync")

    monkeypatch.setattr(migration_journal, "sync_directory", fail)
    with pytest.raises(OSError) as error:
        migration_journal.archive_replace(source, b"old", b"new", archive, write_new_file)
    assert error.value.errno == errno.EIO
    assert source.read_bytes() == b"new"
    assert [p.read_bytes() for p in archive.iterdir()] == [b"old"]


@pytest.mark.parametrize("mode", [0o755, 0o777])
def test_archive_custody_refused_before_moving_source(tmp_path, mode):
    from brainlayer.migration_journal import archive_replace

    source, archive = inputs(tmp_path)
    archive.chmod(mode)
    with pytest.raises(ValueError, match="archive"):
        archive_replace(source, b"old", b"new", archive, write_new_file)
    assert source.read_bytes() == b"old" and not list(archive.iterdir())


def test_archive_names_order_complete_preimages(tmp_path):
    import re

    from brainlayer.migration_journal import archive_replace

    source, archive = inputs(tmp_path)
    archive_replace(source, b"old", b"new", archive, write_new_file)
    archive_replace(source, b"new", b"final", archive, write_new_file)
    copies = sorted(archive.iterdir())
    assert [p.read_bytes() for p in copies] == [b"old", b"new"]
    assert all(re.fullmatch(r"\d{8}T\d{6}Z-\d{20}-[a-f0-9]{32}-journal", p.name) for p in copies)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "permissions"])
def test_recovery_refuses_ambiguous_staging_and_preserves_personal_bytes(tmp_path, kind):
    from brainlayer.migration_journal import recover_staging

    source, archive = inputs(tmp_path)
    personal = tmp_path / "personal"
    personal.write_bytes(b"personal bytes")
    stage = tmp_path / (".migration-" + "a" * 32)
    if kind == "symlink":
        stage.symlink_to(personal)
    elif kind == "hardlink":
        stage.hardlink_to(personal)
    else:
        stage.write_bytes(b"unknown read-only file")
        stage.chmod(0o400)
    with pytest.raises(ValueError, match="staging"):
        recover_staging(source, archive)
    assert stage.exists() and personal.read_bytes() == b"personal bytes"
    assert source.read_bytes() == b"old"


def test_marker_update_retains_launch_agent_permissions_and_private_backup(tmp_path):
    from brainlayer.migration_custody import write_launch_agent
    from brainlayer.migration_journal import archive_replace

    agents = tmp_path / "Library/LaunchAgents"
    agents.mkdir(parents=True)
    agents.chmod(0o755)
    archive = agents / ".archive"
    archive.mkdir(mode=0o700)
    marker = agents / "com.brainlayer.daemon.plist.loaded-keg"
    marker.write_bytes(b"old")
    marker.chmod(0o644)
    archive_replace(marker, b"old", b"new", archive, write_launch_agent)
    assert marker.read_bytes() == b"new" and marker.stat().st_mode & 0o777 == 0o644
    assert all(p.read_bytes() == b"old" and p.stat().st_mode & 0o077 == 0 for p in archive.iterdir())
