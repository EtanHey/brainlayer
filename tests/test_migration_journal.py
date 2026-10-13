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
    archive_replace(source, b"old", b"new", archive, write_new_file)
    assert source.read_bytes() == b"new"
    assert [p.read_bytes() for p in archive.iterdir()] == [b"old"]
    assert source.stat().st_mode & 0o077 == 0


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
        path.write_bytes(b"foreign arrival")
        raise FileExistsError(path)

    with pytest.raises(FileExistsError):
        archive_replace(source, b"old", b"new", archive, arrive)
    assert source.read_bytes() == b"foreign arrival"
    assert [p.read_bytes() for p in archive.iterdir()] == [b"old"]


def test_move_sync_failure_restores_source_and_preserves_primary_error(tmp_path, monkeypatch):
    from brainlayer import migration_journal

    source, archive = inputs(tmp_path)

    def fail(directory):
        raise OSError(errno.EIO, "directory sync")

    monkeypatch.setattr(migration_journal, "sync_directory", fail)
    with pytest.raises(OSError) as error:
        migration_journal.archive_replace(source, b"old", b"new", archive, write_new_file)
    assert error.value.errno == errno.EIO
    assert source.read_bytes() == b"old"


@pytest.mark.parametrize("mode", [0o755, 0o777])
def test_archive_custody_refused_before_moving_source(tmp_path, mode):
    from brainlayer.migration_journal import archive_replace

    source, archive = inputs(tmp_path)
    archive.chmod(mode)
    with pytest.raises(ValueError, match="archive"):
        archive_replace(source, b"old", b"new", archive, write_new_file)
    assert source.read_bytes() == b"old" and not list(archive.iterdir())
