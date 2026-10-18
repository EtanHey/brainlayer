"""VFS fault injection proves sync ordering, not physical power-loss behavior."""

import json
from contextlib import contextmanager

import apsw
from typer.testing import CliRunner

import brainlayer.cli as cli
import brainlayer.index_new as adapter
from tests.test_index_checkpoint_boundary import checkpoint, synthetic_vectors
from tests.test_index_completion import calendar_fixture as calendar_fixture
from tests.test_index_completion import readback


@contextmanager
def wal_faults(*, fail_sync=False):
    base = apsw.vfs_names()[0]
    events = []
    state = {"fail": fail_sync, "frames": False}

    class File(apsw.VFSFile):
        def __init__(self, filename, flags):
            super().__init__(base, filename, flags)
            self.wal = str(filename).endswith("-wal")

        def xWrite(self, data, offset):
            super().xWrite(data, offset)
            if self.wal:
                state["frames"] |= offset >= 32
                events.append("wal-write")

        def xSync(self, flags):
            if self.wal and state["frames"] and state["fail"]:
                events.append("wal-sync-fault")
                raise apsw.IOError("synthetic WAL sync failure")
            super().xSync(flags)
            if self.wal:
                events.append("wal-sync")

    class VFS(apsw.VFS):
        def xOpen(self, name, flags):
            return File(name, flags)

    vfs = VFS("idx-correction-sync", base, makedefault=True)
    try:
        yield events, state
    finally:
        apsw.set_default_vfs(base)
        vfs.unregister()


def test_checkpoint_publication_follows_wal_frame_sync(calendar_fixture, monkeypatch):
    source, db, chunks, _ = calendar_fixture
    from brainlayer.drive_credentials import _atomic_write

    monkeypatch.setenv("BRAINLAYER_WRITE_SYNCHRONOUS", "NORMAL")
    monkeypatch.setattr(adapter, "embed_chunks", synthetic_vectors)
    published = []
    with wal_faults() as (events, _):

        def publish(path, payload):
            if path.name.startswith("index-checkpoint-") and payload["files"]:
                assert events and events[-1] == "wal-sync", events
                published.append(next(iter(payload["files"].values()))["next_chunk"])
                print(
                    f"DURABLE_PUBLICATION next={next(iter(payload['files'].values()))['next_chunk']} "
                    f"last_vfs_event={events[-1]}"
                )
            return _atomic_write(path, payload)

        monkeypatch.setattr("brainlayer.drive_credentials._atomic_write", publish)
        result = CliRunner().invoke(cli.app, ["index", str(source)])
    assert result.exit_code == 0, result.output
    print(f"PUBLICATIONS {published} all_follow_synced_wal_frames=true")
    readback(db, chunks)


def test_failed_wal_sync_cannot_advance_checkpoint_and_retry_recovers(calendar_fixture, monkeypatch):
    source, db, chunks, _ = calendar_fixture
    monkeypatch.setenv("BRAINLAYER_WRITE_SYNCHRONOUS", "NORMAL")
    monkeypatch.setattr(adapter, "embed_chunks", synthetic_vectors)
    with wal_faults(fail_sync=True) as (events, state):
        result = CliRunner().invoke(cli.app, ["index", str(source)])
        state["fail"] = False  # Let SQLite recovery and the fixture's own closes sync.
    assert result.exit_code == 1, result.output
    assert "wal-sync-fault" in events
    assert "INDEX_CHECKPOINTED" not in result.output and checkpoint(db) == {}
    print("SYNC_FAULT rc=1 checkpoint_unadvanced=true replay_required=true")
    retry = CliRunner().invoke(cli.app, ["index", str(source)])
    assert retry.exit_code == 0, retry.output
    readback(db, chunks)


def test_legacy_checkpoint_cannot_skip_simulated_lost_rows(calendar_fixture, monkeypatch):
    source, db, chunks, _ = calendar_fixture
    saved = cli._CalendarIndexCheckpoint(db, source, None, False)
    # Simulated recovery: the old NORMAL sidecar survived, its rows did not.
    payload = dict(
        version=1,
        database=saved.database,
        files={
            str((source / "a.jsonl").resolve()): dict(
                signature=saved.signature(source / "a.jsonl"), next_chunk=66, done=True
            )
        },
    )
    saved.path.write_text(json.dumps(payload))
    monkeypatch.setattr(adapter, "embed_chunks", synthetic_vectors)
    result = CliRunner().invoke(cli.app, ["index", str(source)])
    assert result.exit_code == 0, result.output
    readback(db, chunks)
