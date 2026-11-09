import sqlite3

import pytest

from brainlayer.watcher_bridge import _archive_reconciliation_stored


def test_unavailable_reconciliation_lookup_holds_the_checkpoint(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise sqlite3.OperationalError("forced read failure")

    monkeypatch.setattr(sqlite3, "connect", unavailable)
    with pytest.raises(OSError, match="checkpoint held"):
        _archive_reconciliation_stored(tmp_path / "x.db", "id", "content", None, "repo", "user_message")
