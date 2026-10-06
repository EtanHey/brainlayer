"""Legacy enrichment options cannot stop local hotlane vector progress."""

import importlib
from unittest.mock import MagicMock

import pytest

from brainlayer.vector_store import VectorStore


@pytest.mark.parametrize("split", [False, True])
def test_hotlane_retired_enrichment_preserves_local_vectors(tmp_path, caplog, split):
    hotlane = importlib.import_module("scripts.hotlane_brainbar_daemon")
    db = tmp_path / "synthetic.db"
    store = VectorStore(db)
    store.conn.cursor().execute(
        "INSERT INTO chunks (id, content, metadata, source_file, source, created_at) "
        "VALUES ('synthetic-hot', 'Local embedding fixture', '{}', 'brainbar-store', 'mcp', '2099-01-01')"
    )
    callback = MagicMock()
    options = dict(
        embed_fn=lambda _text: [0.125] * 1024,
        recent_limit=5,
        backlog_batch=0,
        enrich_limit=25,
        enrich_since_hours=8760,
        enrich_fn=callback,
    )
    try:
        if split:
            store.close()
            result = hotlane._run_split_cycle(db_path=db, vector_store_cls=VectorStore, **options)
            store = VectorStore(db)
        else:
            result = hotlane.run_cycle(store=store, **options)
        callback.assert_not_called()
        assert result == hotlane.CycleResult(embedded=1)
        assert list(store.conn.cursor().execute("SELECT id FROM chunk_vectors_rowids")) == [("synthetic-hot",)]
        assert list(store.conn.cursor().execute("SELECT summary, tags, enriched_at FROM chunks")) == [
            (None, None, None)
        ]
        assert "enrichment has been retired" in caplog.text
    finally:
        store.close()


def test_hotlane_has_no_enrichment_controller_dependency(monkeypatch):
    import builtins
    import sys

    original_import = builtins.__import__

    def local_import(name, *args, **kwargs):
        if name == "brainlayer.enrichment_controller" or name.startswith(("google.genai", "google.generativeai")):
            raise AssertionError(f"retired hotlane dependency: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", local_import)
    sys.modules.pop("scripts.hotlane_brainbar_daemon", None)
    hotlane = importlib.import_module("scripts.hotlane_brainbar_daemon")
    assert hotlane.get_embedding_model.__module__ == "brainlayer.embeddings"


def test_hotlane_cli_advertises_only_local_work(monkeypatch, capsys):
    import sys

    hotlane = importlib.import_module("scripts.hotlane_brainbar_daemon")
    monkeypatch.setattr(sys, "argv", ["hotlane", "--help"])
    with pytest.raises(SystemExit) as error:
        hotlane.main()
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--backlog-batch" in help_text
    assert "--enrich-" not in help_text


@pytest.mark.parametrize("option", ["--enrich-limit", "--enrich-interval", "--enrich-since-hours"])
def test_hotlane_cli_rejects_retired_cloud_options(monkeypatch, capsys, option):
    import sys

    hotlane = importlib.import_module("scripts.hotlane_brainbar_daemon")
    monkeypatch.setattr(sys, "argv", ["hotlane", option, "25"])
    monkeypatch.setattr(hotlane, "run", MagicMock())
    with pytest.raises(SystemExit) as error:
        hotlane.main()
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
    hotlane.run.assert_not_called()
