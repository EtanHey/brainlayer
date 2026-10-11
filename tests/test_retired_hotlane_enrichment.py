"""Legacy enrichment options cannot stop local hotlane vector progress."""

import importlib
from unittest.mock import MagicMock

import pytest

from brainlayer.vector_store import VectorStore

pytestmark = pytest.mark.retired_enrichment


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
        # CycleResult retains its legacy tuple shape; neither real cycle produces diagnostics.
        assert result.enrich_attempted == result.enriched == result.enrich_skipped == result.enrich_failed == 0
        assert result.enrich_daily_cap_reached is False
        assert list(store.conn.cursor().execute("SELECT id FROM chunk_vectors_rowids")) == [("synthetic-hot",)]
        assert list(store.conn.cursor().execute("SELECT summary, tags, enriched_at FROM chunks")) == [
            (None, None, None)
        ]
        assert "enrichment has been retired" in caplog.text
    finally:
        store.close()


def test_hotlane_has_no_enrichment_controller_dependency(monkeypatch, tmp_path):
    import builtins
    import sys
    from types import SimpleNamespace

    original_import = builtins.__import__

    def local_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            name == "brainlayer.enrichment_controller"
            or (name == "brainlayer" and "enrichment_controller" in (fromlist or ()))
            or name.startswith(("google.genai", "google.generativeai"))
        ):
            raise AssertionError(f"retired hotlane dependency: {name} {fromlist}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", local_import)
    monkeypatch.delitem(sys.modules, "brainlayer.enrichment_controller", raising=False)
    monkeypatch.delitem(sys.modules, "scripts.hotlane_brainbar_daemon", raising=False)
    hotlane = importlib.import_module("scripts.hotlane_brainbar_daemon")
    assert hotlane.get_embedding_model.__module__ == "brainlayer.embeddings"
    assert "brainlayer.enrichment_controller" not in sys.modules

    run = hotlane.run
    calls = []
    monkeypatch.setattr(hotlane, "run", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(hotlane.signal, "signal", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["hotlane"])
    hotlane.main()
    assert len(calls) == 1
    assert "brainlayer.enrichment_controller" not in sys.modules

    run(
        db_path=tmp_path / "synthetic.db",
        interval=0.25,
        recent_limit=5,
        backlog_interval=1.0,
        backlog_batch=0,
        enrich_interval=0.0,
        enrich_limit=0,
        enrich_since_hours=0,
        max_cycles=1,
        model_factory=lambda: SimpleNamespace(embed_query=lambda text: [0.125] * 1024),
        vector_store_cls=lambda path: SimpleNamespace(close=lambda: None),
        cycle_fn=lambda **kwargs: hotlane.CycleResult(),
        queue_dir=tmp_path / "queue",
        sleep_fn=lambda seconds: None,
    )
    assert "brainlayer.enrichment_controller" not in sys.modules


def test_hotlane_source_has_no_enrichment_controller_reference():
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "scripts/hotlane_brainbar_daemon.py"
    tree = ast.parse(source.read_text())
    assert "enrichment_controller" not in ast.dump(tree)


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


@pytest.mark.parametrize("legacy_values", [("10.0", "0", "87600"), ("0", "0", "0"), ("1", "25", "87600")])
def test_hotlane_cli_accepts_retired_plist_options_as_noops(monkeypatch, caplog, legacy_values):
    import sys

    hotlane = importlib.import_module("scripts.hotlane_brainbar_daemon")
    argv = [
        "hotlane",
        "--interval",
        "1.0",
        "--recent-limit",
        "5",
        "--backlog-interval",
        "7.0",
        "--backlog-batch",
        "16",
        "--enrich-interval",
        legacy_values[0],
        "--enrich-limit",
        legacy_values[1],
        "--enrich-since-hours",
        legacy_values[2],
    ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(hotlane.signal, "signal", lambda *args: None)
    calls = []
    monkeypatch.setattr(hotlane, "run", lambda **kwargs: calls.append(kwargs))
    hotlane.main()
    assert len(calls) == 1
    assert calls[0]["enrich_limit"] == 0
    assert calls[0]["enrich_interval"] == 0.0
    assert calls[0]["enrich_since_hours"] == 0
    assert calls[0]["backlog_batch"] == 16
    warnings = [record for record in caplog.records if "retired" in record.getMessage()]
    assert len(warnings) == int(any(float(value) != 0 for value in legacy_values))
