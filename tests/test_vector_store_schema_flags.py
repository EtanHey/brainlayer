from __future__ import annotations


def test_writer_init_marks_provenance_columns_available(tmp_path, monkeypatch):
    from brainlayer.search_repo import _optional_chunk_expr
    from brainlayer.vector_store import VectorStore

    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(tmp_path / "pidfiles"))
    store = VectorStore(tmp_path / "brainlayer.db")
    try:
        columns = {row[1] for row in store.conn.cursor().execute("PRAGMA table_info(chunks)")}

        assert {"provenance_class", "superseded_by"}.issubset(columns)
        assert _optional_chunk_expr(store, "provenance_class") == "provenance_class"
        assert _optional_chunk_expr(store, "superseded_by") == "superseded_by"
    finally:
        store.close()
