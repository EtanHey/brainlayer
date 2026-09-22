"""Focused CLI coverage for the file-backed digest route."""

from pathlib import Path

from typer.testing import CliRunner

from brainlayer.cli import app


def test_digest_file_uses_embedding_model_query_interface(tmp_path: Path, monkeypatch) -> None:
    """The CLI must use the public single-text embedding interface without loading a model."""
    source = tmp_path / "digest.txt"
    source.write_text("Disposable digest fixture", encoding="utf-8")
    db_path = tmp_path / "brainlayer.db"
    calls: dict[str, object] = {}

    class QueryOnlyModel:
        def embed_query(self, text: str) -> list[float]:
            calls["embedded"] = text
            return [0.25, 0.75]

    class FakeStore:
        def __init__(self, path: Path) -> None:
            assert path == db_path
            calls["store"] = self

        def close(self) -> None:
            calls["closed"] = True

    def fake_digest_content(*, content, store, embed_fn, title, project, participants):
        assert content == "Disposable digest fixture"
        assert store is calls["store"]
        assert embed_fn("probe") == [0.25, 0.75]
        assert title is None
        assert project is None
        assert participants is None
        calls["digested"] = True
        return {
            "digest_id": "digest-disposable",
            "summary": "fixture summary",
            "sentiment": {"label": "neutral", "score": 0.0},
            "tags": [],
            "entities": [],
            "action_items": [],
            "decisions": [],
            "stats": {"entities_found": 0, "relations_found": 0},
        }

    monkeypatch.setattr("brainlayer.embeddings.get_embedding_model", lambda: QueryOnlyModel())
    monkeypatch.setattr("brainlayer.paths.DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr("brainlayer.pipeline.digest.digest_content", fake_digest_content)
    monkeypatch.setattr("brainlayer.vector_store.VectorStore", FakeStore)

    result = CliRunner().invoke(app, ["digest", "--file", str(source)])

    assert result.exit_code == 0, result.output
    assert "Digest complete!" in result.output
    assert calls == {
        "store": calls["store"],
        "embedded": "probe",
        "digested": True,
        "closed": True,
    }
