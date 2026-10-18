"""Empty index chunks must not poison a pass or change surviving identities."""

import json

import pytest
from typer.testing import CliRunner

import brainlayer.cli as cli
import brainlayer.index_new as adapter
from brainlayer.chunk_write import prepare_canonical_insert
from brainlayer.embeddings import EmbeddedChunk
from brainlayer.pipeline.chunk import Chunk, chunk_content
from brainlayer.pipeline.classify import ContentType, ContentValue, classify_content
from brainlayer.vector_store import VectorStore


@pytest.fixture
def synthetic_embeddings(monkeypatch):
    seen = []

    def embed(chunks, **kwargs):
        seen.extend(c.content for c in chunks)
        return [EmbeddedChunk(chunk=c, embedding=[0.1] * 1024) for c in chunks]

    monkeypatch.setattr(adapter, "embed_chunks", embed)
    return seen


@pytest.mark.parametrize("calendar", [False, True])
def test_index_pass_survives_trailing_empty_paragraph(tmp_path, monkeypatch, synthetic_embeddings, calendar):
    source = tmp_path / "projects"
    source.mkdir()
    # Oversized paragraph followed by a blank paragraph produces a zero-length tail.
    entry = {
        "type": "user",
        "message": {"content": "Synthetic long user message explains the index fixture behavior. " * 50 + "\n\n"},
    }
    chunks = chunk_content(classify_content(entry))
    assert chunks[-1].content == ""
    assert all(c.content.strip() for c in chunks[:-1])

    # Repeated synthetic sentences are deliberately duplicates. Keep each row
    # so this regression can assert the exact surviving positions and text.
    def classify_fixture(entry):
        result = classify_content(entry)
        result.metadata["allow_duplicate"] = True
        return result

    monkeypatch.setattr("brainlayer.pipeline.classify.classify_content", classify_fixture)
    path = source / "a.jsonl"
    path.write_text(json.dumps(entry) + "\n")
    db = tmp_path / "fixture.db"
    with VectorStore(db):
        pass
    monkeypatch.setattr(cli, "get_db_path", lambda: db)
    monkeypatch.setenv("BRAINLAYER_LAUNCHD_SERVICE", "index" if calendar else "interactive")
    monkeypatch.setattr(cli, "_calendar_index_budget_s", lambda: 120)
    result = CliRunner().invoke(cli.app, ["index", str(source)])
    assert result.exit_code == 0, result.output
    assert "skipped_empty_chunks=1" in result.output
    assert all(s.strip() for s in synthetic_embeddings)
    with VectorStore(db, readonly=True) as store:
        rows = list(store.conn.execute("SELECT content FROM chunks ORDER BY position"))
        assert [r[0] for r in rows] == [c.content for c in chunks[:-1]]
    if calendar:
        synthetic_embeddings.clear()
        resumed = CliRunner().invoke(cli.app, ["index", str(source)])
        assert resumed.exit_code == 0, resumed.output
        assert "skipped_empty_chunks=0" in resumed.output
        assert not synthetic_embeddings


@pytest.mark.parametrize("empty", ["", " \t\n", "synthetic scrub-only body"])
def test_adapter_filters_after_scrub_without_renumbering(tmp_path, monkeypatch, synthetic_embeddings, empty):
    if empty == "synthetic scrub-only body":
        real_scrub = adapter.scrub_for_storage
        monkeypatch.setattr(
            adapter,
            "scrub_for_storage",
            lambda text, metadata: ("", metadata) if text == empty else real_scrub(text, metadata),
        )
    bodies = ["First synthetic survivor is retained.", empty, "Last synthetic survivor is retained."]
    chunks = [Chunk(s, ContentType.USER_MESSAGE, ContentValue.HIGH, {"allow_duplicate": True}, len(s)) for s in bodies]
    counts = []
    source = str(tmp_path / "fixture.jsonl")
    with VectorStore(tmp_path / "fixture.db") as store:
        assert (
            adapter.index_chunks_to_sqlite(
                chunks,
                source,
                store=store,
                start_position=64,
                on_skipped_empty=counts.append,
            )
            == 2
        )
        assert list(store.conn.execute("SELECT id, position FROM chunks ORDER BY position")) == [
            (f"{source}:64", 64),
            (f"{source}:66", 66),
        ]
        assert store.conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    assert counts == [1]
    assert synthetic_embeddings == [bodies[0], bodies[2]]


def test_empty_only_batch_does_not_embed_or_open_writer(monkeypatch, synthetic_embeddings):
    def forbidden(*args, **kwargs):
        pytest.fail("empty-only batch must not open a writer")

    monkeypatch.setattr(adapter, "open_writer_store", forbidden)
    counts = []
    assert (
        adapter.index_chunks_to_sqlite(
            [Chunk(" \n", ContentType.STACK_TRACE, ContentValue.HIGH, {}, 2)],
            "fixture.jsonl",
            on_skipped_empty=counts.append,
        )
        == 0
    )
    assert counts == [1]
    assert not synthetic_embeddings


def test_canonical_insert_still_refuses_empty_content():
    with pytest.raises(ValueError, match="content must be non-empty"):
        prepare_canonical_insert({"id": "synthetic-empty", "content": " \n"})
