"""embed_pending_chunks skips archived_at/lineage. See test_deferred_embedding.py."""

from brainlayer.store import embed_pending_chunks


def test_embed_pending_chunks_is_callable():
    assert callable(embed_pending_chunks)
