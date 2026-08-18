"""chunk_write helpers — lazy dedupe and explicit zero simhash handling."""

from brainlayer.chunk_write import prepare_canonical_insert
from brainlayer.dedupe import compute_dedupe_fields


def test_prepare_skips_dedupe_when_all_fields_supplied(monkeypatch):
    calls: list[tuple[str, str | None]] = []

    def _spy(content: str, created_at: str | None = None):
        calls.append((content, created_at))
        return compute_dedupe_fields(content, created_at)

    monkeypatch.setattr("brainlayer.chunk_write.compute_dedupe_fields", _spy)
    supplied = compute_dedupe_fields("lazy dedupe payload", "2026-08-18T00:00:00Z")
    prepared = prepare_canonical_insert(
        {
            "id": "lazy-dedupe-1",
            "content": "lazy dedupe payload",
            "source_file": "brainlayer-store",
            "source": "manual",
            "created_at": "2026-08-18T00:00:00Z",
            "dedupe_hash": supplied.dedupe_hash,
            "simhash": "0000000000000000",
            "simhash_band_0": "0000",
            "simhash_band_1": "0000",
            "simhash_band_2": "0000",
            "simhash_band_3": "0000",
        }
    )
    assert calls == []
    assert prepared["simhash"] == "0000000000000000"
    assert prepared["simhash_band_0"] == "0000"


def test_prepare_computes_dedupe_when_missing(monkeypatch):
    calls: list[tuple[str, str | None]] = []

    def _spy(content: str, created_at: str | None = None):
        calls.append((content, created_at))
        return compute_dedupe_fields(content, created_at)

    monkeypatch.setattr("brainlayer.chunk_write.compute_dedupe_fields", _spy)
    prepare_canonical_insert(
        {
            "id": "lazy-dedupe-2",
            "content": "needs dedupe compute",
            "source_file": "brainlayer-store",
            "source": "manual",
            "created_at": "2026-08-18T00:00:01Z",
        }
    )
    assert len(calls) == 1
