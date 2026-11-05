"""Optional runtime user alias source for lexical defense (#995). Every fixture is synthetic."""

import hashlib
import json
import logging
import os
from unittest.mock import patch

import pytest

from brainlayer import lexical_defense as ld
from brainlayer.mcp import search_handler
from brainlayer.mcp.search_handler import _brain_search
from brainlayer.vector_store import VectorStore
from tests.test_search_alias_expansion import _insert_chunk, mock_model  # noqa: F401


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(ld, "_effective_cache", None)


def _write(path, payload):
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    return path


def _entries(*items):
    return {"version": 1, "entries": list(items)}


def test_default_path_is_user_config_and_env_overrides(tmp_path):
    assert ld.default_user_aliases_path({}) == ld.Path.home() / ".config/brainlayer/lexical_defense.local.json"
    override = tmp_path / "aliases.json"
    assert ld.default_user_aliases_path({ld.USER_ALIASES_ENV: str(override)}) == override


def test_absent_file_is_exactly_the_packaged_dictionary(tmp_path):
    packaged = ld.load_lexical_defense_dictionary()
    assert ld.load_effective_lexical_defense_dictionary(tmp_path / "missing.json") is packaged


def test_user_alias_applies_and_packaged_baseline_is_untouched(tmp_path):
    packaged_bytes = hashlib.sha256(ld.DATA_PATH.read_bytes()).hexdigest()
    packaged = ld.load_lexical_defense_dictionary()
    packaged_entries = packaged.entries
    path = _write(
        tmp_path / "a.json", _entries({"canonical": "Zorbleth", "aliases": ["qwvx"], "split_forms": ["zor bleth"]})
    )

    effective = ld.load_effective_lexical_defense_dictionary(path)

    assert effective.lookup("qwvx").canonical == "Zorbleth"
    assert effective.lookup("zor bleth").sources == ("user_config",)
    assert effective.lookup("qwvx").protect_from_split is False  # never enters grammar/prompt exports
    assert effective.entries[: len(packaged_entries)] == packaged_entries
    assert ld.load_lexical_defense_dictionary().lookup("qwvx") is None
    assert hashlib.sha256(ld.DATA_PATH.read_bytes()).hexdigest() == packaged_bytes


MALFORMED = {
    "not_json": "{not json",
    "not_utf8": b"\xff\xfe{}",
    "wrong_version": _entries() | {"version": 2},
    "bool_version": {"version": True, "entries": []},
    "unknown_top_key": _entries() | {"extra": 1},
    "entries_not_list": {"version": 1, "entries": {}},
    "entry_not_object": _entries("Zorbleth"),
    "missing_canonical": _entries({"aliases": ["qwvx"]}),
    "unknown_entry_key": _entries({"canonical": "Zorbleth", "category": "x"}),
    "aliases_not_list": _entries({"canonical": "Zorbleth", "aliases": "qwvx"}),
    "surface_not_string": _entries({"canonical": "Zorbleth", "aliases": [7]}),
    "empty_surface": _entries({"canonical": "Zorbleth", "aliases": ["   "]}),
    "punctuation_only": _entries({"canonical": "Zorbleth", "aliases": ["-.-"]}),
    "surface_too_long": _entries({"canonical": "Zorbleth", "aliases": ["q" * (ld.USER_ALIASES_MAX_SURFACE_CHARS + 1)]}),
    "too_many_surfaces": _entries(
        {"canonical": "Zorbleth", "aliases": [f"qw{i}" for i in range(ld.USER_ALIASES_MAX_SURFACES_PER_ENTRY + 1)]}
    ),
    "too_many_entries": _entries(*({"canonical": f"Zorb{i}"} for i in range(ld.USER_ALIASES_MAX_ENTRIES + 1))),
    # Schema-valid JSON padded with whitespace: only the byte bound can reject it.
    "oversized": json.dumps(_entries({"canonical": "Zorbleth", "aliases": ["qwvx"]})) + " " * ld.USER_ALIASES_MAX_BYTES,
}


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_malformed_file_is_rejected_whole_and_loudly(tmp_path, caplog, case):
    payload = MALFORMED[case]
    path = tmp_path / "bad.json"
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        _write(path, payload)

    with caplog.at_level(logging.WARNING, logger="brainlayer.lexical_defense"):
        effective = ld.load_effective_lexical_defense_dictionary(path)
        ld.load_effective_lexical_defense_dictionary(path)  # same file version: no second warning

    assert effective is ld.load_lexical_defense_dictionary()
    assert [r.message for r in caplog.records].count(caplog.records[0].message) == 1
    assert "ignoring user alias file" in caplog.records[0].message


def test_unreadable_path_is_ignored(tmp_path):
    (tmp_path / "dir.json").mkdir()
    assert ld.load_effective_lexical_defense_dictionary(tmp_path / "dir.json") is ld.load_lexical_defense_dictionary()


def test_conflict_with_packaged_surface_skips_that_entry_only(tmp_path, caplog):
    packaged = ld.load_lexical_defense_dictionary()
    taken = packaged.entries[0].aliases[0]
    path = _write(
        tmp_path / "c.json",
        _entries({"canonical": "Zorbleth", "aliases": [taken.upper()]}, {"canonical": "Quillmoor", "aliases": ["qmr"]}),
    )

    with caplog.at_level(logging.WARNING, logger="brainlayer.lexical_defense"):
        effective = ld.load_effective_lexical_defense_dictionary(path)

    assert effective.lookup(taken) is packaged.lookup(taken)
    assert effective.lookup("Zorbleth") is None
    assert effective.lookup("qmr").canonical == "Quillmoor"
    assert "1 user alias entries skipped on conflict" in caplog.text


def test_cyclic_entries_keep_first_and_expansion_is_one_hop(tmp_path, monkeypatch):
    path = _write(
        tmp_path / "cycle.json",
        _entries(
            {"canonical": "Zorbleth", "aliases": ["qwvx"]},
            {"canonical": "qwvx", "aliases": ["Plimsar"]},
            {"canonical": "Plimsar", "aliases": ["Zorbleth"]},
        ),
    )
    monkeypatch.setenv(ld.USER_ALIASES_ENV, str(path))

    assert search_handler._lexical_defense_variants("Zorbleth") == ["Zorbleth", "qwvx"]
    assert search_handler._lexical_defense_variants("Plimsar") == ["Plimsar"]


def test_edits_apply_on_next_call_without_restart(tmp_path):
    path = _write(tmp_path / "f.json", _entries({"canonical": "Zorbleth", "aliases": ["qwvx"]}))
    first = ld.load_effective_lexical_defense_dictionary(path)
    assert ld.load_effective_lexical_defense_dictionary(path) is first  # unchanged stamp: cached, not re-read

    _write(path, _entries({"canonical": "Zorbleth", "aliases": ["qwvx", "zblt"]}))
    os.utime(path, ns=(1, 1))
    assert ld.load_effective_lexical_defense_dictionary(path).lookup("zblt").canonical == "Zorbleth"

    path.unlink()
    assert ld.load_effective_lexical_defense_dictionary(path) is ld.load_lexical_defense_dictionary()


@pytest.mark.asyncio
async def test_brain_search_expands_user_alias_end_to_end(tmp_path, mock_model, monkeypatch):  # noqa: F811
    store = VectorStore(tmp_path / "user-alias.db")
    try:
        _insert_chunk(store, chunk_id="chunk-zorbleth", content="Zorbleth signed off on the synthetic rollout.")
        store.build_binary_index()
        store.conn.cursor().execute("DELETE FROM chunk_vectors")
        store.conn.cursor().execute("DELETE FROM chunk_vectors_binary")

        async def search(query):
            with (
                patch("brainlayer.mcp.search_handler._get_vector_store", return_value=store),
                patch("brainlayer.mcp.search_handler._get_embedding_model", return_value=mock_model),
            ):
                return (await _brain_search(query=query, project="brainlayer", detail="compact"))[1]

        monkeypatch.setenv(ld.USER_ALIASES_ENV, str(tmp_path / "absent.json"))
        assert (await search("qwvx"))["total"] == 0

        path = _write(tmp_path / "aliases.json", _entries({"canonical": "Zorbleth", "aliases": ["qwvx"]}))
        monkeypatch.setenv(ld.USER_ALIASES_ENV, str(path))
        structured = await search("qwvx")
        assert [r["chunk_id"] for r in structured["results"]] == ["chunk-zorbleth"]
    finally:
        store.close()
