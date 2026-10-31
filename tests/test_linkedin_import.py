"""Synthetic entity import: identity precedence, relations, provenance and reruns."""

import hashlib
import json

import pytest

from brainlayer.linkedin_import import import_connections, require_copy
from brainlayer.vector_store import VectorStore


@pytest.fixture
def export(tmp_path):
    rows = [
        dict(
            full_name="Renamed Example",
            first_name="Renamed",
            last_name="Example",
            linkedin_url="https://www.linkedin.com/in/fake-one/?trk=test",
            company="Existing Co",
            connected_on="2020-01-01",
            relationship_strength="weak",
            has_messages=False,
        ),
        dict(full_name="Email Example", email="fake@example.invalid", company_normalized="New Co"),
        dict(full_name="New Example", first_name="New", last_name="Example", company="New Co"),
        dict(full_name=" new   EXAMPLE ", company="New Co"),
        dict(full_name="", company="Ignored Co"),
    ]
    path = tmp_path / "connections.json"
    path.write_text(json.dumps(rows))
    return path


def seed(store):
    store.upsert_entity("owner", "person", "Etan Heyman")
    store.upsert_entity(
        "url-person",
        "person",
        "Original Example",
        metadata={"linkedin_url": "https://linkedin.com/in/fake-one", "preserved": True},
    )
    store.upsert_entity(
        "email-person", "person", "Email Existing", metadata={"contact_info": {"email": "FAKE@example.invalid"}}
    )
    store.upsert_entity("company", "company", "existing co")


def test_five_rows_twice_preserve_identity_metadata_and_links(tmp_path, export):
    store = VectorStore(tmp_path / "copy.db")
    try:
        seed(store)
        first = import_connections(store, export)
        second = import_connections(store, export)
        assert first["persons"] == {"new": 1, "merged": 2}
        assert first["companies"] == {"new": 1, "merged": 1}
        assert first["relations"] == {"new": 6, "merged": 0}
        assert first["skipped"] == {"missing_name": 1, "duplicate_in_file": 1}
        assert first["duplicates_within_file"] == 1
        assert first["rows_missing_name"] == 1
        assert second["persons"] == {"new": 0, "merged": 3}
        assert second["companies"] == {"new": 0, "merged": 2}
        assert second["relations"] == {"new": 0, "merged": 6}
        person = store.get_entity("url-person")
        assert person["metadata"]["preserved"] is True
        assert person["metadata"]["has_messages"] is False
        assert person["metadata"]["linkedin_imports"][0]["sha256"] == hashlib.sha256(export.read_bytes()).hexdigest()
        assert len(person["metadata"]["linkedin_imports"]) == 1
        assert {a["alias"] for a in store.get_entity_aliases("url-person")} >= {"Renamed", "Example"}
        links = store.get_entity_relations("url-person")
        assert {r["relation_type"] for r in links} == {"works_at", "linkedin_connection"}
        assert any(r["target_id"] == "owner" and r["properties"]["has_messages"] is False for r in links)
    finally:
        store.close()


def test_ambiguous_identity_skips_instead_of_guessing(tmp_path):
    store = VectorStore(tmp_path / "copy.db")
    try:
        for n in range(2):
            store.upsert_entity(str(n), "person", f"Fake {n}", metadata={"email": "same@example.invalid"})
        path = tmp_path / "connections.json"
        path.write_text(json.dumps([dict(full_name="Other", email="same@example.invalid")]))
        result = import_connections(store, path)
        assert result["skipped"] == {"ambiguous_identity": 1}
        assert result["persons"]["new"] == 0
    finally:
        store.close()


def test_missing_owner_stops_connection_relations(tmp_path, export):
    store = VectorStore(tmp_path / "copy.db")
    try:
        result = import_connections(store, export)
        assert result["owner_found"] is False
        assert not list(store.conn.execute("SELECT 1 FROM kg_relations WHERE relation_type='linkedin_connection'"))
    finally:
        store.close()


def test_refuse_live_path_and_hardlink(tmp_path):
    live = tmp_path / "live.db"
    live.touch()
    alias = tmp_path / "alias.db"
    alias.hardlink_to(live)
    for target in (live, alias):
        with pytest.raises(ValueError, match="copy"):
            require_copy(target, live)


def test_existing_relation_evidence_survives_import(tmp_path, export):
    store = VectorStore(tmp_path / "copy.db")
    try:
        seed(store)
        store.add_relation(
            "prior",
            "url-person",
            "owner",
            "linkedin_connection",
            properties={"preserved": True},
            confidence=0.8,
            importance=0.9,
        )
        result = import_connections(store, export)
        assert result["relations"] == {"new": 5, "merged": 1}
        prior = next(r for r in store.get_entity_relations("url-person") if r["id"] == "prior")
        assert prior["properties"]["preserved"] is True
        assert prior["confidence"] == 0.8
        assert prior["importance"] == 0.9
    finally:
        store.close()
