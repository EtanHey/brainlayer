"""Synthetic entity import: identity precedence, relations, provenance and reruns."""

import hashlib
import json
from pathlib import Path

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
        assert first["persons"] == {
            "new": 1,
            "merged": 2,
            "new_name_conflict": 0,
            "merged_by": {"linkedin_url": 1, "email": 1, "name": 0},
        }
        assert first["companies"] == {"new": 1, "merged": 1, "merged_by": {"name": 1}}
        assert first["relations"] == {"new": 6, "merged": 0}
        assert first["skipped"] == {"missing_name": 1, "duplicate_in_file": 1}
        assert first["duplicates_within_file"] == 1
        assert first["rows_missing_name"] == 1
        assert second["persons"] == {
            "new": 0,
            "merged": 3,
            "new_name_conflict": 0,
            "merged_by": {"linkedin_url": 1, "email": 1, "name": 1},
        }
        assert second["companies"] == {"new": 0, "merged": 2, "merged_by": {"name": 2}}
        assert second["relations"] == {"new": 0, "merged": 6}
        person = store.get_entity("url-person")
        assert person["metadata"]["preserved"] is True
        assert person["metadata"]["has_messages"] is False
        assert person["metadata"]["linkedin_imports"][0]["sha256"] == hashlib.sha256(export.read_bytes()).hexdigest()
        assert len(person["metadata"]["linkedin_imports"]) == 1
        assert person["metadata"]["first_name"] == "Renamed"
        assert person["metadata"]["last_name"] == "Example"
        assert {a["alias"] for a in store.get_entity_aliases("url-person")} == {
            "Renamed Example",
            "https://linkedin.com/in/fake-one",
        }
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


@pytest.mark.parametrize(
    "stored,row,conflict,match_kind",
    [
        ({"linkedin_url": "https://linkedin.com/in/old"}, {"linkedin_url": "https://linkedin.com/in/new"}, True, None),
        ({}, {"linkedin_url": "https://linkedin.com/in/new"}, False, "name"),
        (
            {"linkedin_url": "https://linkedin.com/in/same"},
            {"linkedin_url": "https://linkedin.com/in/same"},
            False,
            "linkedin_url",
        ),
        ({"email": "old@example.invalid"}, {"email": "new@example.invalid"}, True, None),
    ],
)
@pytest.mark.parametrize("alias_only", [False, True])
def test_same_name_respects_identity_and_reruns(tmp_path, stored, row, conflict, match_kind, alias_only):
    store = VectorStore(tmp_path / "copy.db")
    try:
        store.upsert_entity("existing", "person", "Collision Example", metadata={} if alias_only else stored)
        # Cover typed identity aliases even when the metadata has no identity fields.
        for key, value in stored.items():
            store.add_entity_alias(value, "existing", alias_type=key)
        export = tmp_path / "connections.json"
        export.write_text(json.dumps([dict(full_name="Collision Example", **row)]))
        first = import_connections(store, export)
        assert first["persons"]["new"] == int(conflict)
        assert first["persons"]["merged"] == int(not conflict)
        assert first["persons"]["new_name_conflict"] == int(conflict)
        assert first["persons"]["merged_by"] == {
            key: int(key == match_kind) for key in ("linkedin_url", "email", "name")
        }
        if conflict:
            assert store.get_entity("existing")["metadata"] == ({} if alias_only else stored)
        second = import_connections(store, export)
        assert second["persons"]["new"] == 0
        # Another identity owns the original name: reruns use the conservative skip path.
        assert second["persons"]["merged"] == int(not conflict)
        assert second["skipped"] == ({"ambiguous_identity": 1} if conflict else {})
        assert len(list(store.conn.execute("SELECT id FROM kg_entities WHERE entity_type='person'"))) == 1 + int(
            conflict
        )
    finally:
        store.close()


def test_transaction_rolls_back_all_imported_entities(tmp_path, export, monkeypatch):
    from brainlayer.kg_repo import KGMixin

    store = VectorStore(tmp_path / "copy.db")
    original = KGMixin.add_entity_alias

    def fail_after_entity_write(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError("synthetic_write_failure")

    monkeypatch.setattr(KGMixin, "add_entity_alias", fail_after_entity_write)
    try:
        with pytest.raises(RuntimeError, match="synthetic_write_failure"):
            import_connections(store, export)
        assert list(store.conn.execute("SELECT id FROM kg_entities")) == []
        assert list(store.conn.execute("SELECT alias FROM kg_entity_aliases")) == []
        assert not store.conn.in_transaction
    finally:
        store.close()


@pytest.mark.parametrize(
    "flags,error",
    [
        (["--live"], "confirmation_required"),
        (["--live", "--confirm-sha256", "0" * 64], "confirmation_sha256_mismatch"),
        (["--live", "--confirm-sha256", "MATCH", "--db-copy", "copy.db"], "not allowed with argument"),
    ],
)
def test_live_refusals_do_not_open_a_database(export, monkeypatch, capsys, flags, error):
    from brainlayer import linkedin_import

    def forbidden(*args, **kwargs):
        pytest.fail("refused live command must not open a database")

    monkeypatch.setattr(linkedin_import, "open_writer_store", forbidden)
    flags = [hashlib.sha256(export.read_bytes()).hexdigest() if f == "MATCH" else f for f in flags]
    monkeypatch.setattr("sys.argv", ["linkedin_import", str(export), *flags])
    try:
        result = linkedin_import.main()
    except SystemExit as exc:
        result = exc.code
    assert result != 0
    output = capsys.readouterr()
    assert error in output.out + output.err


def test_email_identity_preserves_accents(tmp_path):
    store = VectorStore(tmp_path / "copy.db")
    try:
        store.upsert_entity("existing", "person", "Existing", metadata={"email": "jose@example.invalid"})
        export = tmp_path / "connections.json"
        export.write_text(json.dumps([dict(full_name="Other", email=" JOSÉ@example.invalid ")]))
        result = import_connections(store, export)
        assert result["persons"]["new"] == 1
        aliases = list(store.conn.execute("SELECT alias FROM kg_entity_aliases WHERE alias_type='email'"))
        assert aliases == [("josé@example.invalid",)]
    finally:
        store.close()


def test_identity_match_with_another_name_owner_is_skipped(tmp_path):
    store = VectorStore(tmp_path / "copy.db")
    try:
        store.upsert_entity("a", "person", "Original", metadata={"linkedin_url": "https://linkedin.com/in/safe"})
        store.upsert_entity("b", "person", "Conflicting Name", metadata={"preserved": True})
        before = {eid: store.get_entity(eid) for eid in ("a", "b")}
        export = tmp_path / "connections.json"
        export.write_text(json.dumps([dict(full_name="Conflicting Name", linkedin_url="https://linkedin.com/in/safe")]))
        result = import_connections(store, export)
        assert result["skipped"] == {"ambiguous_identity": 1}
        assert {eid: store.get_entity(eid) for eid in ("a", "b")} == before
    finally:
        store.close()


def test_invalid_existing_url_alias_is_counted_and_ignored(tmp_path, export):
    store = VectorStore(tmp_path / "copy.db")
    try:
        store.upsert_entity("existing", "person", "Existing")
        store.add_entity_alias("https://invalid.example/profile", "existing", alias_type="linkedin_url")
        result = import_connections(store, export)
        assert result["invalid_existing_alias"] == 1
        assert result["persons"]["new"] > 0
    finally:
        store.close()


def test_regional_url_alias_matches_canonical_identity(tmp_path):
    store = VectorStore(tmp_path / "copy.db")
    try:
        store.upsert_entity("existing", "person", "Existing")
        store.add_entity_alias("https://il.linkedin.com/in/regional", "existing", alias_type="linkedin_url")
        export = tmp_path / "connections.json"
        export.write_text(
            json.dumps([dict(full_name="New Alias", linkedin_url="https://www.linkedin.com/in/regional/")])
        )
        result = import_connections(store, export)
        assert result["persons"]["merged_by"]["linkedin_url"] == 1
        assert result["invalid_existing_alias"] == 0
    finally:
        store.close()


def test_sha_mismatch_is_typed_and_contains_only_static_reason(export, monkeypatch, capsys):
    from brainlayer import linkedin_import

    monkeypatch.setattr("sys.argv", ["linkedin_import", str(export), "--live", "--confirm-sha256", "0" * 64])
    assert linkedin_import.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "error_type": "ImportRefusal",
        "reason": "confirmation_sha256_mismatch",
    }


def test_unexpected_exception_text_is_never_printed(tmp_path, export, monkeypatch, capsys):
    from brainlayer import linkedin_import

    copy = tmp_path / "copy.db"
    copy.touch()

    def fail(*args, **kwargs):
        raise ValueError("PRIVATE_SYNTHETIC_MARKER")

    monkeypatch.setattr(linkedin_import, "open_writer_store", fail)
    monkeypatch.setattr("sys.argv", ["linkedin_import", str(export), "--db-copy", str(copy)])
    assert linkedin_import.main() == 1
    assert json.loads(capsys.readouterr().out) == {"error_type": "ValueError", "reason": "import_failed"}


def test_bare_name_resolution_does_not_change(tmp_path):
    store = VectorStore(tmp_path / "copy.db")
    try:
        before = store.get_entity_by_alias("David")
        export = tmp_path / "connections.json"
        export.write_text(
            json.dumps([dict(full_name=f"David {last}", first_name="David", last_name=last) for last in ("A", "B")])
        )
        import_connections(store, export)
        assert store.get_entity_by_alias("David") == before
        # Exact alias resolution must stay unchanged; generic FTS fuzzy search is separate.
        for eid, metadata in store.conn.execute("SELECT id, metadata FROM kg_entities"):
            assert json.loads(metadata)["first_name"] == "David"
            assert {a["alias_type"] for a in store.get_entity_aliases(eid)} == {"name"}
    finally:
        store.close()


def test_full_name_alias_without_identifiers_merges_on_rerun(tmp_path):
    store = VectorStore(tmp_path / "copy.db")
    try:
        store.upsert_entity("existing", "person", "Jon A. Smith")
        # Full-name alias persisted by a prior import whose canonical name differed.
        store.add_entity_alias("Jon Smith", "existing", alias_type="name")
        export = tmp_path / "connections.json"
        export.write_text(json.dumps([dict(full_name="Jon Smith")]))
        for _ in range(2):
            result = import_connections(store, export)
            assert result["persons"]["new"] == 0
            assert result["persons"]["merged_by"]["name"] == 1
        assert list(store.conn.execute("SELECT id FROM kg_entities")) == [("existing",)]
    finally:
        store.close()


def test_import_created_full_name_alias_survives_missing_identity_keys(tmp_path):
    store = VectorStore(tmp_path / "copy.db")
    try:
        store.upsert_entity(
            "existing", "person", "Jon A. Smith", metadata={"linkedin_url": "https://linkedin.com/in/name-alias"}
        )
        export = tmp_path / "connections.json"
        export.write_text(json.dumps([dict(full_name="Jon Smith", linkedin_url="https://linkedin.com/in/name-alias")]))
        assert import_connections(store, export)["persons"]["merged_by"]["linkedin_url"] == 1
        export.write_text(json.dumps([dict(full_name="Jon Smith")]))
        for _ in range(2):
            result = import_connections(store, export)
            assert result["persons"]["new"] == 0
            assert result["persons"]["merged_by"]["name"] == 1
        assert list(store.conn.execute("SELECT id FROM kg_entities")) == [("existing",)]
    finally:
        store.close()


def test_live_mode_uses_the_resolved_db_path(export, monkeypatch, tmp_path, capsys):
    """--live writes the database the services use (paths.resolve_db_path), not a hardcoded path."""
    from brainlayer import linkedin_import

    resolved = tmp_path / "configured" / "brainlayer.db"
    opened = []

    class Stop(Exception):
        pass

    def fake_open(path, *args, **kwargs):
        opened.append(Path(path))
        raise Stop

    monkeypatch.setattr(linkedin_import, "resolve_db_path", lambda: resolved)
    monkeypatch.setattr(linkedin_import, "open_writer_store", fake_open)
    digest = hashlib.sha256(export.read_bytes()).hexdigest()
    monkeypatch.setattr("sys.argv", ["linkedin_import", str(export), "--live", "--confirm-sha256", digest])
    assert linkedin_import.main() != 0
    assert opened == [resolved]


def test_url_and_email_owned_by_different_people_is_skipped(tmp_path):
    """A row whose URL belongs to one person and email to another must not join them."""
    store = VectorStore(tmp_path / "copy.db")
    try:
        store.upsert_entity("a", "person", "Person A", metadata={"linkedin_url": "https://linkedin.com/in/person-a"})
        store.upsert_entity("b", "person", "Person B", metadata={"email": "b@example.invalid"})
        before = {eid: store.get_entity(eid) for eid in ("a", "b")}
        export = tmp_path / "connections.json"
        export.write_text(
            json.dumps(
                [
                    dict(
                        full_name="Someone",
                        linkedin_url="https://www.linkedin.com/in/person-a",
                        email="b@example.invalid",
                    )
                ]
            )
        )
        result = import_connections(store, export)
        assert result["skipped"] == {"ambiguous_identity": 1}
        assert result["persons"]["new"] == 0
        assert {eid: store.get_entity(eid) for eid in ("a", "b")} == before
        owners = list(store.conn.execute("SELECT entity_id FROM kg_entity_aliases WHERE alias = 'b@example.invalid'"))
        assert owners in ([], [("b",)])
    finally:
        store.close()


@pytest.mark.parametrize("alias_only", [False, True])
def test_email_match_never_takes_a_new_url_from_a_person_with_another_url(tmp_path, alias_only):
    """Cross-vendor review repro (Codex on #1572): an unowned URL plus an email owned by B must not rewrite B."""
    store = VectorStore(tmp_path / "copy.db")
    try:
        old_url = "https://linkedin.com/in/person-b"
        email = "b@example.invalid"
        metadata = {"email": email, **({} if alias_only else {"linkedin_url": old_url})}
        store.upsert_entity("b", "person", "Person B", metadata=metadata)
        store.add_entity_alias(old_url, "b", alias_type="linkedin_url")
        before, before_aliases = store.get_entity("b"), store.get_entity_aliases("b")
        export = tmp_path / "connections.json"
        export.write_text(
            json.dumps([dict(full_name="New Person A", linkedin_url="https://linkedin.com/in/person-a", email=email)])
        )
        result = import_connections(store, export)
        assert result["skipped"] == {"ambiguous_identity": 1}
        assert store.get_entity("b") == before
        assert store.get_entity_aliases("b") == before_aliases
    finally:
        store.close()
