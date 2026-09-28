import json
import re

from brainlayer.lexical_defense import DATA_PATH, load_lexical_defense_dictionary


def test_dictionary_file_exists():
    assert DATA_PATH.exists()


def test_lookup_matches_split_forms_and_aliases():
    dictionary = load_lexical_defense_dictionary()

    assert dictionary.lookup("brain layer").canonical == "BrainLayer"
    assert dictionary.lookup("repo golden").canonical == "repoGolem"
    assert dictionary.lookup("brain layer").category == "domain_entity"


def test_hebrew_product_entries_are_present():
    dictionary = load_lexical_defense_dictionary()
    assert any(entry.script == "hebrew" and entry.category == "domain_entity" for entry in dictionary.entries)


def test_shipped_dictionary_contains_no_personal_entries():
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    # Synthetic examples document the prohibited shape without putting a real name in a fixture.
    synthetic_surnames = ("Exampleperson", "לדוגמה")
    for entry in payload["entries"]:
        assert entry["category"] not in {"english_name", "hebrew_name"}
        surfaces = [entry["canonical"], *entry.get("aliases", []), *entry.get("split_forms", [])]
        assert not any(re.search(r"[^\s@]+@[^\s@]+\.[^\s@]+", surface) for surface in surfaces)
        assert not any(
            surname.casefold() in surface.casefold() for surname in synthetic_surnames for surface in surfaces
        )


def test_swift_override_patterns_are_priority_sorted():
    dictionary = load_lexical_defense_dictionary()

    patterns = dictionary.swift_override_patterns()

    assert patterns[0]["priority"] >= patterns[-1]["priority"]
    assert {"match": "brain layer", "replacement": "BrainLayer", "priority": 100} in patterns
    assert {"match": "voice layer", "replacement": "VoiceLayer", "priority": 100} in patterns


def test_voicelayer_snapshot_contains_prompt_terms_and_aliases():
    dictionary = load_lexical_defense_dictionary()

    snapshot = dictionary.voicelayer_snapshot()

    assert "BrainLayer" in snapshot["prompt_terms"]
    assert {"from": "brain layer", "to": "BrainLayer"} in snapshot["aliases"]


def test_whisper_entity_gbnf_contains_protected_entities():
    dictionary = load_lexical_defense_dictionary()

    grammar = dictionary.whisper_entity_gbnf()

    assert "root ::= protected_entity" in grammar
    assert '"BrainLayer"' in grammar
    assert '"BrainLayer"' in grammar
