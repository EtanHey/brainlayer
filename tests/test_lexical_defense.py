from brainlayer.lexical_defense import DATA_PATH, load_lexical_defense_dictionary


def test_dictionary_file_exists():
    assert DATA_PATH.exists()


def test_lookup_matches_split_forms_and_aliases():
    dictionary = load_lexical_defense_dictionary()

    assert dictionary.lookup("brain layer").canonical == "BrainLayer"
    assert dictionary.lookup("repo golden").canonical == "repoGolem"
    assert dictionary.lookup("etanheyman").canonical == "Etan Heyman"


def test_hebrew_entries_are_present():
    dictionary = load_lexical_defense_dictionary()

    entry = dictionary.lookup("איתן היימן")

    assert entry is not None
    assert entry.category == "hebrew_name"
    assert entry.protect_from_split is True


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
    assert '"איתן היימן"' in grammar
