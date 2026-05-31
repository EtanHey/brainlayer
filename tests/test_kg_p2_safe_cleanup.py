import importlib.util
from pathlib import Path


def _load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "kg_p2_safe_cleanup.py"
    spec = importlib.util.spec_from_file_location("kg_p2_safe_cleanup_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(entity_id, name, entity_type, *, chunks=1, rels=0):
    return {
        "id": entity_id,
        "name": name,
        "entity_type": entity_type,
        "status": "active",
        "metadata": "{}",
        "rels": rels,
        "chunks": chunks,
        "canonical_name": "",
    }


def test_voice_sources_do_not_match_arbitrary_voicelayerclaude_substrings():
    cleanup = _load_script()
    rows = [
        _row(cleanup.VOICE_TARGET_ID, "voiceClaude", "agent", chunks=100),
        _row("voice-exact", "voicelayerClaude", "tool"),
        _row("voice-worker", "voicelayerClaude (worker) <[EMAIL_1]>", "person"),
        _row("voice-overmatch", "voicelayerClaude archive export", "project"),
    ]

    plan = cleanup.build_plan(rows)

    assert {row["id"] for row in plan["voice_sources"]} == {"voice-exact", "voice-worker"}


def test_person_placeholder_family_sources_include_mistyped_placeholder_rows():
    cleanup = _load_script()
    rows = [
        _row("person-canonical", "PERSON_deadbeef", "person", chunks=10),
        _row("person-duplicate", "PERSON_deadbeef", "person", chunks=2),
        _row("mistyped-tool-placeholder", "PERSON_deadbeef", "tool", chunks=1),
        _row("mistyped-project-placeholder", "[PERSON_deadbeef]", "project", chunks=1),
    ]

    plan = cleanup.build_plan(rows)

    assert len(plan["person_groups"]) == 1
    group = plan["person_groups"][0]
    assert group["target"]["id"] == "person-canonical"
    assert {row["id"] for row in group["sources"]} == {
        "person-duplicate",
        "mistyped-tool-placeholder",
        "mistyped-project-placeholder",
    }
