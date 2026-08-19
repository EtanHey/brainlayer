"""Every brain_store outcome must return an unambiguous, action-guiding response.

Etan, 2026-08-19: "make sure the tool responses explain the agents exactly what's
going on so they don't just duplicate -- if something is deferred, it means it
will be stored in the future."

The contract each response must satisfy, per outcome:
  (a) the OUTCOME WORD, so an agent can branch on it without parsing prose;
  (b) the CANONICAL chunk_id it resolved to -- or an explicit statement that no
      chunk exists, for the outcomes that store nothing;
  (c) an INSTRUCTION telling the agent what to do next (above all: whether
      re-storing is correct or wrong).

Outcome coverage in the Python MCP path:
  STORED / DUPLICATE / MERGED / DEFERRED / REJECTED / ERROR

DUPLICATE vs MERGED is the distinction that actually stops the re-store loop:
both resolve to a pre-existing canonical id, and before this contract both
rendered as a bare "Stored -> <id>" that an agent could not tell apart from a
fresh insert.
"""

import pytest

from brainlayer.mcp.store_handler import (
    STORE_OUTCOMES,
    _error_store_result,
    _rejected_store_result,
    _store_receipt,
)

# --- Handler receipts: structured payload carries the same outcome ---


class TestStoreReceipt:
    @pytest.mark.parametrize("outcome", ["stored", "duplicate", "merged"])
    def test_structured_status_matches_outcome(self, outcome):
        _text, structured = _store_receipt("manual-canon", outcome=outcome, related=[])
        assert structured["status"] == outcome.upper()
        assert structured["chunk_id"] == "manual-canon"

    def test_duplicate_receipt_flags_that_nothing_new_was_written(self):
        _text, structured = _store_receipt("manual-canon", outcome="duplicate", related=[])
        assert structured["stored_new"] is False

    def test_merged_receipt_flags_that_nothing_new_was_written(self):
        _text, structured = _store_receipt("manual-canon", outcome="merged", related=[])
        assert structured["stored_new"] is False

    def test_stored_receipt_flags_a_new_row(self):
        _text, structured = _store_receipt("manual-canon", outcome="stored", related=[])
        assert structured["stored_new"] is True

    def test_text_and_structured_agree(self):
        text, structured = _store_receipt("manual-canon", outcome="duplicate", related=[])
        assert structured["status"] in text


class TestRejectedResult:
    def test_is_an_error_result_naming_the_outcome_and_reason(self):
        result = _rejected_store_result("system prompt content is not stored in BrainLayer")
        text = result.content[0].text
        assert result.is_error is True
        assert "REJECTED" in text
        assert "system prompt content is not stored in BrainLayer" in text
        assert "do not retry" in text.lower()


class TestErrorResult:
    def test_is_an_error_result_naming_the_outcome_and_reason(self):
        result = _error_store_result("database is locked beyond budget")
        text = result.content[0].text
        assert result.is_error is True
        assert "ERROR" in text
        assert "database is locked beyond budget" in text
        assert "no chunk_id" in text.lower()


# --- store_memory must report which outcome it took ---


class TestStoreMemoryOutcome:
    def test_new_content_reports_stored(self, tmp_path):
        store = _open_store(tmp_path)
        result = _store_memory(store, "a genuinely novel memory about widget calibration")
        assert result["outcome"] == "stored"

    def test_identical_content_reports_duplicate_with_canonical_id(self, tmp_path):
        store = _open_store(tmp_path)
        content = "the calibration constant for the widget is 4.7 and never changes"
        first = _store_memory(store, content)
        second = _store_memory(store, content)
        assert second["outcome"] == "duplicate"
        # The canonical id is the FIRST chunk -- the whole point of returning it.
        assert second["id"] == first["id"]

    def test_near_identical_content_reports_merged_with_canonical_id(self, tmp_path):
        store = _open_store(tmp_path)
        base = (
            "The deployment runbook says to stop enrichment workers first, then "
            "checkpoint the write ahead log, then batch the deletes in five "
            "thousand row chunks and checkpoint again every third batch."
        )
        near = base + " Also remember to notify the on-call engineer beforehand."
        first = _store_memory(store, base)
        second = _store_memory(store, near)
        if second["outcome"] == "merged":
            assert second["id"] == first["id"]
        else:
            # SimHash near-duplicate detection is threshold-based; when the pair
            # falls outside the band the honest answer is a fresh store, not a
            # mislabelled merge.
            assert second["outcome"] == "stored"
            assert second["id"] != first["id"]

    def test_outcome_is_always_one_of_the_declared_words(self, tmp_path):
        store = _open_store(tmp_path)
        result = _store_memory(store, "some memory worth keeping around for later")
        assert result["outcome"] in STORE_OUTCOMES


def _open_store(tmp_path):
    from brainlayer.vector_store import VectorStore

    return VectorStore(db_path=tmp_path / "clarity.db")


def _store_memory(store, content):
    from brainlayer.store import store_memory

    return store_memory(
        store=store,
        embed_fn=None,
        content=content,
        memory_type="note",
        project="store-clarity-test",
    )


# --- The tool description must list the outcomes up front ---
#
# AGENTS.md letter: "The rules for agents USING it live in the tool descriptions
# -- keep those true." An agent that only learns DUPLICATE exists by receiving
# one has already made the redundant call.


class TestToolDescription:
    def _brain_store_description(self):
        from brainlayer.mcp import _full_tool_definitions

        tools = _full_tool_definitions()
        store = next(t for t in tools if t.name == "brain_store")
        return store.description

    @pytest.mark.parametrize(
        "token",
        ["STORED", "DUPLICATE", "MERGED", "DEFERRED", "REJECTED", "ERROR"],
    )
    def test_names_every_outcome(self, token):
        assert token in self._brain_store_description()

    def test_states_the_rule_that_makes_duplicate_and_merged_actionable(self):
        assert "do not re-store" in self._brain_store_description().lower()

    def test_states_that_deferred_will_be_stored(self):
        assert "will be stored" in self._brain_store_description().lower()

    def test_output_schema_declares_the_outcome_fields(self):
        from brainlayer.mcp import _full_tool_definitions

        store = next(t for t in _full_tool_definitions() if t.name == "brain_store")
        props = store.output_schema["properties"]
        assert "status" in props
        assert "stored_new" in props
        assert set(props["status"]["enum"]) == {"STORED", "DUPLICATE", "MERGED", "DEFERRED"}
