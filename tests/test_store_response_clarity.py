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

from unittest.mock import patch

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

    def test_reserved_chunk_id_rearriving_with_changed_content_reports_merged(self, tmp_path):
        """MERGED must be asserted somewhere that can actually fail.

        The near-duplicate test above accepts "stored" as an alternative, because
        SimHash banding is threshold-based -- so breaking every merge branch in
        store.py leaves it green (PR725 review, item 6). The reserved-chunk_id
        re-arrival is deterministic: the row already exists, the content changed,
        so it was folded in and nothing new was written.
        """
        store = _open_store(tmp_path)
        reserved = "manual-reserved-merge-1"
        first = _store_memory(
            store,
            "the deployment runbook says to stop enrichment workers first",
            chunk_id=reserved,
        )
        assert first["outcome"] == "stored"
        assert first["id"] == reserved

        second = _store_memory(
            store,
            "the deployment runbook says to stop enrichment workers first, then "
            "checkpoint the write ahead log before batching any deletes",
            chunk_id=reserved,
        )
        assert second["outcome"] == "merged"
        # MERGED resolves to the PRE-EXISTING chunk and writes no second row --
        # that is the whole reason an agent must not re-store on it.
        assert second["id"] == reserved
        assert _row_count(store) == 1

    def test_outcome_is_always_one_of_the_declared_words(self, tmp_path):
        store = _open_store(tmp_path)
        result = _store_memory(store, "some memory worth keeping around for later")
        assert result["outcome"] in STORE_OUTCOMES


def _open_store(tmp_path):
    from brainlayer.vector_store import VectorStore

    return VectorStore(db_path=tmp_path / "clarity.db")


def _store_memory(store, content, **kwargs):
    from brainlayer.store import store_memory

    return store_memory(
        store=store,
        embed_fn=None,
        content=content,
        memory_type="note",
        project="store-clarity-test",
        **kwargs,
    )


def _row_count(store) -> int:
    return int(store.conn.cursor().execute("SELECT COUNT(*) FROM chunks").fetchone()[0])


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


# --- REJECTED is a promise that nothing is on disk ---
#
# PR725 review, MUST-FIX 1: `_store`'s `except ValueError -> REJECTED` wrapped the
# whole body, including the commit. The vocabulary guards this PR added
# (`_store_receipt`, `format_store_result`) raise ValueError AFTER the row is
# durable, so a committed write could answer "nothing was stored ... Do NOT
# retry the same content" -- the exact re-store loop this work exists to close.


class TestRejectedNeverFollowsACommittedWrite:
    @pytest.mark.asyncio
    async def test_unknown_outcome_after_commit_is_never_reported_as_rejected(self, tmp_path, monkeypatch):
        """The reviewer's repro: store_memory returns an outcome word we don't know.

        Whatever the handler answers, it may not be REJECTED -- the row is on disk.
        A loud failure is acceptable here; a confident lie is not.
        """
        import brainlayer.store as store_module
        from brainlayer.mcp.store_handler import _store

        store = _open_store(tmp_path)
        real_store_memory = store_module.store_memory

        def outcome_from_the_future(**kwargs):
            result = real_store_memory(**kwargs)
            # A word `_format.STORE_OUTCOMES` has not learned yet.
            result["outcome"] = "aggregated"
            return result

        monkeypatch.setenv("BRAINLAYER_STORE_BUSY_BUDGET_MS", "400")
        monkeypatch.delenv("BRAINLAYER_ARBITRATED", raising=False)
        monkeypatch.delenv("BRAINLAYER_INTERACTIVE_STORE_QUEUE", raising=False)

        with (
            patch("brainlayer.mcp.store_handler._get_vector_store", return_value=store),
            patch("brainlayer.mcp.store_handler._interactive_queue_reason", return_value=None),
            patch("brainlayer.queue_io.get_queue_dir", return_value=tmp_path / "queue"),
            patch("brainlayer.store.store_memory", side_effect=outcome_from_the_future),
            patch(
                "brainlayer.mcp.store_handler._get_embedding_model",
                side_effect=RuntimeError("no embedding model in this test"),
            ),
        ):
            response = None
            try:
                response = await _store(
                    content="a memory that really does get written to disk before the blowup",
                    memory_type="note",
                    project="store-clarity-test",
                )
            except ValueError:
                # Failing loudly on an unknown outcome word is a legitimate fix.
                response = None

        assert _row_count(store) == 1, "the write must have really committed for this to be the bug"
        if response is not None:
            assert "REJECTED" not in _response_text(response)

    @pytest.mark.asyncio
    async def test_rejected_gates_write_nothing(self, tmp_path, monkeypatch):
        """The invariant itself: REJECTED implies ROWS WRITTEN == 0."""
        from brainlayer.mcp.store_handler import _store

        store = _open_store(tmp_path)
        monkeypatch.delenv("BRAINLAYER_ARBITRATED", raising=False)
        monkeypatch.delenv("BRAINLAYER_INTERACTIVE_STORE_QUEUE", raising=False)

        with (
            patch("brainlayer.mcp.store_handler._get_vector_store", return_value=store),
            patch("brainlayer.mcp.store_handler._interactive_queue_reason", return_value=None),
            patch("brainlayer.queue_io.get_queue_dir", return_value=tmp_path / "queue"),
        ):
            response = await _store(
                content="this content is fine but the memory type is not",
                memory_type="not_a_memory_type",
                project="store-clarity-test",
            )

        assert "REJECTED" in _response_text(response)
        assert _row_count(store) == 0

    def test_validation_gates_raise_a_dedicated_rejection_error(self):
        """A distinct exception type is what lets the handler catch ONLY the
        pre-write gates, instead of every ValueError the write path can raise."""
        from brainlayer.mcp.store_handler import StoreRejected, _validate_store_request

        with pytest.raises(StoreRejected):
            _validate_store_request("   ", "note")
        with pytest.raises(StoreRejected):
            _validate_store_request("valid content here", "not_a_memory_type")


def _response_text(response) -> str:
    """`_store` returns either a CallToolResult (error) or (texts, structured)."""
    content = getattr(response, "content", None)
    if content is None:
        content = response[0]
    return "\n".join(item.text for item in content)
