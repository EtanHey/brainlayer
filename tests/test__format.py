"""format_store_result must render every brain_store outcome unambiguously.

Etan, 2026-08-19: "make sure the tool responses explain the agents exactly what's
going on so they don't just duplicate -- if something is deferred, it means it
will be stored in the future."

Every rendering states three things, because agents that could not tell a fresh
write from a suppressed duplicate kept re-storing the same memory:
  (a) the OUTCOME WORD, so an agent can branch without parsing prose;
  (b) the CANONICAL chunk_id it resolved to -- or an explicit "no chunk_id" for
      the outcomes that store nothing;
  (c) the INSTRUCTION -- above all, whether re-storing is right or wrong.

The end-to-end handler and store_memory contracts live in
tests/test_store_response_clarity.py.
"""

import pytest

from brainlayer.mcp._format import STORE_OUTCOMES, format_store_result

# --- Formatter: one test per outcome ---


class TestFormatStoredNew:
    def test_states_outcome_id_and_instruction(self):
        text = format_store_result("manual-abc123", outcome="stored")
        assert "STORED" in text
        assert "manual-abc123" in text
        # A fresh insert must not read like a duplicate suppression.
        assert "DUPLICATE" not in text
        assert "MERGED" not in text

    def test_names_the_id_as_the_canonical_one(self):
        text = format_store_result("manual-abc123", outcome="stored")
        assert "manual-abc123" in text
        assert "new" in text.lower()


class TestFormatDuplicate:
    def test_states_outcome_canonical_id_and_do_not_restore(self):
        text = format_store_result("manual-canonical1", outcome="duplicate")
        assert "DUPLICATE" in text
        assert "manual-canonical1" in text
        assert "do not re-store" in text.lower()

    def test_does_not_claim_a_new_chunk_was_written(self):
        text = format_store_result("manual-canonical1", outcome="duplicate")
        # "already stored" is the honest phrasing; a bare "Stored ->" is what
        # made agents believe they had written a second copy.
        assert "already stored" in text.lower()


class TestFormatMerged:
    def test_states_outcome_canonical_id_and_do_not_restore(self):
        text = format_store_result("manual-canonical2", outcome="merged")
        assert "MERGED" in text
        assert "manual-canonical2" in text
        assert "do not re-store" in text.lower()

    def test_says_the_content_was_folded_into_the_existing_chunk(self):
        text = format_store_result("manual-canonical2", outcome="merged")
        assert "merged into" in text.lower()


class TestFormatDeferred:
    def test_states_outcome_promised_id_and_will_persist(self):
        text = format_store_result("manual-def456", queued=True)
        # The prefix is "STORED (deferred)", not a bare "DEFERRED:" -- Etan
        # retired that on 2026-08-09 because it read as failure and agents
        # re-stored on it. Machines branch on the structured status "DEFERRED".
        assert "STORED (deferred)" in text
        assert "manual-def456" in text
        # The whole point Etan named on 2026-08-19: deferred means it WILL be stored.
        assert "will be stored" in text.lower()

    def test_keeps_the_success_reading_prefix_retired_in_2026_08_09(self):
        text = format_store_result("manual-def456", queued=True)
        assert "DEFERRED:" not in text

    def test_forbids_retry_and_fallback_copies(self):
        text = format_store_result("manual-def456", queued=True)
        lowered = text.lower()
        assert "do not retry" in lowered
        assert "fallback" in lowered

    def test_reports_the_reason_it_was_deferred(self):
        text = format_store_result("manual-def456", queued=True, queued_reason="SCHEMA_FINGERPRINT_MISMATCH")
        assert "SCHEMA_FINGERPRINT_MISMATCH" in text


class TestFormatRejected:
    def test_states_outcome_absence_of_id_and_do_not_retry(self):
        text = format_store_result(None, outcome="rejected", reason="system prompt content is not stored")
        assert "REJECTED" in text
        assert "no chunk_id" in text.lower()
        assert "system prompt content is not stored" in text
        assert "do not retry" in text.lower()

    def test_never_implies_a_deferred_write(self):
        text = format_store_result(None, outcome="rejected", reason="whatever")
        assert "DEFERRED" not in text
        assert "will be stored" not in text.lower()


class TestFormatError:
    def test_states_outcome_absence_of_id_and_retry_guidance(self):
        text = format_store_result(None, outcome="error", reason="disk I/O error")
        assert "ERROR" in text
        assert "no chunk_id" in text.lower()
        assert "disk I/O error" in text
        assert "not stored" in text.lower()

    def test_never_implies_a_deferred_write(self):
        # An error is the one outcome where the memory is genuinely lost; if it
        # read like DEFERRED the agent would drop content on the floor.
        text = format_store_result(None, outcome="error", reason="boom")
        assert "DEFERRED" not in text
        assert "will be stored" not in text.lower()


# The token each outcome must show in its rendered text. DEFERRED is the one
# that does not simply upper-case: see TestFormatDeferred for why.
OUTCOME_TOKENS = {
    "stored": "STORED",
    "duplicate": "DUPLICATE",
    "merged": "MERGED",
    "deferred": "STORED (deferred)",
    "rejected": "REJECTED",
    "error": "ERROR",
}


class TestFormatOutcomeVocabulary:
    def test_every_declared_outcome_has_a_token(self):
        assert set(OUTCOME_TOKENS) == set(STORE_OUTCOMES)

    def test_every_declared_outcome_renders_its_token(self):
        for outcome, token in OUTCOME_TOKENS.items():
            text = format_store_result("manual-x", outcome=outcome, reason="r")
            assert token in text, outcome

    def test_unknown_outcome_is_rejected_loudly(self):
        # Silently falling back to "Stored" is how ambiguity got in.
        with pytest.raises(ValueError):
            format_store_result("manual-x", outcome="probably-fine")
