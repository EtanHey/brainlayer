"""Tests for PII sanitization pipeline."""

import json

import pytest


def _spacy_available() -> bool:
    try:
        import spacy

        spacy.load("en_core_web_sm")
        return True
    except Exception:
        return False


from brainlayer.pipeline.enrichment import build_external_prompt
from brainlayer.pipeline.sanitize import (
    SanitizeConfig,
    Sanitizer,
    SanitizeResult,
    _strip_nikud,
)


@pytest.fixture
def default_sanitizer():
    """Sanitizer with test config for PII sanitization."""
    return Sanitizer(
        SanitizeConfig(
            owner_names=("Jane Developer", "jane", "JaneDev", "janedev"),
            owner_emails=("jane@example.com",),
            owner_paths=("/Users/testuser",),
        )
    )


@pytest.fixture
def full_sanitizer():
    """Sanitizer with known names and all features."""
    return Sanitizer(
        SanitizeConfig(
            owner_names=("Jane Developer", "jane", "JaneDev", "janedev"),
            owner_emails=("jane@example.com",),
            owner_paths=("/Users/testuser",),
            known_names=frozenset({"David Cohen", "Sarah Miller", "דוד כהן", "שרה לוי"}),
        )
    )


class TestRegexSanitization:
    """Test regex-based PII detection (Layer 1)."""

    def test_owner_name_replaced(self, default_sanitizer):
        result = default_sanitizer.sanitize("Jane Developer wrote this code")
        assert "[OWNER]" in result.sanitized
        assert "Jane Developer" not in result.sanitized
        assert result.pii_detected

    def test_owner_name_case_insensitive(self, default_sanitizer):
        result = default_sanitizer.sanitize("JANE DEVELOPER was here")
        assert "[OWNER]" in result.sanitized

    def test_owner_short_name(self, default_sanitizer):
        result = default_sanitizer.sanitize("jane said hello")
        assert "[OWNER]" in result.sanitized
        assert "jane" not in result.sanitized

    def test_email_replaced(self, default_sanitizer):
        result = default_sanitizer.sanitize("Send to jane@example.com please")
        assert "[OWNER_EMAIL]" in result.sanitized
        assert "jane@example.com" not in result.sanitized

    def test_email_before_owner_name(self, default_sanitizer):
        """Owner email should be caught as whole unit, not partially by owner name."""
        result = default_sanitizer.sanitize("Contact jane@example.com")
        assert "[OWNER_EMAIL]" in result.sanitized
        # Should NOT have "[OWNER]@example.com"
        assert "@example.com" not in result.sanitized

    def test_general_email_replaced(self, default_sanitizer):
        result = default_sanitizer.sanitize("Email someone@other-domain.com for info")
        assert "[EMAIL_1]" in result.sanitized
        assert "someone@other-domain.com" not in result.sanitized

    def test_file_path_replaced(self, default_sanitizer):
        result = default_sanitizer.sanitize("Read /Users/testuser/Projects/my-project/file.ts")
        assert "/Users/[OWNER]/" in result.sanitized
        assert "/Users/testuser" not in result.sanitized

    def test_file_path_before_owner_name(self, default_sanitizer):
        """File path should be caught whole, not partially by owner name."""
        result = default_sanitizer.sanitize("/Users/testuser/code")
        assert "/Users/[OWNER]/code" in result.sanitized

    def test_jwt_replaced(self, default_sanitizer):
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        result = default_sanitizer.sanitize(f"Token: {jwt}")
        assert "[JWT_TOKEN]" in result.sanitized
        assert "eyJ" not in result.sanitized

    def test_op_ref_replaced(self, default_sanitizer):
        result = default_sanitizer.sanitize("Use op://development/my-key/credential")
        assert "[OP_REF]" in result.sanitized
        assert "op://" not in result.sanitized

    def test_ip_replaced(self, default_sanitizer):
        result = default_sanitizer.sanitize("Server at 192.168.1.42")
        assert "[IP_ADDR]" in result.sanitized
        assert "192.168.1.42" not in result.sanitized

    def test_localhost_ip_preserved(self, default_sanitizer):
        """127.x.x.x IPs are not PII."""
        result = default_sanitizer.sanitize("Running on 127.0.0.1:8080")
        assert "127.0.0.1" in result.sanitized

    def test_phone_replaced(self, default_sanitizer):
        result = default_sanitizer.sanitize("Call +972 54 7589755")
        assert "[PHONE]" in result.sanitized
        assert "+972" not in result.sanitized

    def test_github_username_replaced(self, default_sanitizer):
        result = default_sanitizer.sanitize("See @JaneDev on github.com/JaneDev/repo")
        assert "[OWNER_GITHUB]" in result.sanitized
        assert "JaneDev" not in result.sanitized


class TestNameDictionary:
    """Test known-names dictionary matching (Layer 2)."""

    def test_english_name_replaced(self, full_sanitizer):
        result = full_sanitizer.sanitize("David Cohen joined the meeting")
        assert "David Cohen" not in result.sanitized
        assert "[PERSON_" in result.sanitized

    def test_hebrew_name_replaced(self, full_sanitizer):
        result = full_sanitizer.sanitize("דוד כהן כתב הודעה")
        assert "דוד כהן" not in result.sanitized
        assert "[PERSON_" in result.sanitized

    def test_partial_name_not_replaced(self, full_sanitizer):
        """'David' alone should NOT match 'David Cohen' — word boundary."""
        result = full_sanitizer.sanitize("David went to danger zone")
        # "David" alone is not in the dictionary, only "David Cohen"
        # spaCy might catch it though — check name_dict specifically
        dict_replacements = [r for r in result.replacements if r.source == "name_dict"]
        # No name_dict replacement should match just "David" alone
        assert not any(r.original.lower() == "david" for r in dict_replacements)

    def test_consistent_pseudonyms(self, full_sanitizer):
        """Same name should always map to same placeholder."""
        r1 = full_sanitizer.sanitize("David Cohen said hello")
        r2 = full_sanitizer.sanitize("David Cohen said goodbye")
        pseudo1 = [r.placeholder for r in r1.replacements if r.original == "David Cohen"]
        pseudo2 = [r.placeholder for r in r2.replacements if r.original == "David Cohen"]
        assert pseudo1 == pseudo2

    def test_case_insensitive(self, full_sanitizer):
        result = full_sanitizer.sanitize("SARAH MILLER arrived")
        assert "SARAH MILLER" not in result.sanitized
        assert "[PERSON_" in result.sanitized

    def test_hebrew_nikud_stripped(self):
        """Names with nikud should still match."""
        assert _strip_nikud("דָּוִד") == "דוד"
        assert _strip_nikud("hello") == "hello"

    def test_hebrew_nikud_position_mapping(self, full_sanitizer):
        """Sanitizer should correctly replace names in text containing nikud."""
        # "דָּוִד כהן" has nikud on the dalet and vav — more chars than "דוד כהן"
        text_with_nikud = "שָׁלוֹם דָּוִד כהן מה שלומך"
        result = full_sanitizer.sanitize(text_with_nikud)
        # The name should be replaced, not left in or garbled
        assert "דוד כהן" not in _strip_nikud(result.sanitized)
        assert "דָּוִד כהן" not in result.sanitized
        assert "[PERSON_" in result.sanitized
        assert result.pii_detected


class TestSpacyNER:
    """Test spaCy NER for unknown English names (Layer 3)."""

    @pytest.mark.skipif(
        not _spacy_available(),
        reason="spaCy en_core_web_sm model not installed",
    )
    def test_unknown_english_name_detected(self, default_sanitizer):
        """spaCy should catch names not in dictionary or owner list."""
        result = default_sanitizer.sanitize("John Smith and Michael Johnson discussed the architecture")
        spacy_replacements = [r for r in result.replacements if r.source == "spacy"]
        # At least one PERSON entity should be detected
        assert len(spacy_replacements) >= 1

    def test_code_blocks_skipped(self, default_sanitizer):
        """NER should not run inside code blocks."""
        text = "Outside text. ```python\nclass Johnson:\n    pass\n``` End."
        result = default_sanitizer.sanitize(text)
        # "Johnson" inside code block should NOT be replaced
        assert "class Johnson" in result.sanitized or "Johnson" in result.sanitized

    def test_already_caught_names_skipped(self):
        """Names already caught by regex/dict should not be double-replaced."""
        s = Sanitizer(SanitizeConfig(known_names=frozenset({"John Smith"})))
        result = s.sanitize("John Smith was mentioned")
        # Should have exactly one replacement for "John Smith", from name_dict
        john_replacements = [r for r in result.replacements if "john smith" in r.original.lower()]
        assert len(john_replacements) == 1
        assert john_replacements[0].source == "name_dict"


class TestSanitizeResult:
    """Test SanitizeResult metadata."""

    def test_pii_detected_flag(self, default_sanitizer):
        result = default_sanitizer.sanitize("No PII here, just code.")
        # May or may not detect PII depending on spaCy
        # But definitely no PII for very short technical text
        clean = default_sanitizer.sanitize("x = 42")
        assert not clean.pii_detected or len(clean.replacements) > 0

    def test_pii_flag_true_when_found(self, default_sanitizer):
        result = default_sanitizer.sanitize("Jane Developer was here")
        assert result.pii_detected is True

    def test_replacements_logged(self, default_sanitizer):
        result = default_sanitizer.sanitize("Jane Developer at jane@example.com")
        assert len(result.replacements) >= 2
        categories = {r.category for r in result.replacements}
        assert "owner" in categories or "email" in categories

    def test_original_length_preserved(self, default_sanitizer):
        text = "Jane Developer wrote code"
        result = default_sanitizer.sanitize(text)
        assert result.original_length == len(text)

    def test_empty_content(self, default_sanitizer):
        result = default_sanitizer.sanitize("")
        assert result.sanitized == ""
        assert not result.pii_detected
        assert result.original_length == 0


class TestBatchSanitization:
    """Test batch processing."""

    def test_batch_sequential(self, default_sanitizer):
        chunks = [
            {"content": "Jane Developer here"},
            {"content": "No PII at all"},
            {"content": "Email: jane@example.com"},
        ]
        results = default_sanitizer.sanitize_batch(chunks, parallel=1)
        assert len(results) == 3
        assert results[0].pii_detected
        assert results[2].pii_detected

    def test_batch_parallel(self, default_sanitizer):
        chunks = [{"content": f"Jane Developer chunk {i}"} for i in range(10)]
        results = default_sanitizer.sanitize_batch(chunks, parallel=2)
        assert len(results) == 10
        assert all(r.pii_detected for r in results)

    def test_empty_batch(self, default_sanitizer):
        results = default_sanitizer.sanitize_batch([])
        assert results == []


class TestMappingSerialization:
    """Test save/load of name→pseudonym mapping."""

    def test_save_and_load_mapping(self, full_sanitizer, tmp_path):
        # Generate some pseudonyms
        full_sanitizer.sanitize("David Cohen was here")
        mapping_path = tmp_path / "mapping.json"

        # Save
        full_sanitizer.save_mapping(mapping_path)
        assert mapping_path.exists()

        data = json.loads(mapping_path.read_text())
        assert "name_to_pseudonym" in data
        assert "pseudonym_to_name" in data
        assert "david cohen" in data["name_to_pseudonym"]

        # Load into fresh sanitizer
        s2 = Sanitizer(SanitizeConfig(known_names=frozenset({"David Cohen"})))
        s2.load_mapping(mapping_path)
        r = s2.sanitize("David Cohen returned")

        # Should use the same pseudonym
        original_pseudo = data["name_to_pseudonym"]["david cohen"]
        assert original_pseudo in r.sanitized

    def test_load_nonexistent_mapping(self, default_sanitizer, tmp_path):
        """Loading from nonexistent path should be a no-op."""
        default_sanitizer.load_mapping(tmp_path / "nonexistent.json")
        # No error


class TestFromEnv:
    """Test factory method."""

    def test_from_env_defaults(self):
        s = Sanitizer.from_env()
        # Defaults are empty — users configure via env vars
        assert isinstance(s.config.owner_names, tuple)
        assert isinstance(s.config.owner_emails, tuple)
        assert isinstance(s.config.owner_paths, tuple)

    def test_from_env_custom(self, monkeypatch):
        monkeypatch.setenv("BRAINLAYER_SANITIZE_OWNER_NAMES", "Jane Doe,jane")
        monkeypatch.setenv("BRAINLAYER_SANITIZE_OWNER_EMAILS", "jane@doe.com")
        monkeypatch.setenv("BRAINLAYER_SANITIZE_USE_SPACY", "false")

        s = Sanitizer.from_env()
        assert s.config.owner_names == ("Jane Doe", "jane")
        assert s.config.owner_emails == ("jane@doe.com",)
        assert s.config.use_spacy_ner is False


class TestExternalPromptCoupling:
    """Test that build_external_prompt enforces sanitization."""

    def test_external_prompt_sanitizes_owner_name(self, default_sanitizer):
        """External prompt must strip PII from content."""
        chunk = {
            "content": "Jane Developer fixed the auth bug in server.ts",
            "project": "my-project",
            "content_type": "assistant_text",
        }
        prompt, result = build_external_prompt(chunk, default_sanitizer)
        assert "Jane Developer" not in prompt
        assert "[OWNER]" in prompt
        assert result.pii_detected

    def test_external_prompt_sanitizes_email(self, default_sanitizer):
        chunk = {
            "content": "Contact jane@example.com for details",
            "project": "test",
            "content_type": "user_message",
        }
        prompt, result = build_external_prompt(chunk, default_sanitizer)
        assert "jane@example.com" not in prompt
        assert "[OWNER_EMAIL]" in prompt

    def test_external_prompt_sanitizes_file_paths(self, default_sanitizer):
        chunk = {
            "content": "Read /Users/testuser/Projects/my-project/package.json",
            "project": "my-project",
            "content_type": "file_read",
        }
        prompt, result = build_external_prompt(chunk, default_sanitizer)
        assert "/Users/testuser" not in prompt
        assert "/Users/[OWNER]" in prompt

    def test_external_prompt_contains_enrichment_instructions(self, default_sanitizer):
        """Prompt must include the enrichment schema instructions."""
        chunk = {"content": "Some code here", "project": "test", "content_type": "ai_code"}
        prompt, _ = build_external_prompt(chunk, default_sanitizer)
        assert "summary" in prompt
        assert "tags" in prompt
        assert "importance" in prompt
        assert "intent" in prompt

    def test_external_prompt_handles_code_braces(self, default_sanitizer):
        """Content with curly braces (code) must not crash str.format()."""
        chunk = {
            "content": 'function greet() { return "hello"; }',
            "project": "test",
            "content_type": "ai_code",
        }
        prompt, _ = build_external_prompt(chunk, default_sanitizer)
        assert "function greet()" in prompt

    def test_external_prompt_returns_sanitize_result(self, default_sanitizer):
        """Must return the SanitizeResult for audit/mapping."""
        chunk = {"content": "Jane wrote code", "project": "test", "content_type": "ai_code"}
        prompt, result = build_external_prompt(chunk, default_sanitizer)
        assert isinstance(result, SanitizeResult)
        assert result.original_length > 0
        assert len(result.replacements) > 0
