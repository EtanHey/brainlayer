"""Gate tests for the go-forward secret scrubber."""

from __future__ import annotations

import pytest

from brainlayer.pipeline.secret_scrub import MAX_SCAN_BYTES, scrub_secrets


def _secret(*parts: str) -> str:
    return "".join(parts)


LABELED_PROVIDER_SECRETS = [
    # openai
    _secret("sk-", "proj-", "a" * 48),
    _secret("sk-", "a" * 48),
    _secret("sk-", "svcacct-", "a" * 40),
    _secret("sk-", "admin-", "a" * 40),
    _secret("sk-", "org-", "a" * 40),
    # anthropic
    _secret("sk-", "ant-", "api03-", "a" * 80),
    _secret("sk-", "ant-", "admin01-", "a" * 64),
    _secret("sk-", "ant-", "oat01-", "a" * 64),
    _secret("sk-", "ant-", "sid01-", "a" * 64),
    _secret("sk-", "ant-", "a" * 56),
    # aws
    _secret("AK", "IA", "I" * 16),
    _secret("AS", "IA", "J" * 16),
    _secret("AK", "IA", "X" * 16),
    _secret("AS", "IA", "Y" * 16),
    _secret("AK", "IA", "1234567890ABCDEF"),
    # github
    _secret("gh", "p_", "a" * 36),
    _secret("gh", "o_", "a" * 36),
    _secret("gh", "u_", "a" * 36),
    _secret("gh", "s_", "a" * 36),
    _secret("gh", "r_", "a" * 36),
    _secret("github", "_pat_", "11AAAAAAAA0", "a" * 22, "_", "a" * 64),
    # slack
    _secret("xox", "b-", "1" * 12, "-", "2" * 12, "-", "a" * 24),
    _secret("xox", "p-", "1" * 12, "-", "2" * 12, "-", "3" * 12, "-", "a" * 32),
    _secret("xox", "a-", "1" * 12, "-", "2" * 12, "-", "3" * 12, "-", "a" * 32),
    _secret("xox", "r-", "1" * 12, "-", "2" * 12, "-", "3" * 12, "-", "a" * 32),
    _secret("xox", "s-", "1" * 12, "-", "2" * 12, "-", "3" * 12, "-", "a" * 32),
    # google
    _secret("AI", "za", "Sy", "A" * 35),
    _secret("AI", "za", "A" * 36),
    _secret("AI", "za", "Sy", "B" * 35),
    _secret("AI", "za", "Sy", "C" * 35),
    _secret("AI", "za", "Sy", "D" * 35),
    # gitlab
    _secret("gl", "pat-", "a" * 20),
    _secret("gl", "pat-", "b" * 20),
    _secret("gl", "pat-", "c" * 20),
    _secret("gl", "pat-", "d" * 20),
    _secret("gl", "pat-", "e" * 20),
    # stripe
    _secret("sk", "_live_", "a" * 24),
    _secret("sk", "_test_", "a" * 24),
    _secret("rk", "_live_", "a" * 24),
    _secret("rk", "_test_", "a" * 24),
    _secret("wh", "sec_", "a" * 32),
    # sendgrid
    _secret("S", "G.", "a" * 22, ".", "a" * 43),
    _secret("S", "G.", "b" * 22, ".", "b" * 43),
    _secret("S", "G.", "c" * 22, ".", "c" * 43),
    _secret("S", "G.", "d" * 22, ".", "d" * 43),
    _secret("S", "G.", "e" * 22, ".", "e" * 43),
    _secret("S", "G.", "f" * 22, ".", "f" * 43),
    _secret("S", "G.", "g" * 22, ".", "g" * 43),
]


BENIGN_BLOCKS = [
    "/Users/etanheyman/Gits/brainlayer.wt/p1-5-scrubber/src/brainlayer/watcher_bridge.py",
    "session_id=550e8400-e29b-41d4-a716-446655440000 should stay joinable",
    "conversation_id: 123e4567-e89b-12d3-a456-426614174000",
    "chunk_id rt-abcdef12-1234567890abcdef is not a secret",
    "trace_id=018f9f0a-d8f2-7c41-bb28-7f14a8a8924f",
    "node_modules/@scope/package-name/dist/index.js",
    "src/brainlayer/pipeline/secret_scrub.py",
    "normalized_exact_hash(clean_content)[:16]",
    "commit 8f14e45fceea167a5a36dedd4bea2543b17fb2d7",
    "object id 0123456789abcdef0123456789abcdef",
    "tmp path /var/folders/ab/cdefghijklmn/T/session-0d3f7f7c",
    "codex session agent-a97fea32fbaeb2961.jsonl",
    "BRAINLAYER_WATCHER_WRITE_DEADLINE_S=15.0",
    "https://github.com/EtanHey/brainlayer/pull/123",
    "symbol _extract_claude_conversation_id",
    "api_path = /Users/etanheyman/Gits/brainlayer.wt/p1-5-scrubber/session-a97fea32fbaeb2961.jsonl",
] * 67


def test_labeled_provider_secret_recall_gate_is_perfect():
    text = "\n".join(f"token {idx}: {secret}" for idx, secret in enumerate(LABELED_PROVIDER_SECRETS))

    result = scrub_secrets(text)

    leaked = [secret for secret in LABELED_PROVIDER_SECRETS if secret in result.text]
    redaction_recall = (len(LABELED_PROVIDER_SECRETS) - len(leaked)) / len(LABELED_PROVIDER_SECRETS)
    assert redaction_recall == 1.0
    assert len(result.redactions) == len(LABELED_PROVIDER_SECRETS)
    assert not leaked


def test_benign_corpus_over_redaction_gate_is_at_most_point_one_percent():
    changed_blocks = 0

    for block in BENIGN_BLOCKS:
        result = scrub_secrets(block)
        if result.text != block:
            changed_blocks += 1

    over_redaction_rate = changed_blocks / len(BENIGN_BLOCKS)
    assert over_redaction_rate <= 0.001


def test_label_gated_entropy_redacts_assignment_value():
    value = "mF9qP2xL7vR8sK4nT6yB3cD5eG7hJ9kL2mN4pQ6r"

    result = scrub_secrets(f"api_key = {value}")

    assert value not in result.text
    assert result.text == "api_key = [REDACTED:assignment]"
    assert result.redactions[0].provider == "assignment"


@pytest.mark.parametrize(
    "assignment",
    [
        "session_id = 550e8400-e29b-41d4-a716-446655440000",
        "access_token = 0123456789abcdef0123456789abcdef",
    ],
)
def test_label_gated_entropy_skips_uuid_and_hex_join_keys(assignment):
    result = scrub_secrets(assignment)

    assert result.text == assignment
    assert result.redactions == []


def test_label_gated_entropy_skips_file_paths():
    assignment = "api_path = /Users/etanheyman/Gits/brainlayer.wt/p1-5-scrubber/session-a97fea32fbaeb2961.jsonl"

    result = scrub_secrets(assignment)

    assert result.text == assignment
    assert result.redactions == []


def test_unlabeled_high_entropy_token_is_quarantined_but_not_redacted():
    token = "mF9qP2xL7vR8sK4nT6yB3cD5eG7hJ9kL2mN4pQ6r"

    result = scrub_secrets(f"Observed opaque id {token} in the path")

    assert token in result.text
    assert result.redactions == []
    assert [item.value for item in result.quarantine] == [token]


def test_large_input_scans_past_first_window_without_leaking_tail_secrets():
    secret = _secret("gl", "pat-", "a" * 20)
    text = ("x" * (MAX_SCAN_BYTES + 512)) + " " + secret

    result = scrub_secrets(text)

    assert secret not in result.text
    assert result.text.endswith("[REDACTED:gitlab]")


def test_large_input_deduplicates_quarantine_from_window_overlap():
    token = "mF9qP2xL7vR8sK4nT6yB3cD5eG7hJ9kL2mN4pQ6r"
    text = ("x" * (MAX_SCAN_BYTES - 100)) + f" {token} " + ("x" * 700)

    result = scrub_secrets(text)

    assert [item.value for item in result.quarantine] == [token]
