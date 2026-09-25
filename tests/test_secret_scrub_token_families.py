"""Provider token families the scrubber must redact.

Synthetic tokens are assembled from obviously fake characters; none is a real
credential. Groq, Tailscale and Vercel were missing; Anthropic and Slack were
already covered and are pinned here so a pattern edit cannot drop them.
"""

from __future__ import annotations

import pytest

from brainlayer.pipeline.secret_scrub import scrub_secrets

FAMILIES = {
    "groq": ("groq", "gsk_" + "0" * 52),
    "anthropic": ("anthropic", "sk-ant-api03-" + "0" * 40),
    "slack": ("slack", "xoxb-" + "0" * 12 + "-" + "0" * 13 + "-" + "0" * 24),
    "tailscale-auth": ("tailscale", "tskey-auth-" + "k000000CNTRL" + "-" + "0" * 32),
    "tailscale-api": ("tailscale", "tskey-api-" + "k000000CNTRL" + "-" + "0" * 32),
    "tailscale-client": ("tailscale", "tskey-client-" + "k000000CNTRL" + "-" + "0" * 32),
    "vercel-vck": ("vercel", "vck_" + "0" * 40),
    "vercel-vcp": ("vercel", "vcp_" + "0" * 40),
    "vercel-vci": ("vercel", "vci_" + "0" * 40),
}


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_token_family_is_redacted_with_its_provider(family):
    provider, token = FAMILIES[family]

    result = scrub_secrets(f"export KEY_FOR_TEST {token} # rotated")

    assert token not in result.text
    assert f"[REDACTED:{provider}]" in result.text
    assert [redaction.provider for redaction in result.redactions] == [provider]


@pytest.mark.parametrize(
    "prose",
    [
        "the tskey-auth-keys-are-rotated-weekly runbook",
        "gsk_ prefixes mark Groq keys",
        "vck_ is a Vercel prefix",
    ],
)
def test_family_prefixes_in_prose_are_not_redacted(prose):
    result = scrub_secrets(prose)

    assert result.text == prose
    assert result.redactions == []
