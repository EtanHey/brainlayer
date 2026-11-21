"""The anchored provider scan must find exactly what each pattern's finditer finds.

``provider_secret_spans`` drives the scrubber's own provider patterns from literal
anchors so a bulk scan (the transcript backup) runs at str.find speed. A pattern
edit that adds a new prefix without a matching anchor would silently miss keys;
these tests fail on that instead. Tokens are synthetic, built at runtime.
"""

from __future__ import annotations

import random

from brainlayer.pipeline import secret_scrub
from brainlayer.pipeline.secret_scrub import provider_secret_spans, scrub_secrets

UUID = "0" * 8 + "-" + "0" * 4 + "-" + "0" * 4 + "-" + "0" * 4 + "-" + "0" * 12
TOKENS = [
    "ctx7sk-" + "0" * 20,
    "sk-ant-api03-" + "0" * 40,
    "sk_live_" + "0" * 20,
    "rk_test_" + "0" * 20,
    "whsec_" + "0" * 20,
    "sk-" + "0" * 24,
    "sk-proj-" + "0" * 24,
    "AKIA" + "0" * 16,
    "ASIA" + "0" * 16,
    "gh" + "p_" + "0" * 24,
    "gh" + "s_" + "0" * 24,
    "github_pat_" + "0" * 22 + "_" + "0" * 22,
    "xoxb-" + "0" * 12 + "-" + "0" * 16,
    "AIza" + "0" * 35,
    "ya29." + "0" * 24,
    "1//" + "0" * 24,
    "GOCSPX-" + "0" * 24,
    "glpat-" + "0" * 20,
    "sbp_" + "0" * 20,
    "sb_secret_" + "0" * 20,
    "SG." + "0" * 16 + "." + "0" * 32,
    "gsk_" + "0" * 40,
    "tskey-auth-" + "k000000" + "-" + "0" * 16,
    "vck_" + "0" * 20,
    "EXA_API_KEY=" + UUID,
    'exaApiKey: "' + UUID + '"',
    "my.exa-key='" + UUID + "'",
    "ExaKey=" + UUID,
    "eXa_token_key: " + UUID,
    "über-exa-key=" + UUID,
    "EXAMPLE_KEY=" + UUID,
    "nextKey=" + UUID,
]
CONTEXTS = ["{}", " {} ", "x{}", '"{}"', "\\n{}", "task-{}", "{}{}", "a.b-c_{}", "{}\n{}"]


def _slow(text: str):
    return secret_scrub._provider_redactions(text)


def _fast(text: str):
    return secret_scrub._anchored_provider_redactions(text)


def test_every_provider_pattern_has_anchors_and_every_anchor_one_pivot():
    assert set(secret_scrub._PROVIDER_ANCHORS) == {p.provider for p in secret_scrub._PROVIDER_PATTERNS}
    for provider, anchors in secret_scrub._PROVIDER_ANCHORS.items():
        assert anchors or provider == "exa_labeled", provider
        for anchor in anchors:
            assert sum(anchor.startswith(pivot) for pivot in secret_scrub._ANCHOR_PIVOTS) == 1, anchor


def test_anchored_scan_equals_finditer_on_every_family_in_every_context():
    for token in TOKENS:
        for context in CONTEXTS:
            text = context.replace("{}", token)
            assert _fast(text) == _slow(text), text


def test_anchored_scan_equals_finditer_on_fuzzed_text():
    rng = random.Random(1823)
    pieces = [token[: rng.randint(1, len(token))] for token in TOKENS] + TOKENS
    pieces += ["sk-", "ant-", "_", "-", ".", "exa", "EXA", "eXa", "Key", "key", "=", ":", " ", '"', "0" * 8, "ab"]
    for _ in range(3000):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 30)))
        assert _fast(text) == _slow(text), text


def test_provider_secret_spans_matches_scrub_secrets_provider_findings():
    text = " ".join(TOKENS) + " api_key=" + "Zq9XwP7LmN2kJ4vB8" * 2
    expected = scrub_secrets(text, providers=frozenset(p.provider for p in secret_scrub._PROVIDER_PATTERNS))
    spans = provider_secret_spans(text)
    assert [(s.provider, s.start, s.end) for s in spans] == [(r.provider, r.start, r.end) for r in expected.redactions]
    assert all(span.provider != "assignment" for span in spans)


def test_exa_anchor_walk_back_stays_linear_on_one_long_run():
    text = "exa-" * 50_000 + "key=" + UUID
    spans = provider_secret_spans(text)
    assert [(span.provider, span.original) for span in spans] == [("exa_labeled", UUID)]
