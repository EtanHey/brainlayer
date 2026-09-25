"""Go-forward secret scrubber for persisted BrainLayer chunks.

The scrubber is deliberately two-mode:
- provider-prefixed secret shapes are redacted fail-closed;
- unlabeled high-entropy tokens are reported for review but left unchanged.
"""

from __future__ import annotations

import bisect
import itertools
import math
import re
from dataclasses import dataclass, field

MAX_SCAN_BYTES = 128 * 1024
WINDOW_OVERLAP_CHARS = 512
MIN_ENTROPY_TOKEN_LENGTH = 24
ENTROPY_THRESHOLD = 4.0


@dataclass(frozen=True)
class SecretRedaction:
    provider: str
    original: str
    placeholder: str
    start: int
    end: int


@dataclass(frozen=True)
class QuarantinedToken:
    value: str
    start: int
    end: int
    reason: str = "unlabeled_high_entropy"


@dataclass(frozen=True)
class SecretScrubResult:
    text: str
    redactions: list[SecretRedaction] = field(default_factory=list)
    quarantine: list[QuarantinedToken] = field(default_factory=list)


@dataclass(frozen=True)
class _ProviderPattern:
    provider: str
    regex: re.Pattern[str]


_PROVIDER_PATTERNS = (
    _ProviderPattern("anthropic", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    _ProviderPattern("stripe", re.compile(r"\b(?:[sr]k_(?:live|test)|whsec)_[A-Za-z0-9]{16,}\b")),
    _ProviderPattern("openai", re.compile(r"\bsk-(?:proj-|svcacct-|admin-|org-)?[A-Za-z0-9_-]{20,}\b")),
    _ProviderPattern("aws", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    _ProviderPattern(
        "github",
        re.compile(r"\b(?:gh[opusr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}_[A-Za-z0-9_]{20,})\b"),
    ),
    _ProviderPattern("slack", re.compile(r"\bxox[baprs]-(?:[A-Za-z0-9]+-){1,}[A-Za-z0-9]{16,}\b")),
    _ProviderPattern("google", re.compile(r"\bAIza[A-Za-z0-9_-]{32,}\b")),
    _ProviderPattern("gitlab", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b")),
    _ProviderPattern("supabase", re.compile(r"\b(?:sbp_[A-Za-z0-9]{20,}|sb_secret_[A-Za-z0-9_-]{20,})\b")),
    _ProviderPattern("sendgrid", re.compile(r"\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{32,}\b")),
    _ProviderPattern("groq", re.compile(r"\bgsk_[A-Za-z0-9]{40,}\b")),
    # tskey-<kind>-<key id>-<secret>, kind in auth/api/client/webhook/scim.
    _ProviderPattern("tailscale", re.compile(r"\btskey-[a-z]+-[A-Za-z0-9]{6,}-[A-Za-z0-9]{16,}\b")),
    _ProviderPattern("vercel", re.compile(r"\bvc[kpi]_[A-Za-z0-9]{20,}\b")),
)

_LABEL_KEYWORD = r"(?:key|token|secret|password|api|auth|access)"
# The label is one whole run of label characters. The lookbehind stops a match
# from starting mid-run, and the lookahead requires a keyword inside that same
# run, so no quantifier can backtrack across another's territory and the scan
# stays linear. The previous form nested two unbounded [A-Za-z0-9_.-]* around
# the keyword and went roughly cubic on long hyphen-joined text: 4 KB of
# "key-" took ~20 s, and a 128 KB scan window could stall ingest for hours.
# The label still ends on a word character, as the old \b required. The
# optional quote after it accepts JSON / dict keys ("api_key": "...").
_SECRET_LABEL_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    rf"(?=[A-Za-z0-9_.-]*?{_LABEL_KEYWORD})"
    r"(?P<label>[A-Za-z0-9_.-]*[A-Za-z0-9_])"
    r"(?P<label_quote>[\"']?)\s*[=:]\s*"
    r"(?P<quote>[\"']?)"
    rf"(?P<value>[A-Za-z0-9_./+=:-]{{{MIN_ENTROPY_TOKEN_LENGTH},}})"
    r"(?P=quote)",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"\b[A-Za-z0-9_./+=:-]{24,}\b")
_HEX_RE = re.compile(r"\b[0-9a-fA-F]{16,}\b")
_UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")


def scrub_secrets(text: str) -> SecretScrubResult:
    """Redact labeled secrets before persistence while preserving join-key-like tokens."""
    if not text:
        return SecretScrubResult(text=text)

    windows = list(_scan_windows(text))
    spans: list[SecretRedaction] = []
    for start, end in windows:
        spans.extend(_provider_redactions(text[start:end], offset=start))
    for start, end in windows:
        spans.extend(_assignment_redactions(text[start:end], spans, offset=start))
    spans = _without_overlaps(sorted(spans, key=lambda item: (item.start, item.end)))

    scrubbed = _apply_redactions(text, spans) if spans else text
    quarantine: list[QuarantinedToken] = []
    for start, end in windows:
        quarantine.extend(_quarantine_unlabeled_entropy(text[start:end], spans, offset=start))
    quarantine = _without_duplicate_quarantine(sorted(quarantine, key=lambda item: (item.start, item.end)))
    return SecretScrubResult(text=scrubbed, redactions=spans, quarantine=quarantine)


def _scan_windows(text: str) -> list[tuple[int, int]]:
    if len(text) <= MAX_SCAN_BYTES:
        return [(0, len(text))]
    windows: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + MAX_SCAN_BYTES)
        windows.append((start, end))
        if end == len(text):
            break
        start = max(end - WINDOW_OVERLAP_CHARS, start + 1)
    return windows


def _provider_redactions(text: str, *, offset: int = 0) -> list[SecretRedaction]:
    redactions: list[SecretRedaction] = []
    for provider_pattern in _PROVIDER_PATTERNS:
        placeholder = f"[REDACTED:{provider_pattern.provider}]"
        for match in provider_pattern.regex.finditer(text):
            redactions.append(
                SecretRedaction(
                    provider=provider_pattern.provider,
                    original=match.group(0),
                    placeholder=placeholder,
                    start=offset + match.start(),
                    end=offset + match.end(),
                )
            )
    return redactions


def _assignment_redactions(text: str, existing: list[SecretRedaction], *, offset: int = 0) -> list[SecretRedaction]:
    redactions: list[SecretRedaction] = []
    existing_index = _SpanIndex(existing)
    pos = 0
    while (match := _SECRET_LABEL_RE.search(text, pos)) is not None:
        raw_value = match.group("value")
        value = raw_value.rstrip(".,;)")
        value_start = offset + match.start("value")
        value_end = value_start + len(value)
        if match.group("label_quote"):
            # The old rule never matched a quoted label ('"cache_key": "…"'), so
            # it scanned that value and could start its own match inside it
            # (':token=<secret>'). Whether this match is kept or rejected, resume
            # at its value so that match is still found. A value never contains a
            # quote, so each quoted value is rescanned at most once: still linear.
            # Overlapping redactions are resolved later by _without_overlaps.
            pos = match.start("value")
        else:
            pos = match.end()
        if _is_rejected_assignment(value, value_start, value_end, existing_index):
            if not match.group("label_quote") and _looks_like_path_or_url(value):
                # Same for a rejected path ("api_key=/tmp/x:token=<secret>"):
                # resume after its last separator. A later match there has no
                # separator, so it can't be path-rejected again.
                pos = match.start("value") + max(raw_value.rfind("/"), raw_value.rfind("\\")) + 1
            continue
        redactions.append(
            SecretRedaction(
                provider="assignment",
                original=value,
                placeholder="[REDACTED:assignment]",
                start=value_start,
                end=value_end,
            )
        )
    return redactions


def _is_rejected_assignment(value: str, start: int, end: int, existing: _SpanIndex) -> bool:
    return (
        existing.overlaps(start, end)
        or _is_join_key_like(value)
        or _looks_like_path_or_url(value)
        or not _is_high_entropy(value)
    )


def _quarantine_unlabeled_entropy(
    text: str, redactions: list[SecretRedaction], *, offset: int = 0
) -> list[QuarantinedToken]:
    quarantined: list[QuarantinedToken] = []
    redaction_index = _SpanIndex(redactions)
    for match in _TOKEN_RE.finditer(text):
        value = match.group(0).strip(".,;)")
        start = offset + match.start()
        end = start + len(value)
        if redaction_index.overlaps(start, end):
            continue
        if _is_join_key_like(value):
            continue
        if _looks_like_path_or_url(value):
            continue
        if _is_high_entropy(value):
            quarantined.append(QuarantinedToken(value=value, start=start, end=end))
    return quarantined


# Both de-overlap passes take spans sorted by (start, end) and keep the first of
# any overlapping group. Every kept span starts at or before the current one, so
# the current span overlaps some kept span exactly when it starts before the
# furthest kept end. That holds because no span is empty: every pattern has a
# minimum length, and a value stripped to nothing fails the entropy check before
# it becomes a span. This replaces an all-pairs check that was O(k^2) in the
# number of findings (1 MB of `"api_key":"<V>",` lines took ~16 s).
def _without_overlaps(redactions: list[SecretRedaction]) -> list[SecretRedaction]:
    kept: list[SecretRedaction] = []
    furthest_end = -1
    for redaction in redactions:
        if redaction.start >= furthest_end:
            kept.append(redaction)
            furthest_end = max(furthest_end, redaction.end)
    return kept


def _without_duplicate_quarantine(quarantine: list[QuarantinedToken]) -> list[QuarantinedToken]:
    kept: list[QuarantinedToken] = []
    furthest_end = -1
    for token in quarantine:
        if token.start >= furthest_end:
            kept.append(token)
            furthest_end = max(furthest_end, token.end)
    return kept


class _SpanIndex:
    """Answers "does [start, end) overlap any of these spans?" in O(log n).

    Same predicate as the old linear scan, ``start < span.end and end > span.start``:
    spans are sorted by start, the spans that start before ``end`` form a prefix
    found by bisection, and one of them overlaps exactly when the largest end in
    that prefix is past ``start``.
    """

    def __init__(self, spans: list[SecretRedaction]) -> None:
        ordered = sorted(spans, key=lambda span: span.start)
        self._starts = [span.start for span in ordered]
        self._prefix_max_end = list(itertools.accumulate((span.end for span in ordered), max))

    def overlaps(self, start: int, end: int) -> bool:
        prefix = bisect.bisect_left(self._starts, end)
        return prefix > 0 and self._prefix_max_end[prefix - 1] > start


def _apply_redactions(text: str, redactions: list[SecretRedaction]) -> str:
    parts: list[str] = []
    cursor = 0
    for redaction in redactions:
        parts.append(text[cursor : redaction.start])
        parts.append(redaction.placeholder)
        cursor = redaction.end
    parts.append(text[cursor:])
    return "".join(parts)


def _is_join_key_like(value: str) -> bool:
    return bool(_UUID_RE.fullmatch(value) or _HEX_RE.fullmatch(value))


def _looks_like_path_or_url(value: str) -> bool:
    return "/" in value or "\\" in value or "://" in value


def _is_high_entropy(value: str) -> bool:
    if len(value) < MIN_ENTROPY_TOKEN_LENGTH:
        return False
    return _shannon_entropy(value) >= ENTROPY_THRESHOLD


def _shannon_entropy(value: str) -> float:
    counts = {character: value.count(character) for character in set(value)}
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def scrub_for_storage(text: str, metadata: dict | None = None) -> tuple[str, dict]:
    """Scrub ``text`` before it is persisted and record what was found.

    Returns the scrubbed text and a copy of ``metadata`` carrying the same keys
    the watcher writes: ``secret_scrub_redactions`` (sorted provider names,
    merged with any already present) and ``secret_scrub_quarantine_count``.
    Every ingest path that stores raw text goes through this, so the stored
    row, its FTS copy and its content hash are all computed from scrubbed text.
    """
    result = scrub_secrets(text)
    found: dict = {}
    if result.redactions:
        found["secret_scrub_redactions"] = sorted({redaction.provider for redaction in result.redactions})
    if result.quarantine:
        found["secret_scrub_quarantine_count"] = len(result.quarantine)
    return result.text, merge_scrub_metadata(metadata, found)


def merge_scrub_metadata(metadata: dict | None, found: dict) -> dict:
    """Copy ``metadata`` and fold in scrub findings, unioning provider names."""
    merged = dict(metadata or {})
    providers = found.get("secret_scrub_redactions")
    if providers:
        merged["secret_scrub_redactions"] = sorted(set(merged.get("secret_scrub_redactions") or []) | set(providers))
    if "secret_scrub_quarantine_count" in found:
        merged["secret_scrub_quarantine_count"] = found["secret_scrub_quarantine_count"]
    return merged


def scrub_tags(tags: object) -> tuple[object, dict]:
    """Scrub each string tag and report what was found, like ``scrub_for_storage``.

    Returns the tags (any non-list shape passes through unchanged) and the
    findings to fold into the chunk metadata with ``merge_scrub_metadata``.
    BrainBar's store path records tag findings the same way.
    """
    if not isinstance(tags, (list, tuple)):
        return tags, {}
    providers: set[str] = set()
    scrubbed: list = []
    for tag in tags:
        if isinstance(tag, str):
            result = scrub_secrets(tag)
            providers.update(redaction.provider for redaction in result.redactions)
            scrubbed.append(result.text)
        else:
            scrubbed.append(tag)
    return scrubbed, ({"secret_scrub_redactions": sorted(providers)} if providers else {})
