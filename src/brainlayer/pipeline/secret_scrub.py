"""Go-forward secret scrubber for persisted BrainLayer chunks.

The scrubber is deliberately two-mode:
- provider-prefixed secret shapes are redacted fail-closed;
- unlabeled high-entropy tokens are reported for review but left unchanged.
"""

from __future__ import annotations

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
    _ProviderPattern("sendgrid", re.compile(r"\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{32,}\b")),
)

_SECRET_LABEL_RE = re.compile(
    r"(?P<prefix>\b[A-Za-z0-9_.-]*(?:key|token|secret|password|api|auth|access)[A-Za-z0-9_.-]*\b\s*[=:]\s*)"
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
    for match in _SECRET_LABEL_RE.finditer(text):
        value = match.group("value").rstrip(".,;)")
        value_start = offset + match.start("value")
        value_end = value_start + len(value)
        if _span_overlaps(value_start, value_end, existing):
            continue
        if _is_join_key_like(value):
            continue
        if _looks_like_path_or_url(value):
            continue
        if not _is_high_entropy(value):
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


def _quarantine_unlabeled_entropy(
    text: str, redactions: list[SecretRedaction], *, offset: int = 0
) -> list[QuarantinedToken]:
    quarantined: list[QuarantinedToken] = []
    for match in _TOKEN_RE.finditer(text):
        value = match.group(0).strip(".,;)")
        start = offset + match.start()
        end = start + len(value)
        if _span_overlaps(start, end, redactions):
            continue
        if _is_join_key_like(value):
            continue
        if _looks_like_path_or_url(value):
            continue
        if _is_high_entropy(value):
            quarantined.append(QuarantinedToken(value=value, start=start, end=end))
    return quarantined


def _without_overlaps(redactions: list[SecretRedaction]) -> list[SecretRedaction]:
    kept: list[SecretRedaction] = []
    for redaction in redactions:
        if not _span_overlaps(redaction.start, redaction.end, kept):
            kept.append(redaction)
    return kept


def _without_duplicate_quarantine(quarantine: list[QuarantinedToken]) -> list[QuarantinedToken]:
    kept: list[QuarantinedToken] = []
    for token in quarantine:
        if not any(token.start < kept_token.end and token.end > kept_token.start for kept_token in kept):
            kept.append(token)
    return kept


def _apply_redactions(text: str, redactions: list[SecretRedaction]) -> str:
    parts: list[str] = []
    cursor = 0
    for redaction in redactions:
        parts.append(text[cursor : redaction.start])
        parts.append(redaction.placeholder)
        cursor = redaction.end
    parts.append(text[cursor:])
    return "".join(parts)


def _span_overlaps(start: int, end: int, spans: list[SecretRedaction]) -> bool:
    return any(start < span.end and end > span.start for span in spans)


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
