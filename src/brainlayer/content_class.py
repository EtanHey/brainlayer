"""Content-class classification and read-path filtering helpers."""

from __future__ import annotations

import re
from typing import Any

DEFAULT_CONTENT_CLASS = "knowledge"
CONTENT_CLASS_VALUES = frozenset({"knowledge", "decision", "operational", "test"})
DEFAULT_HIDDEN_CONTENT_CLASSES = frozenset({"operational", "test"})

_DECISION_RE = re.compile(
    r"\b("
    r"decision|decided|decide|chose|choose|chosen|because|why|rule|always|never|"
    r"prefer|preference|trade[- ]?off|outcome|learning|learned|lesson"
    r")\b",
    re.IGNORECASE,
)
_OPERATIONAL_RE = re.compile(
    r"("
    r"<task-notification\b|^\s*\[?claude_counter\b|\borc\b.*\btick\b|^\s*loop tick\b|"
    r"\bheartbeat\b|\bbare status\b|\bstatus only\b|\bsurface:\d+\b|"
    r"\btask[_ -]?notification\b|\bmilestone\b|\bretraction\b|\bdone_fixes\b|"
    r"\bworker done\b|\bagent coordination\b"
    r")",
    re.IGNORECASE,
)
_BL_OPERATIONAL_RE = re.compile(
    r"\[bl-[^\]]*(?:tick|status|heartbeat|claude_counter|working|poll|surface|monitor|done)",
    re.IGNORECASE,
)
_TEST_RE = re.compile(
    r"("
    r"\bad[- ]?hoc\b.*\b(test|eval|query)\b|"
    r"\beval test query\b|\btest query\b|\bbenchmark query\b|\bsmoke test\b"
    r")",
    re.IGNORECASE,
)
_PERSONAL_RE = re.compile(
    r"("
    r"\bjournal\b|\bpersonal\b|\bmy life\b|\betan's life\b|\bfamily\b|"
    r"\brelationship\b|\bdating\b|\bpartner\b|\bfriend\b|\bhealth\b|\bdoctor\b|"
    r"\bmedical\b|\btherapy\b|\btherapist\b|\bsleep\b|\bweight\b|\bhabit\b|"
    r"\bfinance\b|\bfinancial\b|\bbank\b|\btax\b|\bsalary\b|\bincome\b|"
    r"\binvoice\b|\brecruiting\b"
    r")",
    re.IGNORECASE,
)
_PERSON_NAME_RE = re.compile(
    r"(\betan\b|etan@heyman\.net|heyman\.net|\bYuval Rapoport\b|"
    r"\bGal Rava\b|\bSamantha Cerqueira\b|\bDaniel Munk\b|\bOren Efraim\b|"
    r"\bOren Ephraim\b|\bAvi Simon\b|\bMichal Cohen\b|\bChase\b)",
    re.IGNORECASE,
)
_PERSONAL_RISK_RE = re.compile(
    r"("
    r"\betan\b|etan@heyman\.net|heyman\.net|\bjournal\b|\bpersonal\b|\bmy life\b|\bfamily\b|"
    r"\brelationship\b|\bdating\b|\bpartner\b|\bfriend\b|\bhealth\b|\bdoctor\b|"
    r"\bmedical\b|\btherapy\b|\btherapist\b|\bsleep\b|\bweight\b|\bhabit\b|"
    r"\bfinance\b|\bfinancial\b|\bbank\b|\btax\b|\bsalary\b|\bincome\b|"
    r"\binvoice\b|\brecruiting\b"
    r")",
    re.IGNORECASE,
)
_HAS_HEBREW = re.compile(r"[\u0590-\u05ff]")
_OPERATIONAL_INTENT_RE = re.compile(
    r"("
    r"\boperational\b|\bheartbeat\b|\btask[-_ ]?notification\b|"
    r"\bclaude_counter\b|\btick\b|\bsurface:\d+\b|"
    r"\bbare status\b|\bstatus only\b|\bstatus notes?\b|"
    r"\bad[- ]?hoc\b.*\b(test|eval|query)\b|\beval test\b|\btest query\b|\bbenchmark query\b"
    r")",
    re.IGNORECASE,
)


def normalize_content_class(value: Any) -> str:
    """Normalize nullable/unknown class values to the safe visible default."""
    if not isinstance(value, str):
        return DEFAULT_CONTENT_CLASS
    normalized = value.strip().lower()
    if normalized in CONTENT_CLASS_VALUES:
        return normalized
    return DEFAULT_CONTENT_CLASS


def has_decision_language(content: str | None, *, content_type: str | None = None) -> bool:
    """Return true when content carries durable decision-style language."""
    if normalize_content_class(content_type) == "decision" or (content_type or "").strip().lower() == "decision":
        return True
    if not content:
        return False
    return bool(_DECISION_RE.search(content))


def has_personal_signal(content: str | None) -> bool:
    """Return true for personal/biographical content that should stay visible by default."""
    return bool(content and _PERSONAL_RE.search(content))


def has_personal_risk_signal(content: str | None) -> bool:
    """Return true for broader personal-risk terms used by backfill gates."""
    return bool(content and _PERSONAL_RISK_RE.search(content))


def has_person_name_signal(content: str | None) -> bool:
    """Return true for known personal-name signals in Etan's corpus."""
    return bool(content and _PERSON_NAME_RE.search(content))


def has_hebrew_signal(content: str | None) -> bool:
    """Return true for Hebrew-script content; keep visible by default."""
    return bool(content and _HAS_HEBREW.search(content))


def keep_visible_signals(content: str | None, *, content_type: str | None = None) -> list[str]:
    """Signals that force would-be hidden rows to remain default-visible."""
    signals: list[str] = []
    if has_decision_language(content, content_type=content_type):
        signals.append("decision_language")
    if has_personal_signal(content) or has_personal_risk_signal(content):
        signals.append("personal_biographical")
    if has_person_name_signal(content):
        signals.append("person_name")
    if has_hebrew_signal(content):
        signals.append("hebrew")
    return signals


def classify_content_class_raw(
    content: str | None,
    *,
    content_type: str | None = None,
    tags: Any = None,
    source: str | None = None,
    source_file: str | None = None,
    project: str | None = None,
) -> str:
    """Classify a chunk before keep-visible safety overrides.

    Decision signals are intentionally evaluated before operational/test signals:
    durable "why we did X" records must stay visible even when they include
    coordination prefixes like "[BL-LEAD]".
    """
    del tags, source, source_file, project  # Reserved for future cheap signals.
    text = (content or "").strip()
    type_key = (content_type or "").strip().lower()

    if type_key == "decision":
        return "decision"
    if type_key != "learning" and has_decision_language(text, content_type=content_type):
        return "decision"

    if _TEST_RE.search(text):
        return "test"
    if _OPERATIONAL_RE.search(text) or _BL_OPERATIONAL_RE.search(text):
        return "operational"

    return DEFAULT_CONTENT_CLASS


def classify_content_class(
    content: str | None,
    *,
    content_type: str | None = None,
    tags: Any = None,
    source: str | None = None,
    source_file: str | None = None,
    project: str | None = None,
) -> str:
    """Classify a chunk for default search visibility."""
    proposed = classify_content_class_raw(
        content,
        content_type=content_type,
        tags=tags,
        source=source,
        source_file=source_file,
        project=project,
    )
    if proposed in DEFAULT_HIDDEN_CONTENT_CLASSES and keep_visible_signals(content, content_type=content_type):
        return DEFAULT_CONTENT_CLASS
    return proposed


def query_signals_operational_intent(query: str | None) -> bool:
    """Cheap opt-in for users explicitly searching operational/test material."""
    return bool(query and _OPERATIONAL_INTENT_RE.search(query))


def content_class_is_default_hidden(value: Any) -> bool:
    """Return true for classes hidden from default read paths."""
    return normalize_content_class(value) in DEFAULT_HIDDEN_CONTENT_CLASSES
