"""Rule-based sentiment analysis for conversation chunks.

Tier 1 of Phase 6: fast pattern matching on known frustration, positive,
confusion, and satisfaction signals. Bilingual (EN + HE).

Tier 2 (LLM enrichment) is handled by extending the enrichment prompt
in enrichment.py — this module covers rule-based detection only.

Usage:
    from brainlayer.pipeline.sentiment import analyze_sentiment, batch_analyze_sentiment

    # Single chunk
    result = analyze_sentiment("what the fuck, this is broken")
    # {"label": "frustration", "score": -0.85, "signals": ["fuck", "broken"]}

    # Batch process
    from brainlayer.vector_store import VectorStore
    store = VectorStore(db_path)
    processed = batch_analyze_sentiment(store, batch_size=500)
"""

import re
from typing import Any, Dict, List

# --- Pattern definitions ---
# Each pattern: (compiled_regex, weight, signal_name)
# Weights: 0.0-1.0 indicating signal strength

FRUSTRATION_PATTERNS = [
    # English - strong
    (re.compile(r"\bwhat the (?:fuck|hell|heck)\b", re.I), 1.0, "wtf"),
    (re.compile(r"\bfuck(?:ing|ed|s)?\b", re.I), 0.9, "fuck"),
    (re.compile(r"\bshit(?:ty)?\b", re.I), 0.8, "shit"),
    (re.compile(r"\bdamn(?:it|ed)?\b", re.I), 0.7, "damn"),
    (re.compile(r"\bcrap\b", re.I), 0.5, "crap"),
    (re.compile(r"\bwtf\b", re.I), 0.9, "wtf"),
    # English - moderate
    (re.compile(r"\bfrustrat(?:ing|ed|ion)\b", re.I), 0.8, "frustrating"),
    (re.compile(r"\bannoying\b", re.I), 0.6, "annoying"),
    (re.compile(r"\brigid(?:iculous)?\b", re.I), 0.5, "ridiculous"),
    (re.compile(r"\bterrible\b", re.I), 0.6, "terrible"),
    (re.compile(r"\bhate this\b", re.I), 0.7, "hate this"),
    (re.compile(r"\bugh+\b", re.I), 0.5, "ugh"),
    (re.compile(r"\bfml\b", re.I), 0.8, "fml"),
    # English - contextual (something is broken)
    (re.compile(r"\b(?:is |still |keeps? )?broken\b", re.I), 0.6, "broken"),
    (re.compile(r"\bnothing works\b", re.I), 0.7, "nothing works"),
    (re.compile(r"\bdoesn'?t work\b", re.I), 0.4, "doesn't work"),
    (re.compile(r"\bwhy (?:the hell |the fuck |on earth )?(?:is|does|won'?t|can'?t|isn'?t)\b", re.I), 0.5, "why..."),
    (re.compile(r"\bagain[!?]+\b", re.I), 0.4, "again!"),
    # Hebrew
    (re.compile(r"\bלעזאזל\b"), 0.9, "לעזאזל"),
    (re.compile(r"\bחרא\b"), 0.8, "חרא"),
    (re.compile(r"\bזה לא עובד\b"), 0.6, "לא עובד"),
    (re.compile(r"\bמעצבן\b"), 0.7, "מעצבן"),
    (re.compile(r"\bנמאס\b"), 0.7, "נמאס"),
    (re.compile(r"\bבלאגן\b"), 0.5, "בלאגן"),
    (re.compile(r"\bשבור\b"), 0.5, "שבור"),
]

POSITIVE_PATTERNS = [
    # English - strong
    (re.compile(r"\bamazing\b", re.I), 0.8, "amazing"),
    (re.compile(r"\bincredible\b", re.I), 0.8, "incredible"),
    (re.compile(r"\bawesome\b", re.I), 0.7, "awesome"),
    (re.compile(r"\bperfect(?:ly)?\b", re.I), 0.8, "perfect"),
    (re.compile(r"\bbrilliant\b", re.I), 0.7, "brilliant"),
    (re.compile(r"\bexcellent\b", re.I), 0.7, "excellent"),
    (re.compile(r"\bfantastic\b", re.I), 0.7, "fantastic"),
    (re.compile(r"\bbeautiful\b", re.I), 0.6, "beautiful"),
    # English - moderate
    (re.compile(r"\bgreat\b", re.I), 0.5, "great"),
    (re.compile(r"\bnice\b", re.I), 0.4, "nice"),
    (re.compile(r"\blove (?:it|this)\b", re.I), 0.7, "love it"),
    (re.compile(r"\bwow\b", re.I), 0.6, "wow"),
    (re.compile(r"\bworks? (?:perfectly|great|well|beautifully)\b", re.I), 0.7, "works great"),
    # Hebrew
    (re.compile(r"\bמדהים\b"), 0.8, "מדהים"),
    (re.compile(r"\bמעולה\b"), 0.7, "מעולה"),
    (re.compile(r"\bנהדר\b"), 0.6, "נהדר"),
    (re.compile(r"\bמושלם\b"), 0.8, "מושלם"),
    (re.compile(r"\bיופי\b"), 0.5, "יופי"),
    (re.compile(r"\bסבבה\b"), 0.4, "סבבה"),
    (re.compile(r"\bאחלה\b"), 0.5, "אחלה"),
]

CONFUSION_PATTERNS = [
    # English
    (re.compile(r"\bi don'?t understand\b", re.I), 0.8, "don't understand"),
    (re.compile(r"\bwait,? what\??\b", re.I), 0.7, "wait what"),
    (re.compile(r"\bconfus(?:ed|ing)\b", re.I), 0.7, "confused"),
    (re.compile(r"\bwhat do you mean\b", re.I), 0.6, "what do you mean"),
    (re.compile(r"\bhow does this work\b", re.I), 0.5, "how does this work"),
    (re.compile(r"\bmakes? no sense\b", re.I), 0.7, "makes no sense"),
    (re.compile(r"\blost\b", re.I), 0.4, "lost"),
    (re.compile(r"\bhuh\?+\b", re.I), 0.5, "huh?"),
    (re.compile(r"\?\?+", re.I), 0.4, "??"),
    # Hebrew
    (re.compile(r"\bלא מבין\b"), 0.8, "לא מבין"),
    (re.compile(r"\bמבלבל\b"), 0.7, "מבלבל"),
    (re.compile(r"\bמה זה\b"), 0.4, "מה זה"),
    (re.compile(r"\bרגע מה\b"), 0.6, "רגע מה"),
]

SATISFACTION_PATTERNS = [
    # English
    (re.compile(r"\bthanks?(?:!| a lot| so much)?\b", re.I), 0.5, "thanks"),
    (re.compile(r"\bthank you\b", re.I), 0.5, "thank you"),
    (re.compile(r"\bexactly what I (?:needed|wanted)\b", re.I), 0.8, "exactly what I needed"),
    (re.compile(r"\bthat'?s? (?:it|right|correct)\b", re.I), 0.4, "that's it"),
    (re.compile(r"\bwell done\b", re.I), 0.6, "well done"),
    (re.compile(r"\bgood job\b", re.I), 0.5, "good job"),
    (re.compile(r"\bnailed it\b", re.I), 0.7, "nailed it"),
    (re.compile(r"\bship it\b", re.I), 0.6, "ship it"),
    (re.compile(r"\blgtm\b", re.I), 0.5, "lgtm"),
    # Hebrew
    (re.compile(r"\bתודה\b"), 0.5, "תודה"),
    (re.compile(r"\bבדיוק מה שרציתי\b"), 0.8, "בדיוק מה שרציתי"),
    (re.compile(r"\bמצוין\b"), 0.6, "מצוין"),
]


def _match_patterns(text: str, patterns: list) -> List[tuple]:
    """Match text against pattern list. Returns [(signal_name, weight), ...]."""
    matches = []
    for regex, weight, signal_name in patterns:
        if regex.search(text):
            matches.append((signal_name, weight))
    return matches


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze sentiment of a text chunk using rule-based pattern matching.

    Returns:
        {"label": str, "score": float, "signals": list[str]}
        label: frustration|confusion|positive|satisfaction|neutral
        score: -1.0 (max negative) to +1.0 (max positive)
        signals: list of matched pattern names
    """
    if not text or not text.strip():
        return {"label": "neutral", "score": 0.0, "signals": []}

    frustration = _match_patterns(text, FRUSTRATION_PATTERNS)
    positive = _match_patterns(text, POSITIVE_PATTERNS)
    confusion = _match_patterns(text, CONFUSION_PATTERNS)
    satisfaction = _match_patterns(text, SATISFACTION_PATTERNS)

    # Calculate weighted scores per category
    frust_score = sum(w for _, w in frustration) if frustration else 0
    pos_score = sum(w for _, w in positive) if positive else 0
    conf_score = sum(w for _, w in confusion) if confusion else 0
    sat_score = sum(w for _, w in satisfaction) if satisfaction else 0

    # Determine dominant category
    # Negative categories: frustration, confusion
    # Positive categories: positive, satisfaction
    neg_total = frust_score + conf_score * 0.5
    pos_total = pos_score + sat_score

    # Normalize score to [-1, 1]
    raw = pos_total - neg_total
    max_magnitude = max(abs(raw), 1.0)
    score = max(-1.0, min(1.0, raw / max_magnitude))

    # Collect all signals
    all_signals = (
        [s for s, _ in frustration]
        + [s for s, _ in confusion]
        + [s for s, _ in positive]
        + [s for s, _ in satisfaction]
    )

    # Label selection: pick the strongest category
    categories = [
        ("frustration", frust_score, frustration),
        ("confusion", conf_score, confusion),
        ("positive", pos_score, positive),
        ("satisfaction", sat_score, satisfaction),
    ]
    categories.sort(key=lambda x: x[1], reverse=True)

    if categories[0][1] == 0:
        return {"label": "neutral", "score": 0.0, "signals": []}

    top_label = categories[0][0]

    # If frustration and satisfaction both present, frustration wins (user frustrated
    # even if they said thanks)
    if frust_score > 0 and sat_score > 0 and frust_score >= sat_score:
        top_label = "frustration"

    # Satisfaction requires positive-like signals too (not just "thanks" alone)
    if top_label == "satisfaction" and sat_score < 0.6 and pos_score == 0:
        top_label = "neutral"
        score = score * 0.5  # dampen

    return {"label": top_label, "score": round(score, 3), "signals": all_signals}


def batch_analyze_sentiment(
    store: Any,
    batch_size: int = 500,
    max_chunks: int = 0,
) -> int:
    """Batch-process user_message chunks through rule-based sentiment analysis.

    Only processes chunks where sentiment_label IS NULL and content_type = 'user_message'.

    Args:
        store: VectorStore instance
        batch_size: chunks per batch
        max_chunks: max total chunks to process (0 = unlimited)

    Returns:
        Number of chunks processed.
    """
    cursor = store.conn.cursor()

    limit_clause = f"LIMIT {max_chunks}" if max_chunks > 0 else ""

    rows = list(
        cursor.execute(
            f"""SELECT id, content FROM chunks
               WHERE content_type = 'user_message'
               AND sentiment_label IS NULL
               ORDER BY created_at DESC
               {limit_clause}"""
        )
    )

    processed = 0
    for chunk_id, content in rows:
        result = analyze_sentiment(content or "")
        store.update_sentiment(
            chunk_id,
            label=result["label"],
            score=result["score"],
            signals=result["signals"],
        )
        processed += 1

    return processed
