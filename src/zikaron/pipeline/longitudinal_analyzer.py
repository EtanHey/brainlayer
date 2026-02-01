"""Longitudinal communication style analyzer.

Analyzes communication patterns over time periods using LLM,
tracks evolution, and generates style rules.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

from .time_batcher import TimeBatch, get_period_weight


@dataclass
class PeriodAnalysis:
    """Analysis results for a single time period."""
    period: str
    language: str  # "hebrew", "english", or "all"
    message_count: int
    avg_length: float
    formality_score: float  # 0-10
    emoji_rate: float
    common_phrases: list[str]
    tone_description: str
    key_characteristics: list[str]
    raw_analysis: str  # Full LLM output
    
    def to_dict(self) -> dict:
        return {
            "period": self.period,
            "language": self.language,
            "message_count": self.message_count,
            "avg_length": self.avg_length,
            "formality_score": self.formality_score,
            "emoji_rate": self.emoji_rate,
            "common_phrases": self.common_phrases,
            "tone_description": self.tone_description,
            "key_characteristics": self.key_characteristics,
            "raw_analysis": self.raw_analysis,
        }


def analyze_batch_with_llm(
    batch: TimeBatch,
    language: str = "all",
    model: str = "qwen3-coder-64k",
    max_samples: int = 250,
    style_collection=None,
) -> PeriodAnalysis:
    """
    Analyze a time batch using LLM.
    
    Args:
        batch: The time batch to analyze
        language: "hebrew", "english", or "all"
        model: Ollama model to use
        max_samples: Maximum messages to include in prompt
    
    Returns:
        PeriodAnalysis with results
    """
    if not HAS_OLLAMA:
        raise ImportError("ollama package required: pip install ollama")
    
    # Filter by language if specified
    if language == "hebrew":
        messages = batch.hebrew_messages
    elif language == "english":
        messages = batch.english_messages
    else:
        messages = batch.messages
    
    if not messages:
        return PeriodAnalysis(
            period=batch.period,
            language=language,
            message_count=0,
            avg_length=0,
            formality_score=5.0,
            emoji_rate=0,
            common_phrases=[],
            tone_description="No messages in this period",
            key_characteristics=[],
            raw_analysis="",
        )
    
    # Sample messages for prompt: cluster-based if style_collection, else first N
    if style_collection:
        from .style_index import get_embeddings_for_batch
        from .cluster_sampling import cluster_sample_messages

        start_epoch = batch.start_date.timestamp()
        end_epoch = batch.end_date.timestamp()
        embs, docs = get_embeddings_for_batch(
            style_collection, start_epoch, end_epoch, language
        )
        if embs and docs:
            class _TextDoc:
                def __init__(self, text): self.text = text
            items = [_TextDoc(d) for d in docs]
            samples = cluster_sample_messages(items, embs, max_total=max_samples)
            samples_text = "\n".join([f"- {m.text[:200]}" for m in samples])
        else:
            samples = messages[:max_samples]
            samples_text = "\n".join([f"- {m.text[:200]}" for m in samples])
    else:
        samples = messages[:max_samples]
        samples_text = "\n".join([f"- {m.text[:200]}" for m in samples])
    
    # Calculate basic metrics
    total_length = sum(len(m.text) for m in messages)
    avg_length = total_length / len(messages)
    
    # Count emojis (simple pattern)
    import re
    emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]')
    emoji_count = sum(len(emoji_pattern.findall(m.text)) for m in messages)
    emoji_rate = emoji_count / len(messages)
    
    relationship_ctx = batch.get_relationship_context()
    relationship_block = f"\n{relationship_ctx}\nConsider how tone may shift by relationship.\n" if relationship_ctx else ""
    
    prompt = f"""Analyze this person's communication style from {len(samples)} messages in the period {batch.period}.

Language focus: {language.upper()}
{relationship_block}
CRITICAL RULES:
- Every phrase, example, and quote MUST appear verbatim in the sample messages below. Do NOT invent or paraphrase.
- Be specific. Cite exact text from the messages.
- If a phrase appears only once, note it as "occasional" not "frequent."

Sample messages:
{samples_text}

Provide a structured analysis:

## 1. FORMALITY SCORE (0-10)
Rate from 0 (very casual) to 10 (very formal). Provide the number and brief justification. Reference specific messages.

## 2. TONE DESCRIPTION
Describe the overall tone in 1-2 sentences. Base this on actual message content.

## 3. KEY CHARACTERISTICS
List 3-5 distinct characteristics. Each must be different—no overlap (e.g., don't list both "short messages" and "concise" as separate items).

## 4. COMMON PHRASES
List 5-10 phrases they use. Copy them EXACTLY from the messages—character for character. Include nothing invented.

## 5. COMMUNICATION PATTERNS
- How do they start messages? (quote real examples)
- How do they make requests? (quote real examples)
- How do they express emotions? (quote real examples)
- Use of punctuation and capitalization
"""

    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={
            'num_ctx': 16000,
            'temperature': 0.1,
        }
    )
    
    raw_analysis = response['response']
    
    # Parse formality score from response (simple extraction)
    formality_score = 5.0  # Default
    import re
    score_match = re.search(r'(?:score|rating)[:\s]*(\d+(?:\.\d+)?)', raw_analysis.lower())
    if score_match:
        try:
            formality_score = float(score_match.group(1))
        except ValueError:
            pass
    
    # Extract key characteristics (lines starting with - or *)
    char_pattern = re.compile(r'^[\-\*]\s*(.+)$', re.MULTILINE)
    characteristics = char_pattern.findall(raw_analysis)[:10]
    
    # Extract common phrases (quoted text)
    phrase_pattern = re.compile(r'"([^"]+)"')
    phrases = phrase_pattern.findall(raw_analysis)[:10]
    
    # Extract tone description (first sentence after "tone" keyword)
    tone_desc = "Professional and direct"  # Default
    tone_match = re.search(r'tone[:\s]+([^.]+\.)', raw_analysis.lower())
    if tone_match:
        tone_desc = tone_match.group(1).strip().capitalize()
    
    return PeriodAnalysis(
        period=batch.period,
        language=language,
        message_count=len(messages),
        avg_length=avg_length,
        formality_score=formality_score,
        emoji_rate=emoji_rate,
        common_phrases=phrases,
        tone_description=tone_desc,
        key_characteristics=characteristics,
        raw_analysis=raw_analysis,
    )


def analyze_evolution(
    analyses: list[PeriodAnalysis],
    model: str = "qwen3-coder-64k",
) -> str:
    """
    Analyze how communication style evolved across periods.
    
    Args:
        analyses: List of period analyses in chronological order
        model: Ollama model to use
    
    Returns:
        Evolution analysis as markdown
    """
    if not HAS_OLLAMA:
        raise ImportError("ollama package required: pip install ollama")
    
    if len(analyses) < 2:
        return "Not enough periods to analyze evolution."
    
    # Build summary of each period
    summaries = []
    for a in analyses:
        summaries.append(f"""
### {a.period} ({a.language})
- Messages: {a.message_count}
- Formality: {a.formality_score}/10
- Avg length: {a.avg_length:.0f} chars
- Emoji rate: {a.emoji_rate:.2f}
- Tone: {a.tone_description}
- Key traits: {', '.join(a.key_characteristics[:3])}
""")
    
    prompt = f"""Analyze how this person's communication style evolved over time.

Period summaries:
{"".join(summaries)}

Provide an evolution analysis:

## 1. OVERALL TRAJECTORY
How has their style changed from earliest to latest period?

## 2. FORMALITY TREND
Are they becoming more or less formal? Why might this be?

## 3. NOTABLE CHANGES
What are the most significant changes between periods?

## 4. CONSISTENCY
What aspects of their style have remained consistent?

## 5. PREDICTIONS
Based on the trajectory, how might their style continue to evolve?

Be specific and reference the data from each period.
"""

    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={
            'num_ctx': 16000,
            'temperature': 0.1,
        }
    )
    
    return response['response']


def _collect_grounded_phrases(analyses: list[PeriodAnalysis]) -> list[str]:
    """Collect unique phrases from analyses (recent-weighted) for grounded examples."""
    seen = set()
    phrases = []
    current_year = datetime.now().year
    for a in reversed(analyses):  # Recent first
        weight = get_period_weight(a.period, current_year)
        for p in a.common_phrases:
            key = p.strip().lower()[:50]
            if key and key not in seen and len(p) > 2:
                seen.add(key)
                phrases.append(p)
                if len(phrases) >= 30:
                    return phrases
    return phrases


def generate_weighted_master_guide(
    analyses: list[PeriodAnalysis],
    evolution_analysis: str,
    model: str = "qwen3-coder-64k",
) -> str:
    """
    Generate a master style guide using two-pass approach:
    Pass 1: Extract raw rules (grounded in actual phrases)
    Pass 2: Consolidate, deduplicate, validate
    """
    if not HAS_OLLAMA:
        raise ImportError("ollama package required: pip install ollama")
    
    current_year = datetime.now().year
    grounded_phrases = _collect_grounded_phrases(analyses)
    phrases_block = "\n".join(f'- "{p}"' for p in grounded_phrases[:25])
    
    weighted_summaries = []
    for a in analyses:
        weight = get_period_weight(a.period, current_year)
        weighted_summaries.append(f"""
### {a.period} (weight: {weight:.0%})
- Formality: {a.formality_score}/10
- Tone: {a.tone_description}
- Key traits: {', '.join(a.key_characteristics[:5])}
- Common phrases: {', '.join(a.common_phrases[:5])}
""")
    
    # Pass 1: Extract with strict grounding
    pass1_prompt = f"""Create a communication style guide based on this person's writing patterns.

PHRASES you MAY use as examples (quote exactly—do not invent):
{phrases_block}

Period analyses:
{"".join(weighted_summaries)}

Evolution context:
{evolution_analysis[:1500]}

Generate a MASTER STYLE GUIDE. CRITICAL RULES:
1. Every example in DO's and DON'Ts MUST be a direct quote from the phrases above or clearly derived from the period analyses. No invented examples.
2. Each DO rule must be DISTINCT—if two rules say the same thing, merge them.
3. Each DON'T rule must be DISTINCT—same principle.
4. Aim for 8-10 DO's and 8-10 DON'Ts. Quality over quantity.

Structure:
## 1. CURRENT VOICE (Who They Are Now)
## 2. DO's - Rules for AI to Follow (with real examples)
## 3. DON'Ts - What to Avoid (with real counter-examples)
## 4. LANGUAGE-SPECIFIC NOTES
## 5. EXAMPLE TRANSFORMATIONS (use phrases from the list above)
"""

    response1 = ollama.generate(
        model=model,
        prompt=pass1_prompt,
        options={'num_ctx': 24000, 'temperature': 0.15}
    )
    raw_guide = response1['response']
    
    # Pass 2: Consolidate and validate
    pass2_prompt = f"""You have a draft style guide. Your task: consolidate and validate it.

GROUNDED PHRASES (these are real—use only these for examples):
{phrases_block}

DRAFT GUIDE:
{raw_guide}

REVISION RULES:
1. Merge any DO rules that convey the same principle (e.g., "short messages" + "be concise" = one rule).
2. Merge any DON'T rules that overlap.
3. Replace any invented example with a phrase from the GROUNDED PHRASES list. If no match exists, remove the example.
4. Ensure each rule is actionable and distinct.
5. Remove redundancy—every rule should add new information.
6. Keep the same structure (CURRENT VOICE, DO's, DON'Ts, LANGUAGE-SPECIFIC, TRANSFORMATIONS).
7. Final output: 8-10 DO's, 8-10 DON'Ts.

Output the revised, consolidated guide. No preamble.
"""

    response2 = ollama.generate(
        model=model,
        prompt=pass2_prompt,
        options={'num_ctx': 24000, 'temperature': 0.1}
    )
    
    return response2['response']


def generate_human_summary(
    analyses: list[PeriodAnalysis],
    evolution_analysis: str,
) -> str:
    """
    Generate a short, human-readable summary.
    
    Args:
        analyses: List of period analyses
        evolution_analysis: The evolution analysis text
    
    Returns:
        Human-friendly summary as markdown
    """
    if not analyses:
        return "# No Data\n\nNo communication data available for analysis."
    
    # Get most recent analysis
    recent = analyses[-1] if analyses else None
    
    # Calculate averages
    avg_formality = sum(a.formality_score for a in analyses) / len(analyses)
    avg_length = sum(a.avg_length for a in analyses) / len(analyses)
    avg_emoji = sum(a.emoji_rate for a in analyses) / len(analyses)
    
    # Determine trends
    if len(analyses) >= 2:
        formality_trend = analyses[-1].formality_score - analyses[0].formality_score
        if formality_trend > 1:
            formality_direction = "becoming more formal"
        elif formality_trend < -1:
            formality_direction = "becoming more casual"
        else:
            formality_direction = "staying consistent"
    else:
        formality_direction = "not enough data for trend"
    
    summary = f"""# Your Communication Style - Summary

*Generated: {datetime.now().strftime('%Y-%m-%d')}*
*Based on {sum(a.message_count for a in analyses):,} messages across {len(analyses)} time periods*

---

## Who You Are Now ({recent.period if recent else 'N/A'})

- **Formality**: {recent.formality_score if recent else 'N/A'}/10
- **Average message length**: {(recent.avg_length if recent else 0):.0f} characters
- **Emoji usage**: {(recent.emoji_rate if recent else 0):.2f} per message
- **Tone**: {recent.tone_description if recent else 'Unknown'}

## Key Characteristics

{chr(10).join(f'- {c}' for c in (recent.key_characteristics[:5] if recent else []))}

## How You've Changed

- **Overall**: {formality_direction}
- **Average formality**: {avg_formality:.1f}/10
- **Average message length**: {avg_length:.0f} chars

## Common Phrases You Use

{chr(10).join(f'- "{p}"' for p in (recent.common_phrases[:5] if recent else []))}

## Quick Rules for AI

1. Match formality level: {recent.formality_score if recent else 5}/10
2. Keep messages around {(recent.avg_length if recent else 100):.0f} characters
3. {'Use emojis sparingly' if (recent and recent.emoji_rate < 0.5) else 'Emojis are acceptable'}
4. Be direct and get to the point
5. Match the {recent.tone_description.lower() if recent else 'neutral'} tone

---

*For detailed rules, see master-style-guide.md*
"""
    
    return summary


def run_full_analysis(
    batches: list[TimeBatch],
    output_dir: Path,
    languages: list[str] = ["hebrew", "english"],
    model: str = "qwen3-coder-64k",
    progress_callback: Optional[callable] = None,
    style_collection=None,
) -> dict:
    """
    Run complete longitudinal analysis and save all outputs.
    
    Args:
        batches: List of time batches
        output_dir: Directory to save output files
        languages: Languages to analyze
        model: Ollama model to use
        progress_callback: Optional callback for progress updates
    
    Returns:
        Dictionary with paths to generated files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    per_period_dir = output_dir / "per-period"
    per_period_dir.mkdir(exist_ok=True)
    
    all_analyses = []
    
    # Analyze each batch
    for i, batch in enumerate(batches):
        if progress_callback:
            progress_callback(f"Analyzing {batch.period}...", i, len(batches))
        
        for lang in languages:
            # Check if there are messages in this language
            if lang == "hebrew" and not batch.hebrew_messages:
                continue
            if lang == "english" and not batch.english_messages:
                continue
            
            analysis = analyze_batch_with_llm(
                batch, language=lang, model=model, style_collection=style_collection
            )
            all_analyses.append(analysis)
            
            # Save individual period analysis
            period_file = per_period_dir / f"{batch.period}-{lang}-style.md"
            with open(period_file, 'w') as f:
                f.write(f"# {batch.period} - {lang.title()} Style\n\n")
                f.write(analysis.raw_analysis)
    
    if progress_callback:
        progress_callback("Analyzing evolution...", len(batches), len(batches))
    
    # Analyze evolution
    evolution = analyze_evolution(all_analyses, model=model)
    evolution_file = output_dir / "evolution-analysis.md"
    with open(evolution_file, 'w') as f:
        f.write("# Communication Style Evolution\n\n")
        f.write(evolution)
    
    if progress_callback:
        progress_callback("Generating master guide...", len(batches), len(batches))
    
    # Generate master guide
    master_guide = generate_weighted_master_guide(all_analyses, evolution, model=model)
    master_file = output_dir / "master-style-guide.md"
    with open(master_file, 'w') as f:
        f.write("# Master Communication Style Guide\n\n")
        f.write(master_guide)
    
    # Generate human summary
    human_summary = generate_human_summary(all_analyses, evolution)
    summary_file = output_dir / "human-summary.md"
    with open(summary_file, 'w') as f:
        f.write(human_summary)
    
    # Save raw analysis data
    data_file = output_dir / "analysis-data.json"
    with open(data_file, 'w') as f:
        json.dump({
            "analyses": [a.to_dict() for a in all_analyses],
            "evolution": evolution,
            "generated_at": datetime.now().isoformat(),
        }, f, indent=2)
    
    return {
        "per_period_dir": str(per_period_dir),
        "evolution_file": str(evolution_file),
        "master_file": str(master_file),
        "summary_file": str(summary_file),
        "data_file": str(data_file),
    }
