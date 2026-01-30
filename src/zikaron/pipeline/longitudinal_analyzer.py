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
    max_samples: int = 200,
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
    
    # Sample messages for prompt
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
    
    prompt = f"""Analyze this person's communication style from {len(samples)} messages in the period {batch.period}.

Language focus: {language.upper()}

Sample messages:
{samples_text}

Provide a structured analysis:

## 1. FORMALITY SCORE (0-10)
Rate from 0 (very casual) to 10 (very formal). Provide the number and brief justification.

## 2. TONE DESCRIPTION
Describe the overall tone in 1-2 sentences.

## 3. KEY CHARACTERISTICS
List 3-5 key characteristics of their writing style.

## 4. COMMON PHRASES
List 5-10 phrases or expressions they frequently use (quote exactly from the messages).

## 5. COMMUNICATION PATTERNS
- How do they start messages?
- How do they make requests?
- How do they express emotions?
- Use of punctuation and capitalization

Be specific and cite examples from the messages.
"""

    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={
            'num_ctx': 16000,
            'temperature': 0.2,
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
            'temperature': 0.3,
        }
    )
    
    return response['response']


def generate_weighted_master_guide(
    analyses: list[PeriodAnalysis],
    evolution_analysis: str,
    model: str = "qwen3-coder-64k",
) -> str:
    """
    Generate a master style guide with recent periods weighted more heavily.
    
    Args:
        analyses: List of period analyses
        evolution_analysis: The evolution analysis text
        model: Ollama model to use
    
    Returns:
        Master style guide as markdown
    """
    if not HAS_OLLAMA:
        raise ImportError("ollama package required: pip install ollama")
    
    # Weight recent analyses more heavily
    current_year = datetime.now().year
    
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
    
    prompt = f"""Create a comprehensive communication style guide based on this person's writing patterns.

Recent periods are weighted more heavily (100% for current year, decreasing for older).

Period analyses:
{"".join(weighted_summaries)}

Evolution context:
{evolution_analysis[:2000]}

Generate a MASTER STYLE GUIDE with:

## 1. CURRENT VOICE (Who They Are Now)
Describe their current communication style based on most recent, heavily-weighted periods.

## 2. DO's - Rules for AI to Follow
List 10 specific, actionable rules. Be concrete with examples.

## 3. DON'Ts - What to Avoid
List 10 things AI should NOT do when writing for this person.

## 4. LANGUAGE-SPECIFIC NOTES
Any differences between Hebrew and English communication.

## 5. CONTEXT-SPECIFIC ADJUSTMENTS
How to adjust for different contexts (social media, professional, casual).

## 6. EXAMPLE TRANSFORMATIONS
Show 3 examples of:
- Generic/AI text
- Transformed to match this person's voice

Make this guide practical and directly usable by any AI assistant.
"""

    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={
            'num_ctx': 24000,
            'temperature': 0.3,
        }
    )
    
    return response['response']


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
- **Average message length**: {recent.avg_length:.0f if recent else 0} characters
- **Emoji usage**: {recent.emoji_rate:.2f if recent else 0} per message
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
2. Keep messages around {recent.avg_length:.0f if recent else 100} characters
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
            
            analysis = analyze_batch_with_llm(batch, language=lang, model=model)
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
