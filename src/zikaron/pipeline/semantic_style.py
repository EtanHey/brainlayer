"""Semantic Style Analysis - Topic-based style pattern extraction.

Uses bge-large embeddings to cluster messages by TOPIC (what you write about),
then analyzes STYLE within each topic cluster (how you write in that context).

This differs from style_embed.py which uses StyleDistance for pure style clustering.
Here we want: "When talking about technical topics, you write like THIS"

Key concepts:
- Topic clusters: technical, casual chat, professional, emotional
- Per-topic style metrics: formality, length, emoji, phrases
- Cross-context comparisons: "more formal when discussing work"
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import re

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# bge-large for semantic/topic clustering (1024 dims, good multilingual)
SEMANTIC_MODEL = "BAAI/bge-large-en-v1.5"
MAX_CHARS = 2000

# Predefined topic seeds for guided clustering
TOPIC_SEEDS = {
    "technical": [
        "debugging the code",
        "implementing the feature",
        "API endpoint",
        "database query",
        "git commit",
        "pull request review",
    ],
    "casual": [
        "haha that's funny",
        "what are you doing",
        "see you later",
        "good morning",
        "how was your day",
    ],
    "professional": [
        "meeting scheduled",
        "project deadline",
        "quarterly review",
        "client presentation",
        "follow up on the proposal",
    ],
    "emotional": [
        "I'm so excited",
        "that's frustrating",
        "really happy about",
        "worried about",
        "love this",
    ],
    "explanatory": [
        "let me explain",
        "the reason is",
        "basically what happens is",
        "think of it like",
        "for example",
    ],
}


@dataclass
class TopicCluster:
    """A cluster of messages grouped by topic."""
    name: str
    messages: list[str] = field(default_factory=list)
    centroid: Optional[list[float]] = None

    # Style metrics for this topic
    message_count: int = 0
    avg_length: float = 0.0
    formality: float = 0.5
    emoji_rate: float = 0.0
    question_rate: float = 0.0
    exclamation_rate: float = 0.0
    common_phrases: list[str] = field(default_factory=list)
    language_mix: dict[str, float] = field(default_factory=dict)


@dataclass
class SemanticStyleAnalysis:
    """Complete semantic style analysis result."""
    topic_clusters: dict[str, TopicCluster] = field(default_factory=dict)
    cross_topic_insights: list[str] = field(default_factory=list)
    style_rules_markdown: str = ""


class SemanticStyleAnalyzer:
    """Analyze writing style patterns by topic/context."""

    def __init__(self, model_name: str = SEMANTIC_MODEL):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers required. Install: pip install sentence-transformers"
            )
        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn required. Install: pip install scikit-learn"
            )
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._topic_seed_embeddings: Optional[dict[str, np.ndarray]] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            print(f"[SemanticStyle] Loading {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _get_topic_seed_embeddings(self) -> dict[str, np.ndarray]:
        """Get/compute embeddings for topic seed phrases."""
        if self._topic_seed_embeddings is None:
            self._topic_seed_embeddings = {}
            for topic, seeds in TOPIC_SEEDS.items():
                embeddings = self.model.encode(seeds, convert_to_numpy=True)
                # Average the seed embeddings to get topic centroid
                self._topic_seed_embeddings[topic] = np.mean(embeddings, axis=0)
        return self._topic_seed_embeddings

    def embed_messages(
        self,
        messages: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Embed messages using bge-large for topic clustering."""
        # Truncate long messages
        truncated = [m[:MAX_CHARS] for m in messages]

        embeddings = self.model.encode(
            truncated,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings

    def assign_topics(
        self,
        messages: list[str],
        embeddings: np.ndarray,
        threshold: float = 0.3,
    ) -> dict[str, list[int]]:
        """Assign messages to topics based on similarity to seed centroids.

        Args:
            messages: List of message texts
            embeddings: Message embeddings from embed_messages()
            threshold: Minimum cosine similarity (0-1) to assign to a topic.
                       Messages below threshold go to "other".

        Returns:
            Dict mapping topic name to list of message indices
        """
        topic_seeds = self._get_topic_seed_embeddings()
        assignments: dict[str, list[int]] = {topic: [] for topic in topic_seeds}
        assignments["other"] = []

        for i, emb in enumerate(embeddings):
            best_topic = "other"
            best_sim = threshold

            for topic, seed_emb in topic_seeds.items():
                sim = cosine_similarity([emb], [seed_emb])[0][0]
                if sim > best_sim:
                    best_sim = sim
                    best_topic = topic

            assignments[best_topic].append(i)

        return assignments

    def analyze_cluster_style(self, messages: list[str]) -> dict[str, Any]:
        """Analyze style patterns within a cluster of messages."""
        if not messages:
            return {}

        # Length analysis
        lengths = [len(m) for m in messages]
        avg_length = sum(lengths) / len(lengths)

        # Formality indicators
        informal_markers = [
            r'\blol\b', r'\bhaha\b', r'\bחח\b', r'\bomg\b', r'\bbtw\b',
            r'\bכן\b', r'\bלא\b', r'\bוואלה\b', r'\bסבבה\b', r'\bיאללה\b',
            r'!!+', r'\?\?+', r'\.\.\.+',
        ]
        formal_markers = [
            r'\bplease\b', r'\bkindly\b', r'\bregards\b', r'\bthank you\b',
            r'\bבבקשה\b', r'\bתודה\b', r'\bלהלן\b',
        ]

        informal_count = 0
        formal_count = 0
        for msg in messages:
            msg_lower = msg.lower()
            informal_count += sum(1 for p in informal_markers if re.search(p, msg_lower))
            formal_count += sum(1 for p in formal_markers if re.search(p, msg_lower))

        informal_ratio = min(1.0, informal_count / max(len(messages), 1))
        formal_ratio = min(1.0, formal_count / max(len(messages), 1))
        formality = 0.5 - (informal_ratio * 0.3) + (formal_ratio * 0.3)
        formality = max(0.1, min(0.9, formality))

        # Emoji rate
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "]+",
            flags=re.UNICODE
        )
        emoji_count = sum(len(emoji_pattern.findall(m)) for m in messages)
        emoji_rate = emoji_count / len(messages)

        # Punctuation rates
        question_count = sum(m.count('?') for m in messages)
        exclamation_count = sum(m.count('!') for m in messages)
        question_rate = question_count / len(messages)
        exclamation_rate = exclamation_count / len(messages)

        # Language detection (simple Hebrew/English check)
        hebrew_pattern = re.compile(r'[\u0590-\u05FF]')
        english_count = 0
        hebrew_count = 0
        for msg in messages:
            has_hebrew = bool(hebrew_pattern.search(msg))
            has_english = bool(re.search(r'[a-zA-Z]', msg))
            if has_hebrew:
                hebrew_count += 1
            if has_english:
                english_count += 1

        language_mix = {
            "hebrew": hebrew_count / len(messages),
            "english": english_count / len(messages),
        }

        # Common phrases (bigrams and trigrams)
        words = []
        for msg in messages:
            words.extend(re.findall(r'\b\w+\b', msg.lower()))

        # Guard against short word lists
        bigrams = (
            [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            if len(words) >= 2 else []
        )
        trigrams = (
            [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
            if len(words) >= 3 else []
        )

        phrase_counts = Counter(bigrams + trigrams)
        # Filter to meaningful phrases (appear 3+ times, not just stopwords)
        min_phrase_count = 3
        common_phrases = [
            phrase for phrase, count in phrase_counts.most_common(20)
            if count >= min_phrase_count and len(phrase.split()) > 1
        ][:10]

        return {
            "avg_length": avg_length,
            "formality": formality,
            "emoji_rate": emoji_rate,
            "question_rate": question_rate,
            "exclamation_rate": exclamation_rate,
            "language_mix": language_mix,
            "common_phrases": common_phrases,
            "message_count": len(messages),
        }

    def analyze(
        self,
        messages: list[str],
        min_cluster_size: int = 10,
    ) -> SemanticStyleAnalysis:
        """Run full semantic style analysis.

        Args:
            messages: List of message texts to analyze
            min_cluster_size: Minimum messages for a topic to be included

        Returns:
            SemanticStyleAnalysis with topic clusters and insights
        """
        print(f"[SemanticStyle] Analyzing {len(messages)} messages...")

        # Embed all messages
        print("[SemanticStyle] Computing embeddings...")
        embeddings = self.embed_messages(messages)

        # Assign to topics
        print("[SemanticStyle] Assigning topics...")
        topic_assignments = self.assign_topics(messages, embeddings)

        # Analyze each topic cluster
        print("[SemanticStyle] Analyzing topic clusters...")
        topic_clusters: dict[str, TopicCluster] = {}

        for topic, indices in topic_assignments.items():
            if len(indices) < min_cluster_size:
                continue

            cluster_messages = [messages[i] for i in indices]
            style = self.analyze_cluster_style(cluster_messages)

            cluster = TopicCluster(
                name=topic,
                messages=cluster_messages[:100],  # Keep sample for reference
                message_count=style.get("message_count", len(cluster_messages)),
                avg_length=style.get("avg_length", 0),
                formality=style.get("formality", 0.5),
                emoji_rate=style.get("emoji_rate", 0),
                question_rate=style.get("question_rate", 0),
                exclamation_rate=style.get("exclamation_rate", 0),
                common_phrases=style.get("common_phrases", []),
                language_mix=style.get("language_mix", {}),
            )
            topic_clusters[topic] = cluster
            print(f"  • {topic}: {len(indices)} messages, formality={cluster.formality:.2f}")

        # Generate cross-topic insights
        insights = self._generate_insights(topic_clusters)

        # Generate markdown rules
        markdown = self._generate_markdown(topic_clusters, insights)

        return SemanticStyleAnalysis(
            topic_clusters=topic_clusters,
            cross_topic_insights=insights,
            style_rules_markdown=markdown,
        )

    def _generate_insights(self, clusters: dict[str, TopicCluster]) -> list[str]:
        """Generate cross-topic insights by comparing clusters."""
        insights = []

        if len(clusters) < 2:
            return insights

        # Find most/least formal topic
        formalities = [(t, c.formality) for t, c in clusters.items()]
        formalities.sort(key=lambda x: x[1])

        if len(formalities) >= 2:
            most_casual = formalities[0]
            most_formal = formalities[-1]
            if most_formal[1] - most_casual[1] > 0.1:
                insights.append(
                    f"Most formal in '{most_formal[0]}' contexts ({most_formal[1]:.2f}), "
                    f"most casual in '{most_casual[0]}' ({most_casual[1]:.2f})"
                )

        # Find where you use most emoji
        emoji_rates = [(t, c.emoji_rate) for t, c in clusters.items()]
        emoji_rates.sort(key=lambda x: x[1], reverse=True)
        if emoji_rates[0][1] > 0.1:
            insights.append(
                f"Uses most emoji in '{emoji_rates[0][0]}' contexts ({emoji_rates[0][1]:.2f} per message)"
            )

        # Language switching patterns
        for topic, cluster in clusters.items():
            hebrew = cluster.language_mix.get("hebrew", 0)
            english = cluster.language_mix.get("english", 0)
            if hebrew > 0.5 and english > 0.5:
                insights.append(f"Frequently code-switches Hebrew/English in '{topic}' context")
            elif hebrew > 0.8:
                insights.append(f"Primarily Hebrew in '{topic}' context")

        # Message length patterns
        lengths = [(t, c.avg_length) for t, c in clusters.items()]
        lengths.sort(key=lambda x: x[1])
        if lengths[-1][1] > lengths[0][1] * 2:
            insights.append(
                f"Writes longest messages in '{lengths[-1][0]}' ({lengths[-1][1]:.0f} chars avg), "
                f"shortest in '{lengths[0][0]}' ({lengths[0][1]:.0f} chars)"
            )

        return insights

    def _generate_markdown(
        self,
        clusters: dict[str, TopicCluster],
        insights: list[str],
    ) -> str:
        """Generate markdown style rules from analysis."""
        lines = [
            "# Your Writing Style (Semantic Analysis)",
            "",
            "Generated from message clustering by topic.",
            "",
            "## Cross-Context Insights",
            "",
        ]

        for insight in insights:
            lines.append(f"- {insight}")

        lines.extend(["", "## By Context", ""])

        for topic, cluster in clusters.items():
            lines.append(f"### {topic.title()}")
            lines.append("")
            lines.append(f"- **Message count:** {cluster.message_count}")
            lines.append(f"- **Average length:** {cluster.avg_length:.0f} characters")
            lines.append(f"- **Formality:** {cluster.formality:.2f} (0=casual, 1=formal)")
            lines.append(f"- **Emoji rate:** {cluster.emoji_rate:.2f} per message")

            if cluster.language_mix:
                lang_str = ", ".join(
                    f"{lang}: {pct:.0%}"
                    for lang, pct in cluster.language_mix.items()
                )
                lines.append(f"- **Language mix:** {lang_str}")

            if cluster.common_phrases:
                phrases = ", ".join(f'"{p}"' for p in cluster.common_phrases[:5])
                lines.append(f"- **Common phrases:** {phrases}")

            lines.append("")

        lines.extend([
            "## For Cover Letters & Professional Outreach",
            "",
            "Based on your patterns:",
            "",
        ])

        # Add recommendations based on analysis
        if "professional" in clusters:
            prof = clusters["professional"]
            lines.append(f"- Use formality level ~{prof.formality:.2f} (matches your work context)")

        if "technical" in clusters:
            tech = clusters["technical"]
            lines.append(f"- Technical explanations avg {tech.avg_length:.0f} chars - keep similar length")
            if tech.common_phrases:
                lines.append(f"- You naturally use phrases like: {', '.join(tech.common_phrases[:3])}")

        if "emotional" in clusters:
            emo = clusters["emotional"]
            if emo.exclamation_rate > 0.3:
                lines.append("- You express enthusiasm with exclamation marks - use sparingly in professional context")

        lines.append("- Avoid Hebrew in English-only professional contexts")
        lines.append("")

        return "\n".join(lines)

    def save_analysis(
        self,
        analysis: SemanticStyleAnalysis,
        output_dir: Path,
    ) -> None:
        """Save analysis results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save markdown rules
        rules_path = output_dir / "semantic-style-rules.md"
        rules_path.write_text(analysis.style_rules_markdown)
        print(f"[SemanticStyle] Saved rules to {rules_path}")

        # Save JSON data for programmatic use
        data = {
            "topics": {
                name: {
                    "message_count": cluster.message_count,
                    "avg_length": cluster.avg_length,
                    "formality": cluster.formality,
                    "emoji_rate": cluster.emoji_rate,
                    "question_rate": cluster.question_rate,
                    "exclamation_rate": cluster.exclamation_rate,
                    "language_mix": cluster.language_mix,
                    "common_phrases": cluster.common_phrases,
                }
                for name, cluster in analysis.topic_clusters.items()
            },
            "insights": analysis.cross_topic_insights,
        }

        json_path = output_dir / "semantic-style-data.json"
        json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"[SemanticStyle] Saved data to {json_path}")


def analyze_semantic_style(
    messages: list[str],
    output_dir: Optional[Path] = None,
    min_cluster_size: int = 10,
) -> SemanticStyleAnalysis:
    """Convenience function for semantic style analysis.

    Args:
        messages: List of message texts
        output_dir: Optional directory to save results
        min_cluster_size: Minimum messages per topic cluster

    Returns:
        SemanticStyleAnalysis result
    """
    analyzer = SemanticStyleAnalyzer()
    analysis = analyzer.analyze(messages, min_cluster_size=min_cluster_size)

    if output_dir:
        analyzer.save_analysis(analysis, output_dir)

    return analysis
