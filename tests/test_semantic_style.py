"""Tests for semantic style analysis module."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

# Skip all tests if dependencies not available
pytest.importorskip("sentence_transformers")
pytest.importorskip("sklearn")


from brainlayer.pipeline.semantic_style import (
    TOPIC_SEEDS,
    SemanticStyleAnalysis,
    SemanticStyleAnalyzer,
    TopicCluster,
    analyze_semantic_style,
)


class TestTopicSeeds:
    """Test topic seed definitions."""

    def test_all_topics_have_seeds(self):
        """Each topic should have multiple seed phrases."""
        expected_topics = {"technical", "casual", "professional", "emotional", "explanatory"}
        assert set(TOPIC_SEEDS.keys()) == expected_topics

    def test_seeds_are_non_empty(self):
        """Each topic should have at least 3 seed phrases."""
        for topic, seeds in TOPIC_SEEDS.items():
            assert len(seeds) >= 3, f"Topic '{topic}' has too few seeds"

    def test_seeds_are_strings(self):
        """All seeds should be strings."""
        for topic, seeds in TOPIC_SEEDS.items():
            for seed in seeds:
                assert isinstance(seed, str), f"Seed in '{topic}' is not a string"


class TestClusterStyleAnalysis:
    """Test style analysis within clusters (no embedding required)."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with mocked model."""
        with patch.object(SemanticStyleAnalyzer, "model", new_callable=lambda: Mock()):
            return SemanticStyleAnalyzer()

    def test_empty_messages_returns_empty(self, analyzer):
        """Empty message list should return empty dict."""
        result = analyzer.analyze_cluster_style([])
        assert result == {}

    def test_length_calculation(self, analyzer):
        """Average length should be calculated correctly."""
        messages = ["hi", "hello", "hey there"]  # lengths: 2, 5, 9
        result = analyzer.analyze_cluster_style(messages)
        expected_avg = (2 + 5 + 9) / 3
        assert abs(result["avg_length"] - expected_avg) < 0.01

    def test_emoji_rate(self, analyzer):
        """Emoji rate should count emoji per message."""
        messages = [
            "hello 😀",  # 1 emoji
            "hi 👋 wow 🎉",  # 2 separate emoji
            "plain text",  # 0 emoji
        ]
        result = analyzer.analyze_cluster_style(messages)
        # Regex finds emoji groups: 1 + 2 + 0 = 3 emoji in 3 messages = 1.0
        expected_rate = 3 / 3
        assert abs(result["emoji_rate"] - expected_rate) < 0.1

    def test_question_rate(self, analyzer):
        """Question rate should count questions per message."""
        messages = [
            "what is this?",
            "how are you?",
            "nice day",
        ]
        result = analyzer.analyze_cluster_style(messages)
        # 2 questions in 3 messages = 0.67
        assert result["question_rate"] == pytest.approx(2 / 3, rel=0.01)

    def test_exclamation_rate(self, analyzer):
        """Exclamation rate should count exclamations per message."""
        messages = [
            "wow!",
            "amazing!!",  # 2 exclamations
            "ok",
        ]
        result = analyzer.analyze_cluster_style(messages)
        # 3 exclamations in 3 messages = 1.0
        assert result["exclamation_rate"] == pytest.approx(3 / 3, rel=0.01)

    def test_formality_casual_markers(self, analyzer):
        """Messages with casual markers should have lower formality."""
        casual_messages = ["lol that's funny", "haha yes", "omg btw"]
        formal_messages = ["Please review this", "Thank you kindly", "Best regards"]

        casual_result = analyzer.analyze_cluster_style(casual_messages)
        formal_result = analyzer.analyze_cluster_style(formal_messages)

        assert casual_result["formality"] < formal_result["formality"]

    def test_hebrew_detection(self, analyzer):
        """Hebrew messages should be detected in language mix."""
        messages = [
            "שלום מה נשמע",  # Hebrew
            "hello there",  # English
            "היי what's up",  # Mixed
        ]
        result = analyzer.analyze_cluster_style(messages)

        assert result["language_mix"]["hebrew"] > 0
        assert result["language_mix"]["english"] > 0

    def test_common_phrases_extraction(self, analyzer):
        """Common phrases should be extracted from repeated patterns."""
        messages = [
            "I think this is great",
            "I think this could work",
            "I think this is the way",
            "I think this makes sense",
            "not sure about that",
        ]
        result = analyzer.analyze_cluster_style(messages)

        # "I think" or "think this" should appear as common phrase
        assert len(result["common_phrases"]) > 0


class TestTopicAssignment:
    """Test topic assignment (requires mocked embeddings)."""

    def test_assignment_returns_all_topics(self):
        """Assignment should return dict with all topic keys plus 'other'."""
        with patch.object(SemanticStyleAnalyzer, "model") as mock_model:
            # Mock the encoder
            mock_model.encode.return_value = np.random.randn(5, 1024)

            analyzer = SemanticStyleAnalyzer()
            messages = ["msg1", "msg2", "msg3", "msg4", "msg5"]
            embeddings = np.random.randn(5, 1024)

            # Need to also mock the seed embeddings computation
            with patch.object(analyzer, "_get_topic_seed_embeddings") as mock_seeds:
                mock_seeds.return_value = {topic: np.random.randn(1024) for topic in TOPIC_SEEDS.keys()}
                result = analyzer.assign_topics(messages, embeddings)

        expected_keys = set(TOPIC_SEEDS.keys()) | {"other"}
        assert set(result.keys()) == expected_keys

    def test_assignment_indices_valid(self):
        """All assigned indices should be valid message indices."""
        with patch.object(SemanticStyleAnalyzer, "model") as mock_model:
            mock_model.encode.return_value = np.random.randn(10, 1024)

            analyzer = SemanticStyleAnalyzer()
            messages = [f"message {i}" for i in range(10)]
            embeddings = np.random.randn(10, 1024)

            with patch.object(analyzer, "_get_topic_seed_embeddings") as mock_seeds:
                mock_seeds.return_value = {topic: np.random.randn(1024) for topic in TOPIC_SEEDS.keys()}
                result = analyzer.assign_topics(messages, embeddings)

        all_indices = []
        for indices in result.values():
            all_indices.extend(indices)

        # Each message should be assigned exactly once
        assert sorted(all_indices) == list(range(10))


class TestSemanticStyleAnalysis:
    """Test the SemanticStyleAnalysis dataclass."""

    def test_default_values(self):
        """Analysis should have sensible defaults."""
        analysis = SemanticStyleAnalysis()
        assert analysis.topic_clusters == {}
        assert analysis.cross_topic_insights == []
        assert analysis.style_rules_markdown == ""

    def test_topic_cluster_default_values(self):
        """TopicCluster should have sensible defaults."""
        cluster = TopicCluster(name="test")
        assert cluster.name == "test"
        assert cluster.messages == []
        assert cluster.formality == 0.5
        assert cluster.emoji_rate == 0.0


class TestInsightGeneration:
    """Test cross-topic insight generation."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with mocked model."""
        with patch.object(SemanticStyleAnalyzer, "model"):
            return SemanticStyleAnalyzer()

    def test_formality_comparison_insight(self, analyzer):
        """Should generate insight about formality differences."""
        clusters = {
            "casual": TopicCluster(name="casual", formality=0.2),
            "professional": TopicCluster(name="professional", formality=0.7),
        }
        insights = analyzer._generate_insights(clusters)

        # Should mention both casual and professional
        combined = " ".join(insights).lower()
        assert "casual" in combined or "professional" in combined

    def test_emoji_usage_insight(self, analyzer):
        """Should generate insight about emoji usage."""
        clusters = {
            "casual": TopicCluster(name="casual", emoji_rate=0.5),
            "technical": TopicCluster(name="technical", emoji_rate=0.01),
        }
        insights = analyzer._generate_insights(clusters)

        combined = " ".join(insights).lower()
        assert "emoji" in combined

    def test_single_cluster_no_comparison(self, analyzer):
        """Single cluster should not generate comparison insights."""
        clusters = {
            "casual": TopicCluster(name="casual", formality=0.2),
        }
        insights = analyzer._generate_insights(clusters)

        # With only one cluster, no meaningful comparisons
        assert len(insights) == 0


class TestMarkdownGeneration:
    """Test markdown output generation."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with mocked model."""
        with patch.object(SemanticStyleAnalyzer, "model"):
            return SemanticStyleAnalyzer()

    def test_markdown_has_header(self, analyzer):
        """Generated markdown should have main header."""
        clusters = {
            "technical": TopicCluster(
                name="technical",
                messages=["msg1", "msg2"],
                formality=0.6,
                avg_length=150,
            ),
        }
        markdown = analyzer._generate_markdown(clusters, [])

        assert "# Your Writing Style" in markdown
        assert "## By Context" in markdown

    def test_markdown_includes_topic_section(self, analyzer):
        """Each topic should have its own section."""
        clusters = {
            "technical": TopicCluster(
                name="technical",
                messages=["msg1", "msg2"],
                formality=0.6,
                avg_length=150,
            ),
        }
        markdown = analyzer._generate_markdown(clusters, [])

        assert "### Technical" in markdown
        assert "formality" in markdown.lower()

    def test_markdown_includes_insights(self, analyzer):
        """Insights should be included in markdown."""
        clusters = {}
        insights = ["Test insight one", "Test insight two"]
        markdown = analyzer._generate_markdown(clusters, insights)

        assert "Test insight one" in markdown
        assert "Test insight two" in markdown


class TestIntegration:
    """Integration tests (requires real model - skipped by default)."""

    @pytest.mark.skip(reason="Requires model download, run manually")
    def test_full_analysis_pipeline(self):
        """Test complete analysis with real model."""
        messages = [
            # Technical
            "debugging this API endpoint",
            "the database query is slow",
            "let me fix this bug",
            "git push to main",
            "reviewing the pull request",
            # Casual
            "haha that's so funny",
            "what are you doing tonight",
            "see you later!",
            "good morning everyone",
            "lol nice",
            # More to meet min cluster size
        ] * 5  # Repeat to get enough messages

        analysis = analyze_semantic_style(messages, min_cluster_size=5)

        # Should have at least some topic clusters
        assert len(analysis.topic_clusters) > 0

        # Should have generated markdown
        assert len(analysis.style_rules_markdown) > 0
