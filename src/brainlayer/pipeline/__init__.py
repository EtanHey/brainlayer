"""Pipeline stages for processing Claude Code conversations."""

from .chunk import chunk_content
from .classify import classify_content
from .enrichment import build_external_prompt
from .extract import extract_system_prompts
from .extract_markdown import (
    classify_by_path,
    extract_markdown_content,
    find_markdown_files,
    parse_markdown,
)
from .sanitize import (
    Replacement,
    SanitizeConfig,
    Sanitizer,
    SanitizeResult,
)
from .semantic_style import (
    SemanticStyleAnalysis,
    SemanticStyleAnalyzer,
    TopicCluster,
    analyze_semantic_style,
)

__all__ = [
    "extract_system_prompts",
    "classify_content",
    "chunk_content",
    # Markdown extraction
    "find_markdown_files",
    "parse_markdown",
    "classify_by_path",
    "extract_markdown_content",
    # Semantic style analysis
    "SemanticStyleAnalyzer",
    "SemanticStyleAnalysis",
    "TopicCluster",
    "analyze_semantic_style",
    # PII sanitization
    "Sanitizer",
    "SanitizeConfig",
    "SanitizeResult",
    "Replacement",
    # External enrichment (sanitized)
    "build_external_prompt",
]
