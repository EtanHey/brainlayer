"""Pipeline stages for processing Claude Code conversations."""

from .extract import extract_system_prompts
from .classify import classify_content
from .chunk import chunk_content
from .embed import embed_chunks
from .index import index_to_chromadb
from .extract_markdown import (
    find_markdown_files,
    parse_markdown,
    classify_by_path,
    extract_markdown_content,
)
from .semantic_style import (
    SemanticStyleAnalyzer,
    SemanticStyleAnalysis,
    TopicCluster,
    analyze_semantic_style,
)

__all__ = [
    "extract_system_prompts",
    "classify_content",
    "chunk_content",
    "embed_chunks",
    "index_to_chromadb",
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
]
