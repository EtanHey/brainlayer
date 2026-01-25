"""Stage 3: Chunk content using AST-aware splitting for code."""

import re
from dataclasses import dataclass
from typing import Any

# tree-sitter for AST parsing
try:
    import tree_sitter_languages
    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False

from .classify import ClassifiedContent, ContentType, ContentValue


@dataclass
class Chunk:
    """A chunk of content ready for embedding."""
    content: str
    content_type: ContentType
    value: ContentValue
    metadata: dict[str, Any]
    char_count: int


# Target chunk size in characters (research suggests ~500 tokens ≈ 2000 chars)
TARGET_CHUNK_SIZE = 2000
MIN_CHUNK_SIZE = 200
MAX_CHUNK_SIZE = 4000


def chunk_content(classified: ClassifiedContent) -> list[Chunk]:
    """
    Chunk classified content appropriately based on type.

    - Code: AST-aware chunking with tree-sitter
    - Stack traces: Never split (preserve exact)
    - Conversation: Turn-based with overlap
    - Large tool outputs: Observation masking or summarization marker
    """
    content = classified.content
    content_type = classified.content_type

    # Never split stack traces
    if content_type == ContentType.STACK_TRACE:
        return [Chunk(
            content=content,
            content_type=content_type,
            value=classified.value,
            metadata=classified.metadata,
            char_count=len(content)
        )]

    # AI-generated code: AST-aware chunking
    if content_type == ContentType.AI_CODE:
        return _chunk_code(classified)

    # Large tool outputs: Use observation masking
    if content_type in (ContentType.FILE_READ, ContentType.BUILD_LOG, ContentType.DIRECTORY_LISTING):
        if len(content) > MAX_CHUNK_SIZE:
            return _mask_large_output(classified)

    # Default: simple character-based chunking with overlap
    return _chunk_text(classified)


def _chunk_code(classified: ClassifiedContent) -> list[Chunk]:
    """
    AST-aware chunking for code content.

    Uses tree-sitter to identify semantic boundaries (functions, classes).
    Falls back to line-based chunking if tree-sitter unavailable.
    """
    content = classified.content
    chunks = []

    # Extract code blocks from markdown
    code_blocks = _extract_code_blocks(content)

    for lang, code in code_blocks:
        if HAS_TREE_SITTER and lang:
            # Try AST-based chunking
            ast_chunks = _ast_chunk(code, lang)
            if ast_chunks:
                for chunk_text in ast_chunks:
                    chunks.append(Chunk(
                        content=chunk_text,
                        content_type=classified.content_type,
                        value=classified.value,
                        metadata={**classified.metadata, "language": lang},
                        char_count=len(chunk_text)
                    ))
                continue

        # Fallback: line-based chunking
        line_chunks = _line_based_chunk(code)
        for chunk_text in line_chunks:
            chunks.append(Chunk(
                content=chunk_text,
                content_type=classified.content_type,
                value=classified.value,
                metadata={**classified.metadata, "language": lang or "unknown"},
                char_count=len(chunk_text)
            ))

    # Also include any text outside code blocks
    non_code = _extract_non_code(content)
    if non_code.strip():
        chunks.append(Chunk(
            content=non_code,
            content_type=ContentType.ASSISTANT_TEXT,
            value=ContentValue.MEDIUM,
            metadata=classified.metadata,
            char_count=len(non_code)
        ))

    return chunks if chunks else [Chunk(
        content=content,
        content_type=classified.content_type,
        value=classified.value,
        metadata=classified.metadata,
        char_count=len(content)
    )]


def _ast_chunk(code: str, language: str) -> list[str] | None:
    """Use tree-sitter to chunk code at semantic boundaries."""
    if not HAS_TREE_SITTER:
        return None

    try:
        parser = tree_sitter_languages.get_parser(language)
        tree = parser.parse(code.encode())

        chunks = []
        current_chunk = []
        current_size = 0

        def traverse(node):
            nonlocal current_chunk, current_size

            # Top-level definitions are natural chunk boundaries
            if node.type in (
                "function_definition", "function_declaration",
                "class_definition", "class_declaration",
                "method_definition", "method_declaration",
                "interface_declaration", "type_alias_declaration"
            ):
                text = code[node.start_byte:node.end_byte]

                # If adding this would exceed max, start new chunk
                if current_size + len(text) > MAX_CHUNK_SIZE and current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0

                current_chunk.append(text)
                current_size += len(text)
                return  # Don't traverse children

            # Recurse into children
            for child in node.children:
                traverse(child)

        traverse(tree.root_node)

        # Don't forget the last chunk
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks if chunks else None

    except Exception:
        return None


def _line_based_chunk(text: str) -> list[str]:
    """Simple line-based chunking with overlap."""
    lines = text.split("\n")
    chunks = []
    current_chunk = []
    current_size = 0

    for line in lines:
        line_size = len(line) + 1  # +1 for newline

        if current_size + line_size > TARGET_CHUNK_SIZE and current_chunk:
            chunks.append("\n".join(current_chunk))
            # Keep last 2 lines for overlap
            current_chunk = current_chunk[-2:] if len(current_chunk) > 2 else []
            current_size = sum(len(l) + 1 for l in current_chunk)

        current_chunk.append(line)
        current_size += line_size

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def _chunk_text(classified: ClassifiedContent) -> list[Chunk]:
    """Simple text chunking for non-code content."""
    content = classified.content

    if len(content) <= TARGET_CHUNK_SIZE:
        return [Chunk(
            content=content,
            content_type=classified.content_type,
            value=classified.value,
            metadata=classified.metadata,
            char_count=len(content)
        )]

    # Chunk by paragraphs first
    paragraphs = content.split("\n\n")
    chunks = []
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para_size = len(para) + 2  # +2 for paragraph separator

        if current_size + para_size > TARGET_CHUNK_SIZE and current_chunk:
            chunks.append(Chunk(
                content="\n\n".join(current_chunk),
                content_type=classified.content_type,
                value=classified.value,
                metadata=classified.metadata,
                char_count=current_size
            ))
            current_chunk = []
            current_size = 0

        current_chunk.append(para)
        current_size += para_size

    if current_chunk:
        chunks.append(Chunk(
            content="\n\n".join(current_chunk),
            content_type=classified.content_type,
            value=classified.value,
            metadata=classified.metadata,
            char_count=current_size
        ))

    return chunks


def _mask_large_output(classified: ClassifiedContent) -> list[Chunk]:
    """
    Apply observation masking to large tool outputs.

    Research shows this often performs as well as LLM summarization.
    """
    content = classified.content
    line_count = content.count("\n") + 1

    # Create a masked summary
    first_lines = "\n".join(content.split("\n")[:5])
    last_lines = "\n".join(content.split("\n")[-3:])

    elided_count = max(0, line_count - 8)
    masked = f"{first_lines}\n\n[... {elided_count} lines elided ...]\n\n{last_lines}"

    return [Chunk(
        content=masked,
        content_type=classified.content_type,
        value=ContentValue.LOW,
        metadata={**classified.metadata, "masked": True, "original_lines": line_count},
        char_count=len(masked)
    )]


def _extract_code_blocks(text: str) -> list[tuple[str | None, str]]:
    """Extract code blocks from markdown-formatted text."""
    pattern = r"```(\w+)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    return [(lang or None, code.strip()) for lang, code in matches]


def _extract_non_code(text: str) -> str:
    """Extract text that's not inside code blocks."""
    return re.sub(r"```\w*\n.*?```", "", text, flags=re.DOTALL).strip()
