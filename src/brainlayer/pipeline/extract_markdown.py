"""Extract and classify markdown files for indexing."""

import re
from pathlib import Path
from typing import Iterator

from .classify import ClassifiedContent, ContentType, ContentValue


def find_markdown_files(
    root: Path, patterns: list[str] | None = None, exclude: list[str] | None = None
) -> Iterator[Path]:
    """
    Find markdown files matching glob patterns.

    Args:
        root: Root directory to search
        patterns: Glob patterns to match (default: ["**/*.md"])
        exclude: Directory names to exclude (default: common non-content dirs)

    Yields:
        Path objects for matching markdown files
    """
    if patterns is None:
        patterns = ["**/*.md"]
    if exclude is None:
        exclude = ["node_modules", ".git", "dist", "__pycache__", ".venv", "venv"]

    exclude_set = set(exclude)

    for pattern in patterns:
        for path in root.glob(pattern):
            # Skip excluded directories
            if any(part in exclude_set for part in path.parts):
                continue
            if path.is_file():
                yield path


def parse_markdown(file_path: Path) -> list[dict]:
    """
    Parse a markdown file into sections.

    Splits on h2 headers (## ) to create logical chunks.
    Each section includes header hierarchy for context.

    Returns:
        List of dicts with keys: content, header, parent_headers, tags
    """
    content = file_path.read_text(encoding="utf-8", errors="replace")
    sections = []

    # Extract frontmatter if present
    frontmatter = {}
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            # Simple YAML parsing for common keys
            for line in parts[1].strip().split("\n"):
                if ":" in line:
                    key, _, value = line.partition(":")
                    frontmatter[key.strip()] = value.strip().strip("\"'")
            content = parts[2]

    # Extract tags from content (#tag patterns not in code blocks)
    tags = set()
    for match in re.finditer(r"(?<!\S)#([a-zA-Z][a-zA-Z0-9_-]*)", content):
        tag = match.group(1).lower()
        # Skip common markdown/code patterns
        if tag not in ("include", "define", "ifdef", "endif", "pragma", "import"):
            tags.add(tag)

    # Split by h2 headers
    h1_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    h1_title = h1_match.group(1) if h1_match else file_path.stem

    # Find all h2 sections
    h2_pattern = r"^##\s+(.+)$"
    h2_positions = [(m.start(), m.group(1)) for m in re.finditer(h2_pattern, content, re.MULTILINE)]

    if not h2_positions:
        # No h2 headers - treat entire file as one section
        sections.append(
            {
                "content": content.strip(),
                "header": h1_title,
                "parent_headers": [],
                "tags": list(tags),
                "frontmatter": frontmatter,
            }
        )
    else:
        # Content before first h2
        if h2_positions[0][0] > 0:
            intro = content[: h2_positions[0][0]].strip()
            if intro and len(intro) > 50:  # Skip trivial intros
                sections.append(
                    {
                        "content": intro,
                        "header": h1_title,
                        "parent_headers": [],
                        "tags": list(tags),
                        "frontmatter": frontmatter,
                    }
                )

        # Each h2 section
        for i, (pos, header) in enumerate(h2_positions):
            end_pos = h2_positions[i + 1][0] if i + 1 < len(h2_positions) else len(content)
            section_content = content[pos:end_pos].strip()

            sections.append(
                {
                    "content": section_content,
                    "header": header,
                    "parent_headers": [h1_title],
                    "tags": list(tags),
                    "frontmatter": frontmatter,
                }
            )

    return sections


def classify_by_path(file_path: Path) -> tuple[ContentType, ContentValue]:
    """
    Classify a markdown file based on its path.

    Returns:
        Tuple of (ContentType, ContentValue) based on path patterns
    """
    path_str = str(file_path).lower()
    name = file_path.name.lower()

    # CLAUDE.md files are high-value project config
    if name == "claude.md":
        return ContentType.PROJECT_CONFIG, ContentValue.HIGH

    # Learnings are curated high-value content
    if "/learnings/" in path_str or "learnings/" in path_str:
        return ContentType.LEARNING, ContentValue.HIGH

    # Skills are high-value reference
    if "/skills/" in path_str or "skills/" in path_str:
        return ContentType.SKILL, ContentValue.HIGH

    # Research docs
    if "/research/" in path_str or "research/" in path_str:
        return ContentType.RESEARCH, ContentValue.HIGH

    # PRD archives - medium value
    if "/prd" in path_str or "prd-" in path_str or "prd_" in path_str:
        return ContentType.PRD_ARCHIVE, ContentValue.MEDIUM

    # Verification rounds - low value (voluminous)
    if "/verification" in path_str or "verification-" in path_str:
        return ContentType.VERIFICATION, ContentValue.LOW

    # Default: documentation
    return ContentType.DOCUMENTATION, ContentValue.MEDIUM


def extract_markdown_content(file_path: Path) -> list[ClassifiedContent]:
    """
    Extract and classify content from a markdown file.

    Combines parsing and classification into ready-to-chunk content.

    Returns:
        List of ClassifiedContent objects for each section
    """
    content_type, value = classify_by_path(file_path)
    sections = parse_markdown(file_path)

    results = []
    for section in sections:
        # Build context string with header hierarchy
        header_context = " > ".join(section["parent_headers"] + [section["header"]])

        results.append(
            ClassifiedContent(
                content=section["content"],
                content_type=content_type,
                value=value,
                metadata={
                    "source_file": str(file_path),
                    "header": section["header"],
                    "header_context": header_context,
                    "tags": section["tags"],
                    "frontmatter": section.get("frontmatter", {}),
                },
            )
        )

    return results
