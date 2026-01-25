"""Tests for markdown extraction pipeline."""

import tempfile
from pathlib import Path

import pytest

from zikaron.pipeline.extract_markdown import (
    find_markdown_files,
    parse_markdown,
    classify_by_path,
    extract_markdown_content,
)
from zikaron.pipeline.classify import ContentType, ContentValue


class TestFindMarkdownFiles:
    """Tests for find_markdown_files function."""

    def test_finds_md_files(self, tmp_path: Path):
        """Should find .md files in directory."""
        # Create test files
        (tmp_path / "readme.md").write_text("# Readme")
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "guide.md").write_text("# Guide")
        (tmp_path / "other.txt").write_text("not markdown")

        files = list(find_markdown_files(tmp_path))
        assert len(files) == 2
        names = {f.name for f in files}
        assert names == {"readme.md", "guide.md"}

    def test_excludes_directories(self, tmp_path: Path):
        """Should exclude specified directories."""
        (tmp_path / "docs.md").write_text("# Docs")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "pkg.md").write_text("# Package")

        files = list(find_markdown_files(tmp_path, exclude=["node_modules"]))
        assert len(files) == 1
        assert files[0].name == "docs.md"

    def test_custom_patterns(self, tmp_path: Path):
        """Should respect custom glob patterns."""
        (tmp_path / "readme.md").write_text("# Readme")
        (tmp_path / "CLAUDE.md").write_text("# Claude")

        # Only match CLAUDE.md
        files = list(find_markdown_files(tmp_path, patterns=["**/CLAUDE.md"]))
        assert len(files) == 1
        assert files[0].name == "CLAUDE.md"


class TestParseMarkdown:
    """Tests for parse_markdown function."""

    def test_parses_h2_sections(self, tmp_path: Path):
        """Should split on h2 headers."""
        content = """# Main Title

This is a longer introduction text that should be preserved as its own section
because it has more than 50 characters.

## Section 1

Content for section 1.

## Section 2

Content for section 2.
"""
        md_file = tmp_path / "test.md"
        md_file.write_text(content)

        sections = parse_markdown(md_file)
        assert len(sections) == 3  # Intro + 2 sections

        # Check section headers
        headers = [s["header"] for s in sections]
        assert "Main Title" in headers
        assert "Section 1" in headers
        assert "Section 2" in headers

    def test_extracts_tags(self, tmp_path: Path):
        """Should extract #tags from content."""
        content = """# Learning

Some text with #python and #debugging tags.
Also #best-practices here.
"""
        md_file = tmp_path / "learning.md"
        md_file.write_text(content)

        sections = parse_markdown(md_file)
        assert len(sections) == 1

        tags = sections[0]["tags"]
        assert "python" in tags
        assert "debugging" in tags
        assert "best-practices" in tags

    def test_handles_frontmatter(self, tmp_path: Path):
        """Should parse YAML frontmatter."""
        content = """---
title: Test Document
author: Test Author
---

# Content

Body text here.
"""
        md_file = tmp_path / "test.md"
        md_file.write_text(content)

        sections = parse_markdown(md_file)
        assert len(sections) == 1

        frontmatter = sections[0]["frontmatter"]
        assert frontmatter.get("title") == "Test Document"
        assert frontmatter.get("author") == "Test Author"

    def test_no_h2_treats_whole_file(self, tmp_path: Path):
        """File without h2 headers becomes single section."""
        content = """# Single Section

Just some content without h2 headers.
More content here.
"""
        md_file = tmp_path / "simple.md"
        md_file.write_text(content)

        sections = parse_markdown(md_file)
        assert len(sections) == 1
        assert "Single Section" in sections[0]["header"]


class TestClassifyByPath:
    """Tests for classify_by_path function."""

    def test_claude_md_high_value(self):
        """CLAUDE.md files should be PROJECT_CONFIG with HIGH value."""
        path = Path("/some/project/CLAUDE.md")
        content_type, value = classify_by_path(path)
        assert content_type == ContentType.PROJECT_CONFIG
        assert value == ContentValue.HIGH

    def test_learnings_high_value(self):
        """Learnings should be LEARNING with HIGH value."""
        path = Path("/docs/learnings/jq-escaping.md")
        content_type, value = classify_by_path(path)
        assert content_type == ContentType.LEARNING
        assert value == ContentValue.HIGH

    def test_skills_high_value(self):
        """Skills should be SKILL with HIGH value."""
        path = Path("/skills/golem-powers/commit/SKILL.md")
        content_type, value = classify_by_path(path)
        assert content_type == ContentType.SKILL
        assert value == ContentValue.HIGH

    def test_research_high_value(self):
        """Research docs should be RESEARCH with HIGH value."""
        path = Path("/docs/research/llm-patterns.md")
        content_type, value = classify_by_path(path)
        assert content_type == ContentType.RESEARCH
        assert value == ContentValue.HIGH

    def test_prd_archive_medium_value(self):
        """PRD archives should be PRD_ARCHIVE with MEDIUM value."""
        path = Path("/prd-json/archives/sprint-1.md")
        content_type, value = classify_by_path(path)
        assert content_type == ContentType.PRD_ARCHIVE
        assert value == ContentValue.MEDIUM

    def test_verification_low_value(self):
        """Verification rounds should be VERIFICATION with LOW value."""
        path = Path("/docs/verification-round-3.md")
        content_type, value = classify_by_path(path)
        assert content_type == ContentType.VERIFICATION
        assert value == ContentValue.LOW

    def test_default_documentation(self):
        """Unknown paths should be DOCUMENTATION with MEDIUM value."""
        path = Path("/docs/readme.md")
        content_type, value = classify_by_path(path)
        assert content_type == ContentType.DOCUMENTATION
        assert value == ContentValue.MEDIUM


class TestExtractMarkdownContent:
    """Tests for extract_markdown_content function."""

    def test_returns_classified_content(self, tmp_path: Path):
        """Should return ClassifiedContent objects."""
        learnings_dir = tmp_path / "learnings"
        learnings_dir.mkdir()

        content = """# JQ Escaping

Use double quotes for jq filters.

## Example

```bash
jq ".foo" file.json
```
"""
        md_file = learnings_dir / "jq-escaping.md"
        md_file.write_text(content)

        results = extract_markdown_content(md_file)
        assert len(results) >= 1

        # Should be classified as learning
        assert all(r.content_type == ContentType.LEARNING for r in results)
        assert all(r.value == ContentValue.HIGH for r in results)

        # Should have source file in metadata
        assert all(str(md_file) in r.metadata.get("source_file", "") for r in results)

    def test_preserves_header_context(self, tmp_path: Path):
        """Should include header hierarchy in metadata."""
        content = """# Parent

## Child Section

Content here.
"""
        md_file = tmp_path / "test.md"
        md_file.write_text(content)

        results = extract_markdown_content(md_file)

        # Find the child section
        child = next((r for r in results if "Child Section" in r.metadata.get("header", "")), None)
        assert child is not None
        assert "Parent" in child.metadata.get("header_context", "")
