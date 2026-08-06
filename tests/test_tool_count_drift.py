"""Guard the documented MCP count against drift between registration surfaces.

The README's dated release notes are historical records, not current claims.  In
particular, the exclusion is anchored to the section heading instead of a line
number so the check cannot accidentally turn line 192 into a current count.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(os.environ.get("B4_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
README = REPO_ROOT / "README.md"
PYTHON_REGISTRATION = REPO_ROOT / "src/brainlayer/mcp/__init__.py"
SWIFT_REGISTRATION = REPO_ROOT / "brain-bar/Sources/BrainBar/MCPRouter.swift"

RELEASE_NOTES_HEADING = "## Recent Hardening (2026-04-15 → 2026-05-17)"
COUNT_LINE = re.compile(
    r"(?:\bMCP\b[^\n]*?\b(\d+)\s*(?:%20|[- ])\s*tools?\b"
    r"|\b(\d+)\s*(?:%20|[- ])\s*tools?\b[^\n]*\bMCP\b"
    r"|\bMCP\s+Tools?\s*\((\d+)\)"
    r"|\bMCP\s+tools?\b[^\n|]*\|\s*(\d+)\b)",
    re.IGNORECASE,
)


def _current_readme_lines() -> list[str]:
    """Return only current README lines, excluding the dated release notes."""
    lines = README.read_text(encoding="utf-8").splitlines()
    start = lines.index(RELEASE_NOTES_HEADING)
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].startswith("## ")),
        len(lines),
    )
    return lines[:start] + lines[end:]


def _has_b3_disambiguation() -> bool:
    current_readme = "\n".join(_current_readme_lines())
    return (
        re.search(r"Python server\s*=\s*13", current_readme, re.IGNORECASE) is not None
        and re.search(r"BrainBar router\s*=\s*17", current_readme, re.IGNORECASE) is not None
    )


# B4 is intentionally red until B3 lands.  The normal suite stays green while
# retaining the assertion; `pytest --runxfail` provides the pinned-SHA RED proof.
pytestmark = pytest.mark.xfail(
    condition=not _has_b3_disambiguation(),
    reason="B4 becomes active after B3 adds the Python/BrainBar count disambiguation",
)


def _python_tool_count() -> int:
    return len(re.findall(r"\bTool\(", PYTHON_REGISTRATION.read_text(encoding="utf-8")))


def _python_literal_tool_name_count() -> int:
    return len(re.findall(r"\bTool\(name=", PYTHON_REGISTRATION.read_text(encoding="utf-8")))


def _swift_tool_names() -> set[str]:
    return set(re.findall(r"\bbrain_[a-z_]+\b", SWIFT_REGISTRATION.read_text(encoding="utf-8")))


def test_documented_counts_match_both_registration_surfaces() -> None:
    assert _python_tool_count() == 13
    assert _python_literal_tool_name_count() == 0
    assert len(_swift_tool_names()) == 17

    documented_counts = [
        int(match.group(1) or match.group(2) or match.group(3) or match.group(4))
        for line in _current_readme_lines()
        for match in COUNT_LINE.finditer(line)
    ]
    assert documented_counts, "README must contain at least one in-scope MCP count"
    assert set(documented_counts) <= {13, 17}
    assert _has_b3_disambiguation()
