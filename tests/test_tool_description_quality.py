"""Tool-description house style.

Ratified doctrine: ~/Gits/skill-creator/docs.local/doctrine/tool-description-house-style.md
(measured over 86 golems skills: median 125 chars, p90 263, max 475). Etan, 2026-08-19:
"the tool descriptions should be extremely short. Just tell them what to do with it."

The original R79 gate (PRs #212/#223) required EVERY active tool to carry a literal
"Use when:" and "Don't use when:" clause. That gate forced prose onto tools with no
confusable neighbour and pushed the palette 8x over the measured norm, which is what
made the 8192-byte wire strip in issue #726 bite. Per the lead ruling of 2026-08-19 it
is replaced by: a hard cap, a budget in the median/p90 spirit, and a not-for clause
required ONLY for tools in the explicitly-named confusable set below.
"""

import statistics

from brainlayer.mcp import _full_tool_definitions

# Tools with a near-neighbour an agent can plausibly pick instead. Only these owe a
# not-for clause naming the sibling; a tool with no near-neighbour does not.
CONFUSABLE_SIBLINGS = {
    "brain_search": ("brain_recall",),
    "brain_recall": ("brain_search",),
    "brain_entity": ("brain_search",),
    "brain_resume": ("brain_search",),
    "brain_store": ("brain_digest",),
    "brain_digest": ("brain_store",),
    "brain_supersede": ("brain_archive",),
    "brain_archive": ("brain_supersede",),
}

# Doctrine max is 475; BrainLayer's palette is small enough to sit under it.
PER_TOOL_CEILING = 320
MEDIAN_CEILING = 175


def _tool_descriptions() -> dict[str, str]:
    tools = _full_tool_definitions()
    return {tool.name: tool.description for tool in tools}


def test_all_descriptions_under_1024_chars():
    descriptions = _tool_descriptions()

    for name, description in descriptions.items():
        assert len(description) < 1024, f"{name} description is {len(description)} chars"


def test_every_tool_has_a_non_empty_description():
    for name, description in _tool_descriptions().items():
        assert description.strip(), f"{name} has no description"


def test_descriptions_stay_within_the_house_style_budget():
    descriptions = _tool_descriptions()

    for name, description in descriptions.items():
        assert len(description) <= PER_TOOL_CEILING, (
            f"{name} is {len(description)} chars, over the {PER_TOOL_CEILING}-char ceiling"
        )

    median = statistics.median(len(d) for d in descriptions.values())
    assert median <= MEDIAN_CEILING, f"median description length is {median} chars"


def test_confusable_tools_name_the_sibling_to_use_instead():
    descriptions = _tool_descriptions()

    for name, siblings in CONFUSABLE_SIBLINGS.items():
        assert name in descriptions, f"{name} not found in available tools"
        description = descriptions[name]
        assert any(sibling in description for sibling in siblings), (
            f"{name} is confusable with {'/'.join(siblings)} and must name the alternative"
        )


def test_brain_store_keeps_its_outcome_contract():
    """PR #725's contract survives the shortening, compressed rather than cut.

    Its ENFORCEMENT is the write-time dedupe, not this sentence; the sentence only
    stops an agent re-storing on an ambiguous response.
    """
    description = _tool_descriptions()["brain_store"]

    for outcome in ("STORED", "DUPLICATE", "MERGED", "DEFERRED", "REJECTED", "ERROR"):
        assert outcome in description, f"brain_store lost the {outcome} outcome"
    assert "do NOT re-store" in description
