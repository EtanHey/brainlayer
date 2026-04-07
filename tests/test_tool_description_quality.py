import asyncio

from brainlayer.mcp import list_tools

ACTIVE_TOOL_NAMES = {
    "brain_search",
    "brain_store",
    "brain_recall",
    "brain_entity",
    "brain_digest",
    "brain_get_person",
}


def _tool_descriptions() -> dict[str, str]:
    tools = asyncio.run(list_tools())
    return {tool.name: tool.description for tool in tools}


def test_all_descriptions_under_1024_chars():
    descriptions = _tool_descriptions()

    for name, description in descriptions.items():
        assert len(description) < 1024, f"{name} description is {len(description)} chars"


def test_active_tools_have_use_when():
    descriptions = _tool_descriptions()

    for name in sorted(ACTIVE_TOOL_NAMES):
        assert name in descriptions, f"{name} not found in available tools"
        assert "Use when:" in descriptions[name], f"{name} is missing 'Use when:'"


def test_active_tools_have_dont_use():
    descriptions = _tool_descriptions()

    for name in sorted(ACTIVE_TOOL_NAMES):
        assert name in descriptions, f"{name} not found in available tools"
        description = descriptions[name]
        assert "Don't use when:" in description or "Does NOT" in description, f"{name} is missing don't-use guidance"
