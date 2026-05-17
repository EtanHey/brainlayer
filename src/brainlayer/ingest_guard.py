"""Write-side guards for content that must never enter BrainLayer."""

from __future__ import annotations

import re

from .chunk_origin import is_precompact_checkpoint_content

_JSONRPC_MESSAGE_RE = re.compile(r'"jsonrpc"\s*:\s*"2\.0"', re.IGNORECASE)
_INVALID_JSONRPC_MARKER = "mcp brainlayer memory: invalid json-rpc message"
_BRAIN_SEARCH_BOX_PREFIX = "┌─ brain_search:"
_BRAINLAYER_BOX_PREFIX_RE = re.compile(
    r"^┌─\s*(?:brain_[a-z_]+|entity(?:\s+search)?):",
    re.IGNORECASE,
)
_RT_AGENT_CHUNK_ID_RE = re.compile(r"^rt-agent-a[0-9a-f]+-[A-Za-z0-9_-]+$", re.IGNORECASE)
_RT_AGENT_SOURCE_FILE_RE = re.compile(r"/subagents/agent-a[0-9a-f]+\.jsonl$", re.IGNORECASE)
_STRONG_JUDGE_NOTE_MARKERS = (
    "judge_agent_name",
    "failure_modes_observed",
    "judge_reasoning",
    "grade distribution",
)
_QID_JUDGE_NOTE_MARKERS = (
    *_STRONG_JUDGE_NOTE_MARKERS,
    "fm11",
)
_F_INFRA_NOISE_PREFIXES = (
    "brainlayer mcp not connected",
    "brainlayer mcp is down",
    "brainlayer mcp tools are not callable",
    "brainlayer unavailable",
    "mcp__brainlayer__brain_search is not available",
    "i can't call `brain_search` or `brain_store` directly",
    "i cannot brain_store",
    'toolsearch found zero matches for "mcp brain"',
)
_F_INFRA_MAX_PREFIX_CHARS = 500
_ASSISTANT_PREFIX_RE = re.compile(r"^assistant\s*:\s*", re.IGNORECASE)


def _looks_like_rt_agent_judge_notes(content: str, chunk_id: str | None = None, source_file: str | None = None) -> bool:
    folded = content.casefold()

    chunk_id_s = chunk_id if isinstance(chunk_id, str) else None
    source_file_s = source_file if isinstance(source_file, str) else None
    has_rt_agent_context = bool(
        (chunk_id_s and _RT_AGENT_CHUNK_ID_RE.match(chunk_id_s))
        or (source_file_s and _RT_AGENT_SOURCE_FILE_RE.search(source_file_s))
    )
    has_qid = bool(re.search(r"\bqid\s*=", folded))
    has_strong_judge_marker = any(marker in folded for marker in _STRONG_JUDGE_NOTE_MARKERS)
    if has_qid and has_strong_judge_marker:
        return True
    if has_rt_agent_context and has_qid and any(marker in folded for marker in _QID_JUDGE_NOTE_MARKERS):
        return True
    return has_rt_agent_context and has_strong_judge_marker


def _looks_like_brainlayer_infra_noise(content: str) -> bool:
    if not content:
        return False

    stripped = content.lstrip()
    stripped = _ASSISTANT_PREFIX_RE.sub("", stripped, count=1)
    stripped = re.sub(r"^[^A-Za-z0-9]+", "", stripped)
    stripped = stripped[:_F_INFRA_MAX_PREFIX_CHARS].casefold()

    return any(re.match(rf"^(?:{re.escape(prefix)})(?:$|[^a-z0-9])", stripped) for prefix in _F_INFRA_NOISE_PREFIXES)


def recursive_mcp_output_reason(
    content: str | None,
    *,
    chunk_id: str | None = None,
    source_file: str | None = None,
    reject_precompact: bool = False,
) -> str | None:
    """Return a reason when content is BrainLayer MCP output being re-ingested."""
    if not content:
        return None

    stripped = str(content).lstrip()
    if stripped.startswith(_BRAIN_SEARCH_BOX_PREFIX):
        return "brain_search_output"
    if _BRAINLAYER_BOX_PREFIX_RE.match(stripped):
        return "brainlayer_mcp_output"

    folded = stripped.casefold()
    if _INVALID_JSONRPC_MARKER in folded:
        return "invalid_jsonrpc_mcp_output"
    if _looks_like_brainlayer_infra_noise(stripped):
        return "brainlayer_mcp_unavailable_diagnostic"
    if reject_precompact and is_precompact_checkpoint_content(stripped):
        return "precompact_checkpoint_noise"
    if _JSONRPC_MESSAGE_RE.search(stripped):
        return "jsonrpc_message"
    if _looks_like_rt_agent_judge_notes(stripped, chunk_id=chunk_id, source_file=source_file):
        return "rt-agent judge notes"

    return None


def reject_recursive_mcp_output(
    content: str | None,
    *,
    chunk_id: str | None = None,
    source_file: str | None = None,
    reject_precompact: bool = False,
) -> None:
    """Raise ValueError when content is recursive BrainLayer MCP output."""
    reason = recursive_mcp_output_reason(
        content,
        chunk_id=chunk_id,
        source_file=source_file,
        reject_precompact=reject_precompact,
    )
    if reason:
        raise ValueError(f"recursive MCP output is not stored in BrainLayer: {reason}")
