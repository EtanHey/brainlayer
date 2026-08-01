"""Per-agent-class provenance decisions for transcript quarantine planning.

The classifier is pure and side-effect-free. Recon Agent-tool signatures are
detectable only when caller-provided content is available; source-only report
runs will classify those rows by path provenance instead.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .content_class import normalize_content_class
from .ingest_denylist import is_denylisted
from .t3_provenance import T3_APP_SESSION, is_t3_app_initiated_codex_session

SearchPolicy = Literal["KEEP", "ISOLATE", "OUT"]
EffectiveVisibility = Literal["default", "operational", "cold"]

RECON_BRAIN_WORKER_RE = re.compile(r"\bbrain[-_ ]?worker\b", re.IGNORECASE)
RECON_WEAVE_RE = re.compile(r"(?<!\w)/weave\b|\bweave[-_ ]+(?:worker|agent|recon)\b", re.IGNORECASE)
RECON_SESSION_MINER_RE = re.compile(r"\bsession[-_ ]?miner\b", re.IGNORECASE)
RECON_AGENT_SIGNATURES = (
    RECON_BRAIN_WORKER_RE,
    RECON_WEAVE_RE,
    RECON_SESSION_MINER_RE,
)

FLEET_REPOS = frozenset(
    {
        "orchestrator",
        "golems",
        "skill-creator",
        "brainlayer",
        "voicelayer",
        "cmuxlayer",
        "dashboard",
        "cmux",
    }
)


@dataclass(frozen=True)
class ProvenanceDecision:
    provenance_tag: str
    search_policy: SearchPolicy
    reason: str


def _abspath(source_file: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(source_file))))


def _has_segment(path: Path, pattern: str) -> bool:
    return any(fnmatch.fnmatchcase(part, pattern) for part in path.parts)


def _under_provider_sessions(path: Path, provider_dir: str) -> bool:
    parts = path.parts
    for index, part in enumerate(parts[:-1]):
        if part == provider_dir and index + 1 < len(parts) and parts[index + 1] == "sessions":
            return True
    return False


def _under_cursor_agent_transcripts(path: Path) -> bool:
    parts = path.parts
    try:
        cursor_index = parts.index(".cursor")
    except ValueError:
        return False
    return "agent-transcripts" in parts[cursor_index + 1 :]


def _project_segment(path: Path) -> str | None:
    parts = path.parts
    for index, part in enumerate(parts[:-1]):
        if part == "projects" and index > 0 and parts[index - 1] == ".claude":
            return parts[index + 1]
    return None


def _repo_from_project_segment(segment: str | None) -> str | None:
    if not segment:
        return None
    normalized = segment.strip("-")
    for marker in ("-Gits-", "--config-"):
        if marker not in normalized:
            continue
        repo = normalized.rsplit(marker, 1)[-1].strip("-")
        return repo.casefold() if repo else None
    return normalized.rsplit("-", 1)[-1].casefold() if normalized else None


def _is_subagent(path: Path) -> bool:
    return "subagents" in path.parts


def _is_direct_claude_session(path: Path) -> bool:
    if is_denylisted(path):
        return False
    parts = path.parts
    if ".claude" not in parts or "projects" not in parts:
        return False
    if _is_subagent(path) or _has_segment(path, "wf_*"):
        return False
    return path.suffix == ".jsonl"


def has_recon_agent_signature(content: str | None) -> bool:
    """Return True when first-chunk/task text identifies recon Agent-tool work."""
    if not content:
        return False
    return any(regex.search(content) for regex in RECON_AGENT_SIGNATURES)


def _has_recon_path_signature(path: Path) -> bool:
    markers = {"brain-worker", "session-miner", "weave"}
    parts = path.parts
    marker_indexes = [index for index, part in enumerate(parts) if part in markers]
    if not marker_indexes:
        return False

    root_indexes: list[int] = []
    if "subagents" in parts:
        root_indexes.append(parts.index("subagents"))
    if _under_cursor_agent_transcripts(path):
        root_indexes.append(parts.index("agent-transcripts"))

    return any(marker_index > root_index for marker_index in marker_indexes for root_index in root_indexes)


def classify_provenance(
    source_file: str,
    content_class: str | None = None,
    *,
    content: str | None = None,
    t3_state_db: str | Path | None = None,
) -> ProvenanceDecision:
    """Classify a source path into an auditable provenance search policy."""
    del content_class
    path = _abspath(source_file)

    if has_recon_agent_signature(content) or _has_recon_path_signature(path):
        return ProvenanceDecision("recon-agent", "OUT", "recon Agent-tool signature wins precedence")

    if _has_segment(path, "wf_*"):
        return ProvenanceDecision("workflow-agent", "ISOLATE", "workflow wf_* path segment")

    if _under_provider_sessions(path, ".codex"):
        is_t3_app_session = (
            is_t3_app_initiated_codex_session(source_file, state_db=t3_state_db)
            if t3_state_db is not None
            else is_t3_app_initiated_codex_session(source_file)
        )
        if is_t3_app_session:
            return ProvenanceDecision(T3_APP_SESSION, "KEEP", "T3 runtime cursor links Codex session")
        return ProvenanceDecision("codex-session", "KEEP", "Codex session root stays searchable")

    if _under_cursor_agent_transcripts(path):
        return ProvenanceDecision("cursor-gather", "ISOLATE", "Cursor agent-transcripts root")

    if _under_provider_sessions(path, ".gemini"):
        return ProvenanceDecision("gemini-session", "KEEP", "Gemini session root stays searchable")

    if _is_subagent(path):
        repo = _repo_from_project_segment(_project_segment(path))
        if repo in FLEET_REPOS:
            return ProvenanceDecision("fleet-subagent", "KEEP", f"fleet subagent project token {repo}")
        return ProvenanceDecision("product-subagent", "KEEP", "non-fleet product subagent")

    if _is_direct_claude_session(path):
        return ProvenanceDecision("direct-session", "KEEP", "direct/control Claude session")

    return ProvenanceDecision("unknown", "KEEP", "no agent provenance rule matched")


def effective_visibility(decision: ProvenanceDecision, content_class: str | None) -> EffectiveVisibility:
    """Map provenance + content class onto P1.4 default/operational/cold visibility."""
    if decision.search_policy == "OUT":
        return "cold"
    if decision.search_policy == "ISOLATE":
        return "operational"

    normalized = normalize_content_class(content_class)
    if normalized in {"knowledge", "decision"}:
        return "default"
    if normalized == "operational":
        return "operational"
    return "cold"
