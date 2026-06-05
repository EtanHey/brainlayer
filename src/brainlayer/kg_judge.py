"""Entity-context judge harness for KG flag-batch clusters.

The judge never mutates BrainLayer. It gathers read-only evidence, builds a
self-contained prompt packet, and validates evidence-cited verdict JSON.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from ._helpers import _escape_fts5_query
from .paths import get_db_path

JUDGE_TYPES = (
    "Person",
    "Organization",
    "Place",
    "Event",
    "Project",
    "Tool",
    "Technology",
    "D1 Concept→tag",
    "D2 Transient→drop",
)

MERGE_DISPOSITIONS = ("merge", "keep", "split")
CONFIDENCE_VALUES = ("high", "medium", "low")
BACKGROUND_RELATION_TYPES = ("worked_at", "works_at", "uses", "owns", "created")

CLOSED_ENUM_TEXT = (
    'Person ("could I message/meet this individual?", never auto-merge) · '
    "Organization (group acting as one agent; incl. communities/meetups; incl. DEFUNCT/renamed companies) · "
    "Place (coordinates) · "
    "Event (specific dated occurrence) · "
    "Project (repo/deliverable ETAN owns or tracks — OWNERSHIP required, mere usage/repo-presence is NOT ownership) · "
    "Tool (he/agents run, call, or build WITH it; functions-of-tools = part_of facet of parent) · "
    "Technology (what tools are made of/run on) · "
    'D1 Concept→tag (fails rigid-designator: "a/an ___" reads naturally) · '
    "D2 Transient→drop (meaningless in 30 days)."
)

REQUIRED_SCHEMA_TEXT = """Required verdict JSON schema:
{
  "stem": "cluster stem",
  "proposed_type": "Person|Organization|Place|Event|Project|Tool|Technology|D1 Concept→tag|D2 Transient→drop",
  "identity": "one line: what this ACTUALLY is",
  "merge_disposition": "merge|keep|split",
  "canonical_suggestion": "non-empty cluster member name, or null only for D1/D2/split verdicts",
  "confidence": "high|medium|low",
  "evidence_cited": ["refs to supplied packet evidence and/or worker-found evidence"],
  "reasoning": "3 sentences max",
  "evidence_degraded": false
}
"""


class JudgeSchemaError(ValueError):
    """Raised when a worker or LLM verdict violates the judge schema."""


@dataclass(frozen=True)
class EvidenceItem:
    ref: str
    kind: str
    title: str
    data: dict[str, Any]

    def to_prompt_text(self) -> str:
        return f"[{self.ref}] {self.kind}: {self.title}\n{json.dumps(self.data, ensure_ascii=False, indent=2)}"


@dataclass(frozen=True)
class EvidencePacket:
    stem: str
    category: str | None
    members: list[dict[str, Any]]
    evidence: list[EvidenceItem]
    db_path: str
    gits_root: str

    def evidence_refs(self) -> list[str]:
        return [item.ref for item in self.evidence]

    def to_prompt_text(self) -> str:
        members_text = json.dumps(self.members, ensure_ascii=False, indent=2)
        evidence_text = "\n\n".join(item.to_prompt_text() for item in self.evidence) or "(no evidence gathered)"
        return (
            f"Stem: {self.stem}\n"
            f"Category: {self.category or 'unknown'}\n"
            f"DB: {self.db_path}\n"
            f"Gits root: {self.gits_root}\n\n"
            f"Cluster members:\n{members_text}\n\n"
            f"Evidence packet:\n{evidence_text}"
        )


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?", (table,)
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(conn, table):
        return set()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        return set()
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _snippet(value: Any, limit: int = 200) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _member_ids(cluster: dict[str, Any]) -> list[str]:
    ids = []
    for member in cluster.get("members", []):
        entity_id = member.get("id")
        if isinstance(entity_id, str) and entity_id:
            ids.append(entity_id)
    return ids


def _member_display(member: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": member.get("id"),
        "name": member.get("name"),
        "type": member.get("type", member.get("entity_type")),
        "chunks": member.get("chunks", member.get("n_chunks", member.get("chunk_count", 0))),
    }


def _placeholders(values: Iterable[Any]) -> str:
    return ",".join("?" for _ in values)


def _normalize_repo_match(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _clean_git_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "cluster"


def _fetch_linked_chunks(conn: sqlite3.Connection, member_ids: list[str]) -> list[EvidenceItem]:
    if not member_ids or not all(_table_exists(conn, table) for table in ("kg_entity_chunks", "kg_entities", "chunks")):
        return []
    placeholders = _placeholders(member_ids)
    rows = conn.execute(
        f"""
        SELECT ec.entity_id, e.name AS entity_name, e.entity_type, ec.relevance, ec.context,
               c.id AS chunk_id, c.content, c.created_at, c.project
        FROM kg_entity_chunks ec
        JOIN kg_entities e ON e.id = ec.entity_id
        JOIN chunks c ON c.id = ec.chunk_id
        WHERE ec.entity_id IN ({placeholders})
        ORDER BY COALESCE(ec.relevance, 0) DESC
        LIMIT 5
        """,
        member_ids,
    ).fetchall()
    evidence = []
    for index, row in enumerate(rows, start=1):
        evidence.append(
            EvidenceItem(
                ref=f"linked-{index}",
                kind="BrainLayer linked chunk",
                title=f"{row['entity_name']} -> {row['chunk_id']}",
                data={
                    "entity_id": row["entity_id"],
                    "entity_name": row["entity_name"],
                    "entity_type": row["entity_type"],
                    "chunk_id": row["chunk_id"],
                    "relevance": row["relevance"],
                    "context": row["context"],
                    "snippet": _snippet(row["content"]),
                    "date": row["created_at"],
                    "project": row["project"],
                },
            )
        )
    return evidence


def _fetch_fts_chunks(conn: sqlite3.Connection, stem: str) -> list[EvidenceItem]:
    if not all(_table_exists(conn, table) for table in ("chunks_fts", "chunks")):
        return []
    query = _escape_fts5_query(stem, match_mode="or")
    if not query:
        return []
    try:
        rows = conn.execute(
            """
            SELECT f.chunk_id, bm25(chunks_fts) AS rank, c.content, c.created_at, c.project
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.chunk_id
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT 5
            """,
            (query,),
        ).fetchall()
    except sqlite3.Error:
        return []
    evidence = []
    for index, row in enumerate(rows, start=1):
        evidence.append(
            EvidenceItem(
                ref=f"fts-{index}",
                kind="BrainLayer FTS5 chunk",
                title=f"{stem} -> {row['chunk_id']}",
                data={
                    "chunk_id": row["chunk_id"],
                    "rank": row["rank"],
                    "snippet": _snippet(row["content"]),
                    "date": row["created_at"],
                    "project": row["project"],
                },
            )
        )
    return evidence


def _fetch_background_relations(conn: sqlite3.Connection, member_ids: list[str]) -> list[EvidenceItem]:
    if not member_ids or not all(_table_exists(conn, table) for table in ("kg_relations", "kg_entities")):
        return []
    relation_columns = _table_columns(conn, "kg_relations")
    expired_filter = "AND (r.expired_at IS NULL OR r.expired_at = '')" if "expired_at" in relation_columns else ""
    placeholders = _placeholders(member_ids)
    relation_placeholders = _placeholders(BACKGROUND_RELATION_TYPES)
    rows = conn.execute(
        f"""
        SELECT r.id, r.relation_type, r.fact, r.confidence, r.source_chunk_id,
               se.name AS source_name, se.entity_type AS source_type,
               te.name AS target_name, te.entity_type AS target_type
        FROM kg_relations r
        JOIN kg_entities se ON se.id = r.source_id
        JOIN kg_entities te ON te.id = r.target_id
        WHERE (r.source_id IN ({placeholders}) OR r.target_id IN ({placeholders}))
          AND r.relation_type IN ({relation_placeholders})
          {expired_filter}
        ORDER BY r.relation_type, r.id
        LIMIT 20
        """,
        member_ids + member_ids + list(BACKGROUND_RELATION_TYPES),
    ).fetchall()
    evidence = []
    for index, row in enumerate(rows, start=1):
        evidence.append(
            EvidenceItem(
                ref=f"relation-{index}",
                kind="KG relation context",
                title=f"{row['source_name']} --{row['relation_type']}--> {row['target_name']}",
                data={
                    "relation_id": row["id"],
                    "relation_type": row["relation_type"],
                    "source": {"name": row["source_name"], "type": row["source_type"]},
                    "target": {"name": row["target_name"], "type": row["target_type"]},
                    "fact": row["fact"],
                    "confidence": row["confidence"],
                    "source_chunk_id": row["source_chunk_id"],
                },
            )
        )
    return evidence


def _read_repo_evidence(stem: str, gits_root: Path) -> list[EvidenceItem]:
    if not gits_root.exists() or not gits_root.is_dir():
        return []
    stem_key = _normalize_repo_match(stem)
    if not stem_key:
        return []
    matches = []
    for child in sorted(gits_root.iterdir(), key=lambda path: path.name.casefold()):
        if not child.is_dir():
            continue
        child_key = _normalize_repo_match(child.name)
        if stem_key in child_key or child_key in stem_key:
            matches.append(child)
        if len(matches) >= 5:
            break

    evidence = []
    for index, repo in enumerate(matches, start=1):
        readme_lines: list[str] = []
        for candidate in ("README.md", "README", "readme.md"):
            readme_path = repo / candidate
            if readme_path.exists():
                readme_lines = readme_path.read_text(encoding="utf-8", errors="replace").splitlines()[:30]
                break
        try:
            git_log = subprocess.run(
                ["git", "log", "--oneline", "-3"],
                cwd=repo,
                env=_clean_git_env(),
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            git_log = ""
        evidence.append(
            EvidenceItem(
                ref=f"repo-{index}",
                kind="Local repo match",
                title=f"{repo.name} fuzzy-matches {stem}",
                data={
                    "repo_path": str(repo),
                    "README.md first lines": "\n".join(readme_lines),
                    "git log --oneline -3": git_log,
                },
            )
        )
    return evidence


def gather_evidence_for_cluster(
    cluster: dict[str, Any],
    *,
    db_path: str | Path | None = None,
    gits_root: str | Path | None = None,
) -> EvidencePacket:
    """Gather read-only evidence for one flag-batch cluster."""
    resolved_db = Path(db_path).expanduser() if db_path is not None else get_db_path()
    resolved_gits = Path(gits_root).expanduser() if gits_root is not None else Path.home() / "Gits"
    stem = str(cluster.get("stem") or "").strip()
    member_ids = _member_ids(cluster)
    evidence: list[EvidenceItem] = []
    conn = _connect_readonly(resolved_db)
    try:
        evidence.extend(_fetch_linked_chunks(conn, member_ids))
        evidence.extend(_fetch_fts_chunks(conn, stem))
        evidence.extend(_fetch_background_relations(conn, member_ids))
    finally:
        conn.close()
    evidence.extend(_read_repo_evidence(stem, resolved_gits))
    return EvidencePacket(
        stem=stem,
        category=cluster.get("category"),
        members=[_member_display(member) for member in cluster.get("members", [])],
        evidence=evidence,
        db_path=str(resolved_db),
        gits_root=str(resolved_gits),
    )


def build_judge_prompt(packet: EvidencePacket, *, worker_mode: bool = False) -> str:
    worker_instruction = ""
    if worker_mode:
        worker_instruction = (
            "\nWorker extra evidence instructions:\n"
            "- Run brain_search on the stem. If brain_search times out, retry once; if it still times out, "
            "proceed on packet evidence and set evidence_degraded:true.\n"
            "- Also grep ~/Gits yourself for the stem, case-insensitive, and cite what you find.\n"
            "- In reasoning, distinguish packet-supplied evidence from worker-found brain_search/grep evidence.\n"
        )
    return (
        "You are the BrainLayer entity-context judge. Think carefully, but output only JSON.\n"
        "No verdict without evidence. Unevidenced typing is the failure being corrected.\n\n"
        "7+2 closed enum:\n"
        f"{CLOSED_ENUM_TEXT}\n\n"
        f"{REQUIRED_SCHEMA_TEXT}\n"
        "Rules:\n"
        "- proposed_type must be exactly one closed-enum label.\n"
        "- split means the cluster contains 2+ real referents.\n"
        "- canonical_suggestion must be a non-empty cluster member name, or null only for D1/D2/split verdicts.\n"
        "- evidence_cited must cite supplied evidence refs and any worker-found evidence refs.\n"
        "- reasoning must be 3 sentences or fewer.\n"
        f"{worker_instruction}\n"
        "Supplied evidence packet:\n"
        f"{packet.to_prompt_text()}\n"
    )


def _extract_json_object(raw: str) -> dict[str, Any]:
    if not raw:
        raise JudgeSchemaError("empty verdict response")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise JudgeSchemaError("verdict response does not contain a JSON object") from None
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError as exc:
            raise JudgeSchemaError(f"invalid verdict JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise JudgeSchemaError("verdict JSON must be an object")
    return parsed


def _sentence_count(value: str) -> int:
    sentences = [part for part in re.split(r"[.!?]+(?:\s+|$)", value.strip()) if part.strip()]
    return len(sentences)


def validate_verdict(
    verdict: dict[str, Any],
    *,
    allowed_evidence_refs: Iterable[str] | None = None,
    cluster_member_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    required = (
        "stem",
        "proposed_type",
        "identity",
        "merge_disposition",
        "canonical_suggestion",
        "confidence",
        "evidence_cited",
        "reasoning",
        "evidence_degraded",
    )
    missing = [key for key in required if key not in verdict]
    if missing:
        raise JudgeSchemaError(f"missing required verdict keys: {', '.join(missing)}")

    cleaned = dict(verdict)
    if cleaned["proposed_type"] not in JUDGE_TYPES:
        raise JudgeSchemaError(f"proposed_type must be one of {', '.join(JUDGE_TYPES)}")
    if cleaned["merge_disposition"] not in MERGE_DISPOSITIONS:
        raise JudgeSchemaError("merge_disposition must be merge|keep|split")
    if cleaned["confidence"] not in CONFIDENCE_VALUES:
        raise JudgeSchemaError("confidence must be high|medium|low")
    if not isinstance(cleaned["evidence_cited"], list) or not cleaned["evidence_cited"]:
        raise JudgeSchemaError("evidence_cited must be a non-empty list")
    if not all(isinstance(ref, str) and ref.strip() for ref in cleaned["evidence_cited"]):
        raise JudgeSchemaError("evidence_cited must contain non-empty strings")
    allowed = set(allowed_evidence_refs or [])
    if allowed:
        unknown = [ref for ref in cleaned["evidence_cited"] if ref not in allowed]
        if unknown:
            raise JudgeSchemaError(f"evidence_cited contains unknown refs: {', '.join(unknown)}")
    for key in ("stem", "identity", "reasoning"):
        if not isinstance(cleaned[key], str) or not cleaned[key].strip():
            raise JudgeSchemaError(f"{key} must be a non-empty string")
    if "\n" in cleaned["identity"]:
        raise JudgeSchemaError("identity must be one line")
    if _sentence_count(cleaned["reasoning"]) > 3:
        raise JudgeSchemaError("reasoning must be 3 sentences or fewer")
    if not isinstance(cleaned["evidence_degraded"], bool):
        raise JudgeSchemaError("evidence_degraded must be boolean")
    _validate_canonical_suggestion(cleaned, cluster_member_names=cluster_member_names)
    return cleaned


def parse_verdict_json(
    raw: str,
    *,
    allowed_evidence_refs: Iterable[str] | None = None,
    cluster_member_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    return validate_verdict(
        _extract_json_object(raw),
        allowed_evidence_refs=allowed_evidence_refs,
        cluster_member_names=cluster_member_names,
    )


def _null_canonical_allowed(verdict: dict[str, Any]) -> bool:
    return verdict["merge_disposition"] == "split" or verdict["proposed_type"] in {
        "D1 Concept→tag",
        "D2 Transient→drop",
    }


def _validate_canonical_suggestion(
    verdict: dict[str, Any],
    *,
    cluster_member_names: Iterable[str] | None,
) -> None:
    canonical = verdict["canonical_suggestion"]
    if canonical is None:
        if not _null_canonical_allowed(verdict):
            raise JudgeSchemaError("canonical_suggestion can be null only for D1/D2/split verdicts")
        return
    if not isinstance(canonical, str) or not canonical.strip():
        raise JudgeSchemaError("canonical_suggestion must be a non-empty string")
    canonical = canonical.strip()
    member_names = {name.strip() for name in cluster_member_names or [] if isinstance(name, str) and name.strip()}
    if member_names and canonical not in member_names:
        raise JudgeSchemaError("canonical_suggestion must match one of the cluster member names")
    verdict["canonical_suggestion"] = canonical


def _packet_member_names(packet: EvidencePacket) -> list[str]:
    return [str(member["name"]) for member in packet.members if isinstance(member.get("name"), str) and member["name"]]


def judge_cluster_with_llm(
    packet: EvidencePacket,
    *,
    llm: Callable[[str], str],
    max_attempts: int = 2,
) -> dict[str, Any]:
    prompt = build_judge_prompt(packet)
    last_error: JudgeSchemaError | None = None
    for attempt in range(1, max_attempts + 1):
        retry_note = ""
        if last_error is not None:
            retry_note = (
                "\n\nYour previous verdict violated the schema: "
                f"{last_error}. Return a corrected JSON object with cited evidence."
            )
        raw = llm(prompt + retry_note)
        try:
            return parse_verdict_json(
                raw,
                allowed_evidence_refs=packet.evidence_refs(),
                cluster_member_names=_packet_member_names(packet),
            )
        except JudgeSchemaError as exc:
            last_error = exc
            if attempt == max_attempts:
                raise
    raise JudgeSchemaError("judge failed to produce a valid verdict")


def _load_clusters_from_flag_batch(
    flag_batch: Path,
    *,
    categories: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    data = json.loads(flag_batch.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("flag batch must be a JSON object keyed by category")
    wanted = set(categories or data.keys())
    clusters: list[dict[str, Any]] = []
    for category, category_clusters in data.items():
        if category not in wanted:
            continue
        if not isinstance(category_clusters, list):
            continue
        for cluster in category_clusters:
            if not isinstance(cluster, dict):
                continue
            clusters.append({**cluster, "category": category})
            if limit is not None and len(clusters) >= limit:
                return clusters
    return clusters


def load_flag_batch_clusters(
    flag_batch: str | Path,
    *,
    categories: str | list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if isinstance(categories, str):
        parsed_categories = [part.strip() for part in categories.split(",") if part.strip()]
    else:
        parsed_categories = categories
    return _load_clusters_from_flag_batch(Path(flag_batch), categories=parsed_categories, limit=limit)


def _cluster_member_names(cluster: dict[str, Any]) -> list[str]:
    names = []
    for member in cluster.get("members", []):
        name = member.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name)
    return names


def _member_names_by_stem(clusters: Iterable[dict[str, Any]] | None) -> dict[str, list[str]]:
    if clusters is None:
        return {}
    names_by_stem: dict[str, list[str]] = {}
    for cluster in clusters:
        stem = cluster.get("stem")
        if isinstance(stem, str) and stem.strip():
            names_by_stem.setdefault(stem, []).extend(_cluster_member_names(cluster))
    return names_by_stem


def emit_prompt_files(
    clusters: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    db_path: str | Path | None = None,
    gits_root: str | Path | None = None,
) -> list[Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    written = []
    for index, cluster in enumerate(clusters, start=1):
        packet = gather_evidence_for_cluster(cluster, db_path=db_path, gits_root=gits_root)
        category = cluster.get("category") or "cluster"
        prompt_path = out_path / f"{index:03d}-{_slug(str(category))}-{_slug(packet.stem)}.prompt.md"
        prompt_path.write_text(build_judge_prompt(packet, worker_mode=True), encoding="utf-8")
        written.append(prompt_path)
    return written


def _read_verdict_payloads(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        payloads = []
        for line in raw.splitlines():
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    payloads.append(payload)
        return payloads
    payload = json.loads(raw)
    if isinstance(payload, dict) and isinstance(payload.get("verdicts"), list):
        return [item for item in payload["verdicts"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def render_markdown_table(verdicts: list[dict[str, Any]]) -> str:
    lines = [
        "| stem | proposed type | identity | disposition | confidence | top evidence |",
        "|---|---|---|---|---|---|",
    ]
    for verdict in verdicts:
        evidence = ", ".join(verdict.get("evidence_cited", [])[:3])
        lines.append(
            "| {stem} | {proposed_type} | {identity} | {merge_disposition} | {confidence} | {evidence} |".format(
                stem=_md_cell(verdict["stem"]),
                proposed_type=_md_cell(verdict["proposed_type"]),
                identity=_md_cell(verdict["identity"]),
                merge_disposition=_md_cell(verdict["merge_disposition"]),
                confidence=_md_cell(verdict["confidence"]),
                evidence=_md_cell(evidence),
            )
        )
    return "\n".join(lines) + "\n"


def _md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _default_out_path() -> Path:
    today = datetime.now(timezone.utc).date().isoformat()
    return Path("eval_results") / f"kg-judge-verdicts-{today}.json"


def write_verdict_outputs(
    verdicts: list[dict[str, Any]],
    *,
    out_json: str | Path | None = None,
    markdown_path: str | Path | None = None,
    mode: str = "collect",
) -> Path:
    out_path = Path(out_json) if out_json is not None else _default_out_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "verdicts": verdicts,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if markdown_path is not None:
        markdown = Path(markdown_path)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(render_markdown_table(verdicts), encoding="utf-8")
    return out_path


def collect_worker_verdicts(
    verdict_dir: str | Path,
    *,
    out_json: str | Path | None = None,
    markdown_path: str | Path | None = None,
    clusters: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    root = Path(verdict_dir)
    if not root.exists():
        raise FileNotFoundError(root)
    files = sorted([*root.glob("*.json"), *root.glob("*.jsonl")])
    names_by_stem = _member_names_by_stem(clusters)
    verdicts = []
    for path in files:
        for payload in _read_verdict_payloads(path):
            stem = payload.get("stem")
            member_names = names_by_stem.get(stem) if isinstance(stem, str) else None
            verdicts.append(validate_verdict(payload, cluster_member_names=member_names))
    if out_json is not None or markdown_path is not None:
        write_verdict_outputs(verdicts, out_json=out_json, markdown_path=markdown_path, mode="collect")
    return verdicts


def _llm_for_backend(backend: str) -> Callable[[str], str]:
    from .pipeline import enrichment

    resolved_backend = os.environ.get("BRAINLAYER_JUDGE_BACKEND", backend)

    def call(prompt: str) -> str:
        response = enrichment.call_llm(prompt, timeout=60, backend=resolved_backend)
        if response is None:
            raise RuntimeError(f"judge backend {resolved_backend} returned no response")
        return response

    return call


def judge_clusters_with_backend(
    clusters: list[dict[str, Any]],
    *,
    backend: str,
    db_path: str | Path | None = None,
    gits_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    llm = _llm_for_backend(backend)
    verdicts = []
    for cluster in clusters:
        packet = gather_evidence_for_cluster(cluster, db_path=db_path, gits_root=gits_root)
        verdicts.append(judge_cluster_with_llm(packet, llm=llm, max_attempts=2))
        time.sleep(0.01)
    return verdicts


__all__ = [
    "CLOSED_ENUM_TEXT",
    "CONFIDENCE_VALUES",
    "EvidenceItem",
    "EvidencePacket",
    "JUDGE_TYPES",
    "JudgeSchemaError",
    "MERGE_DISPOSITIONS",
    "build_judge_prompt",
    "collect_worker_verdicts",
    "emit_prompt_files",
    "gather_evidence_for_cluster",
    "judge_cluster_with_llm",
    "judge_clusters_with_backend",
    "load_flag_batch_clusters",
    "parse_verdict_json",
    "render_markdown_table",
    "validate_verdict",
    "write_verdict_outputs",
]
