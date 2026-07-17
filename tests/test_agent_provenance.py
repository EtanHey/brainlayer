import json
import subprocess
import sys
from pathlib import Path

import apsw

from brainlayer.agent_provenance import (
    classify_provenance,
    effective_visibility,
    has_recon_agent_signature,
)
from scripts.provenance_classify_report import build_report


def test_classifies_provider_roots_and_workflow_paths(tmp_path: Path) -> None:
    home = tmp_path / "home"

    cases = [
        (
            home / ".codex" / "sessions" / "2026" / "07" / "worker.jsonl",
            "codex-session",
            "KEEP",
        ),
        (
            home / ".cursor" / "projects" / "brainlayer" / "agent-transcripts" / "session.jsonl",
            "cursor-gather",
            "ISOLATE",
        ),
        (
            home / ".gemini" / "sessions" / "session.jsonl",
            "gemini-session",
            "KEEP",
        ),
        (
            home
            / ".claude"
            / "projects"
            / "-Users-etanheyman-Gits-brainlayer"
            / "session"
            / "subagents"
            / "workflows"
            / "wf_c83ce37d"
            / "agent.jsonl",
            "workflow-agent",
            "ISOLATE",
        ),
    ]

    for source_file, expected_tag, expected_policy in cases:
        decision = classify_provenance(str(source_file))
        assert decision.provenance_tag == expected_tag
        assert decision.search_policy == expected_policy
        assert decision.reason


def test_classifies_fleet_and_product_subagents_by_transferable_project_token(tmp_path: Path) -> None:
    home = tmp_path / "alt-home"

    fleet = classify_provenance(
        str(
            home
            / ".claude"
            / "projects"
            / "-private-tmp-Gits-brainlayer"
            / "session"
            / "subagents"
            / "agent-a25d6b3aa6880db8e.jsonl"
        )
    )
    hyphenated_fleet = classify_provenance(
        str(
            home
            / ".claude"
            / "projects"
            / "-Users-someone-Gits-skill-creator"
            / "session"
            / "subagents"
            / "agent-fleet.jsonl"
        )
    )
    product = classify_provenance(
        str(
            home
            / ".claude"
            / "projects"
            / "-Users-someone-Gits-domica"
            / "session"
            / "subagents"
            / "agent-product.jsonl"
        )
    )
    product_with_fleet_prefix = classify_provenance(
        str(
            home
            / ".claude"
            / "projects"
            / "-Users-someone-Gits-golems-packages-coach"
            / "session"
            / "subagents"
            / "agent-product.jsonl"
        )
    )
    mehayom = classify_provenance(
        str(
            home
            / ".claude"
            / "projects"
            / "-Users-someone-Gits-Mehayom"
            / "session"
            / "subagents"
            / "agent-product.jsonl"
        )
    )

    assert (fleet.provenance_tag, fleet.search_policy) == ("fleet-subagent", "KEEP")
    assert (hyphenated_fleet.provenance_tag, hyphenated_fleet.search_policy) == ("fleet-subagent", "KEEP")
    assert (product.provenance_tag, product.search_policy) == ("product-subagent", "KEEP")
    assert (product_with_fleet_prefix.provenance_tag, product_with_fleet_prefix.search_policy) == (
        "product-subagent",
        "KEEP",
    )
    assert (mehayom.provenance_tag, mehayom.search_policy) == ("product-subagent", "KEEP")


def test_cursor_agent_transcripts_requires_cursor_ancestor(tmp_path: Path) -> None:
    reversed_segments = tmp_path / "agent-transcripts" / "archive" / ".cursor" / "rules.jsonl"

    decision = classify_provenance(str(reversed_segments))

    assert decision.provenance_tag == "unknown"
    assert decision.search_policy == "KEEP"


def test_direct_claude_sessions_stay_keep_and_never_isolate_or_out(tmp_path: Path) -> None:
    direct_session = (
        tmp_path / "home" / ".claude" / "projects" / "-Users-etanheyman-Gits-brainlayer" / "4220c177-8816-446d.jsonl"
    )

    decision = classify_provenance(str(direct_session))

    assert decision.provenance_tag == "direct-session"
    assert decision.search_policy == "KEEP"


def test_recon_agent_signature_is_out_and_has_precedence_over_fleet_subagent(tmp_path: Path) -> None:
    fleet_subagent = (
        tmp_path
        / "home"
        / ".claude"
        / "projects"
        / "-Users-etanheyman-Gits-brainlayer"
        / "session"
        / "subagents"
        / "agent-a25d6b3aa6880db8e.jsonl"
    )

    decision = classify_provenance(
        str(fleet_subagent),
        content_class="knowledge",
        content="Task for brain-worker: mine workflow JSONL transcripts and write the recon synthesis.",
    )

    assert has_recon_agent_signature("session-miner should inspect these transcripts")
    assert decision.provenance_tag == "recon-agent"
    assert decision.search_policy == "OUT"


def test_recon_agent_signature_does_not_match_plain_weave_verb(tmp_path: Path) -> None:
    direct_session = (
        tmp_path / "home" / ".claude" / "projects" / "-Users-etanheyman-Gits-brainlayer" / "4220c177-8816-446d.jsonl"
    )

    decision = classify_provenance(
        str(direct_session),
        content="Weave together the launch notes into one summary.",
    )

    assert has_recon_agent_signature("/weave run the recon")
    assert has_recon_agent_signature("Send this to the weave agent.")
    assert not has_recon_agent_signature("Weave together the launch notes into one summary.")
    assert decision.provenance_tag == "direct-session"
    assert decision.search_policy == "KEEP"


def test_recon_path_signature_is_scoped_to_agent_transcript_roots(tmp_path: Path) -> None:
    normal_project = tmp_path / "home" / "Gits" / "weave" / "notes" / "memory.md"
    direct_weave_repo_session = (
        tmp_path / "home" / ".claude" / "projects" / "-Users-etanheyman-Gits-weave" / "session.jsonl"
    )
    recon_subagent = (
        tmp_path
        / "home"
        / ".claude"
        / "projects"
        / "-Users-etanheyman-Gits-brainlayer"
        / "session"
        / "subagents"
        / "session-miner"
        / "agent.jsonl"
    )

    assert classify_provenance(str(normal_project)).search_policy == "KEEP"
    assert classify_provenance(str(direct_weave_repo_session)).provenance_tag == "direct-session"
    assert classify_provenance(str(recon_subagent)).provenance_tag == "recon-agent"


def test_effective_visibility_preserves_keep_knowledge_and_decision_but_isolates_operational(
    tmp_path: Path,
) -> None:
    keep = classify_provenance(str(tmp_path / ".codex" / "sessions" / "session.jsonl"))
    isolate = classify_provenance(
        str(tmp_path / ".cursor" / "projects" / "repo" / "agent-transcripts" / "session.jsonl")
    )
    gemini = classify_provenance(str(tmp_path / ".gemini" / "sessions" / "session.jsonl"))

    assert effective_visibility(keep, "knowledge") == "default"
    assert effective_visibility(keep, "decision") == "default"
    assert effective_visibility(keep, "operational") == "operational"
    assert effective_visibility(isolate, "knowledge") == "operational"
    assert effective_visibility(gemini, "knowledge") == "default"


def test_report_summarizes_tags_policies_and_effective_visibility_without_real_db(tmp_path: Path) -> None:
    db_path = tmp_path / "brainlayer-test.db"
    conn = apsw.Connection(str(db_path))
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, source_file TEXT, content_class TEXT)")
    rows = [
        ("codex-knowledge", str(tmp_path / ".codex" / "sessions" / "session.jsonl"), "knowledge"),
        (
            "cursor-operational",
            str(tmp_path / ".cursor" / "projects" / "repo" / "agent-transcripts" / "session.jsonl"),
            "knowledge",
        ),
        ("gemini-cold", str(tmp_path / ".gemini" / "sessions" / "session.jsonl"), "knowledge"),
        (
            "direct-decision",
            str(tmp_path / ".claude" / "projects" / "-Users-x-Gits-brainlayer" / "session.jsonl"),
            "decision",
        ),
    ]
    cursor.executemany("INSERT INTO chunks (id, source_file, content_class) VALUES (?, ?, ?)", rows)
    conn.close()

    report = build_report(db_path)

    assert json.loads(json.dumps(report)) == report
    assert report["dry_run"] is True
    assert report["total_chunks"] == 4
    assert report["tags"] == {
        "codex-session": 1,
        "cursor-gather": 1,
        "direct-session": 1,
        "gemini-session": 1,
    }
    assert report["policies"] == {"KEEP": 3, "ISOLATE": 1, "OUT": 0}
    assert report["effective_visibility"] == {"cold": 0, "default": 3, "operational": 1}


def test_report_script_bootstraps_src_when_run_directly(tmp_path: Path) -> None:
    fake_apsw = tmp_path / "apsw.py"
    fake_apsw.write_text("", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "scripts/provenance_classify_report.py",
            "--help",
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        env={"PYTHONPATH": str(tmp_path)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Read-only provenance classifier report" in result.stdout
