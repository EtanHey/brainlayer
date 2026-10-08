import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts import ci_ratchet_table as table

ROOT = Path(__file__).resolve().parents[1]
MEMBERS = {"gate", "signatures", "quiesce", "retirement", "no-enrichment-render", "heartbeat"}
ARTIFACTS = {
    "retirement": "global-model-retirement",
    "no-enrichment-render": "brainbar-no-enrichment-render",
    "heartbeat": "watcher-heartbeat-evidence",
}


def test_six_job_and_three_artifact_contract():
    jobs = yaml.safe_load((ROOT / ".github/workflows/ratchet.yml").read_text())["jobs"]
    assert MEMBERS <= jobs.keys()
    assert set(jobs["table"]["needs"]) == MEMBERS
    downloads = {
        s["with"]["name"] for s in jobs["table"]["steps"] if s.get("uses", "").startswith("actions/download-artifact@")
    }
    assert downloads == set(ARTIFACTS.values())
    for job, name in ARTIFACTS.items():
        uploads = [s for s in jobs[job]["steps"] if s.get("uses", "").startswith("actions/upload-artifact@")]
        assert len(uploads) == 1 and uploads[0]["with"]["name"] == name
        assert uploads[0]["with"]["if-no-files-found"] == "error"
    for job in ("heartbeat", "no-enrichment-render", "retirement", "table"):
        checkout = next(s for s in jobs[job]["steps"] if s.get("uses", "").startswith("actions/checkout@"))
        assert checkout["with"]["ref"] == "${{ github.event.pull_request.head.sha }}"
        assert checkout["with"]["persist-credentials"] is False
    assert jobs["table"]["if"] == "${{ !cancelled() }}"
    upload = next(s for s in jobs["heartbeat"]["steps"] if s.get("uses", "").startswith("actions/upload-artifact@"))
    assert upload["with"]["path"] == "${{ runner.temp }}/heartbeat/"
    collector = next(s for s in jobs["table"]["steps"] if s.get("id") == "collect")
    assert '"${HEARTBEAT_ARGS[@]}"' in collector["run"]
    assert '"$HEARTBEAT_RESULT" != success' in collector["run"]
    assert '"$RENDER_RESULT" != success' in collector["run"]
    assert '"$TRIGGER_RESULT" != success' in collector["run"]


def test_normal_collector_keeps_three_missing_evidence_rows(monkeypatch, capsys):
    monkeypatch.setattr(table.Probe, "detect", lambda *_: SimpleNamespace(head_sha="a" * 40))
    monkeypatch.setattr(table, "collect", lambda *_: [])
    monkeypatch.setattr(table, "render", lambda rows, *_: "\n".join(r.name for r in rows))
    assert table.main(["--measured-sha", "a" * 40]) == 1
    error = capsys.readouterr().err
    for name in (
        "Watcher heartbeat freshness and recovery",
        "BrainBar renders no enrichment status",
        "no cloud model call reachable anywhere",
    ):
        assert name in error


@pytest.mark.parametrize("failure", ["heartbeat", "render", "retirement", "gate", "missing-heartbeat", "untriggered"])
def test_actual_shell_handoff_refuses_failed_or_missing_producers(tmp_path, failure):
    jobs = yaml.safe_load((ROOT / ".github/workflows/ratchet.yml").read_text())["jobs"]
    collector = next(s for s in jobs["table"]["steps"] if s.get("id") == "collect")["run"]
    block = collector[collector.index('if [[ "$RETIREMENT_RESULT"') : collector.index("QUIESCE_ARGS=")]
    reports = {
        "heartbeat": tmp_path / "heartbeat/heartbeat.json",
        "render": tmp_path / "brainbar-render/brainbar-render-report.json",
        "retirement": tmp_path / "retirement/retirement-report.json",
    }
    for path in reports.values():
        path.parent.mkdir(parents=True)
        path.write_text("previous valid-looking receipt")
    env = dict(
        os.environ,
        RUNNER_TEMP=str(tmp_path),
        RETIREMENT_RESULT="success",
        RENDER_RESULT="success",
        HEARTBEAT_RESULT="success",
        HEARTBEAT_GATE="true",
        TRIGGER_RESULT="success",
    )
    if failure in {"heartbeat", "render", "retirement", "gate"}:
        env[
            {
                "heartbeat": "HEARTBEAT_RESULT",
                "render": "RENDER_RESULT",
                "retirement": "RETIREMENT_RESULT",
                "gate": "TRIGGER_RESULT",
            }[failure]
        ] = "failure"
    elif failure == "missing-heartbeat":
        reports["heartbeat"].unlink()
    else:
        env.update(HEARTBEAT_GATE="false", HEARTBEAT_RESULT="skipped")
    result = subprocess.run(
        ["/bin/bash", "-eu", "-c", block + "declare -p HEARTBEAT_ARGS"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    if failure in {"heartbeat", "render", "retirement"}:
        assert reports[failure].read_text() == "{}"
    elif failure == "gate":
        assert reports["heartbeat"].read_text() == "{}"
    elif failure == "missing-heartbeat":
        assert not reports["heartbeat"].exists()
    assert (
        "--watcher-heartbeat-unavailable" if failure == "untriggered" else "--watcher-heartbeat-report"
    ) in result.stdout
