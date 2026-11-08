"""Run the workflow shell handoffs: gated rows must replace prior measurements with n/a."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts import ci_ratchet_table as ratchet

ROOT = Path(__file__).resolve().parents[1]
JOBS = yaml.safe_load((ROOT / ".github/workflows/ratchet.yml").read_text())["jobs"]


def step(job, name):
    return next(s for s in JOBS[job]["steps"] if s.get("name") == name)


@pytest.mark.parametrize(
    "path,retirement,render",
    [
        ("docs/enrichment.md", False, False),
        ("src/brainlayer/search.py", False, False),
        ("scripts/retirement_policy.json", True, False),
        ("scripts/retirement_native.py", True, False),
        ("pyproject.toml", True, False),
        ("MANIFEST.in", True, False),
        ("setup.py", True, False),
        ("setup.cfg", True, False),
        ("brain-bar/Sources/BrainBar/MCPRouter.swift", True, True),
        ("brain-bar/Tests/BrainBarTests/RetirementRatchetTests.swift", True, True),
        ("brain-bar/Package.swift", True, True),
        ("scripts/brainbar_no_enrichment_ratchet.py", False, True),
        ("scripts/brainbar_source_identity.py", False, True),
        ("scripts/ci_ratchet_table.py", True, True),
        (".github/workflows/ratchet.yml", True, True),
    ],
)
def test_gate_paths(tmp_path, path, retirement, render):
    assert decide(tmp_path, path, []) == (retirement, render)


@pytest.mark.parametrize(
    "label,expected", [("ratchet:retirement", (True, False)), ("ratchet:no-enrichment", (False, True))]
)
def test_gate_labels(tmp_path, label, expected):
    assert decide(tmp_path, "docs/enrichment.md", [label]) == expected


def decide(tmp_path, path, labels):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    git = fake_bin / "git"
    git.write_text('#!/bin/bash\nif [[ "$1" == diff ]]; then printf "%s\\n" "$CHANGED"; fi\n')
    git.chmod(0o755)
    output = tmp_path / "output"
    env = dict(
        os.environ,
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        RUNNER_TEMP=str(tmp_path),
        GITHUB_OUTPUT=str(output),
        BASE_SHA="base",
        CHANGED=path,
        LABELS=json.dumps(labels),
    )
    subprocess.run(
        ["bash", "-c", step("gate", "Decide whether this PR pays for a macOS runner")["run"]],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    values = output.read_text().splitlines()
    return ("retirement=true" in values, "no_enrichment=true" in values)


@pytest.mark.parametrize("job,output", [("retirement", "retirement"), ("no-enrichment-render", "no_enrichment")])
def test_jobs_and_downloads_are_gated(job, output):
    assert JOBS[job]["needs"] == "gate"
    assert f"needs.gate.outputs.{output} == 'true'" in JOBS[job]["if"]
    assert output in JOBS["gate"]["outputs"]
    name = "Read exact-head retirement outcome" if job == "retirement" else "Download actual-view render receipt"
    assert f"needs.gate.outputs.{output} != 'false'" in step("table", name)["if"]
    assert JOBS["table"]["if"] == "${{ !cancelled() }}"


@pytest.mark.parametrize(
    "gate,result,trigger,unavailable",
    [
        ("false", "skipped", "success", True),
        ("true", "success", "success", False),
        ("true", "failure", "success", False),
        ("true", "skipped", "success", False),
        ("", "skipped", "failure", False),
        ("false", "skipped", "failure", False),
    ],
)
def test_collector_handoff(tmp_path, gate, result, trigger, unavailable):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python = fake_bin / "python"
    python.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" > "$RUNNER_TEMP/argv"\nprintf table > "$RUNNER_TEMP/ratchet.md"\n'
    )
    python.chmod(0o755)
    for name in ("signature", "commit", "attestation"):
        (tmp_path / f"{name}-args.txt").write_text("--fixture\nvalue\n")
    env = dict(
        os.environ,
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        RUNNER_TEMP=str(tmp_path),
        GITHUB_OUTPUT=str(tmp_path / "output"),
        RETIREMENT_GATE=gate,
        RENDER_GATE=gate,
        RETIREMENT_RESULT=result,
        RENDER_RESULT=result,
        TRIGGER_RESULT=trigger,
        QUIESCE_GATE="false",
        HEARTBEAT_GATE="false",
        HEARTBEAT_RESULT="skipped",
        GITHUB_SERVER_URL="https://github.com",
        GITHUB_REPOSITORY="EtanHey/brainlayer",
        GITHUB_RUN_ID="1",
    )
    subprocess.run(
        ["bash", "-c", step("table", "Collect ratchet rows")["run"]],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    args = (tmp_path / "argv").read_text().splitlines()
    for prefix in ("retirement", "brainbar-render"):
        assert (f"--{prefix}-unavailable" in args) is unavailable
        assert (f"--{prefix}-report" in args) is not unavailable
        if unavailable:
            assert args[args.index(f"--{prefix}-unavailable") + 1].startswith("gated: ")
        elif result != "success" or trigger != "success":
            assert json.loads(Path(args[args.index(f"--{prefix}-report") + 1]).read_text()) == {}


@pytest.mark.parametrize("builder", [ratchet.row_retirement, ratchet.row_brainbar_no_enrichment])
def test_gated_row_is_na_but_missing_or_invalid_report_is_red(tmp_path, builder):
    reason = "gated: no affected path or opt-in label"
    row = builder(None, "a" * 40, reason)
    assert row.status == ratchet.NA
    assert row.value == f"n/a — {reason}"
    assert builder(None, "a" * 40).status == ratchet.RED
    report = tmp_path / "report.json"
    report.write_text("{}")
    assert builder(report, "a" * 40, reason).status == ratchet.RED


def test_cli_renders_both_gated_rows_without_reading_reports(monkeypatch, capsys):
    from types import SimpleNamespace

    monkeypatch.setattr(ratchet.Probe, "detect", lambda *_: SimpleNamespace(head_sha="a" * 40))
    monkeypatch.setattr(ratchet, "collect", lambda *_: [])
    monkeypatch.setattr(ratchet, "render", lambda rows, *_: "\n".join(row.value for row in rows))
    assert (
        ratchet.main(
            [
                "--retirement-unavailable",
                "gated: no retirement path",
                "--brainbar-render-unavailable",
                "gated: no render path",
                "--watcher-heartbeat-unavailable",
                "gated: no heartbeat path",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "n/a — gated: no retirement path" in output
    assert "n/a — gated: no render path" in output
