"""Report-validation unit fixtures; native render and mutation receipts are separate."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from scripts import brainbar_no_enrichment_ratchet as native
from scripts import ci_ratchet_table as table

SHA = "a" * 40
IDENTITY = {
    "schema_version": 1,
    "root": "/synthetic/checkout",
    "head": SHA,
    "tree": "c" * 40,
    "dirty": False,
    "source_sha256": "d" * 64,
    "source_files": 10,
}


@pytest.fixture(autouse=True)
def synthetic_source_manifest(monkeypatch):
    monkeypatch.setattr(native, "source_identity", lambda *_: copy.deepcopy(IDENTITY))


def fixture_report():
    captures = []
    for name in sorted(native.CAPTURES):
        text = "loading" if name == "dashboard-loading" else "Details Memory on this Mac Pending stores Replay debt"
        item = {"name": name, "png": name + ".png", "text": text, "png_sha256": "a" * 64}
        if name in native.ICONS:
            item["text"] = ""
            item["pixel_check"] = {
                "series": ["Agent", "Watcher"],
                "matches_reference": True,
                "actual_rgba_sha256": "e" * 64,
                "reference_rgba_sha256": "e" * 64,
                "nontransparent_pixels": 100,
                "red_pixels": 10,
            }
        captures.append(item)
    return {
        "schema_version": 1,
        "row": native.ROW,
        "status": "PASS",
        "mode": "synthetic-source-build",
        "measured_sha": SHA,
        "historical_backlog": 274847,
        "captures": captures,
        "violations": [],
        "active_pending_store_visible": True,
        "positive_control": {
            "name": "ocr-positive-control",
            "png": "ocr-positive-control.png",
            "text": "Enrichment paused · 274,847 queued\nEnrichment retired",
            "png_sha256": "f" * 64,
        },
        "build_identity": copy.deepcopy(IDENTITY),
        "render_exit": 0,
        "binary_sha256": "b" * 64,
    }


def test_complete_report_and_row(tmp_path):
    report = tmp_path / "receipt.json"
    report.write_text(json.dumps(fixture_report()))
    row = table.row_brainbar_no_enrichment(report, SHA)
    assert row.status == table.GREEN and row.name == native.ROW


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "garbage",
        "wrong_head",
        "omitted_page",
        "duplicate_page",
        "blank_ocr",
        "no_pixels",
        "no_control",
        "no_active_queue",
    ],
)
def test_missing_or_incomplete_evidence_is_red(tmp_path, fault):
    path = tmp_path / "receipt.json"
    payload = fixture_report()
    if fault == "garbage":
        path.write_text("[]")
    elif fault != "missing":
        if fault == "wrong_head":
            payload["measured_sha"] = "c" * 40
        if fault == "omitted_page":
            payload["captures"].pop()
        if fault == "duplicate_page":
            payload["captures"][0] = copy.deepcopy(payload["captures"][1])
        if fault == "blank_ocr":
            payload["captures"][0]["text"] = ""
        if fault == "no_pixels":
            payload["captures"][0].pop("png_sha256")
        if fault == "no_control":
            payload["positive_control"] = {}
        if fault == "no_active_queue":
            payload["active_pending_store_visible"] = False
        path.write_text(json.dumps(payload))
    assert table.row_brainbar_no_enrichment(path, SHA).status == table.RED
    assert table.row_brainbar_no_enrichment(None, SHA).status == table.RED


@pytest.mark.parametrize(
    "text",
    [
        "Enrichment paused · 274,847 queued",
        "Enrichment retired",
        "Enrichment history",
        "Enrichment off",
        "274,847 queued",
    ],
)
def test_any_restored_status_count_or_job_row_is_red(tmp_path, text):
    payload = fixture_report()
    payload["captures"][0]["text"] += "\n" + text
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(payload))
    assert table.row_brainbar_no_enrichment(path, SHA).status == table.RED


def test_ci_always_hands_missing_render_to_fail_closed_row():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/ratchet.yml").read_text())
    job = workflow["jobs"]["no-enrichment-render"]
    assert job["runs-on"] == "macos-15" and "if" not in job
    collect = next(step for step in workflow["jobs"]["table"]["steps"] if step.get("name") == "Collect ratchet rows")
    assert "--brainbar-render-report" in collect["run"]
    upload = next(step for step in job["steps"] if step.get("name") == "Retain synthetic render receipt only")
    assert upload["with"]["path"].endswith("brainbar-render-report.json")


def test_normal_collector_missing_report_cannot_exit_green(monkeypatch, capsys):
    from types import SimpleNamespace

    monkeypatch.setattr(table.Probe, "detect", lambda *_: SimpleNamespace(head_sha=SHA))
    monkeypatch.setattr(table, "collect", lambda *_: [])
    monkeypatch.setattr(table, "render", lambda rows, *_: rows[0].name)
    assert table.main([]) == 1
    assert native.ROW in capsys.readouterr().err


@pytest.mark.parametrize(
    "fault",
    [
        "nonhex_binary",
        "exit_1",
        "missing_exit",
        "bool_exit",
        "weak_control",
        "missing_control_digest",
        "nonhex_control_digest",
        "missing_paused",
        "missing_count",
        "missing_retired",
        "wrong_tree",
        "wrong_source_digest",
        "dirty_build",
        "missing_identity",
        "icon_third_series",
        "icon_mismatch",
        "icon_bad_hash",
        "icon_blank",
        "missing_badge",
    ],
)
def test_provenance_controls_and_icon_pixels_fail_closed(fault):
    p = fixture_report()
    icon = next(x for x in p["captures"] if x["name"] == "status-icon-badged")["pixel_check"]
    if fault == "nonhex_binary":
        p["binary_sha256"] = "z" * 64
    if fault == "exit_1":
        p["render_exit"] = 1
    if fault == "missing_exit":
        p.pop("render_exit")
    if fault == "bool_exit":
        p["render_exit"] = False
    if fault == "weak_control":
        p["positive_control"]["text"] = "Enrichment"
    if fault == "missing_control_digest":
        p["positive_control"].pop("png_sha256")
    if fault == "nonhex_control_digest":
        p["positive_control"]["png_sha256"] = "z" * 64
    if fault == "missing_paused":
        p["positive_control"]["text"] = "Enrichment · 274,847 queued Enrichment retired"
    if fault == "missing_count":
        p["positive_control"]["text"] = "Enrichment paused Enrichment retired"
    if fault == "missing_retired":
        p["positive_control"]["text"] = "Enrichment paused · 274,847 queued"
    if fault == "wrong_tree":
        p["build_identity"]["tree"] = "e" * 40
    if fault == "wrong_source_digest":
        p["build_identity"]["source_sha256"] = "e" * 64
    if fault == "dirty_build":
        p["build_identity"]["dirty"] = True
    if fault == "missing_identity":
        p.pop("build_identity")
    if fault == "icon_third_series":
        icon["series"].append("Enrichment")
    if fault == "icon_mismatch":
        icon["matches_reference"] = False
    if fault == "icon_bad_hash":
        icon["actual_rgba_sha256"] = "z" * 64
    if fault == "icon_blank":
        icon["nontransparent_pixels"] = 0
    if fault == "missing_badge":
        icon["red_pixels"] = 0
    assert native.validate_report(p, SHA) is not None
