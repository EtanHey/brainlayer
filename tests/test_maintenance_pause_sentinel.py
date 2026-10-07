"""Maintenance must not resume a service a human deliberately paused.

RED reproduces the 2026-08-04/05 incident: `com.brainlayer.maintenance-nightly` runs at
04:00 and its teardown unconditionally re-installs DEFAULT_SERVICES — including
enrichment — via `scripts/launchd/install.sh`. That re-created the enrichment plist four
times, and on 2026-08-04 the resulting run cost 5,137 rows of source `provenance_class`.

The pause sentinel shipped in #638 already records the intent. Maintenance never read it.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from brainlayer import maintenance
from brainlayer.scrub_at_rest import LIVE_SERVICES


def _write_sentinel(path: Path, *labels: str) -> None:
    path.write_text(json.dumps({"paused_at": datetime.now(UTC).isoformat(), "labels": list(labels)}))


@pytest.fixture
def sentinel(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "pause.sentinel"
    monkeypatch.setattr(maintenance, "PAUSE_SENTINEL_PATH", path, raising=False)
    return path


def test_paused_service_is_not_resumed(tmp_path, monkeypatch, sentinel):
    _write_sentinel(sentinel, "com.brainlayer.enrichment")
    resumed: list[str] = []
    monkeypatch.setattr(maintenance, "_resume_service", lambda root, svc: resumed.append(svc))

    failures = maintenance._resume_services(tmp_path, ("watch", "enrichment", "drain"))

    assert "enrichment" not in resumed, "deliberately paused service was resumed"
    assert resumed == ["watch", "drain"], "unpaused services must still resume"
    assert failures == [], "skipping a paused service is not a resume failure"


def test_no_sentinel_resumes_only_active_services(tmp_path, monkeypatch, sentinel):
    resumed: list[str] = []
    monkeypatch.setattr(maintenance, "_resume_service", lambda root, svc: resumed.append(svc))

    maintenance._resume_services(tmp_path, ("watch", "enrichment", "drain"))

    assert resumed == ["watch", "drain"], "retired services never resume"


def test_keep_down_accepts_service_names_and_launchd_labels(tmp_path, monkeypatch, sentinel):
    resumed: list[str] = []
    monkeypatch.setenv(
        "BRAINLAYER_MAINTENANCE_KEEP_DOWN",
        "watch, com.brainlayer.enrichment",
    )
    monkeypatch.setattr(maintenance, "_resume_service", lambda root, svc: resumed.append(svc))

    failures = maintenance._resume_services(
        tmp_path,
        ("watch", "enrichment", "index", "drain"),
        {"watch": True, "enrichment": True, "index": True, "drain": True},
    )

    assert resumed == ["index", "drain"]
    assert failures == []


@pytest.mark.parametrize("service", LIVE_SERVICES)
def test_pause_check_uses_launchd_label_for_every_live_service(service, sentinel, monkeypatch):
    label = maintenance._launchd_label(service)
    _write_sentinel(sentinel, label)
    checked_labels = []
    pause_applies = maintenance.pause_applies_to_label

    def record_pause_check(payload, checked_label):
        checked_labels.append(checked_label)
        return pause_applies(payload, checked_label)

    monkeypatch.setattr(maintenance, "pause_applies_to_label", record_pause_check)

    assert maintenance._service_is_deliberately_paused(service)
    assert checked_labels == [label]


@pytest.mark.parametrize("service", ["enrich", "enrichment"])
def test_retired_services_stay_down_even_when_loaded_before(tmp_path, monkeypatch, sentinel, service):
    resumed = []
    monkeypatch.setattr(maintenance, "_resume_service", lambda root, name: resumed.append(name))
    assert maintenance._resume_services(tmp_path, (service, "watch"), {service: True, "watch": True}) == []
    assert resumed == ["watch"]
