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


def test_no_sentinel_resumes_everything(tmp_path, monkeypatch, sentinel):
    resumed: list[str] = []
    monkeypatch.setattr(maintenance, "_resume_service", lambda root, svc: resumed.append(svc))

    maintenance._resume_services(tmp_path, ("watch", "enrichment", "drain"))

    assert resumed == ["watch", "enrichment", "drain"], "no sentinel means resume all"


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
