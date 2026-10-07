"""Retired tuning must not erase the installed-hotlane credential gate."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_configuration_keeps_installed_credential_gate():
    text = (ROOT / "docs/configuration.md").read_text()
    for retained in ("GOOGLE_API_KEY", "BRAINLAYER_REQUIRE_GOOGLE_API_KEY", "brainlayer-env-run.sh", "78", "op read"):
        assert retained in text
    assert "re-render" in text.lower()
    assert "retired" in text.lower()


def test_configuration_does_not_offer_producer_activation():
    text = (ROOT / "docs/configuration.md").read_text()
    assert not re.search(r"BRAINLAYER_ENRICH\w*\s*=\s*(?:1|remote|gemini)", text)
    assert not re.search(r"install.sh\s+(?:load\s+)?enrichment", text)
    assert "KeepAlive supervisor" not in text
    for retained in ("BRAINLAYER_DB", "BRAINLAYER_LAUNCHD_WATCH_ENABLED", "BRAINLAYER_LAUNCHD_BACKUP_DAILY_ENABLED"):
        assert retained in text


def test_restore_does_not_restart_retired_writer_and_preserves_legacy_stop():
    text = (ROOT / "docs/backup-strategy.md").read_text()
    assert not re.search(r"launchctl load[^\n]*com\.brainlayer\.enrichment", text)
    assert "launchctl unload ~/Library/LaunchAgents/com.brainlayer.enrichment.plist" in text
    assert "retired" in text.lower()
