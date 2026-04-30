"""Tests for the multi-agent ingest launchd wiring."""

from pathlib import Path
import xml.etree.ElementTree as ET


def test_agent_ingest_plist_has_correct_label_and_command():
    plist_path = Path(__file__).parent.parent / "scripts" / "launchd" / "com.brainlayer.agent-ingest.plist"
    tree = ET.parse(plist_path)
    root = tree.getroot()

    strings = [elem.text for elem in root.iter("string") if elem.text]

    assert "com.brainlayer.agent-ingest" in strings
    assert "__BRAINLAYER_BIN__" in strings
    assert "watch-agents" in strings


def test_launchd_installer_mentions_agent_ingest():
    install_script = (Path(__file__).parent.parent / "scripts" / "launchd" / "install.sh").read_text()

    assert "install_plist agent-ingest" in install_script
    assert "remove_plist agent-ingest" in install_script
