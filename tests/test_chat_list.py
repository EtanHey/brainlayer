"""Test chat list extraction."""
import pytest
from pathlib import Path

from zikaron.pipeline.unified_timeline import UnifiedTimeline, UnifiedMessage
from datetime import datetime


def test_get_chat_list_from_messages():
    """Test get_chat_list groups by chat_id/contact_name."""
    timeline = UnifiedTimeline()
    timeline.messages = [
        UnifiedMessage(datetime.now(), "wa", "he", "hi", True, "Maman", "jid1", "Maman"),
        UnifiedMessage(datetime.now(), "wa", "he", "bye", True, "Maman", "jid1", "Maman"),
        UnifiedMessage(datetime.now(), "wa", "he", "ok", True, "Dad", "jid2", "Dad"),
    ]
    chats = timeline.get_chat_list()
    assert len(chats) == 2
    # Maman has 2, Dad has 1
    assert chats[0][2] == 2  # first by count
    assert chats[0][1] == "Maman"


def test_unified_message_has_chat_fields():
    """UnifiedMessage has chat_id and contact_name."""
    m = UnifiedMessage(datetime.now(), "wa", "en", "hi", True, None, "jid1", "Mom")
    assert m.chat_id == "jid1"
    assert m.contact_name == "Mom"
