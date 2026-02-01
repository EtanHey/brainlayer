"""Chat tagging for relationship context (family, friends, co-workers)."""

from pathlib import Path
from typing import Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def get_chat_tags_path() -> Path:
    """Default path for chat-tags config."""
    return Path.home() / ".config" / "zikaron" / "chat-tags.yaml"


def load_chat_tags(config_path: Optional[Path] = None) -> dict[str, str]:
    """
    Load chat_id/contact_name -> tag mapping from YAML.
    
    Returns:
        Dict mapping contact_name or chat_id to tag (family, friends, co-workers, etc.)
    """
    if not HAS_YAML:
        return {}
    
    path = config_path or get_chat_tags_path()
    if not path.exists():
        return {}
    
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    if not data or "tags" not in data:
        return {}
    
    mapping = {}
    for entry in data["tags"]:
        tag = entry.get("tag")
        if not tag:
            continue
        if "contact" in entry:
            mapping[entry["contact"]] = tag
        if "jid" in entry or "chat_id" in entry:
            jid = entry.get("jid") or entry.get("chat_id")
            mapping[jid] = tag
    
    return mapping


def get_tag_for_message(
    contact_name: Optional[str],
    chat_id: Optional[str],
    tags: dict[str, str],
) -> Optional[str]:
    """Get relationship tag for a message. Checks contact_name first, then chat_id."""
    if contact_name and contact_name in tags:
        return tags[contact_name]
    if chat_id and chat_id in tags:
        return tags[chat_id]
    return None
