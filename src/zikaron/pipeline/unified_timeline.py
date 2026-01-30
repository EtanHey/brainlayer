"""Unified timeline for multi-source communication analysis.

Normalizes messages from WhatsApp, Claude, Instagram, Gemini into a single format
for longitudinal analysis.
"""

import json
import re
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional


@dataclass
class UnifiedMessage:
    """Normalized message format for all sources."""
    timestamp: datetime
    source: str  # "whatsapp", "claude", "instagram", "gemini"
    language: str  # "hebrew", "english", "mixed"
    text: str
    is_from_user: bool
    context: Optional[str] = None  # conversation name, thread, contact, etc.
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> 'UnifiedMessage':
        """Create from dictionary."""
        d['timestamp'] = datetime.fromisoformat(d['timestamp'])
        return cls(**d)


# Hebrew character detection
HEBREW_PATTERN = re.compile(r'[\u0590-\u05FF]')


def detect_language(text: str) -> str:
    """Detect if text is primarily Hebrew, English, or mixed."""
    if not text:
        return "unknown"
    
    hebrew_chars = len(HEBREW_PATTERN.findall(text))
    alpha_chars = len([c for c in text if c.isalpha()])
    
    if alpha_chars == 0:
        return "unknown"
    
    hebrew_ratio = hebrew_chars / alpha_chars
    
    if hebrew_ratio > 0.7:
        return "hebrew"
    elif hebrew_ratio < 0.3:
        return "english"
    else:
        return "mixed"


def load_whatsapp_messages(
    db_path: Optional[Path] = None,
    only_from_user: bool = True,
    exclude_groups: bool = True,
) -> Iterator[UnifiedMessage]:
    """
    Load messages from WhatsApp SQLite database.
    
    Args:
        db_path: Path to ChatStorage.sqlite (auto-detected if None)
        only_from_user: Only include messages sent by user
        exclude_groups: Exclude group chat messages
    
    Yields:
        UnifiedMessage objects
    """
    if db_path is None:
        db_path = Path.home() / "Library/Group Containers/group.net.whatsapp.WhatsApp.shared/ChatStorage.sqlite"
    
    if not db_path.exists():
        raise FileNotFoundError(f"WhatsApp database not found: {db_path}")
    
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Core Data epoch (2001-01-01)
    core_data_epoch = 978307200
    
    query = """
        SELECT 
            ZTEXT as text,
            ZISFROMME as is_from_me,
            ZMESSAGEDATE as message_date,
            ZPUSHNAME as contact_name,
            ZTOJID as to_jid
        FROM ZWAMESSAGE
        WHERE ZTEXT IS NOT NULL AND ZTEXT != ''
    """
    
    if only_from_user:
        query += " AND ZISFROMME = 1"
    
    if exclude_groups:
        query += " AND (ZTOJID NOT LIKE '%@g.us' OR ZTOJID IS NULL)"
    
    query += " ORDER BY ZMESSAGEDATE ASC"
    
    cursor.execute(query)
    
    for row in cursor:
        timestamp = datetime.fromtimestamp(row['message_date'] + core_data_epoch)
        text = row['text']
        
        yield UnifiedMessage(
            timestamp=timestamp,
            source="whatsapp",
            language=detect_language(text),
            text=text,
            is_from_user=bool(row['is_from_me']),
            context=row['contact_name'],
        )
    
    conn.close()


def load_claude_messages(
    export_path: Path,
    only_from_user: bool = True,
) -> Iterator[UnifiedMessage]:
    """
    Load messages from Claude.ai export.
    
    Args:
        export_path: Path to conversations.json from Claude export
        only_from_user: Only include user messages (not assistant)
    
    Yields:
        UnifiedMessage objects
    """
    if not export_path.exists():
        raise FileNotFoundError(f"Claude export not found: {export_path}")
    
    with open(export_path, 'r') as f:
        data = json.load(f)
    
    for conv in data:
        conv_name = conv.get('name', 'Untitled')
        created_at = conv.get('created_at', '')
        
        for msg in conv.get('chat_messages', []):
            sender = msg.get('sender', '')
            is_user = sender == 'human'
            
            if only_from_user and not is_user:
                continue
            
            text = msg.get('text', '')
            if not text:
                continue
            
            # Parse timestamp
            msg_created = msg.get('created_at', created_at)
            try:
                timestamp = datetime.fromisoformat(msg_created.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                # Fallback to conversation creation time
                try:
                    timestamp = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    timestamp = datetime.now()
            
            yield UnifiedMessage(
                timestamp=timestamp,
                source="claude",
                language=detect_language(text),
                text=text,
                is_from_user=is_user,
                context=conv_name,
            )


def load_instagram_messages(export_path: Path) -> Iterator[UnifiedMessage]:
    """
    Load messages from Instagram JSON export.
    
    Args:
        export_path: Path to Instagram export directory or comments.json
    
    Yields:
        UnifiedMessage objects
    """
    # Handle directory or direct file
    if export_path.is_dir():
        comments_path = export_path / "comments" / "post_comments_1.json"
        if not comments_path.exists():
            # Try alternative structure
            comments_path = export_path / "your_instagram_activity" / "comments" / "post_comments_1.json"
    else:
        comments_path = export_path
    
    if not comments_path.exists():
        raise FileNotFoundError(f"Instagram comments not found: {comments_path}")
    
    with open(comments_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Instagram export format varies, handle common structures
    comments = data if isinstance(data, list) else data.get('comments', [])
    
    for comment in comments:
        # Extract text and timestamp based on Instagram's JSON structure
        text = comment.get('string_list_data', [{}])[0].get('value', '') if 'string_list_data' in comment else comment.get('text', '')
        timestamp_val = comment.get('string_list_data', [{}])[0].get('timestamp', 0) if 'string_list_data' in comment else comment.get('timestamp', 0)
        
        if not text:
            continue
        
        try:
            timestamp = datetime.fromtimestamp(timestamp_val)
        except (ValueError, TypeError, OSError):
            timestamp = datetime.now()
        
        yield UnifiedMessage(
            timestamp=timestamp,
            source="instagram",
            language=detect_language(text),
            text=text,
            is_from_user=True,  # Comments are always from user
            context="instagram_comment",
        )


def load_gemini_messages(export_path: Path) -> Iterator[UnifiedMessage]:
    """
    Load messages from Gemini/Google Takeout export.
    
    Args:
        export_path: Path to Gemini activity JSON or Takeout directory
    
    Yields:
        UnifiedMessage objects
    """
    # Handle directory or direct file
    if export_path.is_dir():
        activity_path = export_path / "My Activity" / "Gemini Apps" / "MyActivity.json"
        if not activity_path.exists():
            activity_path = export_path / "Takeout" / "My Activity" / "Gemini Apps" / "MyActivity.json"
    else:
        activity_path = export_path
    
    if not activity_path.exists():
        raise FileNotFoundError(f"Gemini activity not found: {activity_path}")
    
    with open(activity_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    activities = data if isinstance(data, list) else [data]
    
    for activity in activities:
        # Google Takeout format for Gemini
        text = activity.get('title', '') or activity.get('query', '')
        time_str = activity.get('time', '')
        
        if not text:
            continue
        
        try:
            timestamp = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            timestamp = datetime.now()
        
        yield UnifiedMessage(
            timestamp=timestamp,
            source="gemini",
            language=detect_language(text),
            text=text,
            is_from_user=True,
            context="gemini_chat",
        )


class UnifiedTimeline:
    """Container for messages from all sources."""
    
    def __init__(self):
        self.messages: list[UnifiedMessage] = []
        self.sources_loaded: set[str] = set()
    
    def load_whatsapp(self, db_path: Optional[Path] = None, **kwargs) -> int:
        """Load WhatsApp messages."""
        count = 0
        for msg in load_whatsapp_messages(db_path, **kwargs):
            self.messages.append(msg)
            count += 1
        self.sources_loaded.add("whatsapp")
        return count
    
    def load_claude(self, export_path: Path, **kwargs) -> int:
        """Load Claude messages."""
        count = 0
        for msg in load_claude_messages(export_path, **kwargs):
            self.messages.append(msg)
            count += 1
        self.sources_loaded.add("claude")
        return count
    
    def load_instagram(self, export_path: Path) -> int:
        """Load Instagram messages."""
        count = 0
        for msg in load_instagram_messages(export_path):
            self.messages.append(msg)
            count += 1
        self.sources_loaded.add("instagram")
        return count
    
    def load_gemini(self, export_path: Path) -> int:
        """Load Gemini messages."""
        count = 0
        for msg in load_gemini_messages(export_path):
            self.messages.append(msg)
            count += 1
        self.sources_loaded.add("gemini")
        return count
    
    def sort_by_time(self):
        """Sort all messages by timestamp."""
        self.messages.sort(key=lambda m: m.timestamp)
    
    def filter_by_language(self, language: str) -> list[UnifiedMessage]:
        """Get messages in a specific language."""
        return [m for m in self.messages if m.language == language]
    
    def filter_by_source(self, source: str) -> list[UnifiedMessage]:
        """Get messages from a specific source."""
        return [m for m in self.messages if m.source == source]
    
    def get_date_range(self) -> tuple[datetime, datetime]:
        """Get the date range of all messages."""
        if not self.messages:
            return (datetime.now(), datetime.now())
        
        dates = [m.timestamp for m in self.messages]
        return (min(dates), max(dates))
    
    def get_stats(self) -> dict:
        """Get statistics about the timeline."""
        stats = {
            "total_messages": len(self.messages),
            "sources": list(self.sources_loaded),
            "by_source": {},
            "by_language": {},
            "date_range": None,
        }
        
        for source in self.sources_loaded:
            stats["by_source"][source] = len(self.filter_by_source(source))
        
        for lang in ["hebrew", "english", "mixed"]:
            count = len(self.filter_by_language(lang))
            if count > 0:
                stats["by_language"][lang] = count
        
        if self.messages:
            start, end = self.get_date_range()
            stats["date_range"] = {
                "start": start.isoformat(),
                "end": end.isoformat(),
            }
        
        return stats
    
    def save(self, path: Path):
        """Save timeline to JSON file."""
        data = {
            "sources": list(self.sources_loaded),
            "messages": [m.to_dict() for m in self.messages],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'UnifiedTimeline':
        """Load timeline from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        timeline = cls()
        timeline.sources_loaded = set(data.get("sources", []))
        timeline.messages = [
            UnifiedMessage.from_dict(m) for m in data.get("messages", [])
        ]
        return timeline
