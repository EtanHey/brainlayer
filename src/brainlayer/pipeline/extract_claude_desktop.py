"""Extract Claude chat transcripts from desktop/web app IndexedDB.

Claude desktop app stores conversations in LevelDB (IndexedDB backend).
This module extracts chat data for analysis.
"""

import json
import subprocess
from pathlib import Path
from typing import Iterator

try:
    import plyvel
    HAS_PLYVEL = True
except ImportError:
    HAS_PLYVEL = False


def get_claude_indexeddb_path() -> Path:
    """Get path to Claude desktop app's IndexedDB."""
    return Path.home() / "Library/Application Support/Claude/IndexedDB/https_claude.ai_0.indexeddb.leveldb"


def extract_with_plyvel(db_path: Path) -> Iterator[dict]:
    """Extract data using plyvel (requires: pip install plyvel)."""
    if not HAS_PLYVEL:
        raise ImportError("plyvel not installed. Run: pip install plyvel")
    
    db = plyvel.DB(str(db_path), create_if_missing=False)
    
    try:
        for key, value in db:
            # IndexedDB stores data with metadata prefixes
            # We need to parse the binary format
            try:
                # Try to decode as JSON (some values are JSON strings)
                decoded = value.decode('utf-8', errors='ignore')
                if decoded.startswith('{') or decoded.startswith('['):
                    data = json.loads(decoded)
                    yield data
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    finally:
        db.close()


def extract_with_chrome_devtools() -> Iterator[dict]:
    """
    Extract using Chrome DevTools Protocol (alternative method).
    
    This requires the Claude desktop app to be running and accessible.
    More reliable than parsing LevelDB directly.
    """
    # TODO: Implement CDP extraction
    # This would connect to the running app and extract via JavaScript
    raise NotImplementedError("Chrome DevTools extraction not yet implemented")


def extract_conversations_from_export(export_path: Path) -> Iterator[dict]:
    """
    Extract from Claude chat export (if available).
    
    Claude.ai may offer export functionality - this handles that format.
    """
    if not export_path.exists():
        raise FileNotFoundError(f"Export file not found: {export_path}")
    
    with open(export_path, 'r') as f:
        data = json.load(f)
        
    # Format depends on Claude's export structure
    # Assuming it's a list of conversations
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict) and 'conversations' in data:
        yield from data['conversations']


def extract_claude_chats(method: str = "manual") -> Iterator[dict]:
    """
    Extract Claude chat transcripts.
    
    Args:
        method: Extraction method
            - "manual": User manually exports from Claude.ai
            - "plyvel": Direct LevelDB access (requires plyvel)
            - "cdp": Chrome DevTools Protocol (requires running app)
    
    Yields:
        Chat conversation dictionaries
    """
    if method == "manual":
        # Guide user to export manually
        print("\n" + "="*70)
        print("MANUAL EXPORT INSTRUCTIONS")
        print("="*70)
        print("\n1. Open Claude.ai in your browser")
        print("2. Go to Settings > Data & Privacy")
        print("3. Click 'Export my data'")
        print("4. Download the export file")
        print("5. Save it to: ~/.local/share/brainlayer/claude-export.json")
        print("\nAlternatively, copy conversations manually from the web interface.")
        print("="*70 + "\n")
        
        export_path = Path.home() / ".local/share/brainlayer/claude-export.json"
        if export_path.exists():
            yield from extract_conversations_from_export(export_path)
        else:
            print(f"Waiting for export at: {export_path}")
            return
            
    elif method == "plyvel":
        db_path = get_claude_indexeddb_path()
        if not db_path.exists():
            raise FileNotFoundError(f"Claude IndexedDB not found at: {db_path}")
        yield from extract_with_plyvel(db_path)
        
    elif method == "cdp":
        yield from extract_with_chrome_devtools()
        
    else:
        raise ValueError(f"Unknown extraction method: {method}")


def format_claude_chat_for_pipeline(conversation: dict) -> dict:
    """
    Format Claude chat data to match brainlayer pipeline format.
    
    Converts Claude's chat format to the JSONL format used by Claude Code.
    """
    # This depends on the actual format from Claude.ai
    # Placeholder structure:
    return {
        "type": "conversation",
        "id": conversation.get("id"),
        "created_at": conversation.get("created_at"),
        "messages": [
            {
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
                "timestamp": msg.get("timestamp"),
            }
            for msg in conversation.get("messages", [])
        ],
        "metadata": {
            "source": "claude_desktop",
            "model": conversation.get("model"),
        }
    }
