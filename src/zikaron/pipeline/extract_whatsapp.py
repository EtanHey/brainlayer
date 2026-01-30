"""Extract WhatsApp messages for communication pattern analysis.

Extracts messages from WhatsApp's SQLite database to analyze:
- User's writing style and patterns
- Response patterns
- Communication preferences
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional


def get_whatsapp_db_path() -> Path:
    """Get path to WhatsApp's ChatStorage database."""
    return Path.home() / "Library/Group Containers/group.net.whatsapp.WhatsApp.shared/ChatStorage.sqlite"


def get_contacts_db_path() -> Path:
    """Get path to WhatsApp's Contacts database."""
    return Path.home() / "Library/Group Containers/group.net.whatsapp.WhatsApp.shared/ContactsV2.sqlite"


def extract_whatsapp_messages(
    db_path: Optional[Path] = None,
    limit: Optional[int] = None,
    only_from_me: bool = False,
    exclude_groups: bool = True,
) -> Iterator[dict]:
    """
    Extract WhatsApp messages from SQLite database.
    
    Args:
        db_path: Path to ChatStorage.sqlite (auto-detected if None)
        limit: Maximum number of messages to extract
        only_from_me: Only extract messages sent by user
        exclude_groups: Exclude group chat messages
    
    Yields:
        Message dictionaries with text, timestamp, sender info
    """
    if db_path is None:
        db_path = get_whatsapp_db_path()
    
    if not db_path.exists():
        raise FileNotFoundError(f"WhatsApp database not found at: {db_path}")
    
    # Open database in read-only mode
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    
    try:
        cursor = conn.cursor()
        
        # Build query
        query = """
            SELECT 
                Z_PK as id,
                ZTEXT as text,
                ZISFROMME as is_from_me,
                ZMESSAGEDATE as message_date,
                ZSENTDATE as sent_date,
                ZFROMJID as from_jid,
                ZTOJID as to_jid,
                ZPUSHNAME as push_name,
                ZMESSAGETYPE as message_type,
                ZMESSAGESTATUS as message_status,
                ZSTARRED as starred,
                ZSTANZAID as stanza_id
            FROM ZWAMESSAGE
            WHERE ZTEXT IS NOT NULL
                AND ZTEXT != ''
        """
        
        # Add filters
        if only_from_me:
            query += " AND ZISFROMME = 1"
        
        if exclude_groups:
            # Group JIDs typically contain '@g.us'
            query += " AND (ZTOJID NOT LIKE '%@g.us' OR ZTOJID IS NULL)"
        
        # Order by date descending (most recent first)
        query += " ORDER BY ZMESSAGEDATE DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        
        for row in cursor:
            # Convert Core Data timestamp to Unix timestamp
            # Core Data uses seconds since 2001-01-01
            core_data_epoch = 978307200  # Unix timestamp for 2001-01-01
            message_timestamp = row['message_date'] + core_data_epoch if row['message_date'] else None
            
            yield {
                'id': row['id'],
                'text': row['text'],
                'is_from_me': bool(row['is_from_me']),
                'timestamp': message_timestamp,
                'datetime': datetime.fromtimestamp(message_timestamp) if message_timestamp else None,
                'from_jid': row['from_jid'],
                'to_jid': row['to_jid'],
                'contact_name': row['push_name'],
                'message_type': row['message_type'],
                'starred': bool(row['starred']),
                'stanza_id': row['stanza_id'],
            }
    
    finally:
        conn.close()


def get_contact_info(jid: str, contacts_db_path: Optional[Path] = None) -> Optional[dict]:
    """
    Get contact information for a JID.
    
    Args:
        jid: WhatsApp JID (e.g., "1234567890@s.whatsapp.net")
        contacts_db_path: Path to ContactsV2.sqlite
    
    Returns:
        Contact info dict or None if not found
    """
    if contacts_db_path is None:
        contacts_db_path = get_contacts_db_path()
    
    if not contacts_db_path.exists():
        return None
    
    conn = sqlite3.connect(f"file:{contacts_db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                ZCONTACTNAME as name,
                ZPHONENUMBER as phone,
                ZSTATUSTEXT as status
            FROM ZWACONTACT
            WHERE ZCONTACTJID = ?
        """, (jid,))
        
        row = cursor.fetchone()
        if row:
            return {
                'name': row['name'],
                'phone': row['phone'],
                'status': row['status'],
            }
        return None
    
    finally:
        conn.close()


def analyze_writing_style(messages: list[dict]) -> dict:
    """
    Analyze user's writing style from messages.
    
    Returns patterns like:
    - Average message length
    - Common phrases
    - Punctuation usage
    - Emoji usage
    - Response time patterns
    """
    user_messages = [m for m in messages if m['is_from_me']]
    
    if not user_messages:
        return {}
    
    total_length = sum(len(m['text']) for m in user_messages)
    avg_length = total_length / len(user_messages)
    
    # Count emoji usage
    emoji_count = sum(
        1 for m in user_messages 
        for char in m['text'] 
        if ord(char) > 0x1F300  # Basic emoji range
    )
    
    # Analyze punctuation
    exclamation_count = sum(m['text'].count('!') for m in user_messages)
    question_count = sum(m['text'].count('?') for m in user_messages)
    
    return {
        'total_messages': len(user_messages),
        'avg_message_length': avg_length,
        'emoji_usage_rate': emoji_count / len(user_messages),
        'exclamation_rate': exclamation_count / len(user_messages),
        'question_rate': question_count / len(user_messages),
        'sample_messages': [m['text'] for m in user_messages[:10]],
    }


def format_whatsapp_for_pipeline(message: dict) -> dict:
    """
    Format WhatsApp message to match zikaron pipeline format.
    
    Converts to a format similar to Claude Code conversations.
    """
    return {
        "type": "whatsapp_message",
        "id": f"whatsapp_{message['id']}",
        "timestamp": message['timestamp'],
        "role": "user" if message['is_from_me'] else "contact",
        "content": message['text'],
        "metadata": {
            "source": "whatsapp",
            "contact": message['contact_name'],
            "jid": message['from_jid'] if not message['is_from_me'] else message['to_jid'],
            "starred": message['starred'],
        }
    }
