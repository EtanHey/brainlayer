"""Extract correction patterns from Claude/Gemini conversations.

Identifies where AI assistants helped draft text and user made corrections,
to learn what the user consistently changes.
"""

import json
import re
from pathlib import Path
from typing import Iterator, Optional


def extract_claude_export_conversations(export_path: Path) -> Iterator[dict]:
    """
    Extract conversations from Claude.ai export.
    
    Args:
        export_path: Path to extracted conversations.json
    
    Yields:
        Conversation dictionaries with messages
    """
    with open(export_path, 'r') as f:
        data = json.load(f)
    
    for conv in data:
        yield {
            'id': conv.get('uuid'),
            'name': conv.get('name', 'Untitled'),
            'created_at': conv.get('created_at'),
            'messages': [
                {
                    'role': 'user' if msg.get('sender') == 'human' else 'assistant',
                    'content': msg.get('text', ''),
                    'created_at': msg.get('created_at'),
                }
                for msg in conv.get('chat_messages', [])
            ]
        }


def is_draft_request(text: str) -> bool:
    """Detect if user is asking for help drafting text."""
    patterns = [
        r'write (me )?a',
        r'draft (me )?a',
        r'help me write',
        r'can you write',
        r'make (me )?a',
        r'does this sound',
        r'is this (good|right|ok)',
        r'how does this sound',
        r'rewrite this',
        r'improve this',
        r'fix this',
        r'better way to say',
    ]
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in patterns)


def is_correction(user_msg: str, prev_assistant_msg: str) -> bool:
    """
    Detect if user message is correcting the assistant's draft.
    
    Patterns:
    - "No, make it..."
    - "Actually, change it to..."
    - "Too formal, use..."
    - User provides alternative text after assistant's draft
    """
    user_lower = user_msg.lower()
    
    correction_markers = [
        'no,', 'actually,', 'instead', 'change it', 'make it',
        'too formal', 'too long', 'too short', 'simpler',
        'more casual', 'more professional', 'shorter',
        'just say', 'better:', 'fix:', 'use this:',
    ]
    
    # Check for correction markers
    if any(marker in user_lower for marker in correction_markers):
        return True
    
    # Check if user provides alternative text (quoted or after colon)
    if '"' in user_msg or "'" in user_msg or ':' in user_msg:
        # User might be providing alternative text
        return True
    
    return False


def extract_correction_pairs(conversation: dict) -> list[dict]:
    """
    Extract (draft, correction) pairs from a conversation.
    
    Returns:
        List of correction dictionaries with:
        - original_draft: AI's draft
        - user_correction: User's correction/feedback
        - context: What they were drafting
    """
    corrections = []
    messages = conversation['messages']
    
    for i in range(len(messages) - 1):
        current = messages[i]
        next_msg = messages[i + 1] if i + 1 < len(messages) else None
        
        # Look for pattern: User asks for draft → AI provides → User corrects
        if current['role'] == 'user' and is_draft_request(current['content']):
            # Find AI's response
            if next_msg and next_msg['role'] == 'assistant':
                ai_draft = next_msg['content']
                
                # Check if user corrects it
                if i + 2 < len(messages):
                    user_response = messages[i + 2]
                    if user_response['role'] == 'user' and is_correction(user_response['content'], ai_draft):
                        corrections.append({
                            'context': current['content'],
                            'ai_draft': ai_draft,
                            'user_correction': user_response['content'],
                            'conversation_name': conversation['name'],
                        })
    
    return corrections


def analyze_correction_patterns(corrections: list[dict]) -> dict:
    """
    Analyze what user consistently changes in AI drafts.
    
    Returns:
        Dictionary with patterns like:
        - length_changes: Shorter/longer
        - formality_changes: More/less formal
        - common_edits: Specific words/phrases changed
        - tone_adjustments: Casual/professional
    """
    if not corrections:
        return {}
    
    patterns = {
        'total_corrections': len(corrections),
        'makes_shorter': 0,
        'makes_longer': 0,
        'removes_formality': 0,
        'adds_emojis': 0,
        'simplifies_language': 0,
        'common_feedback': [],
    }
    
    for corr in corrections:
        draft = corr['ai_draft']
        feedback = corr['user_correction'].lower()
        
        # Length changes
        if 'shorter' in feedback or 'too long' in feedback or 'brief' in feedback:
            patterns['makes_shorter'] += 1
        if 'longer' in feedback or 'more detail' in feedback:
            patterns['makes_longer'] += 1
        
        # Formality
        if any(word in feedback for word in ['casual', 'informal', 'too formal', 'stiff']):
            patterns['removes_formality'] += 1
        
        # Emojis
        if any(char in corr['user_correction'] for char in ['😂', '😊', '🔥', '💪', '👍', '❤️']):
            patterns['adds_emojis'] += 1
        
        # Simplification
        if any(word in feedback for word in ['simpler', 'simple', 'easier', 'plain']):
            patterns['simplifies_language'] += 1
        
        # Extract common feedback phrases
        feedback_phrases = [
            'too formal', 'too long', 'too short', 'more casual',
            'less formal', 'simpler', 'add emoji', 'remove',
        ]
        for phrase in feedback_phrases:
            if phrase in feedback:
                patterns['common_feedback'].append(phrase)
    
    return patterns


def extract_user_final_versions(corrections: list[dict]) -> list[str]:
    """
    Extract the final versions user actually used/approved.
    
    These are the gold standard examples of their style.
    """
    final_versions = []
    
    for corr in corrections:
        # Try to extract quoted text from user's correction
        user_text = corr['user_correction']
        
        # Look for quoted text
        quoted = re.findall(r'"([^"]+)"', user_text)
        if quoted:
            final_versions.extend(quoted)
        
        # Look for text after colon
        if ':' in user_text:
            parts = user_text.split(':', 1)
            if len(parts) > 1:
                final_versions.append(parts[1].strip())
    
    return final_versions
