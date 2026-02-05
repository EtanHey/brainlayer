"""Analyze communication patterns to generate personalized rules.

Extracts patterns from:
1. User's writing style (from WhatsApp, Claude chats)
2. Claude's response patterns that work well for the user
3. Common clarifying questions and their answers

Generates rules for desktop apps to help with text interactions.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import re


class CommunicationAnalyzer:
    """Analyze communication patterns and generate rules."""
    
    def __init__(self):
        self.user_messages = []
        self.assistant_messages = []
        self.clarifying_questions = []
        
    def add_whatsapp_messages(self, messages: list[dict]):
        """Add WhatsApp messages to analysis."""
        for msg in messages:
            if msg['is_from_me']:
                self.user_messages.append({
                    'text': msg['text'],
                    'source': 'whatsapp',
                    'timestamp': msg.get('timestamp'),
                })
    
    def add_claude_conversations(self, conversations: list[dict]):
        """Add Claude conversations to analysis."""
        for conv in conversations:
            for msg in conv.get('messages', []):
                if msg.get('role') == 'user':
                    self.user_messages.append({
                        'text': msg.get('content', ''),
                        'source': 'claude',
                        'timestamp': msg.get('timestamp'),
                    })
                elif msg.get('role') == 'assistant':
                    self.assistant_messages.append({
                        'text': msg.get('content', ''),
                        'source': 'claude',
                        'timestamp': msg.get('timestamp'),
                    })
                    
                    # Detect clarifying questions
                    if self._is_clarifying_question(msg.get('content', '')):
                        self.clarifying_questions.append(msg.get('content', ''))
    
    def _is_clarifying_question(self, text: str) -> bool:
        """Detect if a message is a clarifying question."""
        question_markers = [
            'could you clarify',
            'what do you mean',
            'can you provide more details',
            'would you like me to',
            'should i',
            'do you want',
            'which',
            'how would you like',
        ]
        text_lower = text.lower()
        return any(marker in text_lower for marker in question_markers) and '?' in text
    
    def analyze_writing_style(self) -> dict[str, Any]:
        """Analyze user's writing style."""
        if not self.user_messages:
            return {}
        
        texts = [msg['text'] for msg in self.user_messages]
        
        # Length analysis
        lengths = [len(text) for text in texts]
        avg_length = sum(lengths) / len(lengths)
        
        # Sentence structure
        sentences_per_message = [text.count('.') + text.count('!') + text.count('?') for text in texts]
        avg_sentences = sum(sentences_per_message) / len(sentences_per_message)
        
        # Formality analysis - comprehensive markers
        # English informal markers
        informal_markers_en = [
            'lol', 'haha', 'hahaha', 'btw', 'idk', 'tbh', 'ngl', 'omg', 'wtf',
            'gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'dunno', 'lemme',
            'yeah', 'yep', 'nope', 'nah', 'yup', 'ok', 'okay', 'k',
            'cool', 'awesome', 'dude', 'guys', 'hey', 'yo', 'sup',
            'lmao', 'lmfao', 'rofl', 'brb', 'ttyl', 'imo', 'imho', 'fyi',
            'thx', 'ty', 'np', 'pls', 'plz', 'rn', 'asap', 'afaik',
        ]
        # Hebrew informal markers
        informal_markers_he = [
            'חח', 'חחח', 'חחחח', 'לול', 'אוקי', 'יאללה', 'סבבה',
            'וואלה', 'אחלה', 'יופי', 'בסדר', 'טוב', 'נו', 'מה',
            'כן', 'לא', 'אז', 'רגע', 'שניה', 'בקיצור',
        ]
        informal_markers = informal_markers_en + informal_markers_he

        # Count informal markers
        informal_count = sum(
            1 for text in texts
            for marker in informal_markers
            if marker in text.lower()
        )

        # Also check for formal indicators (increase formality score)
        formal_markers = [
            'dear', 'sincerely', 'regards', 'respectfully', 'hereby',
            'pursuant', 'kindly', 'please find', 'attached herewith',
            'i am writing to', 'to whom it may concern',
        ]
        formal_count = sum(
            1 for text in texts
            for marker in formal_markers
            if marker in text.lower()
        )

        # Check for casual style indicators (count messages with casual traits)
        casual_messages = 0
        for text in texts:
            is_casual = False
            # Short messages are casual
            if len(text) < 30:
                is_casual = True
            # Messages without proper capitalization
            if text and text[0].islower():
                is_casual = True
            # Messages ending without punctuation
            if text and text[-1] not in '.!?':
                is_casual = True
            # Contractions
            if any(c in text.lower() for c in ["don't", "won't", "can't", "i'm", "it's", "that's", "what's"]):
                is_casual = True
            if is_casual:
                casual_messages += 1

        # Calculate ratios (all 0-1 range now, clamped for consistency)
        informal_marker_ratio = min(1.0, informal_count / max(len(texts), 1))
        casual_style_ratio = casual_messages / max(len(texts), 1)
        formal_marker_ratio = min(1.0, formal_count / max(len(texts), 1))

        # Formality score: 0 = very casual, 1 = very formal
        # Start at 0.5, subtract for informal/casual indicators, add for formal
        # Tuned weights: casual=0.15, informal=0.1 (lighter touch to avoid extremes)
        raw_score = 0.5 - (casual_style_ratio * 0.15) - (informal_marker_ratio * 0.1) + (formal_marker_ratio * 0.3)
        formality_score = max(0.1, min(0.9, raw_score))  # Clamp to 0.1-0.9 range
        
        # Punctuation patterns
        exclamation_rate = sum(text.count('!') for text in texts) / len(texts)
        question_rate = sum(text.count('?') for text in texts) / len(texts)
        
        # Emoji usage
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", 
            flags=re.UNICODE
        )
        emoji_count = sum(len(emoji_pattern.findall(text)) for text in texts)
        emoji_rate = emoji_count / len(texts)
        
        # Common phrases
        all_text = ' '.join(texts).lower()
        words = re.findall(r'\b\w+\b', all_text)
        word_freq = Counter(words)
        common_words = [word for word, count in word_freq.most_common(20) if len(word) > 3]
        
        # Greeting patterns
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        greeting_usage = sum(
            1 for text in texts 
            for greeting in greetings 
            if text.lower().startswith(greeting)
        ) / len(texts)
        
        return {
            'avg_message_length': avg_length,
            'avg_sentences_per_message': avg_sentences,
            'formality_score': formality_score,  # 0 = very informal, 1 = very formal
            'exclamation_rate': exclamation_rate,
            'question_rate': question_rate,
            'emoji_rate': emoji_rate,
            'common_words': common_words,
            'greeting_usage': greeting_usage,
            'total_messages_analyzed': len(texts),
        }
    
    def analyze_claude_response_patterns(self) -> dict[str, Any]:
        """Analyze Claude's response patterns that work well."""
        if not self.assistant_messages:
            return {}
        
        texts = [msg['text'] for msg in self.assistant_messages]
        
        # Response structure
        uses_bullet_points = sum('•' in text or '- ' in text for text in texts) / len(texts)
        uses_code_blocks = sum('```' in text for text in texts) / len(texts)
        uses_numbered_lists = sum(bool(re.search(r'\n\d+\.', text)) for text in texts) / len(texts)
        
        # Tone analysis
        encouraging_phrases = ['great', 'excellent', 'perfect', 'nice work', 'well done']
        encouraging_rate = sum(
            1 for text in texts 
            for phrase in encouraging_phrases 
            if phrase in text.lower()
        ) / len(texts)
        
        # Explanation style
        uses_examples = sum('for example' in text.lower() or 'e.g.' in text.lower() for text in texts) / len(texts)
        uses_analogies = sum('like' in text.lower() or 'similar to' in text.lower() for text in texts) / len(texts)
        
        return {
            'uses_bullet_points_rate': uses_bullet_points,
            'uses_code_blocks_rate': uses_code_blocks,
            'uses_numbered_lists_rate': uses_numbered_lists,
            'encouraging_rate': encouraging_rate,
            'uses_examples_rate': uses_examples,
            'uses_analogies_rate': uses_analogies,
            'total_responses_analyzed': len(texts),
        }
    
    def extract_common_clarifications(self) -> list[str]:
        """Extract common clarifying questions Claude asks."""
        if not self.clarifying_questions:
            return []
        
        # Group similar questions
        question_patterns = []
        for q in self.clarifying_questions[:20]:  # Top 20
            # Extract the core question
            core = re.sub(r'^(could you|can you|would you|should i)\s+', '', q.lower())
            question_patterns.append(core)
        
        return question_patterns
    
    def generate_rules(self, output_path: Path):
        """Generate Cursor rules based on analysis."""
        writing_style = self.analyze_writing_style()
        response_patterns = self.analyze_claude_response_patterns()
        clarifications = self.extract_common_clarifications()
        
        rules = self._format_rules(writing_style, response_patterns, clarifications)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(rules)
        
        return rules
    
    def _format_rules(
        self, 
        writing_style: dict, 
        response_patterns: dict, 
        clarifications: list[str]
    ) -> str:
        """Format analysis into Cursor rules."""
        
        # Determine tone
        if writing_style.get('formality_score', 0.5) > 0.7:
            tone = "professional and formal"
        elif writing_style.get('formality_score', 0.5) < 0.3:
            tone = "casual and conversational"
        else:
            tone = "balanced between casual and professional"
        
        # Determine verbosity
        avg_length = writing_style.get('avg_message_length', 100)
        if avg_length < 50:
            verbosity = "concise and to-the-point"
        elif avg_length > 200:
            verbosity = "detailed and thorough"
        else:
            verbosity = "moderately detailed"
        
        # Emoji preference
        emoji_rate = writing_style.get('emoji_rate', 0)
        emoji_pref = "Use emojis occasionally" if emoji_rate > 0.5 else "Avoid emojis unless requested"
        
        # Response structure preference
        structure_prefs = []
        if response_patterns.get('uses_bullet_points_rate', 0) > 0.5:
            structure_prefs.append("Use bullet points for lists")
        if response_patterns.get('uses_numbered_lists_rate', 0) > 0.5:
            structure_prefs.append("Use numbered lists for sequential steps")
        if response_patterns.get('uses_examples_rate', 0) > 0.5:
            structure_prefs.append("Provide concrete examples")
        
        rules_content = f"""# Communication Style Rules
# Auto-generated from analysis of {writing_style.get('total_messages_analyzed', 0)} messages

## User's Writing Style

**Tone**: {tone}
**Verbosity**: {verbosity}
**Average message length**: {writing_style.get('avg_message_length', 0):.0f} characters
**Formality score**: {writing_style.get('formality_score', 0):.2f} (0=informal, 1=formal)

### Preferences

- {emoji_pref}
- Exclamation usage: {writing_style.get('exclamation_rate', 0):.2f} per message
- Question usage: {writing_style.get('question_rate', 0):.2f} per message

## Response Style Guidelines

When helping this user with text interactions:

1. **Match their tone**: Write in a {tone} style
2. **Match their verbosity**: Keep responses {verbosity}
3. **Structure**: 
   {chr(10).join(f'   - {pref}' for pref in structure_prefs) if structure_prefs else '   - Use clear, simple structure'}

## Common Clarifying Questions

When the user's request is ambiguous, consider asking:

{chr(10).join(f'- {q}' for q in clarifications[:10]) if clarifications else '- Ask for specific details about their goal'}

## Desktop App Integration

When helping draft messages or responses:

1. **Analyze the context**: What is the user trying to communicate?
2. **Match their style**: Use the tone and verbosity patterns above
3. **Ask clarifying questions**: If unclear, ask specific questions before drafting
4. **Provide options**: Offer 2-3 variations (formal, casual, concise)

## Example Phrases

Common words/phrases this user uses:
{', '.join(writing_style.get('common_words', [])[:15])}

Use these naturally when appropriate to match their voice.

---

*Generated: {Path.cwd()}*
*Analysis based on WhatsApp and Claude chat history*
"""
        
        return rules_content
    
    def export_analysis(self, output_path: Path):
        """Export full analysis as JSON."""
        analysis = {
            'writing_style': self.analyze_writing_style(),
            'response_patterns': self.analyze_claude_response_patterns(),
            'clarifying_questions': self.extract_common_clarifications(),
            'sample_user_messages': [msg['text'] for msg in self.user_messages[:20]],
            'sample_assistant_messages': [msg['text'] for msg in self.assistant_messages[:20]],
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
