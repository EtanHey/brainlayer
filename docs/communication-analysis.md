# Communication Pattern Analysis

> Extract your writing style from WhatsApp and Claude chats to generate personalized rules for AI assistants.

---

## Overview

This feature analyzes your communication patterns from:

1. **WhatsApp messages** - Your real-world texting style
2. **Claude chat transcripts** - How Claude responds to you effectively

It then generates Cursor rules that help AI assistants:
- Match your writing tone and style
- Ask clarifying questions you typically need
- Structure responses the way you prefer

---

## Quick Start

```bash
# Analyze WhatsApp messages (most recent 1000)
zikaron analyze-style

# Analyze more messages
zikaron analyze-style --whatsapp-limit 5000

# Include Claude chat export
zikaron analyze-style --claude-export ~/Downloads/claude-export.json

# Custom output location
zikaron analyze-style --output ~/.cursor/rules/my-style.md

# Export full analysis as JSON
zikaron analyze-style --export-json
```

---

## Data Sources

### 1. WhatsApp Messages

**Location**: `~/Library/Group Containers/group.net.whatsapp.WhatsApp.shared/ChatStorage.sqlite`

The tool automatically reads from WhatsApp's local database (macOS only). It extracts:
- Your sent messages (to understand your writing style)
- Received messages (for context and conversation patterns)
- Excludes group chats by default (use personal 1-on-1 conversations)

**Privacy**: All analysis happens locally. No data is sent anywhere.

### 2. Claude Chat Transcripts

**Getting your Claude chat data**:

#### Option A: Manual Export (Recommended)

1. Go to [claude.ai](https://claude.ai)
2. Click Settings → Data & Privacy
3. Click "Export my data"
4. Download the export file
5. Run: `zikaron analyze-style --claude-export ~/Downloads/claude-export.json`

#### Option B: Browser Export (Advanced)

If Claude doesn't offer export, you can manually copy conversations:

1. Open a Claude conversation
2. Copy the entire conversation text
3. Save to a JSON file with this format:

```json
{
  "conversations": [
    {
      "id": "conv-1",
      "messages": [
        {
          "role": "user",
          "content": "Your message here",
          "timestamp": "2026-01-30T12:00:00Z"
        },
        {
          "role": "assistant",
          "content": "Claude's response here",
          "timestamp": "2026-01-30T12:00:05Z"
        }
      ]
    }
  ]
}
```

#### Option C: IndexedDB Extraction (Experimental)

Claude desktop app stores chats in IndexedDB (LevelDB format). This is complex to extract:

```bash
# Install plyvel for LevelDB access
pip install plyvel

# Then use the extraction module
python -c "
from zikaron.pipeline.extract_claude_desktop import extract_claude_chats
for chat in extract_claude_chats(method='plyvel'):
    print(chat)
"
```

**Note**: IndexedDB extraction is experimental and may not work reliably.

---

## What Gets Analyzed

### Writing Style Metrics

- **Message length**: Average characters per message
- **Formality score**: 0 (very casual) to 1 (very formal)
- **Sentence structure**: Sentences per message
- **Punctuation patterns**: Exclamation marks, questions
- **Emoji usage**: Frequency of emoji use
- **Common phrases**: Your frequently used words
- **Greeting patterns**: How you start conversations

### Claude Response Patterns

- **Structure preferences**: Bullet points, numbered lists, code blocks
- **Tone**: Encouraging vs neutral
- **Explanation style**: Examples, analogies, step-by-step
- **Clarifying questions**: Common questions Claude asks you

---

## Generated Rules

The tool generates a Cursor rule file with:

### 1. Your Communication Profile

```markdown
## User's Writing Style

**Tone**: casual and conversational
**Verbosity**: concise and to-the-point
**Average message length**: 87 characters
**Formality score**: 0.32 (0=informal, 1=formal)

### Preferences

- Use emojis occasionally
- Exclamation usage: 0.45 per message
- Question usage: 0.23 per message
```

### 2. Response Guidelines

```markdown
## Response Style Guidelines

When helping this user with text interactions:

1. **Match their tone**: Write in a casual and conversational style
2. **Match their verbosity**: Keep responses concise and to-the-point
3. **Structure**: 
   - Use bullet points for lists
   - Provide concrete examples
```

### 3. Clarifying Questions

```markdown
## Common Clarifying Questions

When the user's request is ambiguous, consider asking:

- clarify what you mean by "integrate"?
- provide more details about the expected behavior?
- would you like me to update the existing code or create new?
```

### 4. Desktop App Integration

```markdown
## Desktop App Integration

When helping draft messages or responses:

1. **Analyze the context**: What is the user trying to communicate?
2. **Match their style**: Use the tone and verbosity patterns above
3. **Ask clarifying questions**: If unclear, ask specific questions before drafting
4. **Provide options**: Offer 2-3 variations (formal, casual, concise)
```

---

## Use Cases

### 1. Email/Message Drafting

When you need help writing emails or messages, the AI will:
- Match your natural writing style
- Suggest appropriate tone for the context
- Ask clarifying questions before drafting

### 2. Text Response Assistance

For quick text responses:
- Generate replies in your voice
- Match your typical message length
- Use your emoji/punctuation patterns

### 3. Professional Communication

For work emails:
- Adjust formality based on your baseline
- Maintain your communication patterns
- Suggest appropriate structure

---

## Privacy & Security

- **All local**: Analysis happens on your machine
- **No uploads**: Data never leaves your computer
- **Read-only**: WhatsApp database is accessed in read-only mode
- **Encrypted data**: WhatsApp messages are already encrypted; we just read the local copy
- **No storage**: Original messages aren't stored, only aggregate patterns

---

## Troubleshooting

### WhatsApp Database Not Found

**Error**: `WhatsApp database not found at: ~/Library/Group Containers/...`

**Solutions**:
1. Make sure WhatsApp desktop is installed
2. Open WhatsApp at least once to create the database
3. Check if you're using WhatsApp Business (different path)

### Permission Denied

**Error**: `Permission denied` when accessing WhatsApp database

**Solutions**:
1. Grant Terminal/Cursor full disk access in System Preferences
2. macOS: System Preferences → Security & Privacy → Privacy → Full Disk Access
3. Add Terminal.app or Cursor.app to the list

### No Messages Extracted

**Error**: Analysis runs but finds 0 messages

**Solutions**:
1. Check if WhatsApp has messages (open the app)
2. Try increasing the limit: `--whatsapp-limit 10000`
3. Check if messages are in group chats (excluded by default)
4. Use `--include-groups` flag if you want group messages

### Claude Export Format Issues

**Error**: Failed to parse Claude export

**Solutions**:
1. Verify the JSON format matches the expected structure
2. Check for syntax errors in the JSON file
3. Try the manual conversation copy method instead

---

## Advanced Usage

### Analyze Specific Conversations

```python
from zikaron.pipeline.extract_whatsapp import extract_whatsapp_messages

# Get only your messages
my_messages = list(extract_whatsapp_messages(
    only_from_me=True,
    limit=1000
))

# Include group chats
all_messages = list(extract_whatsapp_messages(
    exclude_groups=False,
    limit=5000
))
```

### Custom Analysis

```python
from zikaron.pipeline.analyze_communication import CommunicationAnalyzer

analyzer = CommunicationAnalyzer()

# Add your own message data
analyzer.user_messages.append({
    'text': 'Your message here',
    'source': 'custom',
    'timestamp': None
})

# Generate analysis
style = analyzer.analyze_writing_style()
print(style)
```

### Export Analysis Data

```bash
# Export full analysis as JSON for further processing
zikaron analyze-style --export-json

# Output: ~/.cursor/rules/communication-style.json
```

The JSON contains:
- Raw metrics
- Sample messages
- Clarifying questions
- All analysis data

---

## Integration with Cursor

The generated rules are automatically saved to `~/.cursor/rules/communication-style.md`.

Cursor will automatically load these rules and apply them when:
- You ask for help drafting messages
- You request text/email assistance
- You use the AI for communication tasks

You can also reference the rules explicitly:

```
@communication-style.md Help me write a response to this email
```

---

## Future Enhancements

- [ ] Support for other messaging platforms (Telegram, Signal, iMessage)
- [ ] Slack workspace analysis
- [ ] Email analysis (Gmail, Outlook)
- [ ] Conversation threading detection
- [ ] Sentiment analysis
- [ ] Time-based patterns (morning vs evening style)
- [ ] Contact-specific styles (formal with boss, casual with friends)

---

## Examples

### Example 1: Casual Communicator

**Input**: 1000 WhatsApp messages, casual style

**Generated Rule**:
```markdown
**Tone**: casual and conversational
**Verbosity**: concise and to-the-point
**Emoji rate**: 1.2 per message

When drafting messages, use:
- Short sentences
- Casual language ("gonna", "wanna")
- Emojis for emphasis
- Minimal punctuation
```

### Example 2: Professional Communicator

**Input**: 1000 WhatsApp messages, formal style

**Generated Rule**:
```markdown
**Tone**: professional and formal
**Verbosity**: detailed and thorough
**Emoji rate**: 0.1 per message

When drafting messages, use:
- Complete sentences
- Formal language
- Proper punctuation
- Structured paragraphs
```

---

## Contributing

Have ideas for improving communication analysis? Open an issue or PR!

Areas for contribution:
- Additional messaging platforms
- Better pattern detection
- More sophisticated style analysis
- UI for reviewing/editing generated rules

---

## License

Same as zikaron project (see main README).
