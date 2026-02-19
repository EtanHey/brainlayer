# Chat-Based Analysis Design

## Overview

Extend the longitudinal analysis to group by chat first, then optionally by time. Add relationship tags so the model understands *who* you're speaking with, not just *when* and *what*.

---

## 1. Data: What We Have

### WhatsApp
- **Per chat**: `ZCONTACTJID` (e.g. `972501234567@s.whatsapp.net`)
- **Display name**: `ZPARTNERNAME` from `ZWACHATSESSION` (e.g. "Mom", "Dad", "Friend A")
- **Join**: Messages have `ZCHATSESSION` → links to `ZWACHATSESSION`

### Claude
- **Per conversation**: `conv.name` (e.g. "Instagram comment draft")
- No contact JID; we have conversation title

### Gemini (Google Takeout)
- TBD — need to inspect export format

---

## 2. Changes to Extraction

### 2.1 UnifiedMessage schema (add fields)
```python
@dataclass
class UnifiedMessage:
    timestamp: datetime
    source: str
    language: str
    text: str
    is_from_user: bool
    context: Optional[str]       # existing: conversation name
    chat_id: Optional[str]       # NEW: stable ID (JID for WA, conv_id for Claude)
    contact_name: Optional[str]  # NEW: "Mom", "Dad", "Instagram draft"
    # tag filled later from user config
```

### 2.2 WhatsApp: join with ZWACHATSESSION
- Join `ZWAMESSAGE` ↔ `ZWACHATSESSION` on `ZCHATSESSION`
- Populate `chat_id` = `ZCONTACTJID`, `contact_name` = `ZPARTNERNAME`

### 2.3 Fallback for unknown contacts
If `ZPARTNERNAME` is a phone number or empty:
- Option A: Use as-is (`+1 (555) 012-3456`)
- Option B: User provides a mapping file or searches for a unique sentence to identify the chat

---

## 3. User Tagging (Relationship Labels)

### Config file: `~/.config/brainlayer/chat-tags.yaml` (or similar)

```yaml
# Map contact_name or chat_id to relationship tag
tags:
  - contact: "Mom"
    tag: family
  - contact: "Dad"
    tag: family
  - contact: "Friend A"
    tag: friends
  - contact: "Jane Smith"
    tag: co-workers
  # Or by JID if name is ambiguous
  - jid: "972501234567@s.whatsapp.net"
    tag: family
```

### Workflow for user
1. We generate `chats.csv`: `chat_id, contact_name, message_count`
2. User adds tags (manually or via script)
3. Re-run analysis with tags

### "Search for a sentence" fallback
- For chats with no clear name: user searches their WhatsApp for a unique phrase from that chat
- We could output: "Chat 972501234567 has 500 msgs, contact_name: +972 50-123-4567. Search for a unique message to identify."
- User finds it, tells us "that's John from work" → we add to tags

---

## 4. Batching Strategy

### Option A: Chat-first, then time within chat
```
Chat: Mom (family)         → 2024-H1, 2024-H2, 2025-H1, ...
Chat: Dad (family)         → 2024-H1, 2024-H2, ...
Chat: Jane (co-workers)    → 2024-H1, 2024-H2, ...
```

### Option B: Tag-first, then time
```
Tag: family    → 2024-H1, 2024-H2, 2025-H1, ... (all family chats combined)
Tag: friends   → 2024-H1, 2024-H2, ...
Tag: co-workers → 2024-H1, 2024-H2, ...
```

### Option C: Both (recommended)
- **Primary**: Tag + time (e.g. "family, 2024-H2")
- **Secondary**: Per-chat analysis for top N chats (optional)

---

## 5. Prompt Enrichment

**Current:**
> "Analyze 250 messages from 2024-H2. Language: HEBREW."

**New:**
> "Analyze 250 messages from 2024-H2.
> - Relationship context: ~60% family (Mom, Dad), ~40% friends (Friend A, Friend B)
> - Contact names in messages: Mom, Dad, Friend A, Friend B
> - Language: HEBREW.
> Consider how they may adapt tone by relationship."

---

## 6. Implementation Order

| Step | Task | Output |
|------|------|--------|
| 1 | Add `chat_id`, `contact_name` to WhatsApp extraction (join ZWACHATSESSION) | Messages with contact names |
| 2 | Add same fields to Claude extraction (use conv name as contact_name) | Unified schema |
| 3 | Create `chats.csv` / `list-chats` CLI command | User sees who we have |
| 4 | Add `chat-tags.yaml` config + loader | Tagged messages |
| 5 | Implement tag+time batching | New batch structure |
| 6 | Update analysis prompts with relationship context | Richer LLM input |
| 7 | Run new analysis | Improved style guide |
| 8 | Add Gemini/Google Takeout extractor | One more source |

---

## 7. Gemini / Google Takeout

- User has downloaded Takeout
- Need to: locate the archive, inspect structure, add `extract_gemini()` or `load_gemini_messages()` to unified_timeline
- Likely format: JSON in `Takeout/My Activity/Gemini` or similar
- Do this after chat-based analysis is working

---

## 8. Open Questions

1. **Default for untagged chats**: Skip? Use "unknown"? Or "unlabeled" and still include?
2. **Group chats**: Include with tag "group" or exclude? (Currently we exclude.)
3. **Claude conversations**: Many are "Instagram draft" style — tag as "instagram_ai" or similar?
