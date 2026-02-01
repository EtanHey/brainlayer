# Deep Research: Chat-Based Style Analysis

*For: extraction design, user tagging, fallback strategies, prompt enrichment*

---

## 1. Should We Group by Chat / Relationship? (Linguistic & Psych Research)

### Verdict: **Yes — strong evidence**

**Register & addressee effects (linguistics):**
- **Register** = way of speaking tied to situation and audience
- **Style** = formality level, shaped by addressee
- **Addressee influence**: familiarity, age, social context all change formality, vocabulary, and structure
- Source: Cambridge Handbook of English Corpus Linguistics, Annual Review of Linguistics

**Communication accommodation theory (digital messaging):**
- In IM, people **converge** on message length, timing, and style
- Effects differ by **friends vs strangers** and **task vs social** context
- Style accommodation improves rapport and impressions
- Source: "Communication Accommodation in Instant Messaging" (Sage), "Linguistic Style Accommodation Shapes Impression Formation..." (2017)

**Friends vs strangers:**
- Friends: more exploratory, varied topics
- Strangers: more convergent, similar patterns
- Source: Nature 2024 hyperscanning study

**Implications for our design:**
- Grouping by relationship (family, friends, co-workers) is supported by research
- Batching only by time loses important variation
- Including relationship context in prompts should improve style modeling

---

## 2. User Tagging: How Much, What Friction?

### Verdict: **Minimize manual tagging; use programmatic + LLM-assisted where possible**

**Annotation burden research:**
- **Programmatic labeling** (rules, not per-item clicks): 10–100x faster than manual
- **Selective labeling**: user confirms uncertain cases, not every item — 10x cost reduction
- **LLM-assisted baselines**: model proposes labels, user corrects — 2.8x faster, ~45% better than hand labeling
- Source: Snorkel AI, Google "Selective Labeling" (2024), FreeAL / ActiveLLM

**What to do:**
1. **Do**: Provide a chat list (contact, message count) so the user can tag in bulk
2. **Consider**: LLM pre-labeling — e.g. infer "family" from names like "Maman", "Dad", "Aunt Lisa"
3. **Do**: Use templates — e.g. tag by regex on contact name
4. **Don’t**: Require per-message tagging

**Recommendation:**
- Generate `chats.csv` with contact_name, chat_id, message_count
- User edits a YAML/CSV mapping: `Maman → family`, `Dad → family`, etc.
- Optional: LLM suggests tags for top N contacts; user reviews and fixes
- Untagged chats: either "unlabeled" (included but flagged) or excluded — make it configurable

---

## 3. Fallback for Unknown Contacts

### Verdict: **Multi-step fallback; avoid blocking on full identification**

**Options, in order of preference:**

| Approach | Pros | Cons |
|----------|------|------|
| **1. Use contact_name when available** | Already in DB (ZPARTNERNAME) | Sometimes phone number or empty |
| **2. "Search for sentence"** | User can disambiguate | Extra step, may be skipped |
| **3. Include as "unlabeled"** | No user effort | Dilutes relationship signal |
| **4. Exclude untagged** | Clean labels | Loses data |

**Recommendation:**
- **Primary**: Use contact_name from ZWACHATSESSION (we have Maman, Dad, etc.)
- **Fallback A**: If name is only a phone number, keep as-is and let user tag in YAML
- **Fallback B**: Emit a report: "Chats needing tags: [list with sample message]" — user can search for a phrase to identify
- **Default for untagged**: Include in analysis as "unlabeled" so we don’t discard data; optionally allow exclusion in config

---

## 4. Prompt Enrichment: What Context Helps?

### Verdict: **Include relationship mix, contact names, and temporal context**

**Context engineering (LLMs):**
- Context = selection, compression, and ordering of information
- Metadata helps build query-specific context
- Source: "Everything is Context" (arxiv 2025), Readme_AI

**What to add to prompts:**

| Context | Format | Rationale |
|---------|--------|-----------|
| **Relationship mix** | "~60% family, ~40% friends" | Register/addressee effects |
| **Contact names** | "Maman, Dad, אביאל, Mercy" | Grounds examples, reduces hallucination |
| **Temporal** | "2024-H2, mostly evening/weekend" | Style varies by time of day |
| **Source** | "WhatsApp 1:1" | Medium affects formality |

**Don’t overload:**
- Keep prompts within context limits
- Prefer summary stats over raw dumps

**Example enriched prompt:**
```
Analyze 250 messages from 2024-H2.
Relationship context: ~55% family (Maman, Dad), ~30% friends (אביאל, לימון), ~15% unlabeled.
Contact names you may reference: Maman, Dad, אביאל, לימון, Mercy Akede.
Language: HEBREW.
Consider how tone may shift by relationship (family vs friends vs work).
```

---

## 5. Extraction: Technical Notes

### WhatsApp
- **Chat ID**: ZTOJID (e.g. 972544667708@s.whatsapp.net)
- **Display name**: ZPARTNERNAME from ZWACHATSESSION (join on ZCHATSESSION)
- **Confirmed**: Maman, Dad, אביאל, Mercy Akede, etc. are available

### Google Takeout / Gemini
- **Location**: `Takeout/My Activity/Gemini Apps/`
- **Format**: HTML (MyActivity.html, ~1.3MB) plus JSON-like and binary files
- **Gemini folder**: Only gems/scheduled actions (tiny); main data is in My Activity
- **Parser**: `google-takeout-parser` supports My Activity; may need a Gemini-specific handler
- **Note**: My Activity schema uses `header`, `title`, `time`, `products`; exact structure for Gemini turns needs inspection

---

## 6. Recommendations Summary

| Area | Do | Don’t |
|------|-----|-------|
| **Extraction** | Join with ZWACHATSESSION for contact names; add chat_id, contact_name to schema | Rely on ZPUSHNAME from messages (often encoded) |
| **Tagging** | Bulk YAML/CSV mapping; optional LLM suggestions | Per-message or per-chat manual clicks |
| **Fallback** | Use contact_name; report "needs tags"; include unlabeled by default | Block analysis on full tagging |
| **Prompts** | Add relationship mix, contact names, optional time-of-day | Dump full metadata into every prompt |
| **Batching** | Tag + time (e.g. family-2024-H2) | Time-only batching |

---

## 7. Open Questions

1. **LLM pre-tagging**: Use local model to propose family/friends/work from names? Worth the complexity?
2. **Group chats**: Include with tag "group" or keep excluding?
3. **Gemini HTML**: Parsing strategy — use google-takeout-parser or custom HTML parser?
4. **Claude conversations**: Tag "instagram_ai", "general", etc. by conversation name patterns?
