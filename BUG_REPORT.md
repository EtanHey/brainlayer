# Bug Report: BrainBar Stub Tool Implementation

**PR:** feat/brainbar-implement-stub-tools  
**Reviewer:** @bugbot  
**Date:** 2026-03-29

## Executive Summary

Reviewed the implementation of 6 BrainBar stub tools (`brain_tags`, `brain_update`, `brain_expand`, `brain_entity`, `brain_recall`, `brain_digest`). Found **7 critical bugs** and **5 moderate issues** that could cause runtime failures, data corruption, or incorrect behavior.

---

## 🔴 CRITICAL BUGS

### 1. **brain_update: Schema Mismatch with Implementation**
**Location:** `MCPRouter.swift:479-490`  
**Severity:** CRITICAL  
**Impact:** Tool will fail validation when Claude tries to call it

**Problem:**
The `brain_update` tool schema declares `action` as a **required** parameter with enum values `["update", "archive", "merge"]`:

```swift
"inputSchema": [
    "type": "object",
    "properties": [
        "action": ["type": "string", "enum": ["update", "archive", "merge"], "description": "Action to perform"],
        "chunk_id": ["type": "string", "description": "Chunk ID to update"],
    ] as [String: Any],
    "required": ["action", "chunk_id"]
]
```

But the implementation at lines 275-288 **completely ignores** the `action` parameter and only implements the "update" action:

```swift
private func handleBrainUpdate(_ args: [String: Any]) throws -> String {
    guard let db = database else { throw ToolError.noDatabase }
    let chunkId = args["chunk_id"] as? String ?? ""
    if chunkId.isEmpty {
        throw ToolError.missingParameter("chunk_id")
    }
    let importance = args["importance"] as? Int
    let tags = args["tags"] as? [String]
    if importance == nil && tags == nil {
        throw ToolError.missingParameter("importance or tags")
    }
    try db.updateChunk(id: chunkId, importance: importance, tags: tags)
    return "✔ Updated \(chunkId)" + ...
}
```

**Expected Behavior:**
- If `action == "archive"`, should set `archived_at` timestamp (per CLAUDE.md lifecycle columns)
- If `action == "merge"`, should handle chunk merging/aggregation
- If `action == "update"`, should update importance/tags (current behavior)

**Fix Required:**
```swift
let action = args["action"] as? String ?? "update"
switch action {
case "update":
    // current implementation
case "archive":
    try db.archiveChunk(id: chunkId)
case "merge":
    guard let targetId = args["target_id"] as? String else {
        throw ToolError.missingParameter("target_id")
    }
    try db.mergeChunks(sourceId: chunkId, targetId: targetId)
default:
    throw ToolError.invalidParameter("action must be update, archive, or merge")
}
```

---

### 2. **brain_update: Missing Schema Properties**
**Location:** `MCPRouter.swift:479-490`  
**Severity:** CRITICAL  
**Impact:** Claude cannot pass importance/tags parameters

**Problem:**
The schema declares `action` and `chunk_id` as the only properties, but the implementation expects `importance` and `tags`:

```swift
"properties": [
    "action": [...],
    "chunk_id": [...]
]
```

But the handler reads:
```swift
let importance = args["importance"] as? Int
let tags = args["tags"] as? [String]
```

**Fix Required:**
Add missing properties to schema:
```swift
"properties": [
    "action": ["type": "string", "enum": ["update", "archive", "merge"]],
    "chunk_id": ["type": "string"],
    "importance": ["type": "integer", "description": "New importance score (1-10)"],
    "tags": ["type": "array", "items": ["type": "string"], "description": "New tags array"],
    "target_id": ["type": "string", "description": "Target chunk for merge action"]
]
```

---

### 3. **brain_expand: Incorrect Rowid Range Calculation**
**Location:** `BrainDatabase.swift:919-920`  
**Severity:** CRITICAL  
**Impact:** Returns wrong context chunks or misses relevant chunks

**Problem:**
The rowid range calculation multiplies `before` and `after` by 10, which is arbitrary and will fail in sparse rowid spaces:

```swift
sqlite3_bind_int64(ctxStmt, 3, targetRowID - Int64(before * 10))
sqlite3_bind_int64(ctxStmt, 4, targetRowID + Int64(after * 10))
```

If `before=3`, this searches for rowids in range `[targetRowID-30, targetRowID+30]`. But:
- If chunks are inserted sparsely (e.g., rowids 1, 100, 200, 300), this will miss most context
- If chunks are dense (e.g., rowids 1, 2, 3, 4, 5), this will return way more than requested
- The magic number `10` has no documented rationale

**Expected Behavior:**
Use a subquery to find the actual N chunks before/after by rowid:

```sql
-- Get N chunks before target
SELECT id, content, ... FROM chunks
WHERE conversation_id = ? AND rowid < ?
ORDER BY rowid DESC
LIMIT ?

-- Get N chunks after target
SELECT id, content, ... FROM chunks
WHERE conversation_id = ? AND rowid > ?
ORDER BY rowid ASC
LIMIT ?
```

**Fix Required:**
Replace the single query with two separate queries for before/after, or use a window function.

---

### 4. **brain_digest: Regex Pattern Doesn't Escape Special Characters**
**Location:** `BrainDatabase.swift:1084-1115`  
**Severity:** HIGH  
**Impact:** Regex can crash on malformed input or miss valid entities

**Problem:**
The regex patterns are created without error handling for invalid patterns:

```swift
let namePattern = try NSRegularExpression(pattern: "\\b([A-Z][a-z]+(?:\\s+[A-Z][a-z]+){1,2})\\b")
```

If `NSRegularExpression` throws (e.g., due to a bug in the pattern), the entire `digest()` function fails. Additionally:
- The pattern `{1,2}` means 2-3 words total, but comment says "2-3 words" (ambiguous)
- No handling for names with hyphens (e.g., "Jean-Claude")
- No handling for acronyms (e.g., "API Gateway")

**Fix Required:**
1. Wrap pattern compilation in proper error handling
2. Document exact matching behavior
3. Add test cases for edge cases

---

### 5. **brain_digest: Stores Truncated Content Without Warning**
**Location:** `BrainDatabase.swift:1124-1129`  
**Severity:** MODERATE  
**Impact:** Data loss without user notification

**Problem:**
```swift
let stored = try store(
    content: content.prefix(500) + (content.count > 500 ? "..." : ""),
    tags: ["digest"] + entities.prefix(5).map { $0 },
    importance: 5,
    source: "digest"
)
```

If content is longer than 500 chars, it's silently truncated. The returned summary doesn't indicate truncation occurred.

**Fix Required:**
- Either store full content, or
- Return truncation info in the result dict:
```swift
"truncated": content.count > 500,
"original_length": content.count
```

---

### 6. **brain_entity: SQL Injection Risk via LIKE Query**
**Location:** `BrainDatabase.swift:994`  
**Severity:** HIGH  
**Impact:** SQL injection if query contains `%` or `_` characters

**Problem:**
```swift
bindText("%\(query)%", to: stmt, index: 1)
```

If `query` contains `%` or `_`, they will be interpreted as SQL wildcards:
- `query = "test_entity"` will match `"test1entity"`, `"testXentity"`, etc.
- `query = "100%"` will match `"100"`, `"100abc"`, etc.

**Fix Required:**
Escape LIKE wildcards before binding:
```swift
let escapedQuery = query
    .replacingOccurrences(of: "\\", with: "\\\\")
    .replacingOccurrences(of: "%", with: "\\%")
    .replacingOccurrences(of: "_", with: "\\_")
bindText("%\(escapedQuery)%", to: stmt, index: 1)
```

And add `ESCAPE '\\'` to the SQL:
```sql
SELECT ... WHERE name LIKE ? ESCAPE '\\'
```

---

### 7. **brain_tags: Case-Insensitive Deduplication Loses Original Casing**
**Location:** `BrainDatabase.swift:839`  
**Severity:** MODERATE  
**Impact:** Tag display loses original casing

**Problem:**
```swift
let t = tag.trimmingCharacters(in: .whitespaces).lowercased()
```

All tags are lowercased before counting, so if the database has `["Swift", "swift", "SWIFT"]`, they're correctly deduplicated, but the returned tag name will always be lowercase `"swift"`.

**Expected Behavior:**
Keep the most common casing variant, or the first seen.

**Fix Required:**
```swift
var tagCounts: [String: (count: Int, canonical: String)] = [:]
for tag in arr {
    let normalized = tag.trimmingCharacters(in: .whitespaces).lowercased()
    if tagCounts[normalized] == nil {
        tagCounts[normalized] = (1, tag)
    } else {
        tagCounts[normalized]!.count += 1
    }
}
return tagCounts.map { ["tag": $0.value.canonical, "count": $0.value.count] }
```

---

## 🟡 MODERATE ISSUES

### 8. **brain_recall: Misleading Mode Name**
**Location:** `MCPRouter.swift:237-252`  
**Severity:** LOW  
**Impact:** Confusing behavior

**Problem:**
When `mode == "context"` but `session_id` is empty, it falls back to returning stats. This is confusing because the user explicitly requested "context" mode.

**Fix Required:**
Return an error or empty context instead of silently switching modes:
```swift
if mode == "context" {
    let sessionId = args["session_id"] as? String ?? ""
    if sessionId.isEmpty {
        return "⚠ context mode requires session_id parameter"
    }
    ...
}
```

---

### 9. **brain_expand: No Error if Chunk Not Found**
**Location:** `BrainDatabase.swift:887`  
**Severity:** MODERATE  
**Impact:** Throws generic error instead of specific "not found"

**Problem:**
```swift
guard sqlite3_step(stmt) == SQLITE_ROW else { throw DBError.noResult }
```

`DBError.noResult` is generic. Should be more specific:
```swift
guard sqlite3_step(stmt) == SQLITE_ROW else { 
    throw DBError.chunkNotFound(id) 
}
```

---

### 10. **brain_entity: Inefficient Double Query**
**Location:** `BrainDatabase.swift:962-1007`  
**Severity:** LOW  
**Impact:** Performance

**Problem:**
Queries exact match first, then LIKE if no match. Could be combined:
```sql
SELECT ... WHERE name = ? OR name LIKE ?
ORDER BY CASE WHEN name = ? THEN 0 ELSE 1 END
LIMIT 1
```

---

### 11. **Missing kg_entities/kg_relations Index**
**Location:** `BrainDatabase.swift:134-156`  
**Severity:** MODERATE  
**Impact:** Slow entity lookups

**Problem:**
`kg_entities` table has no index on `name` column, which is used in lookups. `kg_relations` has no index on `source_id` or `target_id`.

**Fix Required:**
```sql
CREATE INDEX IF NOT EXISTS idx_kg_entities_name ON kg_entities(name);
CREATE INDEX IF NOT EXISTS idx_kg_relations_source ON kg_relations(source_id);
CREATE INDEX IF NOT EXISTS idx_kg_relations_target ON kg_relations(target_id);
```

---

### 12. **brain_digest: No Deduplication of Extracted Entities**
**Location:** `BrainDatabase.swift:1118-1120`  
**Severity:** LOW  
**Impact:** Duplicate entities in results

**Problem:**
```swift
entities = Array(Set(entities))
```

This deduplicates, but it's done AFTER both regex passes, so if "BrainLayer" appears as both a capitalized name and PascalCase, it's added twice then deduplicated. Should deduplicate incrementally or use a Set from the start.

---

## 📋 TEST COVERAGE GAPS

### Missing Test Cases:
1. **brain_update**: No test for "archive" or "merge" actions
2. **brain_expand**: No test for sparse rowid spaces
3. **brain_entity**: No test for LIKE wildcard escaping
4. **brain_digest**: No test for content > 500 chars
5. **brain_tags**: No test for mixed-case deduplication
6. **brain_recall**: No test for empty session_id in context mode

---

## 🔧 RECOMMENDED FIXES (Priority Order)

1. **CRITICAL:** Fix brain_update schema mismatch (#1, #2)
2. **CRITICAL:** Fix brain_expand rowid calculation (#3)
3. **HIGH:** Fix brain_entity SQL injection (#6)
4. **HIGH:** Fix brain_digest regex error handling (#4)
5. **MODERATE:** Add kg_entities/kg_relations indexes (#11)
6. **MODERATE:** Fix brain_tags casing (#7)
7. **MODERATE:** Improve brain_digest truncation (#5)
8. **LOW:** Improve error messages (#8, #9)
9. **LOW:** Optimize brain_entity query (#10)
10. **LOW:** Improve brain_digest deduplication (#12)

---

## ✅ WHAT'S WORKING WELL

1. **Consistent error handling** with ToolError enum
2. **Good retry logic** for SQLITE_BUSY (3 retries with backoff)
3. **Proper transaction handling** in upsertSubscription
4. **FTS5 integration** is solid
5. **Test coverage** for basic happy paths is good
6. **Schema creation** is idempotent and safe

---

## 📝 NOTES

- Swift environment not available in this cloud agent, so tests couldn't be run
- Analysis based on static code review only
- All line numbers verified against current HEAD
- No security issues beyond SQL injection risk (#6)

---

**Recommendation:** Do NOT merge until bugs #1-6 are fixed. These are blocking issues that will cause runtime failures.
