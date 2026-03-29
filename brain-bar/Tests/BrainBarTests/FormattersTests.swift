// FormattersTests.swift — RED tests for MCP output formatters.
//
// Tests each formatter against known input → expected output.
// Also tests ANSI color toggling.

import XCTest
@testable import BrainBar

final class FormattersTests: XCTestCase {

    // MARK: - Search Results

    func testFormatSearchResultsEmpty() {
        let out = Formatters.formatSearchResults(query: "hello", results: [], total: 0)
        XCTAssertTrue(out.contains("brain_search"))
        XCTAssertTrue(out.contains("hello"))
        XCTAssertTrue(out.contains("No results found"))
        XCTAssertTrue(out.hasPrefix("\u{250c}"))  // ┌
        XCTAssertTrue(out.contains("\u{2514}"))    // └
    }

    func testFormatSearchResultsSingleResult() {
        let result: [String: Any] = [
            "chunk_id": "rt-abc123def4",
            "score": 0.87,
            "project": "brainlayer",
            "created_at": "2026-03-29T19:30:00Z",
            "summary": "BrainBar is a native macOS daemon",
            "importance": 8,
            "tags": "[\"swift\", \"macos\"]"
        ]
        let out = Formatters.formatSearchResults(query: "brainbar", results: [result], total: 1, useColor: false)
        XCTAssertTrue(out.contains("1 result"))
        XCTAssertTrue(out.contains("rt-abc123de"))  // truncated to 12 chars
        XCTAssertTrue(out.contains("0.87"))
        XCTAssertTrue(out.contains("brainlayer"))
        XCTAssertTrue(out.contains("2026-03-29"))
        XCTAssertTrue(out.contains("BrainBar is a native macOS daemon"))
        XCTAssertTrue(out.contains("imp:"))
    }

    func testFormatSearchResultsMultiple() {
        let results: [[String: Any]] = [
            ["chunk_id": "a1", "score": 0.9, "project": "proj1", "created_at": "2026-03-01", "summary": "First result", "importance": 7],
            ["chunk_id": "b2", "score": 0.5, "project": "proj2", "created_at": "2026-03-02", "summary": "Second result", "importance": 3],
        ]
        let out = Formatters.formatSearchResults(query: "test", results: results, total: 2, useColor: false)
        XCTAssertTrue(out.contains("2 results"))
        XCTAssertTrue(out.contains("[1]"))
        XCTAssertTrue(out.contains("[2]"))
    }

    func testFormatSearchResultsWithTags() {
        let result: [String: Any] = [
            "chunk_id": "t1",
            "score": 0.5,
            "project": "test",
            "created_at": "2026-01-01",
            "summary": "Tagged result",
            "importance": 5,
            "tags": "[\"alpha\", \"beta\", \"gamma\"]"
        ]
        let out = Formatters.formatSearchResults(query: "tags", results: [result], total: 1, useColor: false)
        XCTAssertTrue(out.contains("tags:"))
        XCTAssertTrue(out.contains("alpha"))
        XCTAssertTrue(out.contains("beta"))
    }

    // MARK: - Store Result

    func testFormatStoreResult() {
        let out = Formatters.formatStoreResult(chunkId: "rt-abc123-xyz")
        XCTAssertTrue(out.contains("\u{2714}"))  // ✔
        XCTAssertTrue(out.contains("Stored"))
        XCTAssertTrue(out.contains("rt-abc123-xyz"))
    }

    func testFormatStoreResultWithSuperseded() {
        let out = Formatters.formatStoreResult(chunkId: "new-1", superseded: "old-1")
        XCTAssertTrue(out.contains("new-1"))
        XCTAssertTrue(out.contains("superseded"))
        XCTAssertTrue(out.contains("old-1"))
    }

    func testFormatStoreResultQueued() {
        let out = Formatters.formatStoreResult(chunkId: "", queued: true)
        XCTAssertTrue(out.contains("\u{23f3}"))  // ⏳
        XCTAssertTrue(out.contains("queued"))
    }

    // MARK: - Entity Card

    func testFormatEntityCard() {
        let entity: [String: Any] = [
            "name": "Etan Heyman",
            "entity_id": "person-001",
            "entity_type": "person",
            "profile": ["role": "Developer", "location": "Tel Aviv"] as [String: Any],
            "relations": [
                ["relation_type": "works_on", "target": ["name": "BrainLayer"] as [String: Any]] as [String: Any]
            ],
            "memories": [
                ["type": "decision", "date": "2026-03-29", "content": "Chose Swift for BrainBar"] as [String: Any]
            ]
        ]
        let out = Formatters.formatEntityCard(entity: entity, useColor: false)
        XCTAssertTrue(out.contains("Entity: Etan Heyman"))
        XCTAssertTrue(out.contains("person-001"))
        XCTAssertTrue(out.contains("person"))
        XCTAssertTrue(out.contains("role: Developer"))
        XCTAssertTrue(out.contains("location: Tel Aviv"))
        XCTAssertTrue(out.contains("Relations"))
        XCTAssertTrue(out.contains("works_on"))
        XCTAssertTrue(out.contains("BrainLayer"))
        XCTAssertTrue(out.contains("Memories"))
        XCTAssertTrue(out.contains("Chose Swift"))
    }

    func testFormatEntityCardMinimal() {
        let entity: [String: Any] = ["name": "Unknown Entity"]
        let out = Formatters.formatEntityCard(entity: entity, useColor: false)
        XCTAssertTrue(out.contains("Entity: Unknown Entity"))
        XCTAssertTrue(out.hasPrefix("\u{250c}"))
        XCTAssertTrue(out.contains("\u{2514}"))
    }

    // MARK: - Entity Simple

    func testFormatEntitySimple() {
        let entity: [String: Any] = [
            "name": "BrainLayer",
            "id": "proj-bl",
            "entity_type": "project",
            "relations": [
                ["relation_type": "used_by", "target_name": "Claude Code"] as [String: Any]
            ],
            "chunks": [
                ["content": "BrainLayer is a local knowledge pipeline"] as [String: Any]
            ]
        ]
        let out = Formatters.formatEntitySimple(entity: entity, useColor: false)
        XCTAssertTrue(out.contains("Entity: BrainLayer"))
        XCTAssertTrue(out.contains("proj-bl"))
        XCTAssertTrue(out.contains("project"))
        XCTAssertTrue(out.contains("used_by"))
        XCTAssertTrue(out.contains("Claude Code"))
        XCTAssertTrue(out.contains("Associated memories"))
    }

    func testFormatEntitySimpleEmpty() {
        let out = Formatters.formatEntitySimple(entity: [:])
        XCTAssertTrue(out.isEmpty)
    }

    // MARK: - Stats

    func testFormatStats() {
        let stats: [String: Any] = [
            "total_chunks": 284127,
            "projects": ["brainlayer", "orchestrator", "voicelayer"],
            "content_types": ["assistant_text", "user_message", "ai_code"]
        ]
        let out = Formatters.formatStats(stats: stats, useColor: false)
        XCTAssertTrue(out.contains("BrainLayer Stats"))
        XCTAssertTrue(out.contains("284,127"))
        XCTAssertTrue(out.contains("brainlayer"))
        XCTAssertTrue(out.contains("orchestrator"))
        XCTAssertTrue(out.contains("assistant_text"))
    }

    // MARK: - Digest Result

    func testFormatDigestResultEnrich() {
        let result: [String: Any] = [
            "mode": "enrich",
            "attempted": 50,
            "enriched": 45,
            "skipped": 3,
            "failed": 2
        ]
        let out = Formatters.formatDigestResult(result: result, useColor: false)
        XCTAssertTrue(out.contains("brain_digest"))
        XCTAssertTrue(out.contains("enrich"))
        XCTAssertTrue(out.contains("50"))
        XCTAssertTrue(out.contains("45"))
    }

    func testFormatDigestResultDigest() {
        let result: [String: Any] = [
            "mode": "digest",
            "chunks_created": 12,
            "entities_created": 5,
            "relations_created": 8,
            "action_items": [
                ["description": "Fix the login bug"] as [String: Any],
                ["description": "Update the README"] as [String: Any]
            ]
        ]
        let out = Formatters.formatDigestResult(result: result, useColor: false)
        XCTAssertTrue(out.contains("digest"))
        XCTAssertTrue(out.contains("12"))
        XCTAssertTrue(out.contains("Action items"))
        XCTAssertTrue(out.contains("Fix the login bug"))
    }

    // MARK: - KG Search

    func testFormatKGSearch() {
        let results: [[String: Any]] = [
            ["chunk_id": "kg-1", "score": 0.95, "snippet": "BrainBar serves MCP over socket"]
        ]
        let facts: [[String: Any]] = [
            ["source": "BrainBar", "relation": "part_of", "target": "BrainLayer"]
        ]
        let out = Formatters.formatKGSearch(
            entityName: "BrainBar",
            results: results,
            facts: facts,
            query: "what is brainbar",
            useColor: false
        )
        XCTAssertTrue(out.contains("Entity search: \"BrainBar\""))
        XCTAssertTrue(out.contains("Knowledge Graph"))
        XCTAssertTrue(out.contains("BrainBar"))
        XCTAssertTrue(out.contains("part_of"))
        XCTAssertTrue(out.contains("BrainLayer"))
        XCTAssertTrue(out.contains("Memories"))
        XCTAssertTrue(out.contains("0.95"))
    }

    func testFormatKGSearchNoFacts() {
        let results: [[String: Any]] = [
            ["chunk_id": "kg-2", "score": 0.6, "content": "Some content"]
        ]
        let out = Formatters.formatKGSearch(entityName: "Test", results: results, facts: [], query: "test", useColor: false)
        XCTAssertFalse(out.contains("Knowledge Graph"))
        XCTAssertTrue(out.contains("Memories"))
    }

    // MARK: - ANSI Color Tests

    func testANSIColorsPresent() {
        let result: [String: Any] = [
            "chunk_id": "c1", "score": 0.85, "project": "test",
            "created_at": "2026-01-01", "summary": "Color test", "importance": 7
        ]
        let out = Formatters.formatSearchResults(query: "color", results: [result], total: 1, useColor: true)
        // Orange for values
        XCTAssertTrue(out.contains("\u{1b}[38;2;232;121;36m"))
        // Blue for keys
        XCTAssertTrue(out.contains("\u{1b}[38;2;88;166;255m"))
        // Green for numbers
        XCTAssertTrue(out.contains("\u{1b}[38;2;63;185;80m"))
        // Reset
        XCTAssertTrue(out.contains("\u{1b}[0m"))
    }

    func testANSIColorsAbsentWhenDisabled() {
        let result: [String: Any] = [
            "chunk_id": "c1", "score": 0.85, "project": "test",
            "created_at": "2026-01-01", "summary": "No color", "importance": 7
        ]
        let out = Formatters.formatSearchResults(query: "color", results: [result], total: 1, useColor: false)
        XCTAssertFalse(out.contains("\u{1b}["))
    }

    func testStoreResultHasColor() {
        let out = Formatters.formatStoreResult(chunkId: "abc-123", useColor: true)
        XCTAssertTrue(out.contains("\u{1b}["))
    }

    func testStatsHasColor() {
        let stats: [String: Any] = [
            "total_chunks": 100,
            "projects": ["test"],
            "content_types": ["ai_code"]
        ]
        let out = Formatters.formatStats(stats: stats, useColor: true)
        XCTAssertTrue(out.contains("\u{1b}[38;2;63;185;80m"))  // green for numbers
    }

    // MARK: - Box Drawing Characters

    func testAllFormattersUseBoxDrawing() {
        let searchOut = Formatters.formatSearchResults(query: "q", results: [
            ["chunk_id": "x", "score": 0.1, "project": "p", "created_at": "d", "summary": "s", "importance": 1]
        ], total: 1, useColor: false)
        XCTAssertTrue(searchOut.contains("\u{250c}"))  // ┌
        XCTAssertTrue(searchOut.contains("\u{2514}"))  // └
        XCTAssertTrue(searchOut.contains("\u{2502}"))  // │
        XCTAssertTrue(searchOut.contains("\u{251c}"))  // ├

        let entityOut = Formatters.formatEntityCard(entity: ["name": "X"], useColor: false)
        XCTAssertTrue(entityOut.contains("\u{250c}"))
        XCTAssertTrue(entityOut.contains("\u{2514}"))
    }

    // MARK: - Layout: No trailing empty │ lines

    func testSearchResultsNoTrailingEmptyLine() {
        let result: [String: Any] = [
            "chunk_id": "x1", "score": 0.5, "project": "test",
            "created_at": "2026-01-01", "summary": "Some result", "importance": 5
        ]
        let out = Formatters.formatSearchResults(query: "q", results: [result], total: 1, useColor: false)
        let lines = out.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        // The line before └─ should NOT be a bare │
        let closerIdx = lines.lastIndex(where: { $0.hasPrefix("\u{2514}") })!
        let beforeCloser = lines[closerIdx - 1]
        XCTAssertNotEqual(beforeCloser, "\u{2502}", "Should not have a trailing empty │ line before └─")
    }

    func testSearchResultsMultipleNoTrailingGap() {
        let results: [[String: Any]] = [
            ["chunk_id": "a", "score": 0.9, "project": "p", "created_at": "d", "summary": "First", "importance": 7],
            ["chunk_id": "b", "score": 0.5, "project": "p", "created_at": "d", "summary": "Second", "importance": 3],
        ]
        let out = Formatters.formatSearchResults(query: "q", results: results, total: 2, useColor: false)
        let lines = out.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        let closerIdx = lines.lastIndex(where: { $0.hasPrefix("\u{2514}") })!
        let beforeCloser = lines[closerIdx - 1]
        XCTAssertNotEqual(beforeCloser, "\u{2502}", "Last result should not have trailing │ gap")
    }

    // MARK: - Content truncation: 150 chars

    func testSearchResultsSummaryTruncatesAt150() {
        let longSummary = String(repeating: "x", count: 200)
        let result: [String: Any] = [
            "chunk_id": "t1", "score": 0.5, "project": "test",
            "created_at": "2026-01-01", "summary": longSummary, "importance": 5
        ]
        let out = Formatters.formatSearchResults(query: "q", results: [result], total: 1, useColor: false)
        // Exact truncation: 149 chars + ellipsis = 150 total
        let expected = String(repeating: "x", count: 149) + "\u{2026}"
        XCTAssertTrue(out.contains(expected), "Summary should truncate to 149 chars + ellipsis (150 total)")
        XCTAssertFalse(out.contains(longSummary), "200-char summary should be truncated")
    }
}
