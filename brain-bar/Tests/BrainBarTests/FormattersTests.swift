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
        XCTAssertTrue(out.contains("## Search results"))
        XCTAssertTrue(out.contains("hello"))
        XCTAssertTrue(out.contains("No results found"))
        XCTAssertFalse(out.contains("score"))
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
        XCTAssertTrue(out.contains("1 of 1 shown"))
        XCTAssertFalse(out.contains("rt-abc123de"))
        XCTAssertFalse(out.contains("0.87"))
        XCTAssertTrue(out.contains("Source: brainlayer"))
        XCTAssertTrue(out.contains("2026-03-29"))
        XCTAssertTrue(out.contains("BrainBar is a native macOS daemon"))
        XCTAssertFalse(out.contains("imp:"))
    }

    func testFormatSearchResultsMultiple() {
        let results: [[String: Any]] = [
            ["chunk_id": "a1", "score": 0.9, "project": "proj1", "created_at": "2026-03-01", "summary": "First result", "importance": 7],
            ["chunk_id": "b2", "score": 0.5, "project": "proj2", "created_at": "2026-03-02", "summary": "Second result", "importance": 3],
        ]
        let out = Formatters.formatSearchResults(query: "test", results: results, total: 2, useColor: false)
        XCTAssertTrue(out.contains("2 of 2 shown"))
        XCTAssertTrue(out.contains("### 1."))
        XCTAssertTrue(out.contains("### 2."))
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
        XCTAssertTrue(out.contains("Tagged result"))
        XCTAssertFalse(out.contains("tags:"))
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
        XCTAssertTrue(out.contains("## Entity: Etan Heyman"))
        XCTAssertTrue(out.contains("### Profile"))
        XCTAssertTrue(out.contains("- role: Developer"))
        XCTAssertTrue(out.contains("- location: Tel Aviv"))
        XCTAssertTrue(out.contains("### KG Facts"))
        XCTAssertTrue(out.contains("works_on"))
        XCTAssertTrue(out.contains("BrainLayer"))
        XCTAssertTrue(out.contains("### Recent context"))
        XCTAssertTrue(out.contains("Chose Swift"))
    }

    func testFormatEntityCardMinimal() {
        let entity: [String: Any] = ["name": "Unknown Entity"]
        let out = Formatters.formatEntityCard(entity: entity, useColor: false)
        XCTAssertTrue(out.contains("## Entity: Unknown Entity"))
        XCTAssertTrue(out.contains("### KG Facts"))
        XCTAssertTrue(out.contains("- None"))
    }

    func testFormatSearchResultsBasenameHandlesWindowsPaths() {
        let results: [[String: Any]] = [
            [
                "source_file": #"C:\Users\etan\brainlayer\src\auth.py"#,
                "date": "2026-04-12T10:00:00Z",
                "snippet": "Windows paths should not leak full absolute paths."
            ]
        ]

        let out = Formatters.formatSearchResults(query: "path privacy", results: results, total: 1, useColor: false)

        XCTAssertTrue(out.contains("- Source: auth.py"))
        XCTAssertFalse(out.contains(#"C:\Users"#))
    }

    func testFormatKGSearchUsesUntitledForWhitespaceSnippet() {
        let out = Formatters.formatKGSearch(
            entityName: "BrainLayer",
            results: [["snippet": "   "]],
            facts: [],
            query: "brainlayer",
            useColor: false
        )

        XCTAssertTrue(out.contains("### 1. Untitled result"))
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
        XCTAssertTrue(out.contains("## Entity: BrainLayer"))
        XCTAssertTrue(out.contains("### KG Facts"))
        XCTAssertTrue(out.contains("used_by"))
        XCTAssertTrue(out.contains("Claude Code"))
        XCTAssertTrue(out.contains("### Recent context"))
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
        XCTAssertTrue(out.contains("## Search results for \"what is brainbar\""))
        XCTAssertTrue(out.contains("### KG Facts for BrainBar"))
        XCTAssertTrue(out.contains("BrainBar"))
        XCTAssertTrue(out.contains("part_of"))
        XCTAssertTrue(out.contains("BrainLayer"))
        XCTAssertFalse(out.contains("Memories"))
        XCTAssertFalse(out.contains("0.95"))
        XCTAssertFalse(out.contains("kg-1"))
    }

    func testFormatKGSearchNoFacts() {
        let results: [[String: Any]] = [
            ["chunk_id": "kg-2", "score": 0.6, "content": "Some content"]
        ]
        let out = Formatters.formatKGSearch(entityName: "Test", results: results, facts: [], query: "test", useColor: false)
        XCTAssertFalse(out.contains("KG Facts"))
        XCTAssertFalse(out.contains("Memories"))
    }

    // MARK: - ANSI Color Tests

    func testANSIColorsPresent() {
        let result: [String: Any] = [
            "chunk_id": "c1", "score": 0.85, "project": "test",
            "created_at": "2026-01-01", "summary": "Color test", "importance": 7
        ]
        let out = Formatters.formatSearchResults(query: "color", results: [result], total: 1, useColor: true)
        XCTAssertFalse(out.contains("\u{1b}["), "MCP markdown output should remain plain text even when useColor is true.")
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

    // MARK: - Markdown Layout

    func testSearchAndEntityFormattersUseMarkdownHeaders() {
        let searchOut = Formatters.formatSearchResults(query: "q", results: [
            ["chunk_id": "x", "score": 0.1, "project": "p", "created_at": "d", "summary": "s", "importance": 1]
        ], total: 1, useColor: false)
        XCTAssertTrue(searchOut.contains("## Search results"))
        XCTAssertTrue(searchOut.contains("### 1."))
        XCTAssertFalse(searchOut.contains("score:"))

        let entityOut = Formatters.formatEntityCard(entity: ["name": "X"], useColor: false)
        XCTAssertTrue(entityOut.contains("## Entity: X"))
        XCTAssertTrue(entityOut.contains("### KG Facts"))
    }

    // MARK: - Layout: No trailing empty │ lines

    func testSearchResultsNoTrailingEmptyLine() {
        let result: [String: Any] = [
            "chunk_id": "x1", "score": 0.5, "project": "test",
            "created_at": "2026-01-01", "summary": "Some result", "importance": 5
        ]
        let out = Formatters.formatSearchResults(query: "q", results: [result], total: 1, useColor: false)
        let lines = out.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        XCTAssertFalse(lines.last == "", "Markdown search output should not have a trailing empty line")
    }

    func testSearchResultsMultipleNoTrailingGap() {
        let results: [[String: Any]] = [
            ["chunk_id": "a", "score": 0.9, "project": "p", "created_at": "d", "summary": "First", "importance": 7],
            ["chunk_id": "b", "score": 0.5, "project": "p", "created_at": "d", "summary": "Second", "importance": 3],
        ]
        let out = Formatters.formatSearchResults(query: "q", results: results, total: 2, useColor: false)
        let lines = out.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        XCTAssertFalse(lines.last == "", "Markdown search output should not have a trailing empty line")
    }

    // MARK: - Content truncation: 150 chars

    func testSearchResultsSummaryTruncatesAt150() {
        let longSummary = String(repeating: "x", count: 300)
        let result: [String: Any] = [
            "chunk_id": "t1", "score": 0.5, "project": "test",
            "created_at": "2026-01-01", "summary": longSummary, "importance": 5
        ]
        let out = Formatters.formatSearchResults(query: "q", results: [result], total: 1, useColor: false)
        let expected = String(repeating: "x", count: 99) + "\u{2026}"
        XCTAssertTrue(out.contains(expected), "Summary title should truncate to 99 chars + ellipsis (100 total)")
        XCTAssertFalse(out.contains(longSummary), "200-char summary should be truncated")
    }
}
