import XCTest
@testable import BrainBar

final class TextFormatterParityTests: XCTestCase {

    func testSearchResultsMatchPythonStructure() {
        let results = [
            SearchResult(
                chunkID: "rt-abc123def4567890",
                score: 0.87,
                project: "brainlayer",
                date: "2026-03-29T19:30:00Z",
                summary: "BrainBar is a native macOS daemon for BrainLayer MCP routing.",
                snippet: "BrainBar is a native macOS daemon for BrainLayer MCP routing.",
                importance: 8,
                tags: ["swift", "macos", "mcp"]
            )
        ]

        let output = TextFormatter.formatSearchResults(
            query: "brainbar native swift renderer parity should truncate long searches cleanly",
            results: results,
            total: results.count
        )

        XCTAssertEqual(
            output,
            """
            ┌─ brain_search: "brainbar native swift renderer parity should trun…" ─ 1 result
            │
            ├─ [1] rt-abc123def  score:0.87  imp: 8  2026-03-29
            │  brainlayer       │ BrainBar is a native macOS daemon for BrainLayer MCP routing.
            │  tags: swift, macos, mcp
            │
            └─
            """
        )
    }

    func testEntityCardMatchesPythonStructure() {
        let entity = EntityCard(
            id: "person-001",
            name: "Etan Heyman",
            entityType: "person",
            profile: [
                "role": "Developer",
                "company": "BrainLayer",
                "location": "Tel Aviv"
            ],
            hardConstraints: ["timezone": "Asia/Jerusalem"],
            preferences: ["editor": "Neovim"],
            contactInfo: ["email": "etan@example.com"],
            relations: [
                EntityCard.Relation(
                    relationType: "works_on",
                    targetName: "BrainLayer",
                    direction: "outgoing"
                )
            ],
            memories: [
                EntityCard.Memory(type: "decision", date: "2026-03-29", content: "Chose Swift renderers for BrainBar parity.")
            ],
            memoryCount: 1
        )

        let output = TextFormatter.formatEntityCard(entity)

        XCTAssertEqual(
            output,
            """
            ┌─ Entity: Etan Heyman
            │ id: person-001  type: person
            │ role: Developer
            │ company: BrainLayer
            │ location: Tel Aviv
            ├─ Constraints
            │   timezone: Asia/Jerusalem
            ├─ Preferences
            │   editor: Neovim
            ├─ Contact
            │   email: etan@example.com
            ├─ Relations (1)
            │   → works_on: BrainLayer
            ├─ Memories (1)
            │   [decision] 2026-03-29 Chose Swift renderers for BrainBar parity.
            └─
            """
        )
    }

    func testDigestStatsAndKGFormattersMatchPythonStructure() {
        let digest = DigestResult(
            mode: "digest",
            attempted: nil,
            enriched: nil,
            skipped: nil,
            failed: nil,
            chunks: 12,
            entities: 5,
            relations: 8,
            actionItems: [
                "Fix the login bug",
                "Update the README"
            ]
        )
        let stats = StatsResult(
            totalChunks: 284_127,
            projects: ["brainlayer", "orchestrator", "voicelayer"],
            contentTypes: ["assistant_text", "user_message", "ai_code"]
        )
        let kg = KGSearchResult(
            entityName: "BrainBar",
            query: "what is brainbar",
            facts: [
                KGSearchResult.Fact(source: "BrainBar", relation: "part_of", target: "BrainLayer")
            ],
            results: [
                KGSearchResult.MemoryResult(chunkID: "kg-1234567890ab", score: 0.95, snippet: "BrainBar serves MCP over a Unix socket.")
            ]
        )

        XCTAssertEqual(
            TextFormatter.formatDigestResult(digest),
            """
            ┌─ brain_digest (digest)
            │ Chunks: 12  Entities: 5  Relations: 8
            ├─ Action items (2)
            │   • Fix the login bug
            │   • Update the README
            └─
            """
        )

        XCTAssertEqual(
            TextFormatter.formatStats(stats),
            """
            ┌─ BrainLayer Stats
            │ Chunks: 284,127
            │ Projects: brainlayer, orchestrator, voicelayer
            │ Types: assistant_text, user_message, ai_code
            └─
            """
        )

        XCTAssertEqual(
            TextFormatter.formatKGSearch(kg),
            """
            ┌─ Entity search: "BrainBar" (query: "what is brainbar") ─ 1 result
            ├─ Knowledge Graph (1 fact)
            │   BrainBar ─[part_of]→ BrainLayer
            │
            ├─ Memories (1)
            │ [1] kg-123456789  score:0.95
            │     BrainBar serves MCP over a Unix socket.
            └─
            """
        )
    }
}
