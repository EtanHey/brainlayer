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
                tags: ["swift", "macos", "mcp"],
                sourceFile: "/Users/etanheyman/Gits/brainlayer/brain-bar/Sources/BrainBar/MCPRouter.swift"
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
            ## Search results for "brainbar native swift renderer parity should trun…" - 1 of 1 shown

            ### 1. BrainBar is a native macOS daemon for BrainLayer MCP routing.
            - Source: MCPRouter.swift
            - Date: 2026-03-29
            - Preview: BrainBar is a native macOS daemon for BrainLayer MCP routing.
            """
        )
        XCTAssertFalse(output.contains("score:"))
        XCTAssertFalse(output.contains("rt-abc123def"))
    }

    func testSearchResultsFullDetailExposesChunkID() {
        let results = [
            SearchResult(
                chunkID: "rt-abc123def4567890",
                score: 0.87,
                project: "brainlayer",
                date: "2026-03-29T19:30:00Z",
                summary: "BrainBar is a native macOS daemon for BrainLayer MCP routing.",
                snippet: "BrainBar is a native macOS daemon for BrainLayer MCP routing.",
                importance: 8,
                tags: ["swift", "macos", "mcp"],
                sourceFile: "/Users/etanheyman/Gits/brainlayer/brain-bar/Sources/BrainBar/MCPRouter.swift"
            )
        ]

        let output = TextFormatter.formatSearchResults(
            query: "brainbar",
            results: results,
            total: results.count,
            detail: "full"
        )

        XCTAssertTrue(output.contains("- ID: rt-abc123def4567890"), "full detail must expose chunk_id for chaining")
        // Sanity: the ID line should sit under the result header, before Source.
        XCTAssertTrue(
            output.contains("### 1. BrainBar is a native macOS daemon for BrainLayer MCP routing.\n- ID: rt-abc123def4567890\n- Source:"),
            "ID line must precede Source line"
        )
    }

    func testSearchResultsCompactDetailHidesChunkID() {
        let results = [
            SearchResult(
                chunkID: "rt-abc123def4567890",
                score: 0.87,
                project: "brainlayer",
                date: "2026-03-29T19:30:00Z",
                summary: "BrainBar is a native macOS daemon for BrainLayer MCP routing.",
                snippet: "BrainBar is a native macOS daemon for BrainLayer MCP routing.",
                importance: 8,
                sourceFile: "/Users/etanheyman/Gits/brainlayer/brain-bar/Sources/BrainBar/MCPRouter.swift"
            )
        ]

        // Explicit compact and the default must both hide the chunk_id.
        let explicitCompact = TextFormatter.formatSearchResults(
            query: "brainbar", results: results, total: 1, detail: "compact"
        )
        let defaultDetail = TextFormatter.formatSearchResults(query: "brainbar", results: results, total: 1)

        XCTAssertFalse(explicitCompact.contains("rt-abc123def"), "compact must not expose chunk_id")
        XCTAssertFalse(explicitCompact.contains("- ID:"))
        XCTAssertFalse(defaultDetail.contains("rt-abc123def"), "default detail must not expose chunk_id")
        XCTAssertFalse(defaultDetail.contains("- ID:"))
    }

    func testSearchResultSourceBasenameHandlesWindowsPaths() {
        let results = [
            SearchResult(
                chunkID: "rt-windows-path",
                project: "brainlayer",
                date: "2026-04-12T10:00:00Z",
                snippet: "Windows paths should not leak full absolute paths.",
                sourceFile: #"C:\Users\etan\brainlayer\src\auth.py"#
            )
        ]

        let output = TextFormatter.formatSearchResults(query: "path privacy", results: results, total: 1)

        XCTAssertTrue(output.contains("- Source: auth.py"))
        XCTAssertFalse(output.contains(#"C:\Users"#))
    }

    func testEntityCardUsesLabeledKGFactsStructure() {
        let expiredAt = ISO8601DateFormatter().date(from: "2026-05-01T00:00:00Z")
        let entity = EntityCard(
            id: "person-001",
            name: "Etan Heyman",
            entityType: "person",
            description: "Owner of the BrainLayer ecosystem.",
            profile: [
                "role": "Developer",
                "company": "BrainLayer",
                "location": "Tel Aviv"
            ],
            hardConstraints: ["timezone": "Asia/Jerusalem"],
            preferences: ["editor": "Neovim"],
            contactInfo: ["email": "etan@example.com"],
            relations: [
                EntityCard.Relation(relationType: "works_on", targetName: "BrainLayer"),
                EntityCard.Relation(relationType: "depends_on", targetName: "legacy-auth-lib", expiredAt: expiredAt)
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
            ## Entity: Etan Heyman

            Owner of the BrainLayer ecosystem.

            ### Profile
            - company: BrainLayer
            - location: Tel Aviv
            - role: Developer

            ### Constraints
            - timezone: Asia/Jerusalem

            ### Preferences
            - editor: Neovim

            ### Contact
            - email: etan@example.com

            ### KG Facts
            - works_on: BrainLayer
            - depends_on: legacy-auth-lib (expired 2026-05-01)

            ### Recent context
            - Chose Swift renderers for BrainBar parity.

            ### Likely follow-ups
            - BrainLayer
            - legacy-auth-lib
            """
        )
    }

    func testEntityCardDoesNotMarkFutureValidUntilExpired() {
        let validUntil = ISO8601DateFormatter().date(from: "2026-12-31T00:00:00Z")
        let entity = EntityCard(
            id: "tech-001",
            name: "JWT middleware",
            entityType: "technology",
            relations: [
                EntityCard.Relation(
                    relationType: "depends_on",
                    targetName: "auth-service",
                    validUntil: validUntil
                )
            ]
        )

        let output = TextFormatter.formatEntityCard(entity)

        XCTAssertTrue(output.contains("- depends_on: auth-service"))
        XCTAssertFalse(output.contains("expired 2026-12-31"))
    }

    func testRecallContextUsesLabeledChunkMarkdown() {
        let results = [
            SearchResult(
                chunkID: "chunk-auth-1",
                project: "brainlayer",
                date: "2026-04-12T10:00:00Z",
                summary: "",
                snippet: "We chose sliding-window refresh tokens with a short grace window for concurrent tabs.",
                sourceFile: "design-doc/auth-v2.md"
            )
        ]

        let output = TextFormatter.formatRecalledContext(query: "how did we handle session expiry", results: results)

        XCTAssertEqual(
            output,
            """
            ## Recalled context for "how did we handle session expiry"

            ### Chunk 1 - auth-v2.md
            We chose sliding-window refresh tokens with a short grace window for concurrent tabs.
            """
        )
        XCTAssertFalse(output.contains("score"))
        XCTAssertFalse(output.contains("[{"))
    }

    func testRecallContextFallsBackToProjectWhenSourceFileIsMissing() {
        let results = [
            SearchResult(
                chunkID: "chunk-session-1",
                project: "brainlayer",
                snippet: "Session recall row did not carry source_file."
            )
        ]

        let output = TextFormatter.formatRecalledContext(query: "session context", results: results)

        XCTAssertTrue(output.contains("### Chunk 1 - brainlayer"))
        XCTAssertFalse(output.contains("unknown"))
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
            ## Search results for "what is brainbar" - 1 of 1 shown

            ### KG Facts for BrainBar
            - BrainBar part_of BrainLayer

            ### 1. BrainBar serves MCP over a Unix socket.
            - Source: unknown
            - Date: unknown
            - Preview: BrainBar serves MCP over a Unix socket.
            """
        )
    }
}
