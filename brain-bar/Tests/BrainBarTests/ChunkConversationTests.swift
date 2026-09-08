import XCTest
@testable import BrainBar

final class ChunkConversationTests: XCTestCase {
    private var db: BrainDatabase!
    private var tempDBPath: String!

    override func setUp() {
        super.setUp()
        tempDBPath = NSTemporaryDirectory() + "brainbar-chunk-conversation-\(UUID().uuidString).db"
        db = BrainDatabase(path: tempDBPath)
    }

    override func tearDown() {
        db.close()
        try? FileManager.default.removeItem(atPath: tempDBPath)
        try? FileManager.default.removeItem(atPath: tempDBPath + "-wal")
        try? FileManager.default.removeItem(atPath: tempDBPath + "-shm")
        super.tearDown()
    }

    func testExpandedConversationPlacesTargetBetweenBeforeAndAfterContext() throws {
        for index in 1...5 {
            try db.insertChunk(
                id: "conv-\(index)",
                content: "Conversation message \(index)",
                sessionId: "conversation-thread",
                project: "brainlayer",
                contentType: index.isMultiple(of: 2) ? "assistant_text" : "user_message",
                importance: 5
            )
        }

        let conversation = try db.expandedConversation(id: "conv-3", before: 2, after: 2)

        XCTAssertEqual(
            conversation.entries.map(\.chunkID),
            ["conv-1", "conv-2", "conv-3", "conv-4", "conv-5"]
        )
        XCTAssertEqual(conversation.target.chunkID, "conv-3")
        XCTAssertTrue(conversation.entries[2].isTarget)
        XCTAssertEqual(conversation.origin, .indexedExcerpt)
    }

    func testExpandedConversationPreservesSenderIndependentOfContentType() throws {
        try db.insertChunk(
            id: "role-user",
            content: "Can you write the copy?",
            sessionId: "role-thread",
            project: "brainlayer",
            contentType: "user_message",
            importance: 5,
            sender: "user"
        )
        try db.insertChunk(
            id: "role-assistant",
            content: "Make your next big call with confidence.",
            sessionId: "role-thread",
            project: "brainlayer",
            contentType: "user_message",
            importance: 5,
            sender: "assistant"
        )
        try db.insertChunk(
            id: "role-user-2",
            content: "Only this line was genuinely user-authored.",
            sessionId: "role-thread",
            project: "brainlayer",
            contentType: "user_message",
            importance: 5,
            sender: "user"
        )

        let conversation = try db.expandedConversation(id: "role-assistant", before: 1, after: 1)

        XCTAssertEqual(conversation.entries.map(\.sender), ["user", "assistant", "user"])
    }

    func testExpandedConversationCapsHugeThreadWorkForResponsiveOpen() throws {
        let hugeContent = String(repeating: "Long transcript line with enough content to stress SwiftUI rendering.\n", count: 120)
        for index in 1...180 {
            try db.insertChunk(
                id: "huge-conv-\(index)",
                content: "\(index): \(hugeContent)",
                sessionId: "huge-conversation-thread",
                project: "brainlayer",
                contentType: index.isMultiple(of: 2) ? "assistant_text" : "user_message",
                importance: 5
            )
        }

        let conversation = try db.expandedConversation(id: "huge-conv-90", before: 10_000, after: 10_000)

        XCTAssertLessThanOrEqual(conversation.entries.count, 81)
        XCTAssertTrue(conversation.entries.contains { $0.chunkID == "huge-conv-90" && $0.isTarget })
        XCTAssertTrue(
            conversation.entries.allSatisfy { $0.content.count <= 4_200 },
            "Conversation expansion must not hand full multi-thousand-line payloads to SwiftUI synchronously."
        )
    }

    func testExpandedConversationReconstructsFullSourceJSONLAndNamesStoringAgent() throws {
        let sourceURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("brainbar-source-conversation-\(UUID().uuidString).jsonl")
        defer { try? FileManager.default.removeItem(at: sourceURL) }

        let rows: [[String: Any]] = [
            [
                "type": "user",
                "sessionId": "source-session",
                "timestamp": "2026-09-08T09:00:00Z",
                "message": ["role": "user", "content": "First question"]
            ],
            [
                "type": "assistant",
                "sessionId": "source-session",
                "timestamp": "2026-09-08T09:01:00Z",
                "attributionAgent": "wrong-duplicate-agent",
                "message": [
                    "role": "assistant",
                    "content": [["type": "text", "text": "## Decision\n**Ship it**"]]
                ]
            ],
            [
                "type": "assistant",
                "sessionId": "source-session",
                "timestamp": "2026-09-08T09:01:30Z",
                "attributionAgent": "brainlayerClaude-source",
                "message": [
                    "role": "assistant",
                    "content": [["type": "text", "text": "## Decision\n**Ship it**"]]
                ]
            ],
            [
                "type": "user",
                "sessionId": "source-session",
                "timestamp": "2026-09-08T09:02:00Z",
                "message": ["role": "user", "content": "Final confirmation"]
            ]
        ]
        let serializedRows = try rows.map { row in
            String(data: try JSONSerialization.data(withJSONObject: row), encoding: .utf8)!
        }
        let jsonl = serializedRows.joined(separator: "\n") + "\n"
        let targetEndOffset = serializedRows.prefix(3).reduce(0) { partial, row in
            partial + row.utf8.count + 1
        }
        try Data(jsonl.utf8).write(to: sourceURL)

        try db.insertChunk(
            id: "source-target",
            content: "## Decision\n**Ship it**",
            sessionId: "source-session",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 9,
            sender: "assistant"
        )
        db.exec("ALTER TABLE chunks ADD COLUMN source_end_offset INTEGER")
        db.exec(
            "UPDATE chunks SET source_file = '\(sourceURL.path)', "
                + "source_end_offset = \(targetEndOffset), tags = '[\"decision\"]' "
                + "WHERE id = 'source-target'"
        )

        let conversation = try db.expandedConversation(id: "source-target")

        XCTAssertEqual(conversation.origin, .sourceJSONL)
        XCTAssertEqual(conversation.entries.map(\.content), [
            "First question",
            "## Decision\n**Ship it**",
            "## Decision\n**Ship it**",
            "Final confirmation"
        ])
        XCTAssertEqual(conversation.entries.map(\.sender), ["user", "assistant", "assistant", "user"])
        XCTAssertEqual(conversation.target.createdAt, "2026-09-08T09:01:30Z")
        XCTAssertEqual(conversation.storingAgent, "brainlayerClaude-source")
        XCTAssertEqual(conversation.sourceFile, sourceURL.path)
        XCTAssertEqual(conversation.target.tags, ["decision"])
    }

    func testSourceConversationReaderReconstructsCodexRowsAndUsesContractSeat() throws {
        let sourceURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("brainbar-codex-conversation-\(UUID().uuidString).jsonl")
        defer { try? FileManager.default.removeItem(at: sourceURL) }
        let rows: [[String: Any]] = [
            [
                "type": "response_item",
                "payload": [
                    "type": "message",
                    "role": "user",
                    "content": [[
                        "type": "input_text",
                        "text": "cmuxlayer contract for brainlayerCodex-source: read it"
                    ]]
                ]
            ],
            [
                "type": "response_item",
                "payload": [
                    "type": "message",
                    "role": "assistant",
                    "content": [["type": "output_text", "text": "Codex final answer"]]
                ]
            ]
        ]
        let jsonl = try rows.map { row in
            String(data: try JSONSerialization.data(withJSONObject: row), encoding: .utf8)!
        }.joined(separator: "\n") + "\n"
        try Data(jsonl.utf8).write(to: sourceURL)
        let target = BrainDatabase.ConversationChunk(
            chunkID: "codex-target",
            content: "Codex final answer",
            contentType: "assistant_text",
            sender: "assistant",
            importance: 8,
            createdAt: "",
            summary: "",
            isTarget: true,
            sourceFile: sourceURL.path
        )

        let conversation = try XCTUnwrap(SourceConversationReader.reconstruct(target: target))

        XCTAssertEqual(conversation.entries.map(\.sender), ["user", "assistant"])
        XCTAssertEqual(conversation.entries.map(\.content), [
            "cmuxlayer contract for brainlayerCodex-source: read it",
            "Codex final answer"
        ])
        XCTAssertEqual(conversation.storingAgent, "brainlayerCodex-source")
    }
}
