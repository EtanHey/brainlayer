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
}
