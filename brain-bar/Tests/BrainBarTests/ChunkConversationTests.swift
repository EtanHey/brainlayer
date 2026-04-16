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
}
