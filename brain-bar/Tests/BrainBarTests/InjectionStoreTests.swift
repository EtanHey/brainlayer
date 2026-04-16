import XCTest
@testable import BrainBar

final class InjectionStoreTests: XCTestCase {
    private var tempDBPath: String!
    private var db: BrainDatabase!

    override func setUp() {
        super.setUp()
        tempDBPath = NSTemporaryDirectory() + "brainbar-injection-store-\(UUID().uuidString).db"
        db = BrainDatabase(path: tempDBPath)
    }

    override func tearDown() {
        db.close()
        try? FileManager.default.removeItem(atPath: tempDBPath)
        try? FileManager.default.removeItem(atPath: tempDBPath + "-wal")
        try? FileManager.default.removeItem(atPath: tempDBPath + "-shm")
        super.tearDown()
    }

    func testStartPublishesExistingEventsImmediately() async throws {
        try db.recordInjectionEvent(
            sessionID: "session-1",
            query: "existing event",
            chunkIDs: ["chunk-1"],
            tokenCount: 11
        )

        let databasePath = tempDBPath!
        let store = try await MainActor.run { try InjectionStore(databasePath: databasePath) }
        defer { Task { @MainActor in store.stop() } }
        await MainActor.run { store.start() }

        try await Task.sleep(for: .milliseconds(150))

        let queries = await MainActor.run { store.events.map(\.query) }
        XCTAssertEqual(queries, ["existing event"])
    }

    func testObservationPublishesNewEventsAfterInsert() async throws {
        let databasePath = tempDBPath!
        let store = try await MainActor.run { try InjectionStore(databasePath: databasePath) }
        defer { Task { @MainActor in store.stop() } }
        await MainActor.run { store.start() }

        try db.recordInjectionEvent(
            sessionID: "session-2",
            query: "new event",
            chunkIDs: ["chunk-a", "chunk-b"],
            tokenCount: 22
        )

        try await Task.sleep(for: .milliseconds(250))

        let firstQuery = await MainActor.run { store.events.first?.query }
        let firstChunkIDs = await MainActor.run { store.events.first?.chunkIDs }
        XCTAssertEqual(firstQuery, "new event")
        XCTAssertEqual(firstChunkIDs, ["chunk-a", "chunk-b"])
    }

    func testExpandedConversationLoadsTargetChunkAndContext() async throws {
        for index in 1...4 {
            try db.insertChunk(
                id: "inject-\(index)",
                content: "Injection conversation \(index)",
                sessionId: "inject-session",
                project: "brainlayer",
                contentType: index.isMultiple(of: 2) ? "assistant_text" : "user_message",
                importance: 5
            )
        }

        let databasePath = tempDBPath!
        let store = try await MainActor.run { try InjectionStore(databasePath: databasePath) }
        defer { Task { @MainActor in store.stop() } }
        let conversation = try await MainActor.run {
            try store.expandedConversation(chunkID: "inject-2")
        }

        XCTAssertEqual(conversation.target.chunkID, "inject-2")
        XCTAssertEqual(
            conversation.entries.map(\.chunkID),
            ["inject-1", "inject-2", "inject-3", "inject-4"]
        )
    }
}
