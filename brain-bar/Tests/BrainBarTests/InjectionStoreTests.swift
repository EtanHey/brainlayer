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

    @MainActor
    func testDegradedStoreStaysDegradedUntilEventQuerySucceeds() throws {
        // Regression guard: Bugbot PR #315 flagged that a forced
        // listInjectionEvents failure could mark the store degraded, then a
        // later unchanged dataVersion poll could skip the event query and
        // incorrectly clear the badge. Recovery must require a successful
        // event query, not only a successful PRAGMA data_version read.
        let reader = ScriptedInjectionEventReader(
            versions: [1, 1, 1],
            eventResults: [
                .success([]),
                .failure(ScriptedInjectionError.listFailed),
                .success([
                    InjectionEvent(
                        id: 1,
                        sessionID: "session-recovered",
                        timestamp: "2026-05-22T18:05:00Z",
                        query: "recovered query",
                        chunkIDs: ["chunk-1"],
                        tokenCount: 7
                    )
                ]),
            ]
        )
        let store = InjectionStore(reader: reader)

        store.refreshForTesting(force: true)
        XCTAssertEqual(store.degradationState, .healthy)

        store.refreshForTesting(force: true)
        XCTAssertTrue(store.degradationState.isDegraded)

        store.refreshForTesting(force: false)
        XCTAssertTrue(
            store.degradationState.isDegraded,
            "An unchanged dataVersion poll skipped the event query; that is not positive recovery evidence."
        )

        store.refreshForTesting(force: false)
        XCTAssertEqual(store.degradationState, .healthy)
        XCTAssertEqual(store.events.map(\.query), ["recovered query"])
    }
}

private enum ScriptedInjectionError: Error {
    case listFailed
    case unexpectedListCall
}

private final class ScriptedInjectionEventReader: InjectionEventReading {
    deinit {}

    var versions: [Int]
    var eventResults: [Result<[InjectionEvent], Error>]

    init(versions: [Int], eventResults: [Result<[InjectionEvent], Error>]) {
        self.versions = versions
        self.eventResults = eventResults
    }

    func dataVersion() throws -> Int {
        if versions.count > 1 {
            return versions.removeFirst()
        }
        return versions.first ?? 1
    }

    func listInjectionEvents(sessionID: String?, limit: Int) throws -> [InjectionEvent] {
        guard !eventResults.isEmpty else {
            throw ScriptedInjectionError.unexpectedListCall
        }
        switch eventResults.removeFirst() {
        case .success(let events):
            return events
        case .failure(let error):
            throw error
        }
    }

    func expandedConversation(
        chunkID: String,
        before: Int,
        after: Int
    ) throws -> BrainDatabase.ExpandedConversation {
        throw ScriptedInjectionError.listFailed
    }

    func close() {}
}
