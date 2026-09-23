// StabilityFixTests.swift — RED tests for BrainBar stability fixes.
//
// TDD: Written before implementation.
// Covers: async store, search result dates, Enter key behavior, popover sizing.

import XCTest
@testable import BrainBar

// MARK: - (1) Store should be async (non-blocking)

final class AsyncStoreTests: XCTestCase {
    var db: BrainDatabase!
    var tempDBPath: String!

    override func setUp() {
        super.setUp()
        tempDBPath = NSTemporaryDirectory() + "brainbar-async-test-\(UUID().uuidString).db"
        db = BrainDatabase(path: tempDBPath)
    }

    override func tearDown() {
        db.close()
        try? FileManager.default.removeItem(atPath: tempDBPath)
        try? FileManager.default.removeItem(atPath: tempDBPath + "-wal")
        try? FileManager.default.removeItem(atPath: tempDBPath + "-shm")
        super.tearDown()
    }

    func testStoreAsyncReturnsChunkID() async throws {
        let stored = try await db.storeAsync(
            content: "Test content",
            tags: ["test"],
            importance: 5,
            source: "unit-test"
        )
        XCTAssertFalse(stored.chunkID.isEmpty)
    }

    func testStoreAsyncContentRetrievable() async throws {
        let stored = try await db.storeAsync(
            content: "Async stored content for retrieval test",
            tags: ["async-test"],
            importance: 7,
            source: "unit-test"
        )
        XCTAssertFalse(stored.chunkID.isEmpty)
        // Verify the stored content is searchable
        let results = try db.search(query: "retrieval test", limit: 5)
        XCTAssertFalse(results.isEmpty)
    }
}

// MARK: - (2) Search results must include dates

final class SearchResultDateTests: XCTestCase {
    var db: BrainDatabase!
    var tempDBPath: String!

    override func setUp() {
        super.setUp()
        tempDBPath = NSTemporaryDirectory() + "brainbar-date-test-\(UUID().uuidString).db"
        db = BrainDatabase(path: tempDBPath)
    }

    override func tearDown() {
        db.close()
        try? FileManager.default.removeItem(atPath: tempDBPath)
        try? FileManager.default.removeItem(atPath: tempDBPath + "-wal")
        try? FileManager.default.removeItem(atPath: tempDBPath + "-shm")
        super.tearDown()
    }

    func testSearchCandidateIncludesDate() throws {
        try db.insertChunk(
            id: "dated-chunk", content: "React Server Components architecture",
            sessionId: "s1", project: "brainlayer", contentType: "ai_code", importance: 7
        )
        let candidates = try db.searchCandidates(query: "React", limit: 5)
        XCTAssertFalse(candidates.isEmpty)
        XCTAssertFalse(candidates.first!.date.isEmpty, "Candidate must include a date")
    }

    func testSearchCandidateIncludesProject() throws {
        try db.insertChunk(
            id: "proj-chunk", content: "BrainLayer memory pipeline",
            sessionId: "s1", project: "brainlayer", contentType: "ai_code", importance: 5
        )
        let candidates = try db.searchCandidates(query: "memory pipeline", limit: 5)
        XCTAssertFalse(candidates.isEmpty)
        XCTAssertEqual(candidates.first!.project, "brainlayer")
    }

    func testSearchCandidateIncludesImportance() throws {
        try db.insertChunk(
            id: "imp-chunk", content: "Important decision about database",
            sessionId: "s1", project: "test", contentType: "ai_code", importance: 8
        )
        let candidates = try db.searchCandidates(query: "decision database", limit: 5)
        XCTAssertFalse(candidates.isEmpty)
        XCTAssertEqual(candidates.first!.importance, 8)
    }
}

// MARK: - (3) Enter in search should select result, not switch to capture

@MainActor
final class EnterKeySearchTests: XCTestCase {


}
