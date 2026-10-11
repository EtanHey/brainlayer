// StabilityFixTests.swift — RED tests for BrainBar stability fixes.
//
// TDD: Written before implementation.
// Covers: async store.

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
