// DatabaseTests.swift — RED tests for BrainBar SQLite database layer.
//
// BrainBar embeds SQLite with:
// - WAL mode
// - FTS5 for full-text search
// - busy_timeout=5000
// - cache_size=-64000 (64MB)
// - synchronous=NORMAL
// - Single-writer architecture (no concurrent writes)

import XCTest
@testable import BrainBar

final class DatabaseTests: XCTestCase {
    var db: BrainDatabase!
    var tempDBPath: String!

    override func setUp() {
        super.setUp()
        tempDBPath = NSTemporaryDirectory() + "brainbar-test-\(UUID().uuidString).db"
        db = BrainDatabase(path: tempDBPath)
    }

    override func tearDown() {
        db.close()
        try? FileManager.default.removeItem(atPath: tempDBPath)
        try? FileManager.default.removeItem(atPath: tempDBPath + "-wal")
        try? FileManager.default.removeItem(atPath: tempDBPath + "-shm")
        super.tearDown()
    }

    // MARK: - PRAGMAs

    func testWALModeEnabled() throws {
        let mode = try db.pragma("journal_mode")
        XCTAssertEqual(mode, "wal")
    }

    func testBusyTimeoutSet() throws {
        let timeout = try db.pragma("busy_timeout")
        XCTAssertEqual(timeout, "5000")
    }

    func testCacheSizeSet() throws {
        let cacheSize = try db.pragma("cache_size")
        XCTAssertEqual(cacheSize, "-64000")
    }

    func testSynchronousNormal() throws {
        let sync = try db.pragma("synchronous")
        // NORMAL = 1
        XCTAssertEqual(sync, "1")
    }

    // MARK: - Schema

    func testChunksTableExists() throws {
        let exists = try db.tableExists("chunks")
        XCTAssertTrue(exists, "chunks table must exist")
    }

    func testFTSTableExists() throws {
        let exists = try db.tableExists("chunks_fts")
        XCTAssertTrue(exists, "chunks_fts FTS5 table must exist")
    }

    // MARK: - Search (FTS5)

    func testFTSSearchReturnsResults() throws {
        // Insert a test chunk
        try db.insertChunk(
            id: "test-chunk-1",
            content: "Authentication was implemented using JWT tokens with refresh rotation",
            sessionId: "session-1",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 7
        )

        let results = try db.search(query: "authentication JWT", limit: 10)
        XCTAssertFalse(results.isEmpty, "FTS search should find the inserted chunk")
        XCTAssertEqual(results.first?["chunk_id"] as? String, "test-chunk-1")
    }

    func testSearchReturnsEmptyForNoMatch() throws {
        let results = try db.search(query: "xyznonexistent123", limit: 10)
        XCTAssertTrue(results.isEmpty)
    }

    // MARK: - Store

    func testStoreCreatesChunk() throws {
        let id = try db.store(
            content: "Decision: Use GRDB for SQLite access",
            tags: ["decision", "architecture"],
            importance: 8,
            source: "mcp"
        )

        XCTAssertFalse(id.isEmpty, "store should return a chunk ID")

        // Verify it's searchable
        let results = try db.search(query: "GRDB SQLite", limit: 10)
        XCTAssertFalse(results.isEmpty)
    }

    // MARK: - Concurrent reads

    func testConcurrentReadsDoNotBlock() throws {
        // Insert test data
        try db.insertChunk(
            id: "concurrent-1",
            content: "Test chunk for concurrent reads",
            sessionId: "session-1",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 5
        )

        let expectation = XCTestExpectation(description: "concurrent reads")
        expectation.expectedFulfillmentCount = 10

        for _ in 0..<10 {
            DispatchQueue.global().async {
                do {
                    let results = try self.db.search(query: "concurrent", limit: 5)
                    XCTAssertFalse(results.isEmpty)
                } catch {
                    XCTFail("Concurrent read failed: \(error)")
                }
                expectation.fulfill()
            }
        }

        wait(for: [expectation], timeout: 5.0)
    }
}
