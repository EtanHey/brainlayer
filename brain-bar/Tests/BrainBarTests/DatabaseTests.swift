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

    // MARK: - Filter: project

    func testSearchFiltersByProject() throws {
        try db.insertChunk(id: "proj-a-1", content: "Authentication uses JWT tokens", sessionId: "s1", project: "alpha", contentType: "assistant_text", importance: 5)
        try db.insertChunk(id: "proj-b-1", content: "Authentication uses OAuth tokens", sessionId: "s2", project: "beta", contentType: "assistant_text", importance: 5)

        let filtered = try db.search(query: "authentication tokens", limit: 10, project: "alpha")
        XCTAssertEqual(filtered.count, 1, "Should return only alpha project chunk")
        XCTAssertEqual(filtered.first?["project"] as? String, "alpha")
    }

    func testSearchWithoutProjectReturnsAll() throws {
        try db.insertChunk(id: "all-a", content: "Database migration script", sessionId: "s1", project: "alpha", contentType: "assistant_text", importance: 5)
        try db.insertChunk(id: "all-b", content: "Database migration tool", sessionId: "s2", project: "beta", contentType: "assistant_text", importance: 5)

        let all = try db.search(query: "database migration", limit: 10)
        XCTAssertEqual(all.count, 2, "Without filter, both projects should be returned")
    }

    // MARK: - Filter: importance_min

    func testSearchFiltersByImportanceMin() throws {
        try db.insertChunk(id: "imp-low", content: "Logging configuration setup", sessionId: "s1", project: "test", contentType: "assistant_text", importance: 3)
        try db.insertChunk(id: "imp-high", content: "Logging security audit", sessionId: "s2", project: "test", contentType: "assistant_text", importance: 8)

        let filtered = try db.search(query: "logging", limit: 10, importanceMin: 7.0)
        XCTAssertEqual(filtered.count, 1, "Should return only high-importance chunk")
        XCTAssertEqual(filtered.first?["chunk_id"] as? String, "imp-high")
    }

    // MARK: - Filter: tag

    func testSearchFiltersByTag() throws {
        try db.insertChunk(id: "tag-1", content: "Fixed the authentication bug in login flow", sessionId: "s1", project: "test", contentType: "assistant_text", importance: 5, tags: "[\"bug-fix\", \"auth\"]")
        try db.insertChunk(id: "tag-2", content: "Fixed the authentication bug in signup flow", sessionId: "s2", project: "test", contentType: "assistant_text", importance: 5)

        let filtered = try db.search(query: "authentication bug", limit: 10, tag: "bug-fix")
        XCTAssertEqual(filtered.count, 1, "Should return only tagged chunk")
        XCTAssertEqual(filtered.first?["chunk_id"] as? String, "tag-1")
    }

    // MARK: - Filter: combined

    func testSearchCombinesFilters() throws {
        try db.insertChunk(id: "combo-1", content: "API rate limiting implementation", sessionId: "s1", project: "alpha", contentType: "assistant_text", importance: 9)
        try db.insertChunk(id: "combo-2", content: "API rate limiting design", sessionId: "s2", project: "beta", contentType: "assistant_text", importance: 9)
        try db.insertChunk(id: "combo-3", content: "API rate limiting notes", sessionId: "s3", project: "alpha", contentType: "assistant_text", importance: 3)

        let filtered = try db.search(query: "API rate limiting", limit: 10, project: "alpha", importanceMin: 7.0)
        XCTAssertEqual(filtered.count, 1, "Should match only alpha + high importance")
        XCTAssertEqual(filtered.first?["chunk_id"] as? String, "combo-1")
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
