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
import SQLite3
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

    func testBrainbarAgentsTableExists() throws {
        let exists = try db.tableExists("brainbar_agents")
        XCTAssertTrue(exists, "brainbar_agents table must exist")
    }

    func testBrainbarSubscriptionsTableExists() throws {
        let exists = try db.tableExists("brainbar_subscriptions")
        XCTAssertTrue(exists, "brainbar_subscriptions table must exist")
    }

    func testUpsertSubscriptionRecoversMissingPubSubTables() throws {
        db.exec("DROP TABLE IF EXISTS brainbar_subscriptions")
        db.exec("DROP TABLE IF EXISTS brainbar_agents")

        let record = try db.upsertSubscription(agentID: "agent-a", tags: ["agent-message"])

        XCTAssertEqual(record.agentID, "agent-a")
        XCTAssertEqual(record.tags, ["agent-message"])
        XCTAssertTrue(try db.tableExists("brainbar_agents"))
        XCTAssertTrue(try db.tableExists("brainbar_subscriptions"))
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
        let stored = try db.store(
            content: "Decision: Use GRDB for SQLite access",
            tags: ["decision", "architecture"],
            importance: 8,
            source: "mcp"
        )

        XCTAssertFalse(stored.chunkID.isEmpty, "store should return a chunk ID")
        XCTAssertGreaterThan(stored.rowID, 0)

        // Verify it's searchable
        let results = try db.search(query: "GRDB SQLite", limit: 10)
        XCTAssertFalse(results.isEmpty)
    }

    func testStoreRetriesThroughTransientWriteLock() throws {
        var lockDB: OpaquePointer?
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX
        XCTAssertEqual(sqlite3_open_v2(tempDBPath, &lockDB, flags, nil), SQLITE_OK)
        guard let lockDB else {
            XCTFail("Failed to open secondary lock connection")
            return
        }
        defer { sqlite3_close(lockDB) }

        XCTAssertEqual(sqlite3_exec(lockDB, "BEGIN IMMEDIATE", nil, nil, nil), SQLITE_OK)

        let releaseExpectation = expectation(description: "release write lock")
        DispatchQueue.global().asyncAfter(deadline: .now() + 5.5, execute: DispatchWorkItem {
            sqlite3_exec(lockDB, "COMMIT", nil, nil, nil)
            releaseExpectation.fulfill()
        })

        let startedAt = Date()
        let stored = try db.store(
            content: "Store after transient lock",
            tags: ["retry"],
            importance: 5,
            source: "mcp"
        )

        XCTAssertFalse(stored.chunkID.isEmpty)
        XCTAssertGreaterThan(Date().timeIntervalSince(startedAt), 5.0)
        wait(for: [releaseExpectation], timeout: 7.0)
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

    // MARK: - Production rowid divergence

    /// Simulates production DB where FTS5 rowids don't match chunks rowids.
    /// In production, Python's trigger doesn't set explicit rowid, so after
    /// FTS5 table rebuilds, rowids diverge. The JOIN must use chunk_id, not rowid.
    func testImportanceFilterWorksWithDivergedRowids() throws {
        // Insert two chunks normally (synced rowids via trigger)
        try db.insertChunk(id: "div-1", content: "Unimportant chatter about weather", sessionId: "s1", project: "test", contentType: "assistant_text", importance: 2)
        try db.insertChunk(id: "div-2", content: "Critical architecture decision about caching", sessionId: "s2", project: "test", contentType: "assistant_text", importance: 9)

        // Simulate production rebuild: drop FTS5 table + triggers, recreate, re-populate
        // This creates divergent rowids: FTS5 rows get new rowids 1,2 but chunks keep original rowids
        db.exec("DROP TRIGGER IF EXISTS chunks_fts_insert")
        db.exec("DROP TRIGGER IF EXISTS chunks_fts_delete")
        db.exec("DROP TRIGGER IF EXISTS chunks_fts_update")
        db.exec("DROP TABLE IF EXISTS chunks_fts")
        db.exec("""
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                content, summary, tags, resolved_query, chunk_id UNINDEXED
            )
        """)
        // Re-populate WITHOUT explicit rowid (matches production trigger behavior).
        // FTS5 auto-assigns rowid 1 to div-2 and rowid 2 to div-1 (or vice versa),
        // which WON'T match the chunks table rowids.
        // Insert in REVERSE order to guarantee mismatch: FTS5 rowid 1 = high-importance chunk.
        db.exec("""
            INSERT INTO chunks_fts(content, summary, tags, resolved_query, chunk_id)
            SELECT content, summary, tags, NULL, id FROM chunks ORDER BY id DESC
        """)
        // Recreate trigger matching production (no explicit rowid)
        db.exec("""
            CREATE TRIGGER chunks_fts_insert AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(content, summary, tags, resolved_query, chunk_id)
                VALUES (new.content, new.summary, new.tags, NULL, new.id);
            END
        """)

        // Search with importance_min=7 — should return ONLY the high-importance chunk
        let filtered = try db.search(query: "architecture decision caching", limit: 10, importanceMin: 7.0)
        XCTAssertEqual(filtered.count, 1, "Should return only high-importance chunk")
        XCTAssertEqual(filtered.first?["chunk_id"] as? String, "div-2")

        // Also verify the returned importance is correct (not from wrong row)
        let importance = filtered.first?["importance"]
        XCTAssertNotNil(importance, "importance should be present")
        // Should be 9, not 2 (which would happen if rowids mapped to wrong chunk)
        if let imp = importance as? Int {
            XCTAssertGreaterThanOrEqual(imp, 7, "Returned importance must be >= 7")
        } else if let imp = importance as? Double {
            XCTAssertGreaterThanOrEqual(imp, 7.0, "Returned importance must be >= 7")
        }
    }

    /// Verify ALL results from importance_min search actually have importance >= threshold.
    func testAllResultsRespectImportanceMin() throws {
        try db.insertChunk(id: "all-1", content: "Vector database indexing strategies", sessionId: "s1", project: "test", contentType: "assistant_text", importance: 3)
        try db.insertChunk(id: "all-2", content: "Vector database performance tuning", sessionId: "s2", project: "test", contentType: "assistant_text", importance: 7)
        try db.insertChunk(id: "all-3", content: "Vector database scaling patterns", sessionId: "s3", project: "test", contentType: "assistant_text", importance: 9)

        let results = try db.search(query: "vector database", limit: 10, importanceMin: 7.0)
        XCTAssertEqual(results.count, 2, "Should return only chunks with importance >= 7")
        for result in results {
            if let imp = result["importance"] as? Int {
                XCTAssertGreaterThanOrEqual(imp, 7, "Every result must have importance >= 7")
            } else if let imp = result["importance"] as? Double {
                XCTAssertGreaterThanOrEqual(imp, 7.0, "Every result must have importance >= 7")
            }
        }
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
