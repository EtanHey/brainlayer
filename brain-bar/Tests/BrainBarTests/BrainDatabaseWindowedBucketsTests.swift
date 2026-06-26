// BrainDatabaseWindowedBucketsTests.swift
//
// Proves the shared Live(30m)/3h/24h selector returns REAL historical data per
// window, not a relabel of the live buckets. The dashboard redesign removed the
// per-card expand/collapse model in favor of ONE shared timeframe selector that
// must re-fetch genuine DB history when the timeframe widens. The data primitive
// is `BrainDatabase.pipelineWindowBuckets(activityWindowMinutes:bucketCount:now:)`.
//
// Strategy: insert chunks spanning >30m (some inside 30m, some 45m–2h ago), then
// assert the 24h-window buckets contain strictly MORE than the 30m-window
// buckets — i.e. widening the window pulls in actual older rows.

import XCTest
import SQLite3
@testable import BrainBar

final class BrainDatabaseWindowedBucketsTests: XCTestCase {
    private var db: BrainDatabase!
    private var tempDBPath: String!

    override func setUp() {
        super.setUp()
        tempDBPath = NSTemporaryDirectory() + "brainbar-windowed-buckets-\(UUID().uuidString).db"
        db = BrainDatabase(path: tempDBPath)
    }

    override func tearDown() {
        db.close()
        try? FileManager.default.removeItem(atPath: tempDBPath)
        try? FileManager.default.removeItem(atPath: tempDBPath + "-wal")
        try? FileManager.default.removeItem(atPath: tempDBPath + "-shm")
        super.tearDown()
    }

    private static let utcISO8601: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(secondsFromGMT: 0)
        f.dateFormat = "yyyy-MM-dd'T'HH:mm:ss'Z'"
        return f
    }()

    /// Direct write so we control `created_at`, `source`, and enrichment columns
    /// — `insertChunk` does not expose `source`, and the source whereclauses are
    /// the whole point of the per-series split.
    private func insertWrite(
        id: String,
        source: String,
        createdAt: Date,
        enrichedAt: Date? = nil
    ) throws {
        let createdText = Self.utcISO8601.string(from: createdAt)
        let enrichedClause: String
        if let enrichedAt {
            enrichedClause = "'\(Self.utcISO8601.string(from: enrichedAt))', 'success'"
        } else {
            enrichedClause = "NULL, NULL"
        }
        try sqliteExecWriteLocal(
            path: tempDBPath,
            sql: """
                INSERT INTO chunks (id, content, source, created_at, enriched_at, enrich_status, status)
                VALUES ('\(id)', 'windowed bucket probe \(id)', '\(source)', '\(createdText)', \(enrichedClause), 'active');
            """
        )
    }

    private func insertWatcherLiveness(chunkID: String, ingestedAt: Date) throws {
        try sqliteExecWriteLocal(
            path: tempDBPath,
            sql: """
                CREATE TABLE IF NOT EXISTS watcher_liveness_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id TEXT NOT NULL,
                    ingested_at INTEGER NOT NULL
                );
                INSERT INTO watcher_liveness_events (chunk_id, ingested_at)
                VALUES ('\(chunkID)', \(Int(ingestedAt.timeIntervalSince1970)));
            """
        )
    }

    func testWiderWindowReturnsMoreHistoricalDataThanLiveWindow() throws {
        // Force the schema to exist (constructor opens + ensures schema, but make
        // it explicit so an empty-table call cannot be misread).
        XCTAssertTrue(try db.tableExists("chunks"))

        let now = Date()
        func minutesAgo(_ m: Double) -> Date { now.addingTimeInterval(-m * 60) }

        // --- Inside the 30m window (appears in BOTH windows) ---
        try insertWrite(id: "agent-live-1", source: "mcp", createdAt: minutesAgo(5))
        try insertWrite(id: "agent-live-2", source: "brain_store", createdAt: minutesAgo(20))
        try insertWrite(id: "watcher-live-1", source: "realtime_watcher", createdAt: minutesAgo(8))
        try insertWrite(
            id: "enrich-live-1",
            source: "mcp",
            createdAt: minutesAgo(90),
            enrichedAt: minutesAgo(12)
        )

        // --- OUTSIDE the 30m window but INSIDE 3h/24h (only in wider windows) ---
        try insertWrite(id: "agent-old-1", source: "mcp", createdAt: minutesAgo(45))
        try insertWrite(id: "agent-old-2", source: "manual", createdAt: minutesAgo(120))
        try insertWrite(id: "agent-old-3", source: "digest", createdAt: minutesAgo(600)) // 10h ago
        try insertWrite(id: "watcher-old-1", source: "realtime_watcher", createdAt: minutesAgo(70))
        try insertWrite(id: "watcher-old-2", source: "realtime", createdAt: minutesAgo(300)) // 5h ago
        try insertWrite(
            id: "enrich-old-1",
            source: "mcp",
            createdAt: minutesAgo(800),
            enrichedAt: minutesAgo(200) // ~3.3h ago: outside 3h, inside 24h
        )

        let live = try db.pipelineWindowBuckets(activityWindowMinutes: 30, bucketCount: 12, now: now)
        let threeHour = try db.pipelineWindowBuckets(activityWindowMinutes: 180, bucketCount: 12, now: now)
        let day = try db.pipelineWindowBuckets(activityWindowMinutes: 1_440, bucketCount: 12, now: now)

        // 1) The windows must NOT be identical — this is the core "not just a
        //    relabel" assertion. Real older data comes back as the window widens.
        XCTAssertNotEqual(live.agentWriteBuckets, day.agentWriteBuckets,
                          "24h agent buckets must differ from 30m (historical data must come back)")
        XCTAssertNotEqual(live.watcherWriteBuckets, day.watcherWriteBuckets,
                          "24h watcher buckets must differ from 30m")

        // The agent-source whereclause counts ANY row whose `source` is an agent
        // source (mcp/manual/digest/brain_store), regardless of whether it was
        // later enriched. So the two "enrich-*" rows (source mcp) are agent
        // writes whose `created_at` lands them in the window too.

        // 2) Live (30m) sees only the in-window rows.
        //    Agent: agent-live-1(5m) + agent-live-2(20m) = 2.
        XCTAssertEqual(live.agentTotal, 2, "30m agent window = agent-live-1 + agent-live-2")
        XCTAssertEqual(live.watcherTotal, 1, "30m watcher window = watcher-live-1")
        XCTAssertEqual(live.enrichmentTotal, 1, "30m enrichment window = enrich-live-1 (enriched 12m ago)")

        // 3) Wider windows strictly CONTAIN MORE than the live window.
        //    24h agent: agent-live-1/2 + agent-old-1/2/3 + enrich-live-1(90m,mcp)
        //    + enrich-old-1(800m,mcp) = 7.
        XCTAssertEqual(day.agentTotal, 7, "24h agent window pulls in all 7 agent-source writes")
        XCTAssertGreaterThan(day.agentTotal, live.agentTotal,
                             "24h agent total must exceed 30m agent total")
        XCTAssertEqual(day.watcherTotal, 3, "24h watcher window pulls in 2 older watcher writes")
        XCTAssertGreaterThan(day.watcherTotal, live.watcherTotal,
                             "24h watcher total must exceed 30m watcher total")
        XCTAssertEqual(day.enrichmentTotal, 2, "24h enrichment window pulls in the older enrichment")
        XCTAssertGreaterThan(day.enrichmentTotal, live.enrichmentTotal,
                             "24h enrichment total must exceed 30m enrichment total")

        // 4) 3h sits between live and day — monotonic widening, real data.
        //    3h agent: agent-live-1/2 + agent-old-1(45m) + agent-old-2(120m)
        //    + enrich-live-1(90m,mcp) = 5.
        XCTAssertEqual(threeHour.agentTotal, 5, "3h agent window = 2 live + agent-old-1 + agent-old-2 + enrich-live-1")
        XCTAssertGreaterThan(threeHour.agentTotal, live.agentTotal)
        XCTAssertLessThan(threeHour.agentTotal, day.agentTotal)
        XCTAssertEqual(threeHour.watcherTotal, 2, "3h watcher window = watcher-live-1 + watcher-old-1(70m)")
        // enrich-old-1 enriched ~200m ago is OUTSIDE 3h (180m) but INSIDE 24h.
        XCTAssertEqual(threeHour.enrichmentTotal, 1, "3h enrichment still only sees the live enrichment")
    }

    func testEmptyWindowAndZeroBucketsAreSafe() throws {
        let now = Date()
        let zeroBuckets = try db.pipelineWindowBuckets(activityWindowMinutes: 1_440, bucketCount: 0, now: now)
        XCTAssertTrue(zeroBuckets.agentWriteBuckets.isEmpty)

        let emptyDB = try db.pipelineWindowBuckets(activityWindowMinutes: 180, bucketCount: 12, now: now)
        XCTAssertEqual(emptyDB.agentTotal, 0)
        XCTAssertEqual(emptyDB.watcherTotal, 0)
        XCTAssertEqual(emptyDB.enrichmentTotal, 0)
        XCTAssertEqual(emptyDB.agentWriteBuckets.count, 12)
    }

    func testWatcherWindowBucketsUseIngestionLivenessInsteadOfTranscriptCreatedAt() throws {
        XCTAssertTrue(try db.tableExists("chunks"))

        let now = Date()
        try insertWrite(
            id: "watcher-ingested-now",
            source: "realtime_watcher",
            createdAt: now.addingTimeInterval(-22 * 60)
        )
        try insertWatcherLiveness(
            chunkID: "watcher-ingested-now",
            ingestedAt: now.addingTimeInterval(-30)
        )

        let buckets = try db.pipelineWindowBuckets(activityWindowMinutes: 30, bucketCount: 6, now: now)

        XCTAssertEqual(
            buckets.watcherWriteBuckets,
            [0, 0, 0, 0, 0, 1],
            "JSONL watcher graph must show durable ingestion recency, not the older transcript event time."
        )
    }
}

/// Local raw-SQL writer (the DatabaseTests one is file-private to that file).
private func sqliteExecWriteLocal(path: String, sql: String) throws {
    var db: OpaquePointer?
    let rc = sqlite3_open_v2(
        path,
        &db,
        SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX,
        nil
    )
    guard rc == SQLITE_OK, let db else {
        throw NSError(domain: "BrainDatabaseWindowedBucketsTests", code: Int(rc))
    }
    defer { sqlite3_close(db) }

    var errPtr: UnsafeMutablePointer<CChar>?
    let execRC = sqlite3_exec(db, sql, nil, nil, &errPtr)
    if execRC != SQLITE_OK {
        let message = errPtr.map { String(cString: $0) } ?? "unknown"
        sqlite3_free(errPtr)
        throw NSError(
            domain: "BrainDatabaseWindowedBucketsTests",
            code: Int(execRC),
            userInfo: [NSLocalizedDescriptionKey: message]
        )
    }
}
