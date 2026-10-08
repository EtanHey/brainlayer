import Foundation
import SQLite3
import XCTest
@testable import BrainBar

final class BrainBarNoEnrichmentQueryTests: XCTestCase {
    private final class ReadAudit {
        var forbidden: [String] = []
    }

    func testDashboardReadsNoRetiredQueueOrMetadataAndPreservesSentinels() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let restoreWatcher = isolateWatcherHealthEnvironment(root: root)
        let restoreFallback = isolateQueueEnvironment("BRAINBAR_FALLBACK_REPLAY_GITS_ROOT", root)
        let restoreQueue = isolateQueueEnvironment("BRAINLAYER_QUEUE_DIR", root.appendingPathComponent("queue"))
        defer { restoreWatcher(); restoreFallback(); restoreQueue() }
        let db = BrainDatabase(path: root.appendingPathComponent("private.db").path)
        defer { db.close() }
        try db.insertChunk(id: "historical-pending", content: "synthetic historical sentinel",
                           sessionId: "synthetic", project: "synthetic", contentType: "user_message", importance: 5)
        try db.insertChunk(id: "historical-success", content: "synthetic metadata sentinel",
                           sessionId: "synthetic", project: "synthetic", contentType: "user_message", importance: 5)
        db.exec("UPDATE chunks SET enrich_status='success', enriched_at='2026-01-01', summary='historical sentinel' WHERE id='historical-success'")
        // Auxiliary historical table is deliberately synthetic, not a claim about a production table.
        db.exec("CREATE TABLE historical_queue_sentinel (id TEXT PRIMARY KEY, payload TEXT)")
        db.exec("INSERT INTO historical_queue_sentinel VALUES ('private-history', 'unchanged')")
        let before = try snapshot(db)
        let audit = ReadAudit()
        let context = Unmanaged.passUnretained(audit).toOpaque()
        try db.withSQLiteHandleForTesting { handle in
            XCTAssertEqual(sqlite3_set_authorizer(handle, { context, action, table, column, _, _ in
                guard action == SQLITE_READ, let context else { return SQLITE_OK }
                let table = table.map { String(cString: $0) } ?? ""
                let column = column.map { String(cString: $0) } ?? ""
                if table == "historical_queue_sentinel" || (table == "chunks" && ["enrich_status", "enriched_at"].contains(column)) {
                    Unmanaged<ReadAudit>.fromOpaque(context).takeUnretainedValue().forbidden.append("\(table).\(column)")
                    return SQLITE_DENY
                }
                return SQLITE_OK
            }, context), SQLITE_OK)
        }
        defer { try? db.withSQLiteHandleForTesting { sqlite3_set_authorizer($0, nil, nil) } }
        for window in [60, 180, 1440] {
            _ = try db.dashboardStats(activityWindowMinutes: window, bucketCount: 12)
            _ = try db.pipelineWindowBuckets(activityWindowMinutes: window)
            _ = try db.dashboardSignalCoverageSnapshot()
        }
        XCTAssertEqual(audit.forbidden, [], "Actual SQLite authorizer observed a retired read")
        _ = try db.withSQLiteHandleForTesting { sqlite3_set_authorizer($0, nil, nil) }
        XCTAssertEqual(try snapshot(db), before, "Dashboard refresh must preserve historical data byte-for-byte")
    }

    private func isolateQueueEnvironment(_ key: String, _ path: URL) -> () -> Void {
        let previous = ProcessInfo.processInfo.environment[key]
        setenv(key, path.path, 1)
        return {
            if let previous { setenv(key, previous, 1) } else { unsetenv(key) }
        }
    }

    private func snapshot(_ db: BrainDatabase) throws -> String {
        try db.withSQLiteHandleForTesting { handle in
            var statement: OpaquePointer?
            let sql = "SELECT id || '|' || content || '|' || COALESCE(enrich_status,'NULL') || '|' || COALESCE(enriched_at,'NULL') || '|' || COALESCE(summary,'NULL') FROM chunks UNION ALL SELECT id || '|' || payload FROM historical_queue_sentinel ORDER BY 1"
            XCTAssertEqual(sqlite3_prepare_v2(handle, sql, -1, &statement, nil), SQLITE_OK)
            defer { sqlite3_finalize(statement) }
            var rows: [String] = []
            while sqlite3_step(statement) == SQLITE_ROW {
                if let value = sqlite3_column_text(statement, 0) { rows.append(String(cString: value)) }
            }
            return rows.joined(separator: "\n")
        }
    }
}
