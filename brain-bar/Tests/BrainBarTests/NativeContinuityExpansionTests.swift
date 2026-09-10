import SQLite3
import XCTest
@testable import BrainBar

final class NativeContinuityExpansionTests: XCTestCase {
    private var path = ""
    private var db: BrainDatabase!

    override func setUp() {
        super.setUp(); path = NSTemporaryDirectory() + "continuity-\(UUID().uuidString).db"; db = BrainDatabase(path: path)
    }
    override func tearDown() {
        db.close(); for suffix in ["", "-wal", "-shm"] { try? FileManager.default.removeItem(atPath: path + suffix) }; super.tearDown()
    }
    private func sql(_ text: String) throws {
        var handle: OpaquePointer?; XCTAssertEqual(sqlite3_open(path, &handle), SQLITE_OK); defer { sqlite3_close(handle) }
        XCTAssertEqual(sqlite3_exec(handle, text, nil, nil, nil), SQLITE_OK)
    }
    private func schema() throws {
        try sql("""
        CREATE TABLE chunk_source_records(record_key TEXT PRIMARY KEY,canonical_chunk_id TEXT NOT NULL,source_file TEXT NOT NULL,record_start_offset INTEGER NOT NULL,record_end_offset INTEGER NOT NULL,raw_record_hash TEXT NOT NULL,session_id TEXT,observed_at TEXT NOT NULL,record_context_json TEXT NOT NULL DEFAULT '[]');
        CREATE TABLE native_record_links(record_key TEXT PRIMARY KEY,provider TEXT NOT NULL,session_id TEXT NOT NULL,source_file TEXT NOT NULL,record_start_offset INTEGER NOT NULL,record_end_offset INTEGER NOT NULL,raw_record_hash TEXT NOT NULL,native_uuid TEXT,parent_uuid_state TEXT NOT NULL,parent_uuid_value TEXT,logical_parent_uuid_state TEXT NOT NULL,logical_parent_uuid_value TEXT,declared_headers_json TEXT NOT NULL,preserved_segment_json TEXT,preserved_segment_sha256 TEXT,preserved_segment_omitted INTEGER NOT NULL DEFAULT 0);
        CREATE INDEX idx_native_record_links_physical ON native_record_links(provider,session_id,source_file,record_start_offset,record_end_offset,raw_record_hash);
        CREATE INDEX idx_native_record_links_uuid ON native_record_links(provider,session_id,native_uuid,source_file,record_start_offset,record_key);
        """)
    }
    private func record(_ key: String, _ chunk: String, _ file: String, _ start: Int, _ hash: String, _ session: String? = "s") throws {
        let sessionSQL = session.map { "'\($0)'" } ?? "NULL"
        try sql("INSERT INTO chunk_source_records VALUES ('\(key)','\(chunk)','\(file)',\(start),\(start + 1),'\(hash)',\(sessionSQL),'2026-09-10T00:00:00Z','[]')")
    }
    private func native(_ key: String, _ file: String, _ start: Int, _ hash: String, _ uuid: String, _ state: String, _ parent: String? = nil, _ logical: String? = nil) throws {
        let parentSQL = parent.map { "'\($0)'" } ?? "NULL", logicalSQL = logical.map { "'\($0)'" } ?? "NULL"
        try sql("INSERT INTO native_record_links VALUES ('\(key)','claude','s','\(file)',\(start),\(start + 1),'\(hash)','\(uuid)','\(state)',\(parentSQL),'\(logical == nil ? "missing" : "string")',\(logicalSQL),'{}',NULL,NULL,0)")
    }
    private func occurrence(_ chunk: String) throws -> [String: Any] {
        let page = try db.listSourceOccurrences(requestedChunkID: chunk, maxBytes: 8_000)
        return try XCTUnwrap((page["source_occurrences"] as? [[String: Any]])?.first)
    }

    func testExpandServesSameSourceNativeParentWithoutParentChunk() throws {
        try db.insertChunk(id: "short-id", content: "target", sessionId: "s", project: "t", contentType: "assistant_text", importance: 1)
        try schema(); try record("child-record", "short-id", "/a", 10, "h-child")
        try native("child", "/a", 10, "h-child", "child", "string", "root")
        try native("control-only-parent", "/a", 1, "h-root", "root", "null")
        let continuity = try XCTUnwrap((try occurrence("short-id"))["structural_continuity"] as? [String: Any])
        XCTAssertEqual(continuity["status"] as? String, "native_chain_root")
        XCTAssertEqual((continuity["hops"] as? [[String: Any]])?.count, 2)
        XCTAssertTrue(continuity["original_start_offset"] is NSNull)
        XCTAssertNil(continuity["provider"])
    }

    func testExpandMarksUnavailableAndDisputedContinuityWithoutFabricatingProvider() throws {
        try db.insertChunk(id: "short-two", content: "target", sessionId: "s", project: "t", contentType: "assistant_text", importance: 1)
        try schema(); try record("no-catalog", "short-two", "/a", 10, "h-none")
        XCTAssertEqual(((try occurrence("short-two"))["structural_continuity"] as? [String: Any])?["status"] as? String, "unavailable_catalog_record")
        try record("cross", "short-two", "/a", 20, "h-cross")
        try native("cross-child", "/a", 20, "h-cross", "cross-child", "string", "other")
        try native("other-source", "/b", 1, "h-other", "other", "null")
        let rows = try db.listSourceOccurrences(requestedChunkID: "short-two", maxBytes: 8_000)["source_occurrences"] as? [[String: Any]] ?? []
        let cross = try XCTUnwrap(rows.first { ($0["raw_record_hash"] as? String) == "h-cross" }?["structural_continuity"] as? [String: Any])
        XCTAssertEqual(cross["status"] as? String, "missing_parent")
        XCTAssertEqual(cross["diagnostic"] as? String, "cross_source_candidate_unknown")
    }
}

extension NativeContinuityExpansionTests {
    func testExpandPreservesLocatorWhenContinuityIsUnavailableLogicalOrOverBudget() throws {
        try db.insertChunk(id: "short-three", content: "target", sessionId: "s", project: "t", contentType: "assistant_text", importance: 1)
        try schema()
        try record("missing-session", "short-three", "/a", 1, "h-session", nil)
        var rows = try db.listSourceOccurrences(requestedChunkID: "short-three", maxBytes: 8_000)["source_occurrences"] as? [[String: Any]] ?? []
        let noSession = try XCTUnwrap(rows.first { ($0["raw_record_hash"] as? String) == "h-session" })
        XCTAssertEqual((noSession["structural_continuity"] as? [String: Any])?["status"] as? String, "unavailable_missing_session")

        try record("logical", "short-three", "/a", 10, "h-logical")
        try native("logical-child", "/a", 10, "h-logical", "child", "missing", nil, "root")
        try native("logical-control", "/a", 2, "h-root", "root", "null")
        rows = try db.listSourceOccurrences(requestedChunkID: "short-three", maxBytes: 8_000)["source_occurrences"] as? [[String: Any]] ?? []
        let logical = try XCTUnwrap(rows.first { ($0["raw_record_hash"] as? String) == "h-logical" }?["structural_continuity"] as? [String: Any])
        XCTAssertEqual(logical["status"] as? String, "compaction_gap")
        XCTAssertEqual((logical["logical_parent_evidence"] as? [String: Any])?["native_uuid"] as? String, "root")

        let longFile = "/" + String(repeating: "x", count: 300)
        try record("cap", "short-three", longFile, 20, "h-cap")
        try native("cap", longFile, 20, "h-cap", "cap", "null")
        rows = try db.listSourceOccurrences(requestedChunkID: "short-three", maxBytes: 1_024)["source_occurrences"] as? [[String: Any]] ?? []
        let capped = try XCTUnwrap(rows.first { ($0["raw_record_hash"] as? String) == "h-cap" })
        XCTAssertEqual((capped["structural_continuity"] as? [String: Any])?["status"] as? String, "omitted_response_budget")
        XCTAssertEqual(capped["record_start_offset"] as? Int64, 20)
    }
}
