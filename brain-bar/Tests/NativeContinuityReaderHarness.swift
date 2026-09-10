import Foundation
import SQLite3

private let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
private var db: OpaquePointer?

func check(_ condition: @autoclosure () -> Bool, _ message: String) { if !condition() { fatalError(message) } }
func exec(_ sql: String) { check(sqlite3_exec(db, sql, nil, nil, nil) == SQLITE_OK, sql) }
func insert(_ uuid: String?, _ parentState: String, _ parent: String?, logicalState: String = "missing", logical: String? = nil, source: String = "/a", start: Int64, hash: String) {
    var st: OpaquePointer?; let sql = "INSERT INTO native_record_links VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    check(sqlite3_prepare_v2(db, sql, -1, &st, nil) == SQLITE_OK, "prepare")
    defer { sqlite3_finalize(st) }
    let values = ["k\(source)\(start)\(hash)", "claude", "s", source, String(start), String(start + 1), hash, uuid ?? "", parentState, parent ?? "", logicalState, logical ?? "", "{}", "", "", "0"]
    for (i, value) in values.enumerated() { _ = value.withCString { sqlite3_bind_text(st, Int32(i + 1), $0, -1, transient) } }
    check(sqlite3_step(st) == SQLITE_DONE, "insert")
}

@main
struct NativeContinuityReaderHarness {
    static func main() {
        check(sqlite3_open(":memory:", &db) == SQLITE_OK, "open")
        defer { sqlite3_close(db) }
        exec("CREATE TABLE native_record_links(record_key TEXT PRIMARY KEY,provider TEXT,session_id TEXT,source_file TEXT,record_start_offset INTEGER,record_end_offset INTEGER,raw_record_hash TEXT,native_uuid TEXT,parent_uuid_state TEXT,parent_uuid_value TEXT,logical_parent_uuid_state TEXT,logical_parent_uuid_value TEXT,declared_headers_json TEXT,preserved_segment_json TEXT,preserved_segment_sha256 TEXT,preserved_segment_omitted INTEGER)")
        exec("CREATE INDEX idx_native_record_links_physical ON native_record_links(provider,session_id,source_file,record_start_offset,record_end_offset,raw_record_hash)")
        exec("CREATE INDEX idx_native_record_links_uuid ON native_record_links(provider,session_id,native_uuid,source_file,record_start_offset,record_key)")
        insert("child", "string", "root", start: 10, hash: "a")
        insert("root", "null", nil, start: 5, hash: "b")
        exec("ANALYZE native_record_links")
        let chain = NativeContinuityReader.read(db: db, provider: "claude", sessionID: "s", sourceFile: "/a", startOffset: 10, endOffset: 11, rawRecordHash: "a")
        check(chain.status == "native_chain_root" && chain.hops.count == 2 && chain.originalStartOffset == nil, "positive chain")
        check(chain.hops[0].sourceRecordKey == "k/a10a" && chain.hops[1].sourceRecordKey == "k/a5b", "catalog keys exposed")
        let encoded = try! JSONSerialization.jsonObject(with: JSONEncoder().encode(chain)) as! [String: Any]
        check(encoded["original_start_offset"] is NSNull && encoded["original_end_offset"] is NSNull && encoded["original_bounds_status"] as? String == "unknown_not_inferred", "explicit unknown bounds")
        insert("cross-child", "string", "only-other", source: "/a", start: 20, hash: "c")
        insert("only-other", "null", nil, source: "/b", start: 1, hash: "d")
        let cross = NativeContinuityReader.read(db: db, provider: "claude", sessionID: "s", sourceFile: "/a", startOffset: 20, endOffset: 21, rawRecordHash: "c")
        check(cross.status == "missing_parent" && cross.diagnostic == "cross_source_candidate_unknown", "cross source")
        insert("cycle-a", "string", "cycle-b", start: 30, hash: "e")
        insert("cycle-b", "string", "cycle-a", start: 31, hash: "f")
        check(NativeContinuityReader.read(db: db, provider: "claude", sessionID: "s", sourceFile: "/a", startOffset: 30, endOffset: 31, rawRecordHash: "e").status == "cycle", "cycle")
        insert("dup-child", "string", "dup", start: 40, hash: "g")
        insert("dup", "null", nil, start: 41, hash: "h")
        insert("dup", "null", nil, start: 42, hash: "i")
        check(NativeContinuityReader.read(db: db, provider: "claude", sessionID: "s", sourceFile: "/a", startOffset: 40, endOffset: 41, rawRecordHash: "g").status == "repeated_uuid_ambiguous", "duplicate")
        insert("missing", "missing", nil, start: 50, hash: "j")
        check(NativeContinuityReader.read(db: db, provider: "claude", sessionID: "s", sourceFile: "/a", startOffset: 50, endOffset: 51, rawRecordHash: "j").status == "missing_parent", "missing")
        insert("logical-child", "missing", nil, logicalState: "string", logical: "logical-root", start: 53, hash: "l")
        insert("logical-root", "null", nil, start: 54, hash: "m")
        let gap = NativeContinuityReader.read(db: db, provider: "claude", sessionID: "s", sourceFile: "/a", startOffset: 53, endOffset: 54, rawRecordHash: "l")
        check(gap.status == "compaction_gap" && gap.logicalParentEvidence?.nativeUUID == "logical-root", "logical gap evidence")
        insert("null-root", "null", nil, start: 55, hash: "n")
        check(NativeContinuityReader.read(db: db, provider: "claude", sessionID: "s", sourceFile: "/a", startOffset: 55, endOffset: 56, rawRecordHash: "n").status == "native_chain_root", "explicit null root")
        let longSource = "/" + String(repeating: "x", count: NativeContinuityReader.maxResponseBytes)
        insert("cap", "null", nil, source: longSource, start: 60, hash: "o")
        check(NativeContinuityReader.read(db: db, provider: "claude", sessionID: "s", sourceFile: longSource, startOffset: 60, endOffset: 61, rawRecordHash: "o").status == "response_byte_cap", "response cap")
        exec("DROP INDEX idx_native_record_links_uuid")
        check(NativeContinuityReader.read(db: db, provider: "claude", sessionID: "s", sourceFile: "/a", startOffset: 10, endOffset: 11, rawRecordHash: "a").status == "missing_uuid_index", "uuid index guard")
        exec("CREATE INDEX idx_native_record_links_uuid ON native_record_links(provider,session_id,native_uuid,source_file,record_start_offset,record_key)")
        exec("DROP INDEX idx_native_record_links_physical")
        check(NativeContinuityReader.read(db: db, provider: "claude", sessionID: "s", sourceFile: "/a", startOffset: 10, endOffset: 11, rawRecordHash: "a").status == "missing_physical_index", "index guard")
        exec("CREATE INDEX idx_native_record_links_physical ON native_record_links(provider,session_id,source_file,record_start_offset,raw_record_hash,record_end_offset)")
        check(NativeContinuityReader.read(db: db, provider: "claude", sessionID: "s", sourceFile: "/a", startOffset: 10, endOffset: 11, rawRecordHash: "a").status == "missing_physical_index", "wrong index definition guard")
        check(sqlite3_close(db) == SQLITE_OK, "close")
        db = nil
        check(sqlite3_open(":memory:", &db) == SQLITE_OK, "reopen")
        check(NativeContinuityReader.read(db: db, provider: "claude", sessionID: "s", sourceFile: "/a", startOffset: 1, endOffset: 2, rawRecordHash: "z").status == "legacy_schema_unavailable", "legacy schema")
        exec("CREATE TABLE native_record_links(record_key TEXT,provider TEXT,session_id TEXT,source_file TEXT,record_start_offset INTEGER,record_end_offset INTEGER,raw_record_hash TEXT,native_uuid TEXT)")
        exec("CREATE INDEX idx_native_record_links_physical ON native_record_links(provider,session_id,source_file,record_start_offset,record_end_offset,raw_record_hash)")
        exec("CREATE INDEX idx_native_record_links_uuid ON native_record_links(provider,session_id,native_uuid,source_file,record_start_offset,record_key)")
        check(NativeContinuityReader.read(db: db, provider: "claude", sessionID: "s", sourceFile: "/a", startOffset: 1, endOffset: 2, rawRecordHash: "z").status == "database_error", "sql error is explicit")
        print("native continuity harness: PASS")
    }
}
