import Foundation
import SQLite3

/// Bounded declared native-record continuity. This helper is not wired to ingestion or MCP.
enum NativeContinuityReader {
    static let maxHops = 16
    static let maxResponseBytes = 8_192
    private static let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    struct Evidence: Codable, Equatable {
        struct Hop: Codable, Equatable {
            let nativeUUID: String?
            let parentState: String
            let parentUUID: String?
            let logicalParentState: String
            let logicalParentUUID: String?
            let preservedSegmentOmitted: Bool
            let sourceFile: String
            let startOffset: Int64
            let endOffset: Int64
            let rawRecordHash: String
            enum CodingKeys: String, CodingKey {
                case parentState = "parent_state", parentUUID = "parent_uuid"
                case logicalParentState = "logical_parent_state", logicalParentUUID = "logical_parent_uuid"
                case preservedSegmentOmitted = "preserved_segment_omitted"
                case sourceFile = "source_file", startOffset = "start_offset", endOffset = "end_offset"
                case rawRecordHash = "raw_record_hash", nativeUUID = "native_uuid"
            }
        }
        let status: String
        let hops: [Hop]
        let logicalParentEvidence: Hop?
        let diagnostic: String?
        let originalStartOffset: Int64?
        let originalEndOffset: Int64?
        let originalBoundsStatus: String

        enum Keys: String, CodingKey {
            case status, hops, diagnostic
            case logicalParentEvidence = "logical_parent_evidence"
            case originalStartOffset = "original_start_offset"
            case originalEndOffset = "original_end_offset"
            case originalBoundsStatus = "original_bounds_status"
        }
        func encode(to encoder: Encoder) throws {
            var box = encoder.container(keyedBy: Keys.self)
            try box.encode(status, forKey: .status); try box.encode(hops, forKey: .hops)
            try box.encode(logicalParentEvidence, forKey: .logicalParentEvidence)
            try box.encode(diagnostic, forKey: .diagnostic)
            try box.encodeNil(forKey: .originalStartOffset); try box.encodeNil(forKey: .originalEndOffset)
            try box.encode(originalBoundsStatus, forKey: .originalBoundsStatus)
        }
    }

    private struct Row {
        let uuid, parentState: String; let parent: String?
        let logicalState: String; let logicalParent: String?
        let omitted: Bool; let source: String; let start, end: Int64; let hash: String
        var hop: Evidence.Hop { .init(nativeUUID: uuid.isEmpty ? nil : uuid, parentState: parentState, parentUUID: parent, logicalParentState: logicalState, logicalParentUUID: logicalParent, preservedSegmentOmitted: omitted, sourceFile: source, startOffset: start, endOffset: end, rawRecordHash: hash) }
    }
    private enum Bind { case text(String), int(Int64) }
    private enum Query<T> {
        case value(T), failure
        func map<U>(_ transform: (T) -> U) -> Query<U> {
            switch self { case .value(let value): return .value(transform(value)); case .failure: return .failure }
        }
    }

    static func read(db: OpaquePointer?, provider: String, sessionID: String, sourceFile: String, startOffset: Int64, endOffset: Int64, rawRecordHash: String) -> Evidence {
        guard provider == "claude" else { return result("unsupported_provider") }
        guard case .value(let hasTable) = tableExists(db) else { return result("database_error") }
        guard hasTable else { return result("legacy_schema_unavailable") }
        guard case .value(let physicalIndexed) = physicalIndexUsed(db, provider, sessionID, sourceFile, startOffset, endOffset, rawRecordHash) else { return result("database_error") }
        guard physicalIndexed else { return result("missing_physical_index", diagnostic: "idx_native_record_links_physical required") }
        guard case .value(let uuidIndexed) = uuidIndexUsed(db, provider, sessionID, sourceFile, "__index_probe__") else { return result("database_error") }
        guard uuidIndexed else { return result("missing_uuid_index", diagnostic: "idx_native_record_links_uuid required") }
        guard case .value(let initial) = physicalRows(db, provider, sessionID, sourceFile, startOffset, endOffset, rawRecordHash) else { return result("database_error") }
        guard initial.count == 1, let first = initial.first else { return result(initial.count > 1 ? "physical_record_ambiguous" : "physical_record_missing") }
        guard let uuid = first.hop.nativeUUID else { return capped(result("missing_native_uuid", hops: [first.hop])) }
        var hops = [first.hop], current = first, seen = Set([uuid])
        for _ in 0..<maxHops {
            if current.parentState == "null" { return capped(result("native_chain_root", hops: hops)) }
            let directStatus: String
            let parent: String?
            switch current.parentState { case "string": directStatus = "missing_parent"; parent = current.parent; case "missing": directStatus = "missing_parent"; parent = nil; default: directStatus = "invalid_parent"; parent = nil }
            if let parent, !parent.isEmpty {
                guard case .value(let matches) = uuidRows(db, provider, sessionID, sourceFile, parent) else { return result("database_error") }
                if matches.count == 1, let next = matches.first {
                    if !seen.insert(parent).inserted { return capped(result("cycle", hops: hops, diagnostic: "same-source UUID cycle")) }
                    hops.append(next.hop); current = next; continue
                }
                if matches.count > 1 { return capped(result("repeated_uuid_ambiguous", hops: hops)) }
                guard case .value(let cross) = crossSourceCandidate(db, provider, sessionID, sourceFile, parent) else { return result("database_error") }
                return logicalFallback(db, provider, sessionID, sourceFile, current, hops, directStatus, cross ? "cross_source_candidate_unknown" : nil)
            }
            return logicalFallback(db, provider, sessionID, sourceFile, current, hops, directStatus, nil)
        }
        return capped(result("hop_cap", hops: hops))
    }

    private static func logicalFallback(_ db: OpaquePointer?, _ p: String, _ s: String, _ f: String, _ current: Row, _ hops: [Evidence.Hop], _ directStatus: String, _ diagnostic: String?) -> Evidence {
        guard current.logicalState == "string", let logical = current.logicalParent, !logical.isEmpty else { return capped(result(directStatus, hops: hops, diagnostic: diagnostic)) }
        guard case .value(let matches) = uuidRows(db, p, s, f, logical) else { return result("database_error") }
        guard matches.count == 1, let edge = matches.first else { return capped(result(directStatus, hops: hops, diagnostic: diagnostic)) }
        return capped(result("compaction_gap", hops: hops, logical: edge.hop, diagnostic: "logical parent is declared, not contiguous"))
    }

    private static func result(_ status: String, hops: [Evidence.Hop] = [], logical: Evidence.Hop? = nil, diagnostic: String? = nil) -> Evidence { .init(status: status, hops: hops, logicalParentEvidence: logical, diagnostic: diagnostic, originalStartOffset: nil, originalEndOffset: nil, originalBoundsStatus: "unknown_not_inferred") }
    private static func capped(_ evidence: Evidence) -> Evidence {
        guard let data = try? JSONEncoder().encode(evidence), data.count <= maxResponseBytes else { return result("response_byte_cap", diagnostic: "declared continuity evidence omitted at byte cap") }
        return evidence
    }

    private static func tableExists(_ db: OpaquePointer?) -> Query<Bool> { scalar(db, "SELECT 1 FROM sqlite_master WHERE type='table' AND name='native_record_links' LIMIT 1", []).map { $0 != nil } }
    private static func physicalIndexUsed(_ db: OpaquePointer?, _ p: String, _ s: String, _ f: String, _ a: Int64, _ b: Int64, _ h: String) -> Query<Bool> { planUses(db, "EXPLAIN QUERY PLAN SELECT native_uuid FROM native_record_links WHERE provider=? AND session_id=? AND source_file=? AND record_start_offset=? AND record_end_offset=? AND raw_record_hash=? LIMIT 2", [.text(p), .text(s), .text(f), .int(a), .int(b), .text(h)], "idx_native_record_links_physical") }
    private static func uuidIndexUsed(_ db: OpaquePointer?, _ p: String, _ s: String, _ f: String, _ u: String) -> Query<Bool> { planUses(db, "EXPLAIN QUERY PLAN SELECT native_uuid FROM native_record_links WHERE provider=? AND session_id=? AND source_file=? AND native_uuid=? LIMIT 2", [.text(p), .text(s), .text(f), .text(u)], "idx_native_record_links_uuid") }
    private static func planUses(_ db: OpaquePointer?, _ sql: String, _ binds: [Bind], _ index: String) -> Query<Bool> { strings(db, sql, binds).map { $0.contains { $0.contains(index) } } }
    private static func physicalRows(_ db: OpaquePointer?, _ p: String, _ s: String, _ f: String, _ a: Int64, _ b: Int64, _ h: String) -> Query<[Row]> { rows(db, "SELECT native_uuid,parent_uuid_state,parent_uuid_value,logical_parent_uuid_state,logical_parent_uuid_value,preserved_segment_omitted,source_file,record_start_offset,record_end_offset,raw_record_hash FROM native_record_links WHERE provider=? AND session_id=? AND source_file=? AND record_start_offset=? AND record_end_offset=? AND raw_record_hash=? ORDER BY record_key LIMIT 2", [.text(p), .text(s), .text(f), .int(a), .int(b), .text(h)]) }
    private static func uuidRows(_ db: OpaquePointer?, _ p: String, _ s: String, _ f: String, _ u: String) -> Query<[Row]> { rows(db, "SELECT native_uuid,parent_uuid_state,parent_uuid_value,logical_parent_uuid_state,logical_parent_uuid_value,preserved_segment_omitted,source_file,record_start_offset,record_end_offset,raw_record_hash FROM native_record_links WHERE provider=? AND session_id=? AND source_file=? AND native_uuid=? ORDER BY record_start_offset,record_key LIMIT 2", [.text(p), .text(s), .text(f), .text(u)]) }
    private static func crossSourceCandidate(_ db: OpaquePointer?, _ p: String, _ s: String, _ f: String, _ u: String) -> Query<Bool> { scalar(db, "SELECT 1 FROM native_record_links WHERE provider=? AND session_id=? AND source_file<>? AND native_uuid=? LIMIT 1", [.text(p), .text(s), .text(f), .text(u)]).map { $0 != nil } }

    private static func scalar(_ db: OpaquePointer?, _ sql: String, _ binds: [Bind]) -> Query<Int?> { guard let stmt = statement(db, sql, binds) else { return .failure }; defer { sqlite3_finalize(stmt) }; let code = sqlite3_step(stmt); return code == SQLITE_ROW ? .value(Int(sqlite3_column_int(stmt, 0))) : code == SQLITE_DONE ? .value(nil) : .failure }
    private static func strings(_ db: OpaquePointer?, _ sql: String, _ binds: [Bind]) -> Query<[String]> { guard let stmt = statement(db, sql, binds) else { return .failure }; defer { sqlite3_finalize(stmt) }; var out: [String] = []; while true { let code = sqlite3_step(stmt); if code == SQLITE_DONE { return .value(out) }; guard code == SQLITE_ROW else { return .failure }; if let text = sqlite3_column_text(stmt, 3) { out.append(String(cString: text)) } } }
    private static func rows(_ db: OpaquePointer?, _ sql: String, _ binds: [Bind]) -> Query<[Row]> { guard let stmt = statement(db, sql, binds) else { return .failure }; defer { sqlite3_finalize(stmt) }; var out: [Row] = []; while true { let code = sqlite3_step(stmt); if code == SQLITE_DONE { return .value(out) }; guard code == SQLITE_ROW else { return .failure }; out.append(.init(uuid: text(stmt, 0) ?? "", parentState: text(stmt, 1) ?? "invalid", parent: text(stmt, 2), logicalState: text(stmt, 3) ?? "missing", logicalParent: text(stmt, 4), omitted: sqlite3_column_int(stmt, 5) != 0, source: text(stmt, 6) ?? "", start: sqlite3_column_int64(stmt, 7), end: sqlite3_column_int64(stmt, 8), hash: text(stmt, 9) ?? "")) } }
    private static func statement(_ db: OpaquePointer?, _ sql: String, _ binds: [Bind]) -> OpaquePointer? { var stmt: OpaquePointer?; guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK, let stmt else { return nil }; for (i, bind) in binds.enumerated() { switch bind { case .text(let value): _ = value.withCString { sqlite3_bind_text(stmt, Int32(i + 1), $0, -1, transient) }; case .int(let value): sqlite3_bind_int64(stmt, Int32(i + 1), value) } }; return stmt }
    private static func text(_ stmt: OpaquePointer, _ column: Int32) -> String? { sqlite3_column_text(stmt, column).map { String(cString: $0) } }
}
