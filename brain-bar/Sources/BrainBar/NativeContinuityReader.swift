import Foundation
import SQLite3

/// Bounded declared native-record continuity. This helper is not wired to MCP or ingestion.
enum NativeContinuityReader {
    static let maxHops = 16
    static let maxRows = 34
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
        }

        let status: String
        let hops: [Hop]
        let diagnostic: String?
        let originalStartOffset: Int64?
        let originalEndOffset: Int64?
    }

    private struct Row {
        let uuid: String?
        let parentState: String
        let parent: String?
        let logicalState: String
        let logicalParent: String?
        let omitted: Bool
        let source: String
        let start: Int64
        let end: Int64
        let hash: String
        var hop: Evidence.Hop {
            .init(nativeUUID: uuid, parentState: parentState, parentUUID: parent,
                  logicalParentState: logicalState, logicalParentUUID: logicalParent,
                  preservedSegmentOmitted: omitted, sourceFile: source,
                  startOffset: start, endOffset: end, rawRecordHash: hash)
        }
    }

    static func read(
        db: OpaquePointer?, provider: String, sessionID: String, sourceFile: String,
        startOffset: Int64, endOffset: Int64, rawRecordHash: String
    ) -> Evidence {
        guard provider == "claude" else { return result("unsupported_provider") }
        guard tableExists(db) else { return result("legacy_schema_unavailable") }
        guard physicalIndexUsed(db, provider, sessionID, sourceFile, startOffset, endOffset, rawRecordHash) else {
            return result("missing_physical_index", diagnostic: "idx_native_record_links_physical required")
        }
        let initial = physicalRows(db, provider, sessionID, sourceFile, startOffset, endOffset, rawRecordHash)
        guard initial.count == 1, let first = initial.first else {
            return result(initial.count > 1 ? "physical_record_ambiguous" : "physical_record_missing")
        }
        guard let uuid = first.uuid, !uuid.isEmpty else { return result("missing_native_uuid", hops: [first.hop]) }
        var rows = 1
        var hops = [first.hop]
        var current = first
        var seen = Set([uuid])
        for _ in 0..<maxHops {
            switch current.parentState {
            case "null": return capped(result("native_chain_root", hops: hops))
            case "missing": return capped(result("missing_parent", hops: hops))
            case "invalid": return capped(result("invalid_parent", hops: hops))
            case "string": break
            default: return capped(result("invalid_parent", hops: hops))
            }
            guard let parent = current.parent, !parent.isEmpty else { return capped(result("invalid_parent", hops: hops)) }
            let matches = uuidRows(db, provider, sessionID, sourceFile, parent)
            rows += matches.count
            if matches.count == 1, let next = matches.first {
                if !seen.insert(parent).inserted { return capped(result("cycle", hops: hops, diagnostic: "same-source UUID cycle")) }
                hops.append(next.hop)
                current = next
                continue
            }
            if matches.count > 1 { return capped(result("repeated_uuid_ambiguous", hops: hops)) }
            let cross = crossSourceCandidate(db, provider, sessionID, sourceFile, parent)
            let gap = current.logicalState == "string" && uniqueLogicalParent(db, provider, sessionID, sourceFile, current.logicalParent)
            return capped(result(gap ? "compaction_gap" : "missing_parent", hops: hops,
                                 diagnostic: cross ? "cross_source_candidate_unknown" : nil))
        }
        _ = rows // fixed query limits make row use bounded; retained for the contract's row budget.
        return capped(result("hop_cap", hops: hops))
    }

    private static func result(_ status: String, hops: [Evidence.Hop] = [], diagnostic: String? = nil) -> Evidence {
        .init(status: status, hops: hops, diagnostic: diagnostic, originalStartOffset: nil, originalEndOffset: nil)
    }

    private static func capped(_ evidence: Evidence) -> Evidence {
        guard evidence.hops.count <= maxRows,
              let data = try? JSONEncoder().encode(evidence), data.count <= maxResponseBytes else {
            return result("response_byte_cap", diagnostic: "declared continuity evidence omitted at byte cap")
        }
        return evidence
    }

    private static func tableExists(_ db: OpaquePointer?) -> Bool {
        scalar(db, "SELECT 1 FROM sqlite_master WHERE type='table' AND name='native_record_links' LIMIT 1", []) != nil
    }

    private static func physicalIndexUsed(_ db: OpaquePointer?, _ p: String, _ s: String, _ f: String, _ a: Int64, _ b: Int64, _ h: String) -> Bool {
        let sql = "EXPLAIN QUERY PLAN SELECT native_uuid FROM native_record_links WHERE provider=? AND session_id=? AND source_file=? AND record_start_offset=? AND record_end_offset=? AND raw_record_hash=? LIMIT 2"
        return strings(db, sql, [.text(p), .text(s), .text(f), .int(a), .int(b), .text(h)]).contains { $0.contains("idx_native_record_links_physical") }
    }

    private static func physicalRows(_ db: OpaquePointer?, _ p: String, _ s: String, _ f: String, _ a: Int64, _ b: Int64, _ h: String) -> [Row] {
        rows(db, "SELECT native_uuid,parent_uuid_state,parent_uuid_value,logical_parent_uuid_state,logical_parent_uuid_value,preserved_segment_omitted,source_file,record_start_offset,record_end_offset,raw_record_hash FROM native_record_links WHERE provider=? AND session_id=? AND source_file=? AND record_start_offset=? AND record_end_offset=? AND raw_record_hash=? ORDER BY record_key LIMIT 2", [.text(p), .text(s), .text(f), .int(a), .int(b), .text(h)])
    }

    private static func uuidRows(_ db: OpaquePointer?, _ p: String, _ s: String, _ f: String, _ uuid: String) -> [Row] {
        rows(db, "SELECT native_uuid,parent_uuid_state,parent_uuid_value,logical_parent_uuid_state,logical_parent_uuid_value,preserved_segment_omitted,source_file,record_start_offset,record_end_offset,raw_record_hash FROM native_record_links WHERE provider=? AND session_id=? AND source_file=? AND native_uuid=? ORDER BY record_start_offset,record_key LIMIT 2", [.text(p), .text(s), .text(f), .text(uuid)])
    }

    private static func crossSourceCandidate(_ db: OpaquePointer?, _ p: String, _ s: String, _ f: String, _ uuid: String) -> Bool {
        scalar(db, "SELECT 1 FROM native_record_links WHERE provider=? AND session_id=? AND source_file<>? AND native_uuid=? LIMIT 1", [.text(p), .text(s), .text(f), .text(uuid)]) != nil
    }

    private static func uniqueLogicalParent(_ db: OpaquePointer?, _ p: String, _ s: String, _ f: String, _ uuid: String?) -> Bool {
        guard let uuid else { return false }
        return uuidRows(db, p, s, f, uuid).count == 1
    }

    private enum Bind { case text(String), int(Int64) }
    private static func scalar(_ db: OpaquePointer?, _ sql: String, _ binds: [Bind]) -> Int? {
        guard let stmt = statement(db, sql, binds) else { return nil }
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        return Int(sqlite3_column_int(stmt, 0))
    }
    private static func strings(_ db: OpaquePointer?, _ sql: String, _ binds: [Bind]) -> [String] {
        guard let stmt = statement(db, sql, binds) else { return [] }; defer { sqlite3_finalize(stmt) }
        var output: [String] = []; while sqlite3_step(stmt) == SQLITE_ROW { if let p = sqlite3_column_text(stmt, 3) { output.append(String(cString: p)) } }; return output
    }
    private static func rows(_ db: OpaquePointer?, _ sql: String, _ binds: [Bind]) -> [Row] {
        guard let stmt = statement(db, sql, binds) else { return [] }; defer { sqlite3_finalize(stmt) }
        var output: [Row] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            output.append(.init(uuid: text(stmt, 0), parentState: text(stmt, 1) ?? "invalid", parent: text(stmt, 2), logicalState: text(stmt, 3) ?? "missing", logicalParent: text(stmt, 4), omitted: sqlite3_column_int(stmt, 5) != 0, source: text(stmt, 6) ?? "", start: sqlite3_column_int64(stmt, 7), end: sqlite3_column_int64(stmt, 8), hash: text(stmt, 9) ?? ""))
        }; return output
    }
    private static func statement(_ db: OpaquePointer?, _ sql: String, _ binds: [Bind]) -> OpaquePointer? {
        var stmt: OpaquePointer?; guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK, let stmt else { return nil }
        for (i, bind) in binds.enumerated() { switch bind { case .text(let value): _ = value.withCString { sqlite3_bind_text(stmt, Int32(i + 1), $0, -1, transient) }; case .int(let value): sqlite3_bind_int64(stmt, Int32(i + 1), value) } }
        return stmt
    }
    private static func text(_ stmt: OpaquePointer, _ column: Int32) -> String? { sqlite3_column_text(stmt, column).map { String(cString: $0) } }
}
