import CoreFoundation
import Foundation
import SQLite3

/// Bounded, read-only native-record continuity and declared target evidence.
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
            let sourceRecordKey: String
            enum CodingKeys: String, CodingKey {
                case parentState = "parent_state", parentUUID = "parent_uuid"
                case logicalParentState = "logical_parent_state", logicalParentUUID = "logical_parent_uuid"
                case preservedSegmentOmitted = "preserved_segment_omitted"
                case sourceFile = "source_file", startOffset = "start_offset", endOffset = "end_offset"
                case rawRecordHash = "raw_record_hash", nativeUUID = "native_uuid"
                case sourceRecordKey = "source_record_key"
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
        let key, uuid, parentState: String; let parent: String?
        let logicalState: String; let logicalParent: String?
        let omitted: Bool; let source: String; let start, end: Int64; let hash: String
        var hop: Evidence.Hop { .init(nativeUUID: uuid.isEmpty ? nil : uuid, parentState: parentState, parentUUID: parent, logicalParentState: logicalState, logicalParentUUID: logicalParent, preservedSegmentOmitted: omitted, sourceFile: source, startOffset: start, endOffset: end, rawRecordHash: hash, sourceRecordKey: key) }
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
    private static func physicalIndexUsed(_ db: OpaquePointer?, _ p: String, _ s: String, _ f: String, _ a: Int64, _ b: Int64, _ h: String) -> Query<Bool> { indexColumns(db, "idx_native_record_links_physical").map { $0 == ["provider", "session_id", "source_file", "record_start_offset", "record_end_offset", "raw_record_hash"] } }
    private static func uuidIndexUsed(_ db: OpaquePointer?, _ p: String, _ s: String, _ f: String, _ u: String) -> Query<Bool> { indexColumns(db, "idx_native_record_links_uuid").map { $0 == ["provider", "session_id", "native_uuid", "source_file", "record_start_offset", "record_key"] } }
    private static func indexColumns(_ db: OpaquePointer?, _ name: String) -> Query<[String]> { guard let stmt = statement(db, "PRAGMA index_info('\(name)')", []) else { return .failure }; defer { sqlite3_finalize(stmt) }; var out: [String] = []; while true { let code = sqlite3_step(stmt); if code == SQLITE_DONE { return .value(out) }; guard code == SQLITE_ROW, let value = text(stmt, 2) else { return .failure }; out.append(value) } }
    private static func physicalRows(_ db: OpaquePointer?, _ p: String, _ s: String, _ f: String, _ a: Int64, _ b: Int64, _ h: String) -> Query<[Row]> { rows(db, "SELECT native_uuid,parent_uuid_state,parent_uuid_value,logical_parent_uuid_state,logical_parent_uuid_value,preserved_segment_omitted,source_file,record_start_offset,record_end_offset,raw_record_hash,record_key FROM native_record_links INDEXED BY idx_native_record_links_physical WHERE provider=? AND session_id=? AND source_file=? AND record_start_offset=? AND record_end_offset=? AND raw_record_hash=? ORDER BY record_key LIMIT 2", [.text(p), .text(s), .text(f), .int(a), .int(b), .text(h)]) }
    private static func uuidRows(_ db: OpaquePointer?, _ p: String, _ s: String, _ f: String, _ u: String) -> Query<[Row]> { rows(db, "SELECT native_uuid,parent_uuid_state,parent_uuid_value,logical_parent_uuid_state,logical_parent_uuid_value,preserved_segment_omitted,source_file,record_start_offset,record_end_offset,raw_record_hash,record_key FROM native_record_links INDEXED BY idx_native_record_links_uuid WHERE provider=? AND session_id=? AND source_file=? AND native_uuid=? ORDER BY record_start_offset,record_key LIMIT 2", [.text(p), .text(s), .text(f), .text(u)]) }
    private static func crossSourceCandidate(_ db: OpaquePointer?, _ p: String, _ s: String, _ f: String, _ u: String) -> Query<Bool> { scalar(db, "SELECT 1 FROM native_record_links INDEXED BY idx_native_record_links_uuid WHERE provider=? AND session_id=? AND source_file<>? AND native_uuid=? LIMIT 1", [.text(p), .text(s), .text(f), .text(u)]).map { $0 != nil } }

    private static func scalar(_ db: OpaquePointer?, _ sql: String, _ binds: [Bind]) -> Query<Int?> { guard let stmt = statement(db, sql, binds) else { return .failure }; defer { sqlite3_finalize(stmt) }; let code = sqlite3_step(stmt); return code == SQLITE_ROW ? .value(Int(sqlite3_column_int(stmt, 0))) : code == SQLITE_DONE ? .value(nil) : .failure }
    private static func rows(_ db: OpaquePointer?, _ sql: String, _ binds: [Bind]) -> Query<[Row]> { guard let stmt = statement(db, sql, binds) else { return .failure }; defer { sqlite3_finalize(stmt) }; var out: [Row] = []; while true { let code = sqlite3_step(stmt); if code == SQLITE_DONE { return .value(out) }; guard code == SQLITE_ROW else { return .failure }; out.append(.init(key: text(stmt, 10) ?? "", uuid: text(stmt, 0) ?? "", parentState: text(stmt, 1) ?? "invalid", parent: text(stmt, 2), logicalState: text(stmt, 3) ?? "missing", logicalParent: text(stmt, 4), omitted: sqlite3_column_int(stmt, 5) != 0, source: text(stmt, 6) ?? "", start: sqlite3_column_int64(stmt, 7), end: sqlite3_column_int64(stmt, 8), hash: text(stmt, 9) ?? "")) } }
    private static func statement(_ db: OpaquePointer?, _ sql: String, _ binds: [Bind]) -> OpaquePointer? { var stmt: OpaquePointer?; guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK, let stmt else { return nil }; for (i, bind) in binds.enumerated() { switch bind { case .text(let value): _ = value.withCString { sqlite3_bind_text(stmt, Int32(i + 1), $0, -1, transient) }; case .int(let value): sqlite3_bind_int64(stmt, Int32(i + 1), value) } }; return stmt }
    private static func text(_ stmt: OpaquePointer, _ column: Int32) -> String? { sqlite3_column_text(stmt, column).map { String(cString: $0) } }
}

extension NativeContinuityReader {
    private static let targetReportSchemaVersion = 1
    private static let targetReportScope = "supported_tool_calls_v1"
    private static let targetReportBoundary =
        "Declared target handles are unverified; no delivery, recipient identity, relay, or authorship is inferred."
    private static let targetColumns: Set<String> = [
        "record_key", "provider", "session_id", "source_file",
        "record_start_offset", "record_end_offset", "raw_record_hash",
        "target_extraction_version", "declared_targets_json", "target_diagnostics_json",
    ]

    static func readTargetRecord(
        db: OpaquePointer?,
        sourceRecordKey: String,
        maxBytes: Int
    ) -> [String: Any] {
        let budget = min(max(maxBytes, 1_024), 128_000)
        let base: [String: Any] = [
            "schema_version": targetReportSchemaVersion,
            "mode": "source_record",
            "source_record_key": sourceRecordKey,
            "serialized_max_bytes": budget,
            "evidence_boundary": targetReportBoundary,
        ]
        guard isSHA256(sourceRecordKey) else {
            return base.merging(["status": "malformed_source_record_key"]) { _, new in new }
        }
        // Codex assistants are persisted by the source-record binder, not by
        // Claude's native_record_links continuity writer. Try that actual key
        // first; preserve the existing Claude route as its fallback.
        if let nativeHeader = NativeHeaderEvidenceReader.read(db: db, sourceRecordKey: sourceRecordKey, maxBytes: maxBytes) {
            return nativeHeader
        }
        guard case .value(let hasTable) = tableExists(db) else {
            return base.merging(["status": "database_error"]) { _, new in new }
        }
        guard hasTable else {
            return base.merging(["status": "unavailable_legacy_catalog"]) { _, new in new }
        }
        guard case .value(let schema) = catalogSchema(db) else {
            return base.merging(["status": "database_error"]) { _, new in new }
        }
        guard targetColumns.isSubset(of: schema.columns), schema.hasExactRecordKeyPrimary else {
            return base.merging(["status": "unavailable_legacy_target_schema"]) { _, new in new }
        }

        let sql = """
        SELECT provider,session_id,source_file,record_start_offset,record_end_offset,
               raw_record_hash,target_extraction_version,declared_targets_json,target_diagnostics_json
        FROM native_record_links
        WHERE record_key=?
        LIMIT 1
        """
        guard let stmt = statement(db, sql, [.text(sourceRecordKey)]) else {
            return base.merging(["status": "database_error"]) { _, new in new }
        }
        defer { sqlite3_finalize(stmt) }
        let step = sqlite3_step(stmt)
        guard step == SQLITE_ROW else {
            return base.merging([
                "status": step == SQLITE_DONE ? "unavailable_source_record_not_found" : "database_error",
            ]) { _, new in new }
        }
        guard
            let provider = boundedText(stmt, 0, maxBytes: 32), ["claude", "codex"].contains(provider),
            let sessionID = boundedText(stmt, 1, maxBytes: 4_096),
            let sourceFile = boundedText(stmt, 2, maxBytes: 16_384),
            sqlite3_column_type(stmt, 3) == SQLITE_INTEGER,
            sqlite3_column_type(stmt, 4) == SQLITE_INTEGER,
            let rawHash = boundedText(stmt, 5, maxBytes: 64), isSHA256(rawHash),
            sqlite3_column_type(stmt, 6) == SQLITE_INTEGER
        else {
            return base.merging(["status": "malformed_catalog_evidence"]) { _, new in new }
        }
        let start = sqlite3_column_int64(stmt, 3)
        let end = sqlite3_column_int64(stmt, 4)
        guard start >= 0, end > start else {
            return base.merging(["status": "malformed_catalog_evidence"]) { _, new in new }
        }
        let version = Int(sqlite3_column_int64(stmt, 6))
        var result = base.merging([
            "provider": provider,
            "session_id": sessionID,
            "source_file": sourceFile,
            "record_start_offset": start,
            "record_end_offset": end,
            "raw_record_hash": rawHash,
            "target_extraction_version": version,
        ]) { _, new in new }
        let contextEvidence = exactRecordContexts(
            db,
            sessionID: sessionID,
            sourceFile: sourceFile,
            startOffset: start,
            endOffset: end,
            rawRecordHash: rawHash
        )
        result["attribution"] = NativeAttributionAdapter.project(
            contexts: contextEvidence.contexts,
            contextStatus: contextEvidence.status,
            evidenceKeys: contextEvidence.evidenceKeys
        )

        if version == 0 {
            result["status"] = "unavailable_not_extracted"
            result["target_report"] = [
                "version": 0,
                "scope": NSNull(),
                "targets": [],
                "diagnostics": [],
                "interpretation": "Target extraction was not run for this record.",
            ] as [String: Any]
            return cappedTargetResult(result, key: sourceRecordKey, budget: budget)
        }
        guard version == 1 else {
            result["status"] = "unavailable_unknown_target_version"
            result["target_report"] = [
                "version": version,
                "scope": NSNull(),
                "targets": [],
                "diagnostics": [],
            ] as [String: Any]
            return cappedTargetResult(result, key: sourceRecordKey, budget: budget)
        }
        guard
            let targetsJSON = boundedText(stmt, 7, maxBytes: 256 * 1_024),
            let diagnosticsJSON = boundedText(stmt, 8, maxBytes: 256 * 1_024),
            targetsJSON.utf8.count + diagnosticsJSON.utf8.count <= 256 * 1_024,
            let targets = validTargets(targetsJSON),
            let diagnostics = validDiagnostics(diagnosticsJSON)
        else {
            result["status"] = "malformed_catalog_evidence"
            return cappedTargetResult(result, key: sourceRecordKey, budget: budget)
        }
        result["status"] = "available"
        result["target_report"] = [
            "version": 1,
            "scope": targetReportScope,
            "targets": targets,
            "diagnostics": diagnostics,
            "interpretation": targets.isEmpty
                ? "No supported target was declared; recipient absence is not inferred."
                : "Each target is declared and unverified.",
        ] as [String: Any]
        return cappedTargetResult(result, key: sourceRecordKey, budget: budget)
    }

    private static func catalogSchema(
        _ db: OpaquePointer?
    ) -> Query<(columns: Set<String>, hasExactRecordKeyPrimary: Bool)> {
        guard let stmt = statement(db, "PRAGMA table_info('native_record_links')", []) else {
            return .failure
        }
        defer { sqlite3_finalize(stmt) }
        var columns: Set<String> = []
        var primaryColumns: [(ordinal: Int32, name: String)] = []
        var recordKeyHasTextType = false
        while true {
            let step = sqlite3_step(stmt)
            if step == SQLITE_DONE {
                let orderedPrimary = primaryColumns.sorted { $0.ordinal < $1.ordinal }.map(\.name)
                return .value((
                    columns: columns,
                    hasExactRecordKeyPrimary: orderedPrimary == ["record_key"] && recordKeyHasTextType
                ))
            }
            guard
                step == SQLITE_ROW,
                let name = boundedText(stmt, 1, maxBytes: 256),
                let declaredType = boundedText(stmt, 2, maxBytes: 64)
            else { return .failure }
            columns.insert(name)
            let primaryOrdinal = sqlite3_column_int(stmt, 5)
            if primaryOrdinal > 0 {
                primaryColumns.append((ordinal: primaryOrdinal, name: name))
            }
            if name == "record_key" {
                recordKeyHasTextType = declaredType.caseInsensitiveCompare("TEXT") == .orderedSame
            }
        }
    }

    private static func validTargets(_ raw: String) -> [[String: Any]]? {
        guard
            raw.utf8.count <= 256 * 1_024,
            let data = raw.data(using: .utf8),
            let targets = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]],
            targets.count <= 25
        else { return nil }
        let keys: Set<String> = [
            "tool_name", "target", "target_kind", "declared_unverified", "call_id",
            "raw_container_pointer", "decoded_pointer", "pointer_kind",
        ]
        for target in targets {
            guard Set(target.keys) == keys,
                  isTrueBoolean(target["declared_unverified"]),
                  validRequired(target["tool_name"]),
                  validRequired(target["target"]),
                  validRequired(target["decoded_pointer"]),
                  validRequired(target["pointer_kind"]),
                  validNullable(target["call_id"]),
                  validNullable(target["raw_container_pointer"]),
                  let kind = target["target_kind"] as? String,
                  ["agent_handle", "task_handle", "surface"].contains(kind),
                  let pointerKind = target["pointer_kind"] as? String,
                  ["direct_raw_json", "decoded_json_string"].contains(pointerKind)
            else { return nil }
        }
        return targets
    }

    private struct ExactRecordContexts {
        let contexts: [[String: Any]]
        let status: String
        let evidenceKeys: [String]
    }

    private static func exactRecordContexts(
        _ db: OpaquePointer?,
        sessionID: String,
        sourceFile: String,
        startOffset: Int64,
        endOffset: Int64,
        rawRecordHash: String
    ) -> ExactRecordContexts {
        let required = Set([
            "record_key", "source_file", "record_start_offset", "record_end_offset",
            "raw_record_hash", "session_id", "record_context_json",
        ])
        guard let columns = sourceRecordColumns(db) else {
            return .init(contexts: [], status: "unavailable_database_error", evidenceKeys: [])
        }
        guard required.isSubset(of: columns) else {
            return .init(contexts: [], status: "unavailable_legacy_schema", evidenceKeys: [])
        }
        guard case .value(let indexColumns) = indexColumns(db, "idx_chunk_source_records_physical") else {
            return .init(contexts: [], status: "unavailable_database_error", evidenceKeys: [])
        }
        guard indexColumns == [
            "source_file", "record_start_offset", "record_end_offset",
            "raw_record_hash", "session_id", "record_key",
        ] else {
            return .init(contexts: [], status: "unavailable_missing_physical_index", evidenceKeys: [])
        }
        let sql = """
        SELECT record_key,
               CASE WHEN length(CAST(record_context_json AS BLOB)) <= 32768
                    THEN record_context_json ELSE NULL END,
               CASE WHEN length(CAST(record_context_json AS BLOB)) > 32768 THEN 1 ELSE 0 END
        FROM chunk_source_records INDEXED BY idx_chunk_source_records_physical
        WHERE source_file=? AND record_start_offset=? AND record_end_offset=?
          AND raw_record_hash=? AND session_id=?
        ORDER BY record_key
        LIMIT 17
        """
        guard let stmt = statement(db, sql, [
            .text(sourceFile), .int(startOffset), .int(endOffset),
            .text(rawRecordHash), .text(sessionID),
        ]) else {
            return .init(contexts: [], status: "unavailable_database_error", evidenceKeys: [])
        }
        defer { sqlite3_finalize(stmt) }
        var contexts: [[String: Any]] = []
        var evidenceKeys: [String] = []
        var rowCount = 0
        while true {
            let step = sqlite3_step(stmt)
            if step == SQLITE_DONE { break }
            guard step == SQLITE_ROW else {
                return .init(contexts: [], status: "unavailable_database_error", evidenceKeys: [])
            }
            rowCount += 1
            if rowCount > 16 {
                return .init(contexts: [], status: "omitted_match_cap", evidenceKeys: [])
            }
            guard let key = boundedText(stmt, 0, maxBytes: 128) else {
                return .init(contexts: [], status: "unavailable_database_error", evidenceKeys: [])
            }
            evidenceKeys.append(key)
            if sqlite3_column_int(stmt, 2) != 0 {
                return .init(contexts: [], status: "omitted_byte_cap", evidenceKeys: evidenceKeys)
            }
            guard let raw = boundedText(stmt, 1, maxBytes: 32_768),
                  let data = raw.data(using: .utf8),
                  let maps = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
                return .init(contexts: [], status: "unavailable_database_error", evidenceKeys: evidenceKeys)
            }
            contexts.append(contentsOf: maps)
        }
        return contexts.isEmpty
            ? .init(contexts: [], status: "unavailable_missing", evidenceKeys: evidenceKeys)
            : .init(contexts: contexts, status: "declared_unverified", evidenceKeys: evidenceKeys)
    }

    private static func sourceRecordColumns(_ db: OpaquePointer?) -> Set<String>? {
        guard let stmt = statement(db, "PRAGMA table_info('chunk_source_records')", []) else {
            return nil
        }
        defer { sqlite3_finalize(stmt) }
        var columns = Set<String>()
        while true {
            let step = sqlite3_step(stmt)
            if step == SQLITE_DONE { return columns }
            guard step == SQLITE_ROW, let name = boundedText(stmt, 1, maxBytes: 256) else {
                return nil
            }
            columns.insert(name)
        }
    }

    private static func validDiagnostics(_ raw: String) -> [String]? {
        guard
            raw.utf8.count <= 256 * 1_024,
            let data = raw.data(using: .utf8),
            let diagnostics = try? JSONSerialization.jsonObject(with: data) as? [String],
            diagnostics.count <= 32,
            diagnostics.allSatisfy({ validBounded($0, maxBytes: 4_096) })
        else { return nil }
        return diagnostics
    }

    private static func validRequired(_ value: Any?) -> Bool {
        guard let value = value as? String else { return false }
        return validBounded(value, maxBytes: 4_096)
    }

    private static func validNullable(_ value: Any?) -> Bool {
        value is NSNull || (value as? String).map { validBounded($0, maxBytes: 4_096) } == true
    }

    private static func validBounded(_ value: String, maxBytes: Int) -> Bool {
        !value.isEmpty && value.utf8.count <= maxBytes && !value.utf8.contains(0)
    }

    private static func isTrueBoolean(_ value: Any?) -> Bool {
        guard let value else { return false }
        return CFGetTypeID(value as CFTypeRef) == CFBooleanGetTypeID()
            && (value as? Bool) == true
    }

    private static func boundedText(
        _ statement: OpaquePointer,
        _ column: Int32,
        maxBytes: Int
    ) -> String? {
        guard sqlite3_column_type(statement, column) == SQLITE_TEXT else { return nil }
        let byteCount = Int(sqlite3_column_bytes(statement, column))
        guard byteCount > 0, byteCount <= maxBytes, let bytes = sqlite3_column_text(statement, column) else {
            return nil
        }
        let data = Data(bytes: bytes, count: byteCount)
        guard let value = String(data: data, encoding: .utf8), !value.utf8.contains(0) else {
            return nil
        }
        return value
    }

    private static func isSHA256(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy {
            (48...57).contains($0) || (65...70).contains($0) || (97...102).contains($0)
        }
    }

    private static func cappedTargetResult(
        _ result: [String: Any],
        key: String,
        budget: Int
    ) -> [String: Any] {
        if let data = try? JSONSerialization.data(withJSONObject: result), data.count <= budget {
            return result
        }
        return [
            "schema_version": targetReportSchemaVersion,
            "mode": "source_record",
            "status": "omitted_response_budget",
            "source_record_key": key,
            "serialized_max_bytes": budget,
            "omitted": "source record locator and target report",
            "evidence_boundary": targetReportBoundary,
        ]
    }
}
