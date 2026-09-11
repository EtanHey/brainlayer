import CryptoKit
import Foundation
import SQLite3

/// Provider-recorded Codex header evidence joined from the persisted source key.
/// This does not authenticate a sender, person, relay, or original author.
enum NativeHeaderEvidenceReader {
    private static let maxContextBytes = 32 * 1024
    private static let maxRecordBytes = 256 * 1024
    private static let requiredColumns: Set<String> = [
        "record_key", "session_id", "source_file", "record_start_offset",
        "record_end_offset", "raw_record_hash", "native_generation_key", "native_provider",
        "native_watcher_generation", "native_record_role", "native_occurrence_key",
        "native_record_manifest_sha256",
    ]

    static func read(db: OpaquePointer?, sourceRecordKey: String, maxBytes: Int) -> [String: Any]? {
        guard tableExists(db, "chunk_source_records") else { return nil }
        let budget = min(max(maxBytes, 1_024), 128_000)
        let base: [String: Any] = [
            "schema_version": 1,
            "mode": "source_record",
            "source_record_key": sourceRecordKey,
            "serialized_max_bytes": budget,
            "evidence_boundary": "Provider-recorded header metadata is declared and unverified; no person, sender, delivery, relay, or original authorship is inferred.",
        ]
        guard isSHA256(sourceRecordKey) else { return base.withStatus("malformed_source_record_key") }
        // A Claude continuity key is not a source-record binding. Returning
        // nil here is what preserves the established native_record_links path.
        guard sourceProvider(db, sourceRecordKey) == "codex" else { return nil }
        guard let columns = tableColumns(db, "chunk_source_records"), requiredColumns.isSubset(of: columns) else {
            return base.withStatus("unknown_native_header_schema")
        }
        let sql = """
        SELECT csr.record_key,csr.native_provider,csr.session_id,csr.source_file,
               csr.record_start_offset,csr.record_end_offset,csr.raw_record_hash,
               csr.native_generation_key,csr.native_provider,csr.native_watcher_generation,
               csr.native_record_role,csr.native_occurrence_key,csr.native_record_manifest_sha256,
               g.provider,g.source_file,g.watcher_generation,g.root_manifest_sha256,
               g.current_manifest_sha256,g.status,g.canonical_occurrence_key,
               o.occurrence_key,o.generation_key,o.disposition,o.manifest_sha256,
               o.record_start_offset,o.record_end_offset,o.raw_record_hash,o.session_id,
               o.header_context_json,o.header_context_sha256,o.validation_error
        FROM chunk_source_records AS csr
        LEFT JOIN native_source_generations AS g ON g.generation_key=csr.native_generation_key
        LEFT JOIN native_session_header_occurrences AS o
          ON o.occurrence_key=g.canonical_occurrence_key AND o.generation_key=g.generation_key
        WHERE csr.record_key=?
        LIMIT 2
        """
        guard let statement = prepare(db, sql, [sourceRecordKey]) else { return base.withStatus("unknown_native_header_schema") }
        defer { sqlite3_finalize(statement) }
        // A valid binding has one joined row. Read one additional row only so
        // a corrupt catalog cannot be accepted based on whichever twin SQLite
        // happens to return first, while keeping the query bounded.
        let firstRow = sqlite3_step(statement)
        guard firstRow == SQLITE_ROW else { return base.withStatus("unknown_native_header_missing_source_row") }
        let secondRow = sqlite3_step(statement)
        guard secondRow == SQLITE_DONE else {
            return base.withStatus(secondRow == SQLITE_ROW
                ? "unknown_multiple_native_header_bindings"
                : "unknown_native_header_schema")
        }
        guard sqlite3_reset(statement) == SQLITE_OK, sqlite3_step(statement) == SQLITE_ROW else {
            return base.withStatus("unknown_native_header_schema")
        }
        guard let provider = text(statement, 1), let sessionID = text(statement, 2),
              let sourceFile = text(statement, 3), let rawHash = text(statement, 6),
              sqlite3_column_type(statement, 4) == SQLITE_INTEGER,
              sqlite3_column_type(statement, 5) == SQLITE_INTEGER,
              provider == "codex", isCanonicalUUID(sessionID), isSHA256(rawHash) else {
            return base.withStatus("unknown_malformed_native_header_binding")
        }
        let start = sqlite3_column_int64(statement, 4)
        let end = sqlite3_column_int64(statement, 5)
        var result = base.merging([
            "provider": provider, "session_id": sessionID, "source_file": sourceFile,
            "record_start_offset": start, "record_end_offset": end, "raw_record_hash": rawHash,
        ]) { _, value in value }
        func unknown(_ status: String) -> [String: Any] {
            result["status"] = status
            result["attribution"] = NativeAttributionAdapter.project(contexts: [], contextStatus: "unavailable_missing", evidenceKeys: [])
            result["native_header"] = ["status": status, "declared_unverified": true, "evidence_kind": "provider_recorded_header"] as [String: Any]
            return capped(result, budget: budget)
        }
        guard start >= 0, end > start, end - start <= Int64(maxRecordBytes) else { return unknown("unknown_malformed_native_header_binding") }
        guard let generationKey = text(statement, 7), text(statement, 8) == "codex",
              let watcherGeneration = integer(statement, 9), text(statement, 10) == "assistant",
              let recordManifest = text(statement, 12), isSHA256(recordManifest) else {
            return unknown(text(statement, 10) == nil ? "unknown_missing_native_header_binding" : "unknown_malformed_native_header_binding")
        }
        guard let boundOccurrenceKey = text(statement, 11) else {
            return unknown(headerAfterAssistant(db, generationKey: generationKey, assistantStart: start)
                ? "unknown_header_after_assistant" : "unknown_missing_native_header_binding")
        }
        guard let generationProvider = text(statement, 13), let generationSource = text(statement, 14),
              let generationWatcher = integer(statement, 15), let rootManifest = text(statement, 16),
              let currentManifest = text(statement, 17), let generationStatus = text(statement, 18),
              let canonicalOccurrenceKey = text(statement, 19), let occurrenceKey = text(statement, 20),
              let occurrenceGeneration = text(statement, 21), let disposition = text(statement, 22),
              let headerManifest = text(statement, 23), let headerStart = integer(statement, 24),
              let headerEnd = integer(statement, 25), let headerHash = text(statement, 26),
              let headerSessionID = text(statement, 27), let contextJSON = text(statement, 28),
              let contextHash = text(statement, 29) else {
            return unknown("unknown_missing_native_header_binding")
        }
        guard generationProvider == provider, generationSource == sourceFile,
              generationWatcher == watcherGeneration, generationStatus == "active",
              canonicalOccurrenceKey == occurrenceKey, boundOccurrenceKey == occurrenceKey,
              occurrenceGeneration == generationKey, disposition == "canonical",
              headerSessionID == sessionID, headerStart >= 0, headerEnd > headerStart,
              headerEnd <= start, headerEnd - headerStart <= Int64(maxRecordBytes),
              isSHA256(rootManifest), isSHA256(currentManifest), isSHA256(headerManifest),
              isSHA256(headerHash), contextJSON.utf8.count <= maxContextBytes,
              sqlite3_column_type(statement, 30) == SQLITE_NULL,
              contextHash == sha256CanonicalJSON(contextJSON) else {
            let status = generationStatus == "conflicted" || generationStatus == "invalid"
                ? "unknown_tainted_native_header"
                : (headerStart > start ? "unknown_header_after_assistant" : "unknown_malformed_native_header_binding")
            return unknown(status)
        }
        guard let context = parseContext(contextJSON), let recordedThreadID = context["/payload/id"] as? String,
              isCanonicalUUID(recordedThreadID), recordedThreadID == headerSessionID else {
            return unknown("unknown_malformed_native_header_context")
        }
        var header: [String: Any] = [
            "status": "available", "evidence_kind": "provider_recorded_header",
            "declared_unverified": true, "provider_recorded": true,
            "recorded_thread_id": recordedThreadID,
            "provider": provider, "source_file": sourceFile, "generation_key": generationKey,
            "occurrence_key": occurrenceKey, "watcher_generation": watcherGeneration,
            "assistant_record_start_offset": start, "assistant_record_end_offset": end,
            "assistant_raw_record_hash": rawHash, "header_start_offset": headerStart,
            "header_end_offset": headerEnd, "header_raw_record_hash": headerHash,
            "header_manifest_sha256": headerManifest,
            "recorded_manifest_sha256": recordManifest,
            "session_id": headerSessionID, "header_context": context,
            "header_context_sha256": contextHash, "person_principal": NSNull(),
            "original_author_principal": NSNull(),
        ]
        for (contextKey, outputKey) in [
            ("/payload/agent_path", "recorded_agent_path"),
            ("/payload/agent_nickname", "recorded_agent_nickname"),
            ("/payload/parent_thread_id", "recorded_parent_thread_id"),
            ("/payload/agent_role", "recorded_agent_role"),
            ("/payload/source/subagent/thread_spawn/agent_path", "spawn_agent_path"),
            ("/payload/source/subagent/thread_spawn/agent_nickname", "spawn_agent_nickname"),
            ("/payload/source/subagent/thread_spawn/parent_thread_id", "spawn_parent_thread_id"),
            ("/payload/source/subagent/thread_spawn/agent_role", "spawn_agent_role"),
        ] {
            if let value = context[contextKey] as? String { header[outputKey] = value }
        }
        result["status"] = "available"
        result["native_header"] = header
        result["attribution"] = NativeAttributionAdapter.project(contexts: [], contextStatus: "unavailable_missing", evidenceKeys: [])
        return capped(result, budget: budget)
    }

    private static func capped(_ result: [String: Any], budget: Int) -> [String: Any] {
        guard let data = try? JSONSerialization.data(withJSONObject: result), data.count <= budget else {
            return ["schema_version": 1, "mode": "source_record", "status": "omitted_response_budget"]
        }
        return result
    }

    private static func tableExists(_ db: OpaquePointer?, _ name: String) -> Bool {
        guard let statement = prepare(db, "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", [name]) else { return false }
        defer { sqlite3_finalize(statement) }
        return sqlite3_step(statement) == SQLITE_ROW
    }

    private static func sourceProvider(_ db: OpaquePointer?, _ key: String) -> String? {
        guard let statement = prepare(db, "SELECT native_provider FROM chunk_source_records WHERE record_key=? LIMIT 1", [key]) else { return nil }
        defer { sqlite3_finalize(statement) }
        guard sqlite3_step(statement) == SQLITE_ROW else { return nil }
        return text(statement, 0)
    }

    private static func tableColumns(_ db: OpaquePointer?, _ table: String) -> Set<String>? {
        guard let statement = prepare(db, "PRAGMA table_info(\(table))", []) else { return nil }
        defer { sqlite3_finalize(statement) }
        var columns = Set<String>()
        while sqlite3_step(statement) == SQLITE_ROW {
            guard let name = text(statement, 1) else { return nil }
            columns.insert(name)
        }
        return columns
    }

    private static func prepare(_ db: OpaquePointer?, _ sql: String, _ values: [String]) -> OpaquePointer? {
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK, let statement else { return nil }
        for (index, value) in values.enumerated() {
            _ = value.withCString { sqlite3_bind_text(statement, Int32(index + 1), $0, -1, unsafeBitCast(-1, to: sqlite3_destructor_type.self)) }
        }
        return statement
    }

    private static func text(_ statement: OpaquePointer, _ column: Int32) -> String? {
        sqlite3_column_text(statement, column).map { String(cString: $0) }
    }

    private static func integer(_ statement: OpaquePointer, _ column: Int32) -> Int64? {
        sqlite3_column_type(statement, column) == SQLITE_INTEGER ? sqlite3_column_int64(statement, column) : nil
    }

    private static func isSHA256(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy { ($0 >= 48 && $0 <= 57) || ($0 >= 97 && $0 <= 102) }
    }

    private static func isCanonicalUUID(_ value: String) -> Bool {
        guard let uuid = UUID(uuidString: value) else { return false }
        return uuid.uuidString.lowercased() == value.lowercased()
    }

    private static func headerAfterAssistant(_ db: OpaquePointer?, generationKey: String, assistantStart: Int64) -> Bool {
        guard let statement = prepare(db, "SELECT 1 FROM native_session_header_occurrences WHERE generation_key=? AND disposition='canonical' AND record_start_offset>? LIMIT 1", [generationKey, String(assistantStart)]) else { return false }
        defer { sqlite3_finalize(statement) }
        return sqlite3_step(statement) == SQLITE_ROW
    }

    private static func parseContext(_ raw: String) -> [String: Any]? {
        guard let data = raw.data(using: .utf8), let value = try? JSONSerialization.jsonObject(with: data), let object = value as? [String: Any] else { return nil }
        return object
    }

    private static func sha256CanonicalJSON(_ raw: String) -> String? {
        guard let data = raw.data(using: .utf8), let value = try? JSONSerialization.jsonObject(with: data), JSONSerialization.isValidJSONObject(value), let canonical = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys, .withoutEscapingSlashes]) else { return nil }
        return Data(SHA256.hash(data: canonical)).map { String(format: "%02x", $0) }.joined()
    }
}

private extension Dictionary where Key == String, Value == Any {
    func withStatus(_ status: String) -> [String: Any] { merging(["status": status]) { _, value in value } }
}
