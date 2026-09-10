import Foundation
import SQLite3
@testable import BrainBar

private let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

private func require(_ condition: @autoclosure () -> Bool, _ message: String) {
    if !condition() { fatalError(message) }
}

private func execute(_ db: OpaquePointer?, _ sql: String) {
    var error: UnsafeMutablePointer<CChar>?
    let code = sqlite3_exec(db, sql, nil, nil, &error)
    let detail = error.map { String(cString: $0) } ?? ""
    sqlite3_free(error)
    require(code == SQLITE_OK, "SQL failed \(code): \(detail)")
}

private func target(
    tool: String,
    handle: String,
    kind: String,
    callID: String?,
    rawPointer: String?,
    decodedPointer: String,
    pointerKind: String
) -> [String: Any] {
    [
        "tool_name": tool,
        "target": handle,
        "target_kind": kind,
        "declared_unverified": true,
        "call_id": callID ?? NSNull(),
        "raw_container_pointer": rawPointer ?? NSNull(),
        "decoded_pointer": decodedPointer,
        "pointer_kind": pointerKind,
    ]
}

private func compactJSON(_ value: Any) -> String {
    String(data: try! JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]), encoding: .utf8)!
}

private func insertRecord(
    _ db: OpaquePointer?,
    key: String,
    provider: String = "claude",
    session: String = "session-target",
    source: String = "/generated/fixture.jsonl",
    start: Int64,
    hash: String,
    version: Int,
    targetsJSON: String,
    diagnosticsJSON: String = "[]"
) {
    let sql = """
    INSERT INTO native_record_links(
        record_key,provider,session_id,source_file,record_start_offset,record_end_offset,
        raw_record_hash,native_uuid,parent_uuid_state,parent_uuid_value,
        logical_parent_uuid_state,logical_parent_uuid_value,declared_headers_json,
        preserved_segment_json,preserved_segment_sha256,preserved_segment_omitted,
        target_extraction_version,declared_targets_json,target_diagnostics_json
    ) VALUES(?,?,?,?,?,?,?,NULL,'missing',NULL,'missing',NULL,'{}',NULL,NULL,0,?,?,?)
    """
    var statement: OpaquePointer?
    require(sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK, "insert prepare")
    defer { sqlite3_finalize(statement) }
    let strings: [(Int32, String)] = [
        (1, key), (2, provider), (3, session), (4, source), (7, hash),
        (9, targetsJSON), (10, diagnosticsJSON),
    ]
    for (index, value) in strings {
        require(value.withCString { sqlite3_bind_text(statement, index, $0, -1, transient) } == SQLITE_OK, "bind \(index)")
    }
    sqlite3_bind_int64(statement, 5, start)
    sqlite3_bind_int64(statement, 6, start + 17)
    sqlite3_bind_int(statement, 8, Int32(version))
    require(sqlite3_step(statement) == SQLITE_DONE, "insert step")
}

private func request(_ router: MCPRouter, arguments: [String: Any], id: Int = 1) -> [String: Any] {
    router.handle([
        "jsonrpc": "2.0",
        "id": id,
        "method": "tools/call",
        "params": [
            "name": "brain_expand",
            "arguments": arguments,
        ],
    ])
}

private func result(_ response: [String: Any]) -> [String: Any] {
    response["result"] as? [String: Any] ?? [:]
}

private func structured(_ response: [String: Any]) -> [String: Any] {
    result(response)["structuredContent"] as? [String: Any] ?? [:]
}

private func text(_ response: [String: Any]) -> String {
    let content = result(response)["content"] as? [[String: Any]] ?? []
    return content.compactMap { $0["text"] as? String }.joined(separator: "\n")
}

private func status(_ response: [String: Any]) -> String? {
    structured(response)["status"] as? String
}

private func createReaderOnlySchema(
    _ db: OpaquePointer?,
    keyDefinition: String,
    tableConstraint: String = ""
) {
    execute(db, """
    CREATE TABLE native_record_links(
        record_key \(keyDefinition),
        provider TEXT,
        session_id TEXT,
        source_file TEXT,
        record_start_offset INTEGER,
        record_end_offset INTEGER,
        raw_record_hash TEXT,
        target_extraction_version INTEGER,
        declared_targets_json TEXT,
        target_diagnostics_json TEXT
        \(tableConstraint)
    )
    """)
}

private func createTargetSchema(_ db: OpaquePointer?, primaryKey: Bool = true) {
    execute(db, """
    CREATE TABLE native_record_links(
        record_key TEXT \(primaryKey ? "PRIMARY KEY" : ""),
        provider TEXT NOT NULL,
        session_id TEXT NOT NULL,
        source_file TEXT NOT NULL,
        record_start_offset INTEGER NOT NULL,
        record_end_offset INTEGER NOT NULL,
        raw_record_hash TEXT NOT NULL,
        native_uuid TEXT,
        parent_uuid_state TEXT NOT NULL,
        parent_uuid_value TEXT,
        logical_parent_uuid_state TEXT NOT NULL,
        logical_parent_uuid_value TEXT,
        declared_headers_json TEXT NOT NULL,
        preserved_segment_json TEXT,
        preserved_segment_sha256 TEXT,
        preserved_segment_omitted INTEGER NOT NULL DEFAULT 0,
        target_extraction_version INTEGER NOT NULL DEFAULT 0,
        declared_targets_json TEXT NOT NULL DEFAULT '[]',
        target_diagnostics_json TEXT NOT NULL DEFAULT '[]'
    )
    """)
}

private func createSourceRecordSchema(_ db: OpaquePointer?) {
    execute(db, """
    CREATE TABLE chunk_source_records(
        record_key TEXT PRIMARY KEY,
        canonical_chunk_id TEXT NOT NULL,
        source_file TEXT NOT NULL,
        record_start_offset INTEGER NOT NULL,
        record_end_offset INTEGER NOT NULL,
        raw_record_hash TEXT NOT NULL,
        session_id TEXT,
        observed_at TEXT NOT NULL,
        record_context_json TEXT NOT NULL DEFAULT '[]'
    )
    """)
}

private func createSourceRecordPhysicalIndex(_ db: OpaquePointer?) {
    execute(db, """
    CREATE INDEX idx_chunk_source_records_physical ON chunk_source_records(
        source_file, record_start_offset, record_end_offset, raw_record_hash, session_id, record_key
    )
    """)
}

private func sourceRecordPhysicalPlan(_ db: OpaquePointer?) -> [String] {
    let sql = """
    EXPLAIN QUERY PLAN
    SELECT record_key FROM chunk_source_records INDEXED BY idx_chunk_source_records_physical
    WHERE source_file=? AND record_start_offset=? AND record_end_offset=?
      AND raw_record_hash=? AND session_id=?
    ORDER BY record_key LIMIT 17
    """
    var statement: OpaquePointer?
    require(sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK, "plan prepare")
    defer { sqlite3_finalize(statement) }
    for (index, value) in [(1, "/generated/fixture.jsonl"), (4, String(repeating: "9", count: 64)), (5, "session-target")] {
        require(value.withCString { sqlite3_bind_text(statement, Int32(index), $0, -1, transient) } == SQLITE_OK, "plan bind \(index)")
    }
    sqlite3_bind_int64(statement, 2, 101)
    sqlite3_bind_int64(statement, 3, 118)
    var details: [String] = []
    while sqlite3_step(statement) == SQLITE_ROW {
        guard let raw = sqlite3_column_text(statement, 3) else { fatalError("missing plan detail") }
        details.append(String(cString: raw))
    }
    return details
}

private func insertSourceRecord(
    _ db: OpaquePointer?,
    key: String,
    chunk: String,
    source: String = "/generated/fixture.jsonl",
    start: Int64,
    hash: String,
    session: String = "session-target",
    contexts: [[String: Any]]
) {
    let sql = """
    INSERT INTO chunk_source_records(
        record_key,canonical_chunk_id,source_file,record_start_offset,record_end_offset,
        raw_record_hash,session_id,observed_at,record_context_json
    ) VALUES(?,?,?,?,?,?,?,'2026-09-10T00:00:00Z',?)
    """
    var statement: OpaquePointer?
    require(sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK, "source insert prepare")
    defer { sqlite3_finalize(statement) }
    let strings: [(Int32, String)] = [
        (1, key), (2, chunk), (3, source), (6, hash), (7, session), (8, compactJSON(contexts)),
    ]
    for (index, value) in strings {
        require(value.withCString { sqlite3_bind_text(statement, index, $0, -1, transient) } == SQLITE_OK, "source bind \(index)")
    }
    sqlite3_bind_int64(statement, 4, start)
    sqlite3_bind_int64(statement, 5, start + 17)
    require(sqlite3_step(statement) == SQLITE_DONE, "source insert step")
}

@main
struct NativeTargetRecordExpansionHarness {
    static func main() throws {
        let path = NSTemporaryDirectory() + "native-target-\(UUID().uuidString).db"
        defer {
            for suffix in ["", "-wal", "-shm"] {
                try? FileManager.default.removeItem(atPath: path + suffix)
            }
        }
        let database = BrainDatabase(path: path)
        defer { database.close() }
        guard let handle = database.dbHandle else { fatalError("database did not open") }
        let router = MCPRouter(profile: "core")
        router.setDatabase(database)

        let selectedKey = String(repeating: "a", count: 64)
        let missingKey = String(repeating: "b", count: 64)
        let versionZeroKey = String(repeating: "c", count: 64)
        let unknownVersionKey = String(repeating: "d", count: 64)
        let malformedKey = String(repeating: "e", count: 64)
        let budgetKey = String(repeating: "f", count: 64)
        let emptyKey = "1" + String(repeating: "0", count: 63)
        let decoyKey = "2" + String(repeating: "0", count: 63)
        let numericBooleanKey = "3" + String(repeating: "0", count: 63)
        let nulTargetKey = "4" + String(repeating: "0", count: 63)
        let nulSourceKey = "5" + String(repeating: "0", count: 63)
        let invalidUTF8Key = "6" + String(repeating: "0", count: 63)
        let hash = String(repeating: "9", count: 64)

        require(status(request(router, arguments: ["source_record_key": selectedKey])) == "unavailable_legacy_catalog", "missing catalog status")

        execute(handle, "CREATE TABLE native_record_links(record_key TEXT PRIMARY KEY,provider TEXT,session_id TEXT,source_file TEXT,record_start_offset INTEGER,record_end_offset INTEGER,raw_record_hash TEXT)")
        require(status(request(router, arguments: ["source_record_key": selectedKey])) == "unavailable_legacy_target_schema", "legacy target schema status")
        execute(handle, "DROP TABLE native_record_links")
        createReaderOnlySchema(handle, keyDefinition: "TEXT")
        require(status(request(router, arguments: ["source_record_key": selectedKey])) == "unavailable_legacy_target_schema", "record key must be primary")
        execute(handle, "DROP TABLE native_record_links")
        createReaderOnlySchema(handle, keyDefinition: "TEXT", tableConstraint: ", other TEXT, PRIMARY KEY(record_key, other)")
        require(status(request(router, arguments: ["source_record_key": selectedKey])) == "unavailable_legacy_target_schema", "composite primary key accepted")
        execute(handle, "DROP TABLE native_record_links")
        createReaderOnlySchema(handle, keyDefinition: "BLOB PRIMARY KEY")
        require(status(request(router, arguments: ["source_record_key": selectedKey])) == "unavailable_legacy_target_schema", "non-text record key accepted")
        execute(handle, "DROP TABLE native_record_links")
        createTargetSchema(handle)
        createSourceRecordSchema(handle)

        let selectedTargets = [
            target(
                tool: "send_message",
                handle: "task-1",
                kind: "task_handle",
                callID: "call-1",
                rawPointer: "/payload/arguments",
                decodedPointer: "/threadId",
                pointerKind: "decoded_json_string"
            ),
            target(
                tool: "capture_screen",
                handle: "surface-2",
                kind: "surface",
                callID: nil,
                rawPointer: nil,
                decodedPointer: "/message/content/0/input/surface",
                pointerKind: "direct_raw_json"
            ),
        ]
        insertRecord(handle, key: selectedKey, start: 101, hash: hash, version: 1, targetsJSON: compactJSON(selectedTargets), diagnosticsJSON: compactJSON(["fixture diagnostic"]))
        insertSourceRecord(
            handle,
            key: "selected-source-context",
            chunk: "selected-chunk",
            start: 101,
            hash: hash,
            contexts: [["attributionAgent": "workflow-subagent", "declared_message_role": "assistant"]]
        )
        insertRecord(handle, key: decoyKey, source: "/generated/adjacent.jsonl", start: 102, hash: String(repeating: "8", count: 64), version: 1, targetsJSON: compactJSON([target(tool: "send_message", handle: "decoy", kind: "agent_handle", callID: nil, rawPointer: nil, decodedPointer: "/recipient", pointerKind: "direct_raw_json")]))
        insertRecord(handle, key: versionZeroKey, start: 201, hash: hash, version: 0, targetsJSON: "[]")
        insertRecord(handle, key: unknownVersionKey, start: 301, hash: hash, version: 7, targetsJSON: "[]")
        insertRecord(handle, key: malformedKey, start: 401, hash: hash, version: 1, targetsJSON: "{")
        insertRecord(handle, key: budgetKey, start: 501, hash: hash, version: 1, targetsJSON: compactJSON([target(tool: "send_message", handle: String(repeating: "z", count: 3_000), kind: "agent_handle", callID: nil, rawPointer: nil, decodedPointer: "/recipient", pointerKind: "direct_raw_json")]))
        insertRecord(handle, key: emptyKey, start: 601, hash: hash, version: 1, targetsJSON: "[]")
        insertSourceRecord(
            handle,
            key: "conflict-a",
            chunk: "conflict-a",
            start: 601,
            hash: hash,
            contexts: [["attributionAgent": "workflow-subagent", "declared_message_role": "assistant"]]
        )
        insertSourceRecord(
            handle,
            key: "conflict-b",
            chunk: "conflict-b",
            start: 601,
            hash: hash,
            contexts: [["attributionAgent": "source-agent", "declared_message_role": "assistant"]]
        )
        let numericBoolean = compactJSON([[
            "tool_name": "send_message",
            "target": "numeric",
            "target_kind": "agent_handle",
            "declared_unverified": 1,
            "call_id": NSNull(),
            "raw_container_pointer": NSNull(),
            "decoded_pointer": "/recipient",
            "pointer_kind": "direct_raw_json",
        ]])
        insertRecord(handle, key: numericBooleanKey, start: 701, hash: hash, version: 1, targetsJSON: numericBoolean)
        insertRecord(handle, key: nulTargetKey, start: 801, hash: hash, version: 1, targetsJSON: compactJSON([target(tool: "send_message", handle: "bad\u{0}handle", kind: "agent_handle", callID: nil, rawPointer: nil, decodedPointer: "/recipient", pointerKind: "direct_raw_json")]))
        insertRecord(handle, key: nulSourceKey, start: 901, hash: hash, version: 1, targetsJSON: "[]")
        execute(handle, "UPDATE native_record_links SET source_file=CAST(X'2f610062' AS TEXT) WHERE record_key='\(nulSourceKey)'")
        insertRecord(handle, key: invalidUTF8Key, start: 1_001, hash: hash, version: 1, targetsJSON: "[]")
        execute(handle, "UPDATE native_record_links SET source_file=CAST(X'ff' AS TEXT) WHERE record_key='\(invalidUTF8Key)'")

        let unindexedResponse = request(router, arguments: ["source_record_key": selectedKey, "occurrence_max_bytes": 32_000])
        let unindexedEmitter = (structured(unindexedResponse)["attribution"] as? [String: Any])?["recorded_emitter"] as? [String: Any] ?? [:]
        require(unindexedEmitter["declared_agent_label"] is NSNull, "unindexed context was scanned")
        require(unindexedEmitter["declared_agent_label_status"] as? String == "unknown_missing_record_context_physical_index", "missing physical index status")

        execute(handle, "CREATE INDEX idx_chunk_source_records_physical ON chunk_source_records(source_file)")
        let wrongIndexResponse = request(router, arguments: ["source_record_key": selectedKey, "occurrence_max_bytes": 32_000])
        let wrongIndexEmitter = (structured(wrongIndexResponse)["attribution"] as? [String: Any])?["recorded_emitter"] as? [String: Any] ?? [:]
        require(wrongIndexEmitter["declared_agent_label"] is NSNull, "wrong-shaped index admitted context")
        require(wrongIndexEmitter["declared_agent_label_status"] as? String == "unknown_missing_record_context_physical_index", "wrong-shaped physical index status")
        execute(handle, "DROP INDEX idx_chunk_source_records_physical")

        createSourceRecordPhysicalIndex(handle)
        execute(handle, "ANALYZE chunk_source_records")
        let plan = sourceRecordPhysicalPlan(handle)
        require(plan.contains { $0.contains("SEARCH chunk_source_records USING INDEX idx_chunk_source_records_physical") || $0.contains("SEARCH chunk_source_records USING COVERING INDEX idx_chunk_source_records_physical") }, "physical lookup plan: \(plan)")
        require(!plan.contains { $0.contains("SCAN chunk_source_records") }, "physical lookup scanned: \(plan)")

        let selectedResponse = request(router, arguments: ["source_record_key": selectedKey, "occurrence_max_bytes": 32_000])
        let selected = structured(selectedResponse)
        require(result(selectedResponse)["isError"] as? Bool != true, "selected response errored")
        require(selected["status"] as? String == "available", "selected status")
        require(selected["source_record_key"] as? String == selectedKey, "selected key")
        require(selected["provider"] as? String == "claude", "provider")
        require(selected["session_id"] as? String == "session-target", "session")
        require(selected["source_file"] as? String == "/generated/fixture.jsonl", "source")
        require((selected["record_start_offset"] as? Int64) == 101, "start")
        require((selected["record_end_offset"] as? Int64) == 118, "end")
        require(selected["raw_record_hash"] as? String == hash, "hash")
        require(selected["target_extraction_version"] as? Int == 1, "version")
        require(selected["evidence_boundary"] as? String == "Declared target handles are unverified; no delivery, recipient identity, relay, or authorship is inferred.", "evidence boundary")
        let selectedAttribution = selected["attribution"] as? [String: Any] ?? [:]
        let selectedEmitter = selectedAttribution["recorded_emitter"] as? [String: Any] ?? [:]
        require(selectedEmitter["principal"] is NSNull, "class label promoted to principal")
        require(selectedEmitter["principal_status"] as? String == "unknown_no_distinct_identity_binding", "principal status")
        require(selectedEmitter["declared_agent_label"] as? String == "workflow-subagent", "declared agent label")
        require(selectedEmitter["declared_agent_label_status"] as? String == "declared_unverified", "label status")
        require(selectedEmitter["raw_declared_role"] as? String == "assistant", "raw role")
        require((selectedAttribution["observed_sender"] as? [String: Any])?["principal"] is NSNull, "observed sender fabricated")
        require((selectedAttribution["observed_sender"] as? [String: Any])?["status"] as? String == "unknown_no_transport_sender_evidence", "observed sender status")
        require((selectedAttribution["original_content_author"] as? [String: Any])?["principal"] is NSNull, "author fabricated")
        require((selectedAttribution["original_content_author"] as? [String: Any])?["status"] as? String == "unknown_no_authored_source_lineage", "author status")
        require((selectedAttribution["relay"] as? [String: Any])?["status"] as? String == "unknown_no_delivery_receipt", "relay status")
        let report = selected["target_report"] as? [String: Any] ?? [:]
        require(report["scope"] as? String == "supported_tool_calls_v1", "scope")
        let targets = report["targets"] as? [[String: Any]] ?? []
        require(targets.count == 2, "target count")
        require(targets[0]["target"] as? String == "task-1", "first target")
        require(targets[0]["raw_container_pointer"] as? String == "/payload/arguments", "raw pointer")
        require(targets[0]["decoded_pointer"] as? String == "/threadId", "decoded pointer")
        require(targets[1]["target"] as? String == "surface-2", "second target")
        require(Set(targets[0].keys) == Set(["tool_name", "target", "target_kind", "declared_unverified", "call_id", "raw_container_pointer", "decoded_pointer", "pointer_kind"]), "body-free target fields")
        require(!compactJSON(selectedResponse).contains("decoy"), "adjacent record leaked")
        require(text(selectedResponse).contains("task_handle task-1 via send_message [declared/unverified]"), "readable target")
        require(text(selectedResponse).contains("raw /payload/arguments; decoded /threadId"), "readable pointers")

        require(status(request(router, arguments: ["source_record_key": missingKey])) == "unavailable_source_record_not_found", "unknown key")
        let versionZeroResponse = request(router, arguments: ["source_record_key": versionZeroKey])
        require(status(versionZeroResponse) == "unavailable_not_extracted", "version zero")
        let missingEmitter = (structured(versionZeroResponse)["attribution"] as? [String: Any])?["recorded_emitter"] as? [String: Any] ?? [:]
        require(missingEmitter["declared_agent_label"] is NSNull, "missing label fabricated")
        require(missingEmitter["declared_agent_label_status"] as? String == "unknown_no_record_context", "missing label status")
        require(status(request(router, arguments: ["source_record_key": unknownVersionKey])) == "unavailable_unknown_target_version", "unknown version")
        require(status(request(router, arguments: ["source_record_key": malformedKey])) == "malformed_catalog_evidence", "malformed report")
        require(status(request(router, arguments: ["source_record_key": numericBooleanKey])) == "malformed_catalog_evidence", "numeric one accepted as true")
        require(status(request(router, arguments: ["source_record_key": nulTargetKey])) == "malformed_catalog_evidence", "embedded NUL target accepted")
        require(status(request(router, arguments: ["source_record_key": nulSourceKey])) == "malformed_catalog_evidence", "embedded NUL source truncated")
        require(status(request(router, arguments: ["source_record_key": invalidUTF8Key])) == "malformed_catalog_evidence", "invalid UTF-8 source accepted")

        let emptyResponse = request(router, arguments: ["source_record_key": emptyKey])
        require(status(emptyResponse) == "available", "empty extracted status")
        require(((structured(emptyResponse)["target_report"] as? [String: Any])?["targets"] as? [Any])?.isEmpty == true, "empty extracted targets")
        let conflictEmitter = (structured(emptyResponse)["attribution"] as? [String: Any])?["recorded_emitter"] as? [String: Any] ?? [:]
        require(conflictEmitter["principal"] is NSNull, "conflict principal fabricated")
        require(conflictEmitter["declared_agent_label"] is NSNull, "conflict label selected")
        require(conflictEmitter["declared_agent_label_status"] as? String == "unknown_conflicting_declared_agent_labels", "conflict status")
        require(text(emptyResponse).contains("Recipient absence remains unknown."), "empty interpretation")

        let cappedResponse = request(router, arguments: ["source_record_key": budgetKey, "occurrence_max_bytes": 1_024])
        require(status(cappedResponse) == "omitted_response_budget", "budget omission")
        let cappedResultBytes = try JSONSerialization.data(withJSONObject: result(cappedResponse), options: [.sortedKeys]).count
        require(cappedResultBytes <= 1_024, "serialized result exceeds requested budget: \(cappedResultBytes)")

        let ambiguity = request(router, arguments: ["chunk_id": "existing", "source_record_key": selectedKey])
        require(result(ambiguity)["isError"] as? Bool == true, "ambiguous selector accepted")
        require(text(ambiguity).contains("exactly one of chunk_id or source_record_key"), "ambiguous diagnostic")
        let absent = request(router, arguments: [:])
        require(result(absent)["isError"] as? Bool == true, "missing selector accepted")
        let invalid = request(router, arguments: ["source_record_key": "not-a-hash"])
        require(result(invalid)["isError"] as? Bool == true, "malformed selector accepted")

        try database.insertChunk(id: "existing-short", content: "existing chunk expansion body", sessionId: "existing-session", project: "fixture", contentType: "assistant_text", importance: 1)
        let chunkResponse = request(router, arguments: ["chunk_id": "existing-short", "before": 0, "after": 0])
        require(result(chunkResponse)["isError"] as? Bool != true, "existing chunk mode failed")
        require(text(chunkResponse).contains("existing chunk expansion body"), "existing chunk content missing")
        require(structured(chunkResponse).isEmpty, "chunk mode unexpectedly became source-record mode")

        try database.insertChunk(id: "attributed-short", content: "attributed chunk expansion body", sessionId: "session-target", project: "fixture", contentType: "assistant_text", importance: 1)
        insertSourceRecord(
            handle,
            key: "chunk-occurrence-context",
            chunk: "attributed-short",
            source: "/generated/chunk.jsonl",
            start: 1_201,
            hash: String(repeating: "7", count: 64),
            contexts: [["attributionAgent": "brain-worker", "declared_message_role": "user"]]
        )
        let attributedChunkResponse = request(router, arguments: ["chunk_id": "attributed-short", "before": 0, "after": 0])
        let occurrences = result(attributedChunkResponse)["source_occurrences"] as? [[String: Any]] ?? []
        let recordOccurrence = occurrences.first { $0["evidence_key"] as? String == "chunk-occurrence-context" } ?? [:]
        let occurrenceEmitter = (recordOccurrence["attribution"] as? [String: Any])?["recorded_emitter"] as? [String: Any] ?? [:]
        require(occurrenceEmitter["principal"] is NSNull, "occurrence class promoted to principal")
        require(occurrenceEmitter["declared_agent_label"] as? String == "brain-worker", "occurrence label")
        require(occurrenceEmitter["raw_declared_role"] as? String == "user", "occurrence raw role")

        let toolsResponse = router.handle(["jsonrpc": "2.0", "id": 90, "method": "tools/list"])
        let tools = (toolsResponse["result"] as? [String: Any])?["tools"] as? [[String: Any]] ?? []
        let expand = tools.first { $0["name"] as? String == "brain_expand" } ?? [:]
        let schema = expand["inputSchema"] as? [String: Any] ?? [:]
        let properties = schema["properties"] as? [String: Any] ?? [:]
        require(properties["source_record_key"] != nil, "source record selector hidden from core schema")

        print("native target record public expansion harness: PASS")
    }
}
