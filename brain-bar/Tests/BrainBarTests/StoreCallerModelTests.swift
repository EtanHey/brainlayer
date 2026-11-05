// StoreCallerModelTests.swift — #655 Part B: brain_store records the caller's
// self-reported model, on every write path, and never invents one.
//
// Everything here drives the served surface: the brain_store wire arguments, the
// durable pending-stores.jsonl file and the chunks row. Model strings are
// synthetic fixtures, not real model IDs.

import Foundation
import SQLite3
import XCTest
@testable import BrainBar

final class StoreCallerModelTests: XCTestCase {
    private let modelA = "fixture-model-a"
    private let modelB = "fixture-model-b"

    private var tempDir: URL!
    private var dbPath: String!

    override func setUpWithError() throws {
        tempDir = brainBarTestTemporaryDirectory()
            .appendingPathComponent("brainbar-caller-model-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        dbPath = tempDir.appendingPathComponent("brainbar.db").path
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDir)
    }

    private var queuePath: URL { tempDir.appendingPathComponent("pending-stores.jsonl") }

    // MARK: helpers

    /// A router whose deferred drains never run on their own, so each test decides
    /// when replay happens.
    private func makeRouter(_ db: BrainDatabase, drainQueue: DispatchQueue) -> MCPRouter {
        let router = MCPRouter(profile: "full", dbPath: dbPath, pendingStoreDrainQueue: drainQueue)
        router.setDatabase(db)
        return router
    }

    private func heldDrainQueue() -> DispatchQueue {
        let queue = DispatchQueue(label: "caller-model-held-drain-\(UUID().uuidString)")
        queue.suspend()
        return queue
    }

    private func store(
        _ router: MCPRouter,
        _ arguments: [String: Any],
        session: MCPRouter.PaletteSession? = nil,
        deferForBackup: Bool = false,
        id: Int = 1
    ) throws -> [String: Any] {
        let response = router.handle(
            [
                "jsonrpc": "2.0",
                "id": id,
                "method": "tools/call",
                "params": ["name": "brain_store", "arguments": arguments] as [String: Any],
            ],
            session: session,
            deferStoreForBackup: deferForBackup
        )
        return try XCTUnwrap(response["result"] as? [String: Any])
    }

    private func metadata(ofChunk chunkID: String) throws -> [String: Any] {
        var handle: OpaquePointer?
        guard sqlite3_open_v2(dbPath, &handle, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else {
            throw NSError(domain: "StoreCallerModelTests", code: 1)
        }
        defer { sqlite3_close(handle) }
        var statement: OpaquePointer?
        // Chunk IDs here are server-made or fixture literals; no quoting is needed.
        let sql = "SELECT metadata FROM chunks WHERE id = '\(chunkID)'"
        guard sqlite3_prepare_v2(handle, sql, -1, &statement, nil) == SQLITE_OK else {
            throw NSError(domain: "StoreCallerModelTests", code: 2)
        }
        defer { sqlite3_finalize(statement) }
        guard sqlite3_step(statement) == SQLITE_ROW, let text = sqlite3_column_text(statement, 0) else {
            throw NSError(domain: "StoreCallerModelTests", code: 3, userInfo: [NSLocalizedDescriptionKey: "no row \(chunkID)"])
        }
        let json = String(cString: text)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any], json)
    }

    private func chunkCount() throws -> Int {
        var handle: OpaquePointer?
        guard sqlite3_open_v2(dbPath, &handle, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else {
            throw NSError(domain: "StoreCallerModelTests", code: 4)
        }
        defer { sqlite3_close(handle) }
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(handle, "SELECT COUNT(*) FROM chunks", -1, &statement, nil) == SQLITE_OK,
              sqlite3_step(statement) == SQLITE_ROW else {
            sqlite3_finalize(statement)
            throw NSError(domain: "StoreCallerModelTests", code: 5)
        }
        defer { sqlite3_finalize(statement) }
        return Int(sqlite3_column_int64(statement, 0))
    }

    private func assertProvenance(
        _ metadata: [String: Any],
        model: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(metadata["model"] as? String, model, "metadata.model", file: file, line: line)
        XCTAssertEqual(
            metadata["model_source"] as? String, "caller_self_reported", "metadata.model_source", file: file, line: line
        )
    }

    private func assertNoProvenance(_ metadata: [String: Any], file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertNil(metadata["model"], "no model may be recorded: \(metadata)", file: file, line: line)
        XCTAssertNil(metadata["model_source"], "no model_source may be recorded: \(metadata)", file: file, line: line)
    }

    // MARK: schema

    func testBrainStoreSchemaAdvertisesOptionalBoundedCallerModel() throws {
        let router = MCPRouter(profile: "full")
        let response = router.handle(["jsonrpc": "2.0", "id": 1, "method": "tools/list"])
        let result = try XCTUnwrap(response["result"] as? [String: Any])
        let tools = try XCTUnwrap(result["tools"] as? [[String: Any]])
        let store = try XCTUnwrap(tools.first { ($0["name"] as? String) == "brain_store" })
        let schema = try XCTUnwrap(store["inputSchema"] as? [String: Any])
        let properties = try XCTUnwrap(schema["properties"] as? [String: Any])
        let callerModel = try XCTUnwrap(properties["caller_model"] as? [String: Any], "caller_model must be on the schema")
        XCTAssertEqual(callerModel["type"] as? String, "string")
        XCTAssertEqual(callerModel["maxLength"] as? Int, 128)
        XCTAssertEqual(schema["required"] as? [String], ["content"], "caller_model must stay optional")
    }

    // MARK: direct path

    func testDirectStoreRecordsSelfReportedModel() throws {
        // Created first so it is resumed after the database closes, as in the reliability tests.
        let drain = heldDrainQueue()
        defer { drain.resume() }
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }
        let router = makeRouter(db, drainQueue: drain)

        let result = try store(router, ["content": "direct with a model", "caller_model": "  \(modelA)  "])
        XCTAssertEqual(result["status"] as? String, "STORED")
        let chunkID = try XCTUnwrap(result["chunk_id"] as? String)
        assertProvenance(try metadata(ofChunk: chunkID), model: modelA)
    }

    func testAbsentCallerModelRecordsNothing() throws {
        // Created first so it is resumed after the database closes, as in the reliability tests.
        let drain = heldDrainQueue()
        defer { drain.resume() }
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }
        let router = makeRouter(db, drainQueue: drain)

        let result = try store(router, ["content": "direct without a model", "tags": ["agent:someone"]])
        XCTAssertEqual(result["status"] as? String, "STORED")
        assertNoProvenance(try metadata(ofChunk: try XCTUnwrap(result["chunk_id"] as? String)))
    }

    /// Untrustworthy strings are dropped, and the memory is still stored.
    func testUntrustworthyCallerModelIsDroppedButMemoryIsStored() throws {
        // Created first so it is resumed after the database closes, as in the reliability tests.
        let drain = heldDrainQueue()
        defer { drain.resume() }
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }
        let router = makeRouter(db, drainQueue: drain)

        let untrusted: [(String, String)] = [
            ("empty", ""),
            ("blank", "   \t "),
            ("synthetic placeholder", "<synthetic>"),
            ("control character", "fixture\u{0007}model"),
            ("embedded newline", "fixture\nmodel"),
            ("zero-width format character", "fixture\u{200B}model"),
            ("bidi override", "fixture\u{202E}model"),
            ("secret-shaped", "sk-" + String(repeating: "0", count: 40)),
            ("over the scalar cap via combining marks", "m" + String(repeating: "\u{0301}", count: 128)),
        ]
        for (index, (label, value)) in untrusted.enumerated() {
            let result = try store(router, ["content": "memory \(index) with \(label) model", "caller_model": value], id: index + 1)
            XCTAssertEqual(result["status"] as? String, "STORED", "\(label): the memory must still be stored")
            let chunkID = try XCTUnwrap(result["chunk_id"] as? String, label)
            assertNoProvenance(try metadata(ofChunk: chunkID))
        }
    }

    /// Wrong type or over the advertised maxLength is a bad request: REJECTED, nothing stored.
    func testMalformedOrOversizedCallerModelIsRejectedAndStoresNothing() throws {
        // Created first so it is resumed after the database closes, as in the reliability tests.
        let drain = heldDrainQueue()
        defer { drain.resume() }
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }
        let router = makeRouter(db, drainQueue: drain)

        let malformed: [(String, Any)] = [
            ("number", 42),
            ("array", [modelA]),
            ("object", ["name": modelA]),
            ("129 characters", String(repeating: "m", count: 129)),
        ]
        for (label, value) in malformed {
            let result = try store(router, ["content": "rejected \(label)", "caller_model": value])
            XCTAssertEqual(result["isError"] as? Bool, true, "\(label) must be rejected")
            XCTAssertNil(result["status"], "\(label): REJECTED carries no status")
            XCTAssertNil(result["chunk_id"], "\(label): REJECTED carries no chunk_id")
            let text = ((result["content"] as? [[String: Any]]) ?? []).compactMap { $0["text"] as? String }.joined()
            XCTAssertTrue(text.contains("REJECTED"), "\(label): \(text)")
            XCTAssertTrue(text.contains("caller_model"), "\(label): the reason must name the field: \(text)")
        }
        XCTAssertEqual(try chunkCount(), 0)
        XCTAssertFalse(FileManager.default.fileExists(atPath: queuePath.path), "a rejected store must not queue")
    }

    func testRetryInTheSameSessionIsIdempotentAndKeepsTheFirstModel() throws {
        // Created first so it is resumed after the database closes, as in the reliability tests.
        let drain = heldDrainQueue()
        defer { drain.resume() }
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }
        let router = makeRouter(db, drainQueue: drain)
        let session = router.makePaletteSession()

        let first = try store(router, ["content": "same memory", "caller_model": modelA], session: session, id: 1)
        let second = try store(router, ["content": "same memory", "caller_model": modelB], session: session, id: 2)
        XCTAssertEqual(first["status"] as? String, "STORED")
        XCTAssertEqual(second["status"] as? String, "DUPLICATE")
        let chunkID = try XCTUnwrap(first["chunk_id"] as? String)
        XCTAssertEqual(second["chunk_id"] as? String, chunkID)
        XCTAssertEqual(try chunkCount(), 1)
        assertProvenance(try metadata(ofChunk: chunkID), model: modelA)
    }

    // MARK: queue + replay

    func testBusyQueuedStoreCarriesModelThroughDurableQueueAndReplay() throws {
        // Created first so it is resumed after the database closes, as in the reliability tests.
        let drain = heldDrainQueue()
        defer { drain.resume() }
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }
        let router = makeRouter(db, drainQueue: drain)

        db.failNextStoreWithBusyForTesting = true
        let result = try store(router, ["content": "queued under a busy db", "caller_model": modelB])
        XCTAssertEqual(result["status"] as? String, "DEFERRED")
        let chunkID = try XCTUnwrap(result["chunk_id"] as? String)

        let line = try XCTUnwrap(
            String(contentsOf: queuePath, encoding: .utf8).split(separator: "\n").first.map(String.init)
        )
        let queued = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any])
        XCTAssertEqual(queued["caller_model"] as? String, modelB, "the durable queue record must keep the model")

        XCTAssertEqual(db.flushPendingStores().map(\.storedChunk.chunkID), [chunkID])
        assertProvenance(try metadata(ofChunk: chunkID), model: modelB)
        XCTAssertNotNil(try metadata(ofChunk: chunkID)["brainbar_queue_id"], "queue provenance still recorded")
    }

    func testBackupDeferredStoreCarriesModelThroughReplay() throws {
        // Created first so it is resumed after the database closes, as in the reliability tests.
        let drain = heldDrainQueue()
        defer { drain.resume() }
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }
        let router = makeRouter(db, drainQueue: drain)

        let result = try store(router, ["content": "deferred for a backup", "caller_model": modelA], deferForBackup: true)
        XCTAssertEqual(result["status"] as? String, "DEFERRED")
        let chunkID = try XCTUnwrap(result["chunk_id"] as? String)
        XCTAssertEqual(db.flushPendingStores().map(\.storedChunk.chunkID), [chunkID])
        assertProvenance(try metadata(ofChunk: chunkID), model: modelA)
    }

    func testDirectAndReplayedRowsRecordIdenticalProvenance() throws {
        // Created first so it is resumed after the database closes, as in the reliability tests.
        let drain = heldDrainQueue()
        defer { drain.resume() }
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }
        let router = makeRouter(db, drainQueue: drain)

        let direct = try store(router, ["content": "parity direct", "caller_model": modelA], id: 1)
        db.failNextStoreWithBusyForTesting = true
        let deferred = try store(router, ["content": "parity queued", "caller_model": modelA], id: 2)
        XCTAssertEqual(deferred["status"] as? String, "DEFERRED")
        let deferredID = try XCTUnwrap(deferred["chunk_id"] as? String)
        XCTAssertEqual(db.flushPendingStores().map(\.storedChunk.chunkID), [deferredID])

        let directMeta = try metadata(ofChunk: try XCTUnwrap(direct["chunk_id"] as? String))
        let replayMeta = try metadata(ofChunk: deferredID)
        XCTAssertEqual(directMeta["model"] as? String, replayMeta["model"] as? String)
        XCTAssertEqual(directMeta["model_source"] as? String, replayMeta["model_source"] as? String)
        assertProvenance(replayMeta, model: modelA)
    }

    func testLineSkippedByAFlushKeepsItsModelWhenRewritten() throws {
        // Created first so it is resumed after the database closes, as in the reliability tests.
        let drain = heldDrainQueue()
        defer { drain.resume() }
        let db = BrainDatabase(path: dbPath)
        defer { db.close() }
        let router = makeRouter(db, drainQueue: drain)

        db.failNextStoreWithBusyForTesting = true
        let result = try store(router, ["content": "kept queued once", "caller_model": modelB])
        let chunkID = try XCTUnwrap(result["chunk_id"] as? String)

        XCTAssertTrue(db.flushPendingStores(excludingChunkIDs: [chunkID]).isEmpty)
        let rewritten = try String(contentsOf: queuePath, encoding: .utf8)
        XCTAssertTrue(rewritten.contains("\"caller_model\":\"\(modelB)\""), rewritten)

        XCTAssertEqual(db.flushPendingStores().map(\.storedChunk.chunkID), [chunkID])
        assertProvenance(try metadata(ofChunk: chunkID), model: modelB)
    }

    // MARK: hand-written queue lines (old records, malformed records)

    private func writeQueueLines(_ records: [[String: Any]]) throws {
        var data = Data()
        for record in records {
            data.append(try JSONSerialization.data(withJSONObject: record))
            data.append(0x0A)
        }
        try data.write(to: queuePath)
    }

    func testQueuedRecordsReplayWithValidatedModelOnly() throws {
        try writeQueueLines([
            // Written before caller_model existed: replays exactly as before.
            ["content": "legacy line", "tags": [], "importance": 5, "source": "mcp", "chunk_id": "cm-legacy"],
            ["content": "valid model line", "tags": [], "importance": 5, "source": "mcp",
             "chunk_id": "cm-valid", "caller_model": modelA],
            // Malformed model values: the model is dropped, the memory is not stranded.
            ["content": "numeric model line", "tags": [], "importance": 5, "source": "mcp",
             "chunk_id": "cm-number", "caller_model": 42],
            ["content": "null model line", "tags": [], "importance": 5, "source": "mcp",
             "chunk_id": "cm-null", "caller_model": NSNull()],
            ["content": "control model line", "tags": [], "importance": 5, "source": "mcp",
             "chunk_id": "cm-control", "caller_model": "fixture\u{0000}model"],
            ["content": "oversized model line", "tags": [], "importance": 5, "source": "mcp",
             "chunk_id": "cm-oversized", "caller_model": String(repeating: "m", count: 4_096)],
        ])

        let db = BrainDatabase(path: dbPath)
        defer { db.close() }
        let flushed = db.flushPendingStores().map(\.storedChunk.chunkID)
        XCTAssertEqual(
            Set(flushed), ["cm-legacy", "cm-valid", "cm-number", "cm-null", "cm-control", "cm-oversized"],
            "every line must replay; a bad model never strands a memory"
        )
        XCTAssertFalse(FileManager.default.fileExists(atPath: queuePath.path), "nothing may remain queued")

        assertProvenance(try metadata(ofChunk: "cm-valid"), model: modelA)
        for chunkID in ["cm-legacy", "cm-number", "cm-null", "cm-control", "cm-oversized"] {
            assertNoProvenance(try metadata(ofChunk: chunkID))
        }

        // Replay is idempotent: a second flush finds nothing and changes nothing.
        XCTAssertTrue(db.flushPendingStores().isEmpty)
        XCTAssertEqual(try chunkCount(), 6)
        assertProvenance(try metadata(ofChunk: "cm-valid"), model: modelA)
    }
}
