import XCTest
import SQLite3
@testable import BrainBar

final class ReturnedChunkIDContractTests: XCTestCase {
    private func call(_ router: MCPRouter, _ name: String, _ arguments: [String: Any]) -> [String: Any] {
        router.handle([
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": ["name": name, "arguments": arguments] as [String: Any]
        ])
    }

    private func text(_ response: [String: Any], isError: Bool = false) throws -> String {
        XCTAssertNil(response["error"])
        let result = try XCTUnwrap(response["result"] as? [String: Any])
        XCTAssertEqual(result["isError"] as? Bool ?? false, isError)
        let content = try XCTUnwrap(result["content"] as? [[String: Any]])
        return content.compactMap { $0["text"] as? String }.joined(separator: "\n")
    }

    private func withDatabase(_ body: (BrainDatabase, MCPRouter) throws -> Void) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let database = BrainDatabase(path: directory.appendingPathComponent("fixture.db").path)
        defer { database.close() }
        let router = MCPRouter(profile: "full")
        router.setDatabase(database)
        try body(database, router)
    }

    private func insert(_ db: BrainDatabase, _ id: String, _ content: String) throws {
        try db.insertChunk(id: id, content: content, sessionId: "synthetic-contract",
                           project: "brainlayer", contentType: "assistant_text", importance: 5)
    }

    private func importance(_ db: BrainDatabase, _ id: String) throws -> Int {
        var statement: OpaquePointer?
        XCTAssertEqual(sqlite3_prepare_v2(db.dbHandle, "SELECT importance FROM chunks WHERE id = ?", -1, &statement, nil), SQLITE_OK)
        defer { sqlite3_finalize(statement) }
        let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        XCTAssertEqual(sqlite3_bind_text(statement, 1, id, -1, transient), SQLITE_OK)
        XCTAssertEqual(sqlite3_step(statement), SQLITE_ROW)
        return Int(sqlite3_column_int(statement, 0))
    }

    func testSearchReturned142CharacterPathIDExpandsExactly() throws {
        try withDatabase { db, router in
            // Legacy index_new producer: full source_file + ':' + chunk position.
            let id = "/synthetic/" + String(repeating: "p", count: 123) + ".jsonl:0"
            XCTAssertEqual(id.count, 142)
            try insert(db, id, "contractneedle synthetic technical fixture")
            for detail in ["compact", "full"] {
                let search = try text(call(router, "brain_search", ["query": "contractneedle", "detail": detail]))
                let header = try XCTUnwrap(search.split(separator: "\n").first { $0.hasPrefix("- ID: ") })
                let returned = String(header.dropFirst(6).components(separatedBy: detail == "full" ? " | score:" : "|")[0])
                XCTAssertEqual(returned, id)
                let expanded = try text(call(router, "brain_expand", ["chunk_id": returned, "before": 0, "after": 0]))
                XCTAssertTrue(expanded.contains("brain_expand: \(id)"))
                XCTAssertTrue(expanded.contains("contractneedle synthetic technical fixture"))
            }
        }
    }

    func testLongOpaqueIDsPreservedThroughExpandUpdateSupersedeAndArchive() throws {
        try withDatabase { db, router in
            // A 4096-character path plus a 19-digit position fits the protocol budget.
            for length in [142, 4_116, 8_192] {
                let prefix = String(repeating: "x", count: length - 1)
                let old = prefix + "a", new = prefix + "b"
                try insert(db, old, "old technical fixture \(length)")
                try insert(db, new, "new technical fixture \(length)")
                let expanded = try text(call(router, "brain_expand", ["chunk_id": new, "before": 0, "after": 0]))
                XCTAssertTrue(expanded.contains("brain_expand: \(new)"))
                XCTAssertTrue(expanded.contains("new technical fixture \(length)"))
                _ = try text(call(router, "brain_update", ["chunk_id": new, "importance": 8]))
                XCTAssertEqual(try importance(db, new), 8)
                _ = try text(call(router, "brain_supersede", ["old_chunk_id": old, "new_chunk_id": new,
                                                               "safety_check": "confirm", "confirm": true]))
                XCTAssertEqual(try db.getChunk(id: old)?["superseded_by"] as? String, new)
                _ = try text(call(router, "brain_archive", ["chunk_id": new]))
                XCTAssertNotNil(try db.getChunk(id: new)?["archived_at"] as? String)
                XCTAssertEqual(try db.getChunk(id: old)?["content"] as? String, "old technical fixture \(length)")
            }
        }
    }

    func testOverflowRejectedBeforeEveryIdentifierHandlerWithoutMutation() throws {
        try withDatabase { db, router in
            let id = String(repeating: "z", count: 8_193)
            try insert(db, id, "overflow technical fixture")
            try insert(db, "short", "short technical fixture")
            let attempts: [(String, [String: Any], String)] = [
                ("brain_expand", ["chunk_id": id], "chunk_id"),
                ("brain_update", ["chunk_id": id, "importance": 9], "chunk_id"),
                ("brain_archive", ["chunk_id": id], "chunk_id"),
                ("brain_supersede", ["old_chunk_id": id, "new_chunk_id": "short"], "old_chunk_id"),
                ("brain_supersede", ["old_chunk_id": "short", "new_chunk_id": id], "new_chunk_id")
            ]
            for (tool, arguments, field) in attempts {
                let error = try text(call(router, tool, arguments), isError: true)
                XCTAssertTrue(error.contains("\(field) length 8193 exceeds maxLength 8192"), error)
            }
            XCTAssertEqual(try importance(db, id), 5)
            XCTAssertNil(try db.getChunk(id: id)?["archived_at"] as? String)
            XCTAssertNil(try db.getChunk(id: id)?["superseded_by"] as? String)
            XCTAssertNil(try db.getChunk(id: "short")?["superseded_by"] as? String)
        }
    }

    func testAdvertisedIdentifierCapsAndUnrelatedResourceCaps() throws {
        let router = MCPRouter(profile: "full")
        let response = router.handle(["jsonrpc": "2.0", "id": 2, "method": "tools/list"])
        let tools = try XCTUnwrap((response["result"] as? [String: Any])?["tools"] as? [[String: Any]])
        var identifiers = 0
        for tool in tools {
            let schema = try XCTUnwrap(tool["inputSchema"] as? [String: Any])
            let properties = try XCTUnwrap(schema["properties"] as? [String: [String: Any]])
            for (name, field) in properties {
                if ["chunk_id", "old_chunk_id", "new_chunk_id"].contains(name) {
                    XCTAssertEqual(field["maxLength"] as? Int, 8_192)
                    identifiers += 1
                }
                if name == "query" { XCTAssertEqual(field["maxLength"] as? Int, 4_096) }
                if name == "content" { XCTAssertEqual(field["maxLength"] as? Int, 200_000) }
                if name == "agent_id" || name == "session_id" || name == "tag" {
                    XCTAssertEqual(field["maxLength"] as? Int, 128)
                }
                if name == "tags" {
                    XCTAssertEqual(field["maxItems"] as? Int, 100)
                    XCTAssertEqual((field["items"] as? [String: Any])?["maxLength"] as? Int, 128)
                }
            }
        }
        XCTAssertEqual(identifiers, 5)
        let error = try text(call(router, "brain_search", ["query": String(repeating: "q", count: 4_097)]), isError: true)
        XCTAssertTrue(error.contains("query length 4097 exceeds maxLength 4096"))
    }
}
