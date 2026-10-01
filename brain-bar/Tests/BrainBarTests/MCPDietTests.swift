import XCTest
@testable import BrainBar

final class MCPDietTests: XCTestCase {
    private func call(_ router: MCPRouter, _ name: String, _ arguments: [String: Any]) throws -> String {
        let response = router.handle([
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": ["name": name, "arguments": arguments]
        ])
        let result = try XCTUnwrap(response["result"] as? [String: Any])
        XCTAssertNotEqual(result["isError"] as? Bool, true)
        let content = try XCTUnwrap(result["content"] as? [[String: Any]])
        return try XCTUnwrap(content.first?["text"] as? String)
    }

    func testCompactIdentifiersRoundTripAndKeepMetadata() throws {
        let path = NSTemporaryDirectory() + "diet-\(UUID().uuidString).db"
        let db = BrainDatabase(path: path)
        defer { db.close(); try? FileManager.default.removeItem(atPath: path) }
        let ids = ["rt-rollout--" + String(repeating: "a", count: 64),
                   "rt-rollout--" + String(repeating: "b", count: 64)]
        for (index, id) in ids.enumerated() {
            try db.insertChunk(id: id, content: "dietprobe unique content \(index)", sessionId: "s",
                               project: "fixture", contentType: "assistant_text", importance: 5,
                               createdAt: "2026-10-01T12:00:00Z")
        }
        let router = MCPRouter(profile: "full")
        router.setDatabase(db)
        let text = try call(router, "brain_search", ["query": "dietprobe"])
        XCTAssertTrue(text.contains("score:"), text)
        XCTAssertTrue(text.contains("project: fixture"), text)
        XCTAssertTrue(text.contains("2026-10-01"), text)
        XCTAssertFalse(text.contains("###"), text)
        let renderedIDs = text.split(separator: "\n").filter { $0.hasPrefix("- ID: ") }.map {
            String($0.dropFirst(6).components(separatedBy: " | ")[0])
        }
        XCTAssertEqual(Set(renderedIDs), Set(ids))
        for id in renderedIDs {
            let expanded = try call(router, "brain_expand", ["chunk_id": id, "before": 0, "after": 0])
            XCTAssertTrue(expanded.contains("unique content"), expanded)
        }
    }

    func testEntityAndPersonOmitOnlyEmptySections() {
        let empty = EntityCard(lookupPayload: ["name": "Fixture"])
        for text in [TextFormatter.formatEntityCard(empty), TextFormatter.formatEntitySimple(empty),
                     Formatters.formatEntityCard(entity: ["name": "Fixture"], useColor: false),
                     Formatters.formatEntitySimple(entity: ["name": "Fixture"], useColor: false)] {
            XCTAssertEqual(text, "## Entity: Fixture")
        }
    }

    func testRecallDefaultsToEightAndOverrideKeepsIDs() throws {
        let path = NSTemporaryDirectory() + "diet-recall-\(UUID().uuidString).db"
        let db = BrainDatabase(path: path)
        defer { db.close(); try? FileManager.default.removeItem(atPath: path) }
        for index in 0..<12 {
            let id = "recall-diet-\(index)"
            try db.insertChunk(id: id, content: "recall fixture body \(index)", sessionId: "diet-session",
                               project: "fixture", contentType: "assistant_text", importance: 5)
            _ = db.recordInjectionEvent(sessionID: "diet-session", query: "fixture", chunkIDs: [id],
                                        tokenCount: 10, timestamp: "2026-10-01T12:00:\(String(format: "%02d", index))Z")
        }
        let router = MCPRouter(profile: "full")
        router.setDatabase(db)
        for mode in ["context", "injections"] {
            let arguments: [String: Any] = ["mode": mode, "session_id": "diet-session"]
            let text = try call(router, "brain_recall", arguments)
            XCTAssertEqual(text.components(separatedBy: "recall-diet-").count - 1, 8, text)
            var override = arguments
            override["limit"] = 12
            let expanded = try call(router, "brain_recall", override)
            XCTAssertEqual(expanded.components(separatedBy: "recall-diet-").count - 1, 12, expanded)
        }
    }
    func testFixtureWireMeasurements() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("diet-wire-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let path = folder.appendingPathComponent("fixture.db").path
        let socketPath = "/tmp/diet-" + UUID().uuidString + ".sock"
        let db = BrainDatabase(path: path)
        defer { db.close() }
        for index in 0..<10 {
            let query = "DietEntity\(index)"
            try db.insertEntity(id: query, type: "project", name: query)
            try db.insertEntity(id: "target-\(index)", type: "project", name: "Target\(index)")
            try db.insertRelation(sourceId: query, targetId: "target-\(index)", relationType: "depends_on")
            try db.insertEntity(id: "empty-\(index)", type: "project", name: "EmptyEntity\(index)")
            try db.insertEntity(id: "person-\(index)", type: "person", name: "EmptyPerson\(index)")
            for chunk in 0..<12 {
                let id = "fixture-\(index)-\(chunk)"
                try db.insertChunk(id: id, content: "\(query) fixture decision \(chunk): preserve canonical identifiers and context.",
                                   sessionId: query, project: "fixture", contentType: "assistant_text", importance: 5,
                                   createdAt: "2026-10-01T12:00:00Z")
                _ = db.recordInjectionEvent(sessionID: query, query: query, chunkIDs: [id], tokenCount: 10)
            }
        }
        let server = BrainBarServer(socketPath: socketPath, dbPath: path, database: db,
                                    enableHybridSearchHelper: false, diagnostics: .none)
        let ready = expectation(description: "scratch server database ready")
        server.onDatabaseReady = { _ in ready.fulfill() }
        server.start()
        defer { server.stop() }
        wait(for: [ready], timeout: 10)
        let testFolder = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        let root = testFolder.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let receipt = ProcessInfo.processInfo.environment["MCP_DIET_RECEIPT"] ?? folder.appendingPathComponent("receipt.json").path
        let client = Process()
        client.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        client.arguments = ["python3", testFolder.appendingPathComponent("Scripts/mcp_diet_wire.py").path, socketPath, root.path, receipt]
        let exited = expectation(description: "stdio client completed")
        client.terminationHandler = { _ in exited.fulfill() }
        try client.run()
        wait(for: [exited], timeout: 30)
        if client.isRunning { client.terminate() }
        XCTAssertEqual(client.terminationStatus, 0)
        XCTAssertTrue(FileManager.default.fileExists(atPath: receipt))
    }

}
