import XCTest
@testable import BrainBar

private final class DietFixtureHybridClient: HybridSearchClientProtocol, @unchecked Sendable {
    private let responses: [String: [String: Any]]
    private let lock = NSLock()
    private var calls = 0

    init(path: String) throws {
        responses = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) as? [String: [String: Any]] ?? [:]
    }

    var requestCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return calls
    }

    func search(arguments: [String: Any]) throws -> HybridSearchResponse {
        lock.lock()
        calls += 1
        lock.unlock()
        let key = "\(arguments["query"] as? String ?? "")|\(arguments["detail"] as? String ?? "compact")|\(arguments["project"] != nil ? "True" : "False")"
        guard let response = responses[key], let text = response["text"] as? String else {
            throw RecordingHybridSearchClientError.injectedFailure
        }
        return HybridSearchResponse(text: text, metadata: ["structuredContent": response["structuredContent"] ?? [:]])
    }
}

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
        XCTAssertTrue(text.contains("ID|score|project|date|source"), text)
        XCTAssertTrue(text.contains("|fixture|"), text)
        XCTAssertTrue(text.contains("2026-10-01"), text)
        XCTAssertFalse(text.contains("###"), text)
        let renderedIDs = text.split(separator: "\n").filter { $0.hasPrefix("- ID: ") }.map {
            String($0.dropFirst(6).components(separatedBy: "|")[0])
        }
        XCTAssertEqual(Set(renderedIDs), Set(ids))
        for id in renderedIDs {
            let expanded = try call(router, "brain_expand", ["chunk_id": id, "before": 0, "after": 0])
            XCTAssertTrue(expanded.contains("unique content"), expanded)
        }
    }

    func testScorePrecisionAndExactSourceDeduplication() {
        let results = [
            SearchResult(chunkID: "cart-id", score: 0.1234567890123456, snippet: "content", sourceFile: "a"),
            SearchResult(chunkID: "/fixtures/same.jsonl", score: 0, snippet: "content", sourceFile: "/fixtures/same.jsonl")
        ]
        let text = TextFormatter.formatSearchResults(query: "fixture", results: results, total: 2)
        XCTAssertTrue(text.contains("|0.1235|"), text)
        XCTAssertTrue(text.contains("|0.0000|"), text)
        XCTAssertFalse(text.contains("0.123456789"), text)
        XCTAssertTrue(text.contains("|a\n"), text)
        XCTAssertFalse(text.contains("|same.jsonl"), text)
    }

    func testScopedCompactOmitsOnlyTheImpliedProject() throws {
        let path = NSTemporaryDirectory() + "diet-scope-\(UUID().uuidString).db"
        let db = BrainDatabase(path: path)
        defer { db.close(); try? FileManager.default.removeItem(atPath: path) }
        try db.insertChunk(id: "scope-id", content: "scopeprobe content", sessionId: "s",
                           project: "fixture", contentType: "assistant_text", importance: 5)
        let local = MCPRouter(profile: "full")
        local.setDatabase(db)
        let scoped = try call(local, "brain_search", ["query": "scopeprobe", "project": "fixture"])
        XCTAssertFalse(scoped.contains("|fixture|"), scoped)
        let unscoped = try call(local, "brain_search", ["query": "scopeprobe"])
        XCTAssertTrue(unscoped.contains("|fixture|"), unscoped)

        let helper = RecordingHybridSearchClient(response: HybridSearchResponse(text: "unused", metadata: [
            "structuredContent": ["total": 2, "results": [
                ["chunk_id": "scope-id", "project": "fixture", "snippet": "content"],
                ["chunk_id": "different-id", "project": "other", "snippet": "other content"]
            ]]
        ]))
        let hybrid = MCPRouter(profile: "full", hybridSearchClient: helper)
        hybrid.setDatabase(db)
        let text = try call(hybrid, "brain_search", ["query": "scopeprobe", "project": "fixture"])
        XCTAssertFalse(text.contains("|fixture|"), text)
        XCTAssertTrue(text.contains("|other|"), text)
        let full = try call(local, "brain_search", ["query": "scopeprobe", "project": "fixture", "detail": "full"])
        XCTAssertTrue(full.contains("project: fixture"), full)
    }

    func testSummaryOmittedOnlyWhenItsContentIsAlreadyVisible() {
        let results = [
            SearchResult(chunkID: "contained", summary: "shared fact", snippet: "Intro: shared fact. More preview evidence."),
            SearchResult(chunkID: "distinct", summary: "different fact", snippet: "shared fact"),
            SearchResult(chunkID: "beyond-preview", summary: "hidden fact", snippet: String(repeating: "x", count: 200) + "hidden fact"),
            SearchResult(chunkID: "normalized", summary: "shared\nfact", snippet: "Intro: shared fact."),
            SearchResult(chunkID: "long", summary: String(repeating: "x", count: 99) + "distinct tail", snippet: String(repeating: "x", count: 99) + "other tail")
        ]
        let text = TextFormatter.formatSearchResults(query: "fixture", results: results, total: results.count)
        XCTAssertFalse(text.contains("S: shared fact"), text)
        XCTAssertTrue(text.contains("S: different fact"), text)
        XCTAssertTrue(text.contains("S: hidden fact"), text)
        // The existing 100-character summary budget still applies.
        XCTAssertTrue(text.contains("P: Intro: shared fact."), text)
    }

    func testCompactDisplayFieldsEscapeColumnDelimiters() {
        let text = TextFormatter.formatSearchResults(query: "fixture", results: [
            SearchResult(chunkID: "canonical-id", project: "project|part\\name\nnext",
                         date: "bad|date", snippet: "content", sourceFile: "/fixtures/source|part.jsonl")
        ], total: 1)
        XCTAssertTrue(text.contains("project\\|part\\\\name\\nnext"), text)
        XCTAssertTrue(text.contains("bad\\|date"), text)
        XCTAssertTrue(text.contains("source\\|part.jsonl"), text)
        XCTAssertTrue(text.contains("- ID: canonical-id|"), text)
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
        let testFolder = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        let root = testFolder.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        // The release gate runs in ordinary CI; set 0 explicitly for the local FTS receipt.
        let hybridEnabled = ProcessInfo.processInfo.environment["MCP_DIET_HYBRID"] != "0"
        var hybridClient: DietFixtureHybridClient?
        if hybridEnabled {
            let fixturePath = folder.appendingPathComponent("hybrid.json").path
            let builder = Process()
            builder.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            var environment = ProcessInfo.processInfo.environment
            environment["BRAINLAYER_MCP_SOCKET"] = socketPath
            environment["BRAINLAYER_FORBID_BRAINBAR_SOCKET"] = "1"
            builder.environment = environment
            builder.arguments = ["python3", testFolder.appendingPathComponent("Scripts/mcp_diet_wire.py").path,
                                 "--helper-fixture", root.path, fixturePath]
            try builder.run()
            builder.waitUntilExit()
            XCTAssertEqual(builder.terminationStatus, 0)
            hybridClient = try DietFixtureHybridClient(path: fixturePath)
        }
        let server = BrainBarServer(socketPath: socketPath, dbPath: path, database: db,
                                    hybridSearchClient: hybridClient, enableHybridSearchHelper: hybridEnabled, diagnostics: .none)
        let ready = expectation(description: "scratch server database ready")
        server.onDatabaseReady = { _ in ready.fulfill() }
        server.start()
        defer { server.stop() }
        wait(for: [ready], timeout: 10)
        let receipt = ProcessInfo.processInfo.environment["MCP_DIET_RECEIPT"] ?? folder.appendingPathComponent("receipt.json").path
        let client = Process()
        client.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        var environment = ProcessInfo.processInfo.environment
        environment["BRAINLAYER_MCP_SOCKET"] = socketPath
        environment["BRAINLAYER_FORBID_BRAINBAR_SOCKET"] = "1"
        environment["MCP_DIET_HYBRID"] = hybridEnabled ? "1" : "0"
        client.environment = environment
        client.arguments = ["python3", testFolder.appendingPathComponent("Scripts/mcp_diet_wire.py").path, socketPath, root.path, receipt]
        let exited = expectation(description: "stdio client completed")
        client.terminationHandler = { _ in exited.fulfill() }
        try client.run()
        wait(for: [exited], timeout: 30)
        if client.isRunning { client.terminate() }
        XCTAssertEqual(client.terminationStatus, 0)
        XCTAssertTrue(FileManager.default.fileExists(atPath: receipt))
        if let hybridClient {
            XCTAssertEqual(hybridClient.requestCount, 30, "Every search must traverse the fixture hybrid helper")
            var payload = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: receipt))) as? [String: Any] ?? [:]
            payload["hybrid_request_count"] = hybridClient.requestCount
            try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: receipt))
            let rows = try XCTUnwrap(payload["rows"] as? [[String: Any]])
            for (item, ceiling) in [("1 compact IDs", 27_440), ("2 KG default", 29_110)] {
                let measured = rows.filter { $0["item"] as? String == item }
                XCTAssertEqual(measured.count, 10)
                let bytes = try measured.reduce(0) { try $0 + XCTUnwrap($1["response_bytes"] as? Int) }
                XCTAssertLessThanOrEqual(bytes, ceiling, "#1060 hybrid release size gate: \(item)")
            }
        }
    }

}
